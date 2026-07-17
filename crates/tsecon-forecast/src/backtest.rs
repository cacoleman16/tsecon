//! The pseudo-out-of-sample backtesting engine: a model-agnostic
//! rolling-origin driver over expanding and fixed-width rolling schemes.
//!
//! # The alignment convention (pinned)
//!
//! Given a series `y[0..n]`, a forecast **origin** `t` is a time index such
//! that the forecaster has observed `y[0..=t]` (data up to and *including*
//! index `t`) and produces multi-step point forecasts for the future targets
//! `y[t+1], ..., y[t+H]`, where `H` is [`Backtest::horizon`]. Concretely:
//!
//! * **The forecast made at origin `t` for horizon `h` is compared to the
//!   realized value at index `t + h`** (`h` in `1..=H`). The forecast error
//!   stored is `y[t+h] - yhat`, the crate-wide `actual - forecast` sign.
//! * The **training slice** handed to the forecaster at origin `t` is
//!   - expanding ([`Window::Expanding`]): `&y[0..=t]` (grows with `t`);
//!   - rolling ([`Window::Rolling`] with `width`): `&y[t+1-width..=t]` (the
//!     most recent `width` observations ending at `t`).
//!
//! # Which origins are evaluated (rectangular by design)
//!
//! Let `t0` be the first origin whose training window is full:
//! `t0 = min_train - 1` (expanding) or `t0 = width - 1` (rolling). Every
//! origin from `t0` up to and including `n - 1 - H` is evaluated, so that
//! **every** included origin has all `H` targets in sample. The result is
//! therefore a rectangular `origins x horizons` grid: `errors(h)`,
//! `forecasts(h)`, and `targets(h)` all have exactly one entry per origin,
//! for every `h in 1..=H`. Keeping every horizon's evaluation sample
//! identical is what makes the per-horizon loss vectors directly comparable
//! and lets the Diebold-Mariano / Clark-West / Giacomini-White tests consume
//! equal-length, index-aligned error streams. (Origins in `n-H..=n-2`, which
//! could support only shorter horizons, are deliberately dropped rather than
//! producing ragged per-horizon samples.)
//!
//! # Refit cadence (`refit_every`)
//!
//! `refit_every = k` re-estimates the forecaster only once every `k`
//! consecutive origins; between refits its multi-step forecasts are *reused*.
//! Precisely: at a **refit origin** `t_r` (every `k`-th origin, starting at
//! `t0`) the forecaster closure is called **once**, on the training window
//! `window(t_r)`, and asked for enough horizons to cover its whole block —
//! `(block_len - 1) + H` steps, where `block_len <= k` is the number of
//! origins the block spans. For an origin `t_r + s` (`0 <= s < block_len`)
//! inside the block, the horizon-`h` forecast is taken from the `t_r` call at
//! step `s + h`, i.e. the model fit at `t_r` is *not* re-estimated but its
//! direct multi-step path is rolled forward. With `k = 1` (the default and
//! the leakage-free ideal) the closure is called at every origin with exactly
//! `H` horizons, and the forecast at origin `t` uses `window(t)`.
//!
//! All preprocessing that must not peek at the future — scaling, seasonal
//! adjustment, transformation choice, hyperparameter tuning — belongs
//! *inside* the forecaster closure so it only ever sees the training slice;
//! the engine never hands future observations to the closure.

use crate::accuracy::{mae, mape, mase, mdae, me, mse, rmse, rmsse, smape};
use crate::comparison::AccuracyRow;
use crate::error::ForecastError;
use crate::validate::check_finite;

/// The out-of-sample estimation scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// Expanding (recursive) window: the training set always starts at index
    /// `0` and grows by one observation per origin. `min_train` is the number
    /// of observations in the first training window, so the first origin is
    /// index `min_train - 1`.
    Expanding {
        /// Observations in the first (smallest) training window; must be
        /// `>= 1`.
        min_train: usize,
    },
    /// Fixed-width rolling window: the training set is always the most recent
    /// `width` observations ending at the origin. The first origin is index
    /// `width - 1`. Fixed-width estimation is the scheme the Giacomini-White
    /// test's asymptotics require (estimation error does not vanish).
    Rolling {
        /// The (constant) number of observations in every training window;
        /// must be `>= 1`.
        width: usize,
    },
}

impl Window {
    /// Index of the first origin whose training window is full.
    fn first_origin(self) -> usize {
        match self {
            Window::Expanding { min_train } => min_train - 1,
            Window::Rolling { width } => width - 1,
        }
    }

    /// The training slice `&y[..]` for an origin `t` under this scheme.
    fn train_slice(self, y: &[f64], t: usize) -> &[f64] {
        match self {
            Window::Expanding { .. } => &y[..=t],
            Window::Rolling { width } => &y[t + 1 - width..=t],
        }
    }
}

/// A configured pseudo-out-of-sample backtest.
///
/// Build with [`Backtest::new`] (which validates the scheme) and run it over
/// a series with [`Backtest::run`], supplying a forecaster closure. See the
/// [module docs](self) for the alignment convention and refit contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backtest {
    /// The estimation scheme (expanding or fixed-width rolling).
    pub window: Window,
    /// The maximum forecast horizon `H`; horizons `1..=H` are all evaluated.
    pub horizon: usize,
    /// Refit cadence: re-estimate the forecaster once every `refit_every`
    /// origins (`1` = refit at every origin).
    pub refit_every: usize,
}

impl Backtest {
    /// Configure a backtest, validating the scheme parameters.
    ///
    /// # Errors
    ///
    /// [`ForecastError::InvalidBacktestParam`] if `min_train`/`width` is `0`,
    /// `horizon` is `0`, or `refit_every` is `0`.
    pub fn new(window: Window, horizon: usize, refit_every: usize) -> Result<Self, ForecastError> {
        let bt = Backtest {
            window,
            horizon,
            refit_every,
        };
        bt.validate()?;
        Ok(bt)
    }

    fn validate(&self) -> Result<(), ForecastError> {
        match self.window {
            Window::Expanding { min_train } if min_train == 0 => {
                return Err(ForecastError::InvalidBacktestParam {
                    what: "min_train",
                    value: min_train,
                    requirement: "min_train >= 1 (the first training window)",
                });
            }
            Window::Rolling { width } if width == 0 => {
                return Err(ForecastError::InvalidBacktestParam {
                    what: "width",
                    value: width,
                    requirement: "width >= 1 (the rolling training window)",
                });
            }
            _ => {}
        }
        if self.horizon == 0 {
            return Err(ForecastError::InvalidBacktestParam {
                what: "horizon",
                value: self.horizon,
                requirement: "horizon >= 1",
            });
        }
        if self.refit_every == 0 {
            return Err(ForecastError::InvalidBacktestParam {
                what: "refit_every",
                value: self.refit_every,
                requirement: "refit_every >= 1 (1 refits at every origin)",
            });
        }
        Ok(())
    }

    /// Run the backtest over `y`, driving `forecaster` per the refit contract.
    ///
    /// `forecaster` is called as `forecaster(train, max_h)` and must return a
    /// `Vec<f64>` of exactly `max_h` point forecasts for horizons `1..=max_h`
    /// from the end of the `train` slice. It may carry mutable state (it is
    /// `FnMut`); its own errors propagate unchanged. See the
    /// [module docs](self) for the training-slice and refit semantics.
    ///
    /// # Errors
    ///
    /// [`ForecastError::NonFinite`] if `y` has NaN/inf values,
    /// [`ForecastError::NoBacktestOrigins`] if the scheme leaves no evaluable
    /// origin, [`ForecastError::ForecasterOutputLen`] if the closure returns
    /// the wrong number of forecasts, plus whatever error the closure itself
    /// returns.
    pub fn run<F>(&self, y: &[f64], mut forecaster: F) -> Result<BacktestResult, ForecastError>
    where
        F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
    {
        self.validate()?;
        check_finite(y, "backtest series")?;
        let n = y.len();
        let h = self.horizon;
        let t0 = self.window.first_origin();

        // Last origin needs all H targets in sample: t + H <= n - 1.
        // Guard the subtraction (n may be smaller than H + 1).
        if n < h + 1 || t0 > n - 1 - h {
            return Err(ForecastError::NoBacktestOrigins {
                n,
                first_origin: t0,
                horizon: h,
            });
        }
        let last_origin = n - 1 - h;
        let p = last_origin - t0 + 1; // number of origins

        let mut origins = Vec::with_capacity(p);
        let mut forecasts: Vec<Vec<f64>> = vec![Vec::with_capacity(p); h];
        let mut targets: Vec<Vec<f64>> = vec![Vec::with_capacity(p); h];

        // Walk refit blocks of up to `refit_every` origins.
        let mut i = 0usize;
        while i < p {
            let block_len = self.refit_every.min(p - i);
            let t_r = t0 + i; // refit origin
            let max_h = (block_len - 1) + h;
            let train = self.window.train_slice(y, t_r);
            let fc = forecaster(train, max_h)?;
            if fc.len() != max_h {
                return Err(ForecastError::ForecasterOutputLen {
                    origin: t_r,
                    expected: max_h,
                    actual: fc.len(),
                });
            }
            check_finite(&fc, "backtest forecaster output")?;

            for s in 0..block_len {
                let t = t_r + s; // this origin
                origins.push(t);
                for hh in 1..=h {
                    let forecast = fc[s + hh - 1];
                    let target = y[t + hh];
                    forecasts[hh - 1].push(forecast);
                    targets[hh - 1].push(target);
                }
            }
            i += block_len;
        }

        let first_train = self.window.train_slice(y, t0).to_vec();
        Ok(BacktestResult {
            horizon: h,
            origins,
            forecasts,
            targets,
            first_train,
        })
    }
}

/// The tidy output of a [`Backtest::run`]: per-horizon, origin-aligned
/// forecasts, targets, and errors, plus the origin indices.
///
/// Every horizon `h in 1..=horizon` exposes vectors of the same length —
/// one entry per forecast origin, in ascending origin order (see the
/// [module docs](crate::backtest) for the alignment convention).
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestResult {
    horizon: usize,
    origins: Vec<usize>,
    forecasts: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
    first_train: Vec<f64>,
}

impl BacktestResult {
    /// The maximum horizon `H` evaluated (horizons `1..=H` are available).
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// The number of forecast origins (rows of the `origins x horizons` grid).
    #[must_use]
    pub fn n_origins(&self) -> usize {
        self.origins.len()
    }

    /// The forecast origin indices `t`, ascending. The horizon-`h` forecast
    /// at row `i` was made at origin `origins()[i]` and targets index
    /// `origins()[i] + h`.
    #[must_use]
    pub fn origins(&self) -> &[usize] {
        &self.origins
    }

    fn check_h(&self, h: usize) -> Result<usize, ForecastError> {
        if h == 0 || h > self.horizon {
            return Err(ForecastError::HorizonOutOfRange {
                h,
                max_h: self.horizon,
            });
        }
        Ok(h - 1)
    }

    /// The point forecasts for horizon `h`, one per origin.
    ///
    /// # Errors
    ///
    /// [`ForecastError::HorizonOutOfRange`] if `h` is not in `1..=horizon`.
    pub fn forecasts(&self, h: usize) -> Result<&[f64], ForecastError> {
        Ok(&self.forecasts[self.check_h(h)?])
    }

    /// The realized targets `y[t + h]` for horizon `h`, one per origin.
    ///
    /// # Errors
    ///
    /// [`ForecastError::HorizonOutOfRange`] if `h` is not in `1..=horizon`.
    pub fn targets(&self, h: usize) -> Result<&[f64], ForecastError> {
        Ok(&self.targets[self.check_h(h)?])
    }

    /// The forecast errors `y[t + h] - yhat` for horizon `h`, one per origin
    /// (the crate-wide `actual - forecast` sign).
    ///
    /// # Errors
    ///
    /// [`ForecastError::HorizonOutOfRange`] if `h` is not in `1..=horizon`.
    pub fn errors(&self, h: usize) -> Result<Vec<f64>, ForecastError> {
        let idx = self.check_h(h)?;
        Ok(self.targets[idx]
            .iter()
            .zip(self.forecasts[idx].iter())
            .map(|(&y, &f)| y - f)
            .collect())
    }

    /// A per-horizon accuracy table computed with the crate's point-accuracy
    /// measures ([`crate::accuracy`]).
    ///
    /// Row `h - 1` labels the horizon `"h={h}"` and reports every measure of
    /// the horizon-`h` forecasts against their targets. The scaled errors
    /// (MASE/RMSSE) are scaled by the in-sample seasonal-naive error of the
    /// **first training window** at the seasonal period `insample_period`
    /// (`1` for non-seasonal data) — never the evaluation sample, which would
    /// leak the test set into the scale. As in [`crate::comparison`], MAPE and
    /// sMAPE are `None` where undefined (zero actuals / zero denominators)
    /// rather than failing the whole table.
    ///
    /// # Errors
    ///
    /// The validation errors of the scaled measures — in particular
    /// [`ForecastError::InvalidPeriod`] or [`ForecastError::SeriesTooShort`]
    /// if `insample_period` is too large for the first training window, and
    /// [`ForecastError::ZeroScaleDenominator`] if that window is seasonally
    /// constant.
    pub fn accuracy_table(
        &self,
        insample_period: usize,
    ) -> Result<Vec<AccuracyRow>, ForecastError> {
        let mut rows = Vec::with_capacity(self.horizon);
        for h in 1..=self.horizon {
            let idx = h - 1;
            let actual = &self.targets[idx];
            let forecast = &self.forecasts[idx];
            let mape_v = match mape(actual, forecast) {
                Ok(v) => Some(v),
                Err(ForecastError::ZeroActualInMape { .. }) => None,
                Err(e) => return Err(e),
            };
            let smape_v = match smape(actual, forecast) {
                Ok(v) => Some(v),
                Err(ForecastError::ZeroDenominatorInSmape { .. }) => None,
                Err(e) => return Err(e),
            };
            rows.push(AccuracyRow {
                name: format!("h={h}"),
                me: me(actual, forecast)?,
                mse: mse(actual, forecast)?,
                rmse: rmse(actual, forecast)?,
                mae: mae(actual, forecast)?,
                mdae: mdae(actual, forecast)?,
                mape: mape_v,
                smape: smape_v,
                mase: Some(mase(actual, forecast, &self.first_train, insample_period)?),
                rmsse: Some(rmsse(actual, forecast, &self.first_train, insample_period)?),
            });
        }
        Ok(rows)
    }
}
