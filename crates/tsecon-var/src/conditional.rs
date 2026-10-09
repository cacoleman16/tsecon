//! Conditional forecasts of a VAR(p) — the Doan-Litterman-Sims (1984) /
//! Waggoner-Zha (1999) closed form for hard conditions, and its two
//! Gaussian soft-condition readings (Andersson-Palmqvist-Waggoner 2010;
//! Monti 2010).
//!
//! A conditional forecast asks what the model expects for the *free* cells of
//! the future path when some *constrained* cells are pinned to given values
//! ("inflation follows this path for four quarters; what happens to output?").
//! With Gaussian innovations and the coefficients treated as known, the
//! answer is Gaussian conditioning of the joint forecast-error distribution.
//! Writing the `h`-step forecast error through the MA coefficients
//! (Lütkepohl 2005, eq. 2.2.10),
//!
//! ```text
//! y_{T+h} - ŷ_{T+h|T} = Σ_{s=1}^{h} Ψ_{h-s} u_{T+s},        u_t ~ N(0, Σ_u),
//! ```
//!
//! stack the future innovations `u = (u_{T+1}', ..., u_{T+H}')'` and write the
//! constraints as `B u = r`: each row of `B` is the row of the
//! block-lower-triangular MA matrix `R` (block `(h, s) = Ψ_{h-s}`, `s <= h`)
//! that belongs to one constrained `(horizon, series)` cell, and `r` stacks
//! the gaps `condition - unconditional forecast`. With `Σ = I_H ⊗ Σ_u`,
//!
//! ```text
//! u*  = Σ B' (B Σ B')^{-1} r                     the implied shocks,
//! ỹ   = ŷ + R u*                                 the conditional mean path,
//! V   = R (Σ - Σ B' (B Σ B')^{-1} B Σ) R'        the conditional covariance.
//! ```
//!
//! `u*` is at once the minimum-norm shock sequence in the `Σ`-metric
//! (Doan-Litterman-Sims 1984), the Waggoner-Zha (1999) solution whose
//! structural version is `ε* = (I ⊗ P^{-1}) u*` for any factor `P P' = Σ_u`
//! (the path and covariance do not depend on `P`; only the reported
//! structural shocks do), and the conditional expectation `E[u | B u = r]`.
//! Because it is a conditional expectation, exactly the same path and
//! covariance come out of a Kalman smoother run over the future with the
//! unconstrained cells set missing — the Bańbura-Giannone-Lenza (2015)
//! route — which is the second, independent golden leg pinned in
//! `fixtures/generate_var_cf_fixtures.py` (statsmodels `VARMAX(...).smooth`).
//!
//! **Soft (Gaussian, stated-variance) conditions**
//! ([`VarResults::conditional_forecast_soft`]) give conditioned cell `i` a
//! standard deviation `sd_i` (`Ω = diag(sd_i²)`, `sd_i = 0` for a hard
//! cell). The two readings disagree on what the typed value *is*, so the
//! caller names one ([`SoftMode`]):
//!
//! * **density** (Andersson-Palmqvist-Waggoner 2010): the conditioned cells'
//!   forecast distribution IS `N(value, Ω)`, the mixture over `c ~ N(value,
//!   Ω)` of the hard conditional at `c`. The path, the shocks and the
//!   Mahalanobis statistic are the hard ones at the central values, and the
//!   covariance gains the spread of the stated distribution through the gain
//!   `X_h = (B Σ B')^{-1} G_h'`:
//!   `cov_h = MSE(h) - G_h X_h + Σ_i sd_i² X_h[i,:]' X_h[i,:]`, with
//!   `G = R Σ B'`. A soft cell's point is its value and its se its `sd`;
//! * **signal** (Monti 2010): the value is a noisy reading `z = B y + e`,
//!   `e ~ N(0, Ω)` independent of the innovations, so `B Σ B'` becomes
//!   `M = B Σ B' + Ω` throughout: `u* = Σ B' M^{-1} r = E[u | z]`,
//!   `cov_h = MSE(h) - G_h M^{-1} G_h'`, Mahalanobis `r' M^{-1} r ~ χ²(m)`.
//!   A soft cell is the posterior mean, not its typed value — exactly what a
//!   Kalman smoother returns with measurement-noise variance `sd²` on that
//!   cell, the statsmodels leg of `fixtures/generate_var_cf_soft_fixtures.py`.
//!   The soft × soft block of a horizon is taken from the equal,
//!   cancellation-free `Cov(B y | z) = Ω M^{-1} B Σ B'`, so a soft cell
//!   whose sd is far below its forecast se keeps a correct se.
//!
//! Both reduce to the hard closed form at `sd = 0`, and an all-zero (or
//! all-`None`) `condition_sd` *is* the hard path — the same code with no
//! soft term touched.
//!
//! What this is not: it is a coefficients-known object. Parameter
//! uncertainty (the Bayesian per-draw version) and interval/range
//! (truncated, Waggoner-Zha) conditions are out of scope here, and the
//! density reading is a plug-in conditioning, not a Bayesian tilt. The
//! squared `Σ`-norm of the implied shocks, `r' (B Σ B')^{-1} r = ε*'ε*`, is
//! reported with its `χ²(m)` tail probability as a plausibility check of the
//! conditioning path against the model's own unconditional forecast
//! distribution (Waggoner-Zha's measure of how hard the model has to be
//! pushed).

use tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::VarError;
use crate::results::{chol_lower, VarResults};

/// Cap on the number of `f64` the working set may hold (the `B`, `W = Σ B'`
/// and `G = R W` matrices are each `steps·k × m` with `m` constrained
/// cells, plus the `m × m` Gram matrix and the `steps` MA/MSE matrices):
/// 2^24 doubles is 128 MB, far above any policy-scenario use and far below
/// what would make the allocator abort on a `steps` typo.
const MAX_WORKING_DOUBLES: usize = 1 << 24;

/// Which reading of a stated standard deviation a soft condition means
/// (module docs for the closed forms). There is no default: the two
/// disagree on whether a soft cell's point equals its typed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftMode {
    /// Andersson-Palmqvist-Waggoner (2010): the conditioned cell's forecast
    /// distribution IS `N(value, sd²)` — the point is the value and the se
    /// is `sd`, exactly.
    Density,
    /// Monti (2010): the value is a noisy reading `y + N(0, sd²)` of the
    /// cell — point and se are the posterior given the readings, and the
    /// cell is not pinned.
    Signal,
}

/// The conditioning a [`ConditionalForecast`] was computed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionMode {
    /// Every conditioned cell is hard (no standard deviation above 0).
    Hard,
    /// At least one soft cell, read as [`SoftMode::Density`].
    Density,
    /// At least one soft cell, read as [`SoftMode::Signal`].
    Signal,
}

impl ConditionMode {
    /// `"hard"`, `"density"` or `"signal"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Density => "density",
            Self::Signal => "signal",
        }
    }
}

/// A conditional forecast produced by [`VarResults::conditional_forecast`]
/// or [`VarResults::conditional_forecast_soft`].
///
/// Every `steps × k` matrix has row `h` for horizon `h + 1` and one column
/// per series, the layout of [`VarResults::forecast`].
#[derive(Debug, Clone)]
pub struct ConditionalForecast {
    /// Forecast horizon `H` (rows of every path matrix).
    pub steps: usize,
    /// Interval level `alpha` the bounds were built at (`1 - alpha` coverage).
    pub alpha: f64,
    /// Conditional mean path `ỹ`. Hard cells, and every conditioned cell in
    /// density mode, hold their condition exactly (set bitwise, not left to
    /// rounding); a signal-mode soft cell is the posterior mean given the
    /// readings, not its typed value.
    pub point: Mat<f64>,
    /// The unconditional path `ŷ` — identical to [`VarResults::forecast`].
    pub unconditional: Mat<f64>,
    /// Conditional forecast-error covariance at each horizon, `k × k` per
    /// entry: the diagonal blocks of `V`. Hard cells have an exactly zero
    /// row and column; in density mode a soft cell has `sd²` on the diagonal
    /// and an exact 0 against every other conditioned cell of its horizon.
    pub cov: Vec<Mat<f64>>,
    /// `sqrt(diag cov[h])`, `steps × k`; exactly zero at hard cells and
    /// exactly `sd` at a density-mode soft cell.
    pub se: Mat<f64>,
    /// `sqrt(diag MSE(h))` of the unconditional forecast, for reading the
    /// variance reduction the conditions buy.
    pub unconditional_se: Mat<f64>,
    /// `point - z_{1 - alpha/2} se`.
    pub lower: Mat<f64>,
    /// `point + z_{1 - alpha/2} se`.
    pub upper: Mat<f64>,
    /// Implied reduced-form innovations `u*`, `steps × k` (row `s` is
    /// `u_{T+s+1}`): the minimum-`Σ`-norm shock sequence that delivers the
    /// conditions, equivalently `E[u | conditions]` (in signal mode
    /// `E[u | readings]`; in density mode the hard shocks at the central
    /// values).
    pub shocks: Mat<f64>,
    /// The same shocks orthogonalised by the lower Cholesky factor of
    /// `Σ_u` in the data's column order, `ε*_s = P^{-1} u*_s` — the
    /// Waggoner-Zha structural shocks under a recursive identification.
    /// Only this field depends on the ordering.
    pub orth_shocks: Mat<f64>,
    /// `constrained[h][j]` is `true` iff cell `(h, j)` was conditioned on,
    /// hard or soft.
    pub constrained: Vec<Vec<bool>>,
    /// Number of conditioned cells `m`, hard and soft.
    pub n_constrained: usize,
    /// `r' M^{-1} r`, with `M = B Σ B'` (hard and density modes) or
    /// `B Σ B' + Ω` (signal mode): the Mahalanobis distance of the
    /// conditioned values from the unconditional forecast under the model's
    /// own forecast distribution (of the readings, in signal mode). In
    /// density mode it measures the central path only and ignores `sd`.
    pub mahalanobis: f64,
    /// `P(χ²(m) > mahalanobis)`: small when the conditioning path is one the
    /// model would rarely generate on its own.
    pub mahalanobis_pvalue: f64,
    /// The conditioning the forecast was computed under: [`ConditionMode::Hard`]
    /// when no cell has a standard deviation above 0.
    pub mode: ConditionMode,
    /// `steps × k`: the standard deviation applied at each soft cell, `0.0`
    /// at hard and free cells.
    pub condition_sd: Mat<f64>,
    /// Number of conditioned cells with a standard deviation above 0.
    pub n_soft: usize,
}

impl VarResults {
    /// Conditional forecast `steps` periods ahead subject to hard conditions
    /// on some cells of the future path (module docs for the closed form).
    ///
    /// `conditions[h][j]` is `Some(value)` to pin series `j` at horizon
    /// `h + 1` to `value`, `None` (or `Some(NaN)`) to leave it free. Rows
    /// beyond `conditions.len()` up to `steps` are free, so a short list
    /// conditions the near horizons only. At least one cell must be
    /// constrained — the all-free case is [`VarResults::forecast`].
    ///
    /// Intervals are `point ± z_{1 - alpha/2} se` from the conditional
    /// covariance — innovation uncertainty only, coefficients treated as
    /// known, exactly like [`VarResults::forecast_interval`].
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `steps == 0`, `conditions` is
    ///   empty, constrains no cell, or holds an infinite value, or if the
    ///   working set would exceed the memory budget;
    /// * [`VarError::InvalidParameter`] if `alpha` is not inside `(0, 1)`, or
    ///   if `steps` alone puts the forecast path past the memory budget
    ///   (checked before anything is allocated, so a typo cannot abort the
    ///   allocator);
    /// * [`VarError::Dimension`] if `conditions` has more rows than `steps`
    ///   or a row does not have exactly `k` entries;
    /// * [`VarError::NotPositiveDefinite`] if the covariance of the
    ///   constrained cells is numerically singular (an explosive system at a
    ///   long horizon).
    pub fn conditional_forecast(
        &self,
        steps: usize,
        conditions: &[Vec<Option<f64>>],
        alpha: f64,
    ) -> Result<ConditionalForecast, VarError> {
        let (cells, constrained) = self.conditioned_cells(steps, conditions, alpha)?;
        self.conditional_core(steps, &cells, constrained, None, alpha)
    }

    /// Conditional forecast with soft (Gaussian, stated-variance) conditions
    /// on some of the conditioned cells (module docs for both closed forms).
    ///
    /// `conditions` is read exactly as in
    /// [`VarResults::conditional_forecast`]. `condition_sd` has one row per
    /// row of `conditions` and one entry per series: at a conditioned cell a
    /// finite standard deviation (not a variance) `>= 0`, where `0`, `None`
    /// or `NaN` means a hard condition; at a free cell only `None`, `NaN` or
    /// `0`. `mode` says which reading of the standard deviation is meant.
    /// When no cell has a standard deviation above 0 the result is exactly
    /// [`VarResults::conditional_forecast`] with [`ConditionMode::Hard`] —
    /// the same code path, no soft term touched.
    ///
    /// # Errors
    ///
    /// Every error of [`VarResults::conditional_forecast`] (checked first),
    /// and:
    ///
    /// * [`VarError::Dimension`] naming `condition_sd` if it does not have
    ///   exactly `conditions.len()` rows of `k` entries;
    /// * [`VarError::InvalidParameter`] naming `condition_sd` for a
    ///   negative, infinite or square-overflowing standard deviation, for a
    ///   standard deviation above 0 at a free cell, or when the stated
    ///   standard deviations are too large for the forecast covariance to
    ///   stay finite;
    /// * [`VarError::NotPositiveDefinite`] if `B Σ B'` (density) or
    ///   `B Σ B' + Ω` (signal) is numerically singular.
    pub fn conditional_forecast_soft(
        &self,
        steps: usize,
        conditions: &[Vec<Option<f64>>],
        condition_sd: &[Vec<Option<f64>>],
        mode: SoftMode,
        alpha: f64,
    ) -> Result<ConditionalForecast, VarError> {
        let k = self.neqs;
        let (cells, constrained) = self.conditioned_cells(steps, conditions, alpha)?;
        if condition_sd.len() != conditions.len() {
            return Err(VarError::Dimension {
                what: "condition_sd must have exactly one row per row of conditions (the \
                       standard deviation of each conditioned cell; 0, None or NaN for a \
                       hard one and None or NaN at a free cell)",
                expected: conditions.len(),
                got: condition_sd.len(),
            });
        }
        for row in condition_sd {
            if row.len() != k {
                return Err(VarError::Dimension {
                    what: "every row of condition_sd must have exactly one entry per series \
                           (k = the number of columns of the data), like the rows of \
                           conditions",
                    expected: k,
                    got: row.len(),
                });
            }
        }
        for (h, row) in condition_sd.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                let Some(v) = *c else { continue };
                if v.is_nan() {
                    continue;
                }
                if !(v.is_finite() && v >= 0.0) {
                    return Err(VarError::InvalidParameter {
                        name: "condition_sd",
                        value: v,
                        requirement: "a finite standard deviation >= 0 at every conditioned \
                                      cell (0, None or NaN means a hard condition)",
                    });
                }
                if v > 0.0 && !(v * v).is_finite() {
                    return Err(VarError::InvalidParameter {
                        name: "condition_sd",
                        value: v,
                        requirement: "a standard deviation whose square (the variance) is a \
                                      finite double, i.e. below about 1.3e154",
                    });
                }
                if v > 0.0 && !constrained[h][j] {
                    return Err(VarError::InvalidParameter {
                        name: "condition_sd",
                        value: v,
                        requirement: "None, NaN or 0 at a free cell of conditions: a standard \
                                      deviation needs a condition value to be centred on — \
                                      give the cell a value in conditions or drop the sd",
                    });
                }
            }
        }
        // Standard deviation per conditioned cell, in the row-major order of
        // `cells`; 0.0 for a hard cell (None, NaN, 0 or -0).
        let sd: Vec<f64> = cells
            .iter()
            .map(|&(h, j, _)| match condition_sd[h][j] {
                Some(v) if v > 0.0 => v,
                _ => 0.0,
            })
            .collect();
        if sd.iter().all(|&s| s == 0.0) {
            return self.conditional_core(steps, &cells, constrained, None, alpha);
        }
        self.conditional_core(steps, &cells, constrained, Some((mode, &sd)), alpha)
    }

    /// Validates `steps`, `alpha` and `conditions` and lists the conditioned
    /// cells `(h, j, value)` in row-major order with the `steps × k` grid of
    /// conditioned flags.
    #[allow(clippy::type_complexity)]
    fn conditioned_cells(
        &self,
        steps: usize,
        conditions: &[Vec<Option<f64>>],
        alpha: f64,
    ) -> Result<(Vec<(usize, usize, f64)>, Vec<Vec<bool>>), VarError> {
        let k = self.neqs;
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "steps = 0: a conditional forecast needs at least one step ahead; \
                       pass steps >= 1 (the number of rows of conditions is the natural \
                       choice)",
            });
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(VarError::InvalidParameter {
                name: "alpha",
                value: alpha,
                requirement: "a value strictly inside (0, 1) — alpha = 0.05 gives a 95% \
                              conditional forecast interval",
            });
        }
        if conditions.is_empty() {
            return Err(VarError::InvalidArgument {
                what: "conditions is empty: pass one row per forecast horizon, each with \
                       one entry per series — None (or NaN) for a free cell and a number \
                       for a constrained one",
            });
        }
        if conditions.len() > steps {
            return Err(VarError::Dimension {
                what: "conditions has more rows than steps: every row of conditions is one \
                       forecast horizon, so the row count must be at most steps (rows \
                       beyond the last one are free); raise steps or drop rows",
                expected: steps,
                got: conditions.len(),
            });
        }
        // The `steps × k` grid of constrained flags below — and every later
        // buffer, all of which are at least that big — scales with `steps`,
        // a user count. Refuse an impossible horizon BEFORE allocating any of
        // them, so a `steps` typo is a teaching error and not an allocator
        // abort. (`n <= MAX_WORKING_DOUBLES` is implied by the full
        // working-set budget checked once `m` is known, so this only moves
        // the same refusal earlier.)
        let n = steps.saturating_mul(k);
        if n > MAX_WORKING_DOUBLES {
            return Err(VarError::InvalidParameter {
                name: "steps",
                value: steps as f64,
                requirement: "a horizon whose forecast path fits the conditional-forecast \
                              memory budget of 2^24 doubles: steps times the number of \
                              series must not exceed it; reduce steps",
            });
        }
        let mut cells: Vec<(usize, usize, f64)> = Vec::new();
        let mut constrained = vec![vec![false; k]; steps];
        for (h, row) in conditions.iter().enumerate() {
            if row.len() != k {
                return Err(VarError::Dimension {
                    what: "every row of conditions must have exactly one entry per series \
                           (k = the number of columns of the data), None or NaN marking a \
                           free cell",
                    expected: k,
                    got: row.len(),
                });
            }
            for (j, c) in row.iter().enumerate() {
                if let Some(v) = c {
                    if v.is_nan() {
                        continue;
                    }
                    if !v.is_finite() {
                        return Err(VarError::InvalidArgument {
                            what: "conditions contains an infinite value: a constrained cell \
                                   must be a finite number and a free cell None (or NaN)",
                        });
                    }
                    cells.push((h, j, *v));
                    constrained[h][j] = true;
                }
            }
        }
        if cells.is_empty() {
            return Err(VarError::InvalidArgument {
                what: "conditions constrains no cell (every entry is None/NaN), so there is \
                       nothing to condition on: pass at least one finite value, or use the \
                       unconditional forecast (var_forecast) instead",
            });
        }
        Ok((cells, constrained))
    }

    /// The shared body of the hard and soft conditional forecasts. With
    /// `soft = None` it is the 0.10.0 hard path (no soft term is touched);
    /// `soft = Some((mode, sd))` carries the standard deviation of every
    /// conditioned cell in the order of `cells` (0 for a hard one), at least
    /// one of them above 0.
    fn conditional_core(
        &self,
        steps: usize,
        cells: &[(usize, usize, f64)],
        constrained: Vec<Vec<bool>>,
        soft: Option<(SoftMode, &[f64])>,
        alpha: f64,
    ) -> Result<ConditionalForecast, VarError> {
        let k = self.neqs;
        let n = steps.saturating_mul(k);
        let m = cells.len();
        let working = n
            .checked_mul(3 * m + k + 1)
            .and_then(|a| m.checked_mul(m).map(|b| a.saturating_add(b)))
            .unwrap_or(usize::MAX);
        // The soft modes add the m stated variances; signal mode also keeps
        // the m x m matrix M^{-1} B Sigma B' for the soft cells' block.
        let working = match soft {
            None => working,
            Some((SoftMode::Density, _)) => working.saturating_add(m),
            Some((SoftMode::Signal, _)) => working
                .saturating_add(m)
                .saturating_add(m.saturating_mul(m)),
        };
        if working > MAX_WORKING_DOUBLES {
            return Err(VarError::InvalidArgument {
                what: "the conditional-forecast working set (steps * k cells times the \
                       number of constrained cells in conditions) exceeds the memory budget \
                       of 2^24 doubles; reduce steps or constrain fewer cells",
            });
        }
        // Soft standard deviation per (h, j) cell: sd at a soft cell, 0.0 at
        // hard and free cells (all zero on the hard path).
        let mut sd_grid = Mat::<f64>::zeros(steps, k);
        if let Some((_, sd)) = soft {
            for (i, &(h, j, _)) in cells.iter().enumerate() {
                sd_grid[(h, j)] = sd[i];
            }
        }
        let density = matches!(soft, Some((SoftMode::Density, _)));
        // Indices (into `cells`) of the soft cells at each conditioned
        // horizon (`cells` is row-major, so its last entry has the largest h).
        let mut soft_at: Vec<Vec<usize>> = Vec::new();
        if let Some((_, sd)) = soft {
            soft_at = vec![Vec::new(); cells.last().map_or(0, |c| c.0 + 1)];
            for (i, &(h, _, _)) in cells.iter().enumerate() {
                if sd[i] > 0.0 {
                    soft_at[h].push(i);
                }
            }
        }

        let yhat = self.forecast(steps)?;
        let psi = self.ma_rep(steps - 1)?;
        let mse = self.forecast_cov(steps)?;
        let sigma = &self.sigma_u;

        // B (m x n): row i holds the MA row of constrained cell (h_i, j_i):
        // block s (s <= h_i) is row j_i of Psi_{h_i - s}.
        let mut b = Mat::<f64>::zeros(m, n);
        for (i, &(h, j, _)) in cells.iter().enumerate() {
            for s in 0..=h {
                let phi = &psi[h - s];
                for l in 0..k {
                    b[(i, s * k + l)] = phi[(j, l)];
                }
            }
        }
        // W = (I ⊗ Sigma_u) B'  (n x m): block s of column i is Sigma_u B_{i,s}'.
        let mut w = Mat::<f64>::zeros(n, m);
        for i in 0..m {
            for s in 0..steps {
                for l in 0..k {
                    let mut acc = 0.0;
                    for l2 in 0..k {
                        acc += sigma[(l, l2)] * b[(i, s * k + l2)];
                    }
                    w[(s * k + l, i)] = acc;
                }
            }
        }
        // Gram matrix of the constraints, B Sigma B' (m x m), symmetrised.
        let bw = &b * &w;
        let occ = Mat::from_fn(m, m, |i, j| 0.5 * (bw[(i, j)] + bw[(j, i)]));
        let llt = match soft {
            // Signal: the readings' covariance M = B Sigma B' + diag(sd^2).
            Some((SoftMode::Signal, sd)) => Mat::from_fn(m, m, |i, j| {
                if i == j {
                    occ[(i, j)] + sd[i] * sd[i]
                } else {
                    occ[(i, j)]
                }
            })
            .llt(Side::Lower)
            .map_err(|_| VarError::NotPositiveDefinite {
                what: "B Sigma B' + diag(condition_sd^2), the covariance of the readings in \
                   conditions under soft = signal — numerically singular, which happens for \
                   an explosive VAR conditioned at a long horizon; shorten steps or check \
                   is_stable",
            })?,
            _ => occ
                .llt(Side::Lower)
                .map_err(|_| VarError::NotPositiveDefinite {
                    what: "B Sigma B', the forecast-error covariance of the constrained cells \
                   in conditions — numerically singular, which happens for an explosive VAR \
                   conditioned at a long horizon; shorten steps or check is_stable",
                })?,
        };
        // Signal mode: Q = M^{-1} B Sigma B', so that the readings' cells have
        // Cov(B y | z) = Omega Q (= Omega - Omega M^{-1} Omega) — the same
        // block as MSE - G M^{-1} G', but without the cancellation that
        // leaves a nearly hard soft cell's variance at roundoff (and its se
        // at sqrt(roundoff)).
        let q = match soft {
            Some((SoftMode::Signal, _)) => Some(llt.solve(&occ)),
            _ => None,
        };
        let r = Mat::from_fn(m, 1, |i, _| {
            let (h, j, v) = cells[i];
            v - yhat[(h, j)]
        });
        let lam = llt.solve(&r);
        let u = &w * &lam;
        let shocks = Mat::from_fn(steps, k, |s, l| u[(s * k + l, 0)]);

        // Conditional mean: y_cond[h] = yhat[h] + sum_{s <= h} Psi_{h-s} u*_s.
        let mut point = yhat.clone();
        for h in 0..steps {
            for s in 0..=h {
                let phi = &psi[h - s];
                for row in 0..k {
                    let mut acc = 0.0;
                    for l in 0..k {
                        acc += phi[(row, l)] * shocks[(s, l)];
                    }
                    point[(h, row)] += acc;
                }
            }
        }
        // Hard cells hold their condition bitwise; so does every conditioned
        // cell in density mode. A signal-mode soft cell is the posterior mean.
        for &(h, j, v) in cells {
            if !(matches!(soft, Some((SoftMode::Signal, _))) && sd_grid[(h, j)] > 0.0) {
                point[(h, j)] = v;
            }
        }

        // G = R W (n x m): block h = sum_{s <= h} Psi_{h-s} W_s.
        let mut g = Mat::<f64>::zeros(n, m);
        for h in 0..steps {
            for s in 0..=h {
                let phi = &psi[h - s];
                for i in 0..m {
                    for row in 0..k {
                        let mut acc = 0.0;
                        for l in 0..k {
                            acc += phi[(row, l)] * w[(s * k + l, i)];
                        }
                        g[(h * k + row, i)] += acc;
                    }
                }
            }
        }

        let z = inv_norm_cdf(1.0 - alpha / 2.0)?;
        let mut cov = Vec::with_capacity(steps);
        let mut se = Mat::<f64>::zeros(steps, k);
        let mut unconditional_se = Mat::<f64>::zeros(steps, k);
        let mut lower = Mat::<f64>::zeros(steps, k);
        let mut upper = Mat::<f64>::zeros(steps, k);
        for h in 0..steps {
            // cov_h = MSE(h) - G_h (B Sigma B')^{-1} G_h'  (M^{-1} in signal mode).
            let gh_t = Mat::from_fn(m, k, |i, row| g[(h * k + row, i)]);
            let x = llt.solve(&gh_t);
            let raw = Mat::from_fn(k, k, |a, c| {
                let mut acc = mse[h][(a, c)];
                for i in 0..m {
                    acc -= g[(h * k + a, i)] * x[(i, c)];
                }
                acc
            });
            // Density: + sum_i sd_i^2 X_h[i,:]' X_h[i,:], X_h = Occ^{-1} G_h'.
            let raw = match soft {
                Some((SoftMode::Density, sd)) => Mat::from_fn(k, k, |a, c| {
                    let mut acc = raw[(a, c)];
                    for i in 0..m {
                        if sd[i] > 0.0 {
                            acc += sd[i] * sd[i] * x[(i, a)] * x[(i, c)];
                        }
                    }
                    acc
                }),
                _ => raw,
            };
            let mut c = Mat::from_fn(k, k, |a, bb| 0.5 * (raw[(a, bb)] + raw[(bb, a)]));
            for j in 0..k {
                // Hard cells (sd 0) get an exactly zero row and column.
                if constrained[h][j] && sd_grid[(h, j)] <= 0.0 {
                    for l in 0..k {
                        c[(j, l)] = 0.0;
                        c[(l, j)] = 0.0;
                    }
                }
                c[(j, j)] = c[(j, j)].max(0.0);
            }
            if let (Some(q), Some((_, sd)), Some(here)) = (&q, soft, soft_at.get(h)) {
                // Signal: the soft x soft block of this horizon from the
                // cancellation-free Omega Q (hard rows stay exactly zero).
                for &a in here {
                    let ja = cells[a].1;
                    for &bb in here {
                        let jb = cells[bb].1;
                        c[(ja, jb)] =
                            0.5 * (sd[a] * sd[a] * q[(a, bb)] + sd[bb] * sd[bb] * q[(bb, a)]);
                    }
                    c[(ja, ja)] = c[(ja, ja)].max(0.0);
                }
            }
            if density {
                // The conditioned x conditioned block is the stated Omega
                // exactly: sd^2 on the diagonal, 0 between distinct
                // conditioned cells (computed: roundoff, ~1e-17).
                for j in 0..k {
                    let sj = sd_grid[(h, j)];
                    if sj > 0.0 {
                        for l in 0..k {
                            if l != j && constrained[h][l] {
                                c[(j, l)] = 0.0;
                                c[(l, j)] = 0.0;
                            }
                        }
                        c[(j, j)] = sj * sj;
                    }
                }
            }
            for j in 0..k {
                let s = if density && sd_grid[(h, j)] > 0.0 {
                    sd_grid[(h, j)]
                } else {
                    c[(j, j)].sqrt()
                };
                se[(h, j)] = s;
                unconditional_se[(h, j)] = mse[h][(j, j)].max(0.0).sqrt();
                lower[(h, j)] = point[(h, j)] - z * s;
                upper[(h, j)] = point[(h, j)] + z * s;
            }
            cov.push(c);
        }
        if soft.is_some() {
            let finite = |mat: &Mat<f64>| {
                (0..mat.nrows()).all(|i| (0..mat.ncols()).all(|j| mat[(i, j)].is_finite()))
            };
            if !(finite(&point) && finite(&lower) && finite(&upper) && cov.iter().all(finite)) {
                let largest = soft.map_or(0.0, |(_, sd)| sd.iter().fold(0.0f64, |a, &s| a.max(s)));
                return Err(VarError::InvalidParameter {
                    name: "condition_sd",
                    value: largest,
                    requirement: "standard deviations small enough that the forecast \
                                  covariance (which grows with sd^2 under soft = density) \
                                  stays finite; pass smaller ones",
                });
            }
        }

        // Structural (Cholesky-orthogonalised) shocks: P eps_s = u_s.
        let p_chol = chol_lower(sigma.as_ref(), "Sigma_u, the residual covariance")?;
        let mut orth_shocks = Mat::<f64>::zeros(steps, k);
        for s in 0..steps {
            for i in 0..k {
                let mut acc = shocks[(s, i)];
                for l in 0..i {
                    acc -= p_chol[(i, l)] * orth_shocks[(s, l)];
                }
                orth_shocks[(s, i)] = acc / p_chol[(i, i)];
            }
        }

        let mahalanobis: f64 = (0..m).map(|i| r[(i, 0)] * lam[(i, 0)]).sum();
        let mahalanobis_pvalue = chi2_sf(mahalanobis.max(0.0), m as f64)?;

        let (mode, n_soft) = match soft {
            None => (ConditionMode::Hard, 0),
            Some((sm, sd)) => (
                match sm {
                    SoftMode::Density => ConditionMode::Density,
                    SoftMode::Signal => ConditionMode::Signal,
                },
                sd.iter().filter(|&&s| s > 0.0).count(),
            ),
        };

        Ok(ConditionalForecast {
            steps,
            alpha,
            point,
            unconditional: yhat,
            cov,
            se,
            unconditional_se,
            lower,
            upper,
            shocks,
            orth_shocks,
            constrained,
            n_constrained: m,
            mahalanobis,
            mahalanobis_pvalue,
            mode,
            condition_sd: sd_grid,
            n_soft,
        })
    }
}
