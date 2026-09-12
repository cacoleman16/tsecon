//! Conditional (hard-path) forecasts of a VAR(p) — the Doan-Litterman-Sims
//! (1984) / Waggoner-Zha (1999) closed form.
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
//! What this is not: it is a *hard*-conditioning, coefficients-known object.
//! Parameter uncertainty (the Bayesian per-draw version) and soft/interval
//! conditions are out of scope here. The squared `Σ`-norm of the implied
//! shocks, `r' (B Σ B')^{-1} r = ε*'ε*`, is reported with its `χ²(m)` tail
//! probability as a plausibility check of the conditioning path against the
//! model's own unconditional forecast distribution (Waggoner-Zha's measure of
//! how hard the model has to be pushed).

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

/// A conditional forecast produced by [`VarResults::conditional_forecast`].
///
/// Every `steps × k` matrix has row `h` for horizon `h + 1` and one column
/// per series, the layout of [`VarResults::forecast`].
#[derive(Debug, Clone)]
pub struct ConditionalForecast {
    /// Forecast horizon `H` (rows of every path matrix).
    pub steps: usize,
    /// Interval level `alpha` the bounds were built at (`1 - alpha` coverage).
    pub alpha: f64,
    /// Conditional mean path `ỹ`; constrained cells hold their condition
    /// exactly (set bitwise, not left to rounding).
    pub point: Mat<f64>,
    /// The unconditional path `ŷ` — identical to [`VarResults::forecast`].
    pub unconditional: Mat<f64>,
    /// Conditional forecast-error covariance at each horizon, `k × k` per
    /// entry: the diagonal blocks of `V`. Constrained cells have an exactly
    /// zero row and column.
    pub cov: Vec<Mat<f64>>,
    /// `sqrt(diag cov[h])`, `steps × k`; exactly zero at constrained cells.
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
    /// conditions, equivalently `E[u | conditions]`.
    pub shocks: Mat<f64>,
    /// The same shocks orthogonalised by the lower Cholesky factor of
    /// `Σ_u` in the data's column order, `ε*_s = P^{-1} u*_s` — the
    /// Waggoner-Zha structural shocks under a recursive identification.
    /// Only this field depends on the ordering.
    pub orth_shocks: Mat<f64>,
    /// `constrained[h][j]` is `true` iff cell `(h, j)` was conditioned on.
    pub constrained: Vec<Vec<bool>>,
    /// Number of constrained cells `m`.
    pub n_constrained: usize,
    /// `r' (B Σ B')^{-1} r = ε*'ε*`: the squared `Σ`-norm of the implied
    /// shocks — the Mahalanobis distance of the conditioned values from the
    /// unconditional forecast under the model's own forecast distribution.
    pub mahalanobis: f64,
    /// `P(χ²(m) > mahalanobis)`: small when the conditioning path is one the
    /// model would rarely generate on its own.
    pub mahalanobis_pvalue: f64,
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
    /// * [`VarError::InvalidParameter`] if `alpha` is not inside `(0, 1)`;
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
        let m = cells.len();
        if m == 0 {
            return Err(VarError::InvalidArgument {
                what: "conditions constrains no cell (every entry is None/NaN), so there is \
                       nothing to condition on: pass at least one finite value, or use the \
                       unconditional forecast (var_forecast) instead",
            });
        }
        let n = steps.saturating_mul(k);
        let working = n
            .checked_mul(3 * m + k + 1)
            .and_then(|a| m.checked_mul(m).map(|b| a.saturating_add(b)))
            .unwrap_or(usize::MAX);
        if working > MAX_WORKING_DOUBLES {
            return Err(VarError::InvalidArgument {
                what: "the conditional-forecast working set (steps * k cells times the \
                       number of constrained cells in conditions) exceeds the memory budget \
                       of 2^24 doubles; reduce steps or constrain fewer cells",
            });
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
        let llt = occ
            .llt(Side::Lower)
            .map_err(|_| VarError::NotPositiveDefinite {
                what: "B Sigma B', the forecast-error covariance of the constrained cells in \
                   conditions — numerically singular, which happens for an explosive VAR \
                   conditioned at a long horizon; shorten steps or check is_stable",
            })?;
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
        for &(h, j, v) in &cells {
            point[(h, j)] = v;
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
            // cov_h = MSE(h) - G_h (B Sigma B')^{-1} G_h'.
            let gh_t = Mat::from_fn(m, k, |i, row| g[(h * k + row, i)]);
            let x = llt.solve(&gh_t);
            let raw = Mat::from_fn(k, k, |a, c| {
                let mut acc = mse[h][(a, c)];
                for i in 0..m {
                    acc -= g[(h * k + a, i)] * x[(i, c)];
                }
                acc
            });
            let mut c = Mat::from_fn(k, k, |a, bb| 0.5 * (raw[(a, bb)] + raw[(bb, a)]));
            for j in 0..k {
                if constrained[h][j] {
                    for l in 0..k {
                        c[(j, l)] = 0.0;
                        c[(l, j)] = 0.0;
                    }
                }
                c[(j, j)] = c[(j, j)].max(0.0);
            }
            for j in 0..k {
                let s = c[(j, j)].sqrt();
                se[(h, j)] = s;
                unconditional_se[(h, j)] = mse[h][(j, j)].max(0.0).sqrt();
                lower[(h, j)] = point[(h, j)] - z * s;
                upper[(h, j)] = point[(h, j)] + z * s;
            }
            cov.push(c);
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
        })
    }
}
