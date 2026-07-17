//! The two-step FAVAR of Bernanke, Boivin & Eliasz (2005, QJE).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_var::{Trend, VarResults, VarSpec};

use crate::error::FavarError;
use crate::pca::FactorModel;

/// A factor-augmented VAR estimated by the Bernanke-Boivin-Eliasz (2005)
/// two-step procedure.
///
/// Step 1 extracts `r` common factors `F_t` from a large standardized
/// panel `X` (`n x N`) by principal components ([`FactorModel`]). Step 2
/// fits a reduced-form VAR (`tsecon-var`) on the `(r + 1)`-vector
/// `Y_t = [F_t', R_t]'`, where `R_t` is an observed policy variable
/// (e.g. the federal funds rate) ordered **last**. Under a recursive
/// (Cholesky) identification with `R` last, the policy innovation is the
/// structural monetary shock: the factors — and hence, through the
/// observation equation, every series in the panel — may respond to it
/// only with a lag, while `R` responds to the factors within the period.
///
/// The observation equation `X_t = L F_t + e_t` (with `L` the `N x r`
/// loadings on the standardized scale) maps the factor VAR's impulse
/// responses back onto any observed series via
/// [`Favar::series_response`].
///
/// Only the factor-extraction step is externally validated against a
/// golden fixture; the two-step assembly and IRF mapping are validated
/// structurally (dimensions, orthogonality, reconstruction round-trip)
/// and by simulation.
#[derive(Debug, Clone)]
pub struct Favar {
    model: FactorModel,
    n_factors: usize,
    factors: Mat<f64>,
    policy: Vec<f64>,
    var: VarResults,
    slow_fast: bool,
}

impl Favar {
    /// Fits the plain two-step FAVAR: `r` principal-component factors of
    /// `panel`, then a VAR(`lags`) with the given `trend` on
    /// `[factors, policy]` (`policy` ordered last).
    ///
    /// `panel` is `n x N` (observations in rows, oldest first); `policy`
    /// is the length-`n` observed policy series aligned to the same rows.
    ///
    /// # Errors
    ///
    /// * everything [`FactorModel::fit`] can return;
    /// * [`FavarError::DimensionMismatch`] if `policy.len() != n`;
    /// * [`FavarError::NonFinite`] on a NaN/infinite policy entry;
    /// * [`FavarError::InvalidFactorCount`] if `r` is out of range;
    /// * [`FavarError::Var`] if the factor VAR fails to fit.
    pub fn two_step(
        panel: MatRef<'_, f64>,
        policy: &[f64],
        r: usize,
        lags: usize,
        trend: Trend,
    ) -> Result<Self, FavarError> {
        let model = FactorModel::fit(panel)?;
        Self::from_model(model, None, policy, r, lags, trend, false)
    }

    /// Fits the two-step FAVAR with the Bernanke-Boivin-Eliasz slow/fast
    /// factor rotation.
    ///
    /// The raw principal components `C_t` of the full panel are
    /// contaminated by the contemporaneous policy shock. To recover
    /// factors that (under the recursive scheme) `R` can be ordered after,
    /// `C_t` is projected on the slow-moving factors `F^{slow}_t` — the
    /// `r` principal components of the slow-moving subset of series,
    /// indexed by `slow_indices` — together with the policy variable and a
    /// constant, and the fitted policy component is purged:
    ///
    /// ```text
    /// C_t = b_slow' F^{slow}_t + b_R R_t + a + resid_t,
    /// F_t^{clean} = C_t - b_R R_t.
    /// ```
    ///
    /// The cleaned factors `F^{clean}_t` (which no longer load on the
    /// contemporaneous policy shock) enter the VAR with `R` ordered last.
    /// `slow_indices` must be distinct, in `0..N`, and number at least `r`
    /// (the slow subset must admit `r` factors).
    ///
    /// # Errors
    ///
    /// As [`Favar::two_step`], plus [`FavarError::InvalidArgument`] for a
    /// malformed `slow_indices` set and any error from the slow-subset
    /// factor extraction or the projection solve.
    pub fn two_step_slow_fast(
        panel: MatRef<'_, f64>,
        policy: &[f64],
        slow_indices: &[usize],
        r: usize,
        lags: usize,
        trend: Trend,
    ) -> Result<Self, FavarError> {
        let model = FactorModel::fit(panel)?;
        Self::from_model(
            model,
            Some((panel, slow_indices)),
            policy,
            r,
            lags,
            trend,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_model(
        model: FactorModel,
        slow: Option<(MatRef<'_, f64>, &[usize])>,
        policy: &[f64],
        r: usize,
        lags: usize,
        trend: Trend,
        slow_fast: bool,
    ) -> Result<Self, FavarError> {
        let n = model.n_obs();
        if policy.len() != n {
            return Err(FavarError::DimensionMismatch {
                what: "policy length must equal the number of panel observations",
                expected: n,
                got: policy.len(),
            });
        }
        for &p in policy {
            if !p.is_finite() {
                return Err(FavarError::NonFinite { what: "policy" });
            }
        }

        // Raw factors, then (optionally) the slow/fast rotation.
        let factors = match slow {
            None => model.factors(r)?,
            Some((panel, slow_indices)) => {
                slow_fast_rotate(&model, panel, slow_indices, policy, r)?
            }
        };

        // Assemble the VAR endog: r factor columns then the policy column.
        let k = r + 1;
        let endog = Mat::from_fn(n, k, |i, j| if j < r { factors[(i, j)] } else { policy[i] });

        let spec = VarSpec::new(lags, trend)?;
        let var = spec.fit(endog.as_ref())?;

        Ok(Self {
            model,
            n_factors: r,
            factors,
            policy: policy.to_vec(),
            var,
            slow_fast,
        })
    }

    /// The fitted principal-component factor model of the panel.
    pub fn factor_model(&self) -> &FactorModel {
        &self.model
    }

    /// The estimated factor VAR on `[factors, policy]` (policy last).
    pub fn var(&self) -> &VarResults {
        &self.var
    }

    /// Number of common factors `r`.
    pub fn n_factors(&self) -> usize {
        self.n_factors
    }

    /// Number of endogenous VAR variables, `k = r + 1`.
    pub fn n_endog(&self) -> usize {
        self.n_factors + 1
    }

    /// Index of the policy variable within the VAR (`= r`, the last
    /// equation).
    pub fn policy_index(&self) -> usize {
        self.n_factors
    }

    /// The factor matrix that entered the VAR (`n x r`); the cleaned
    /// factors when the slow/fast rotation was used.
    pub fn factors(&self) -> MatRef<'_, f64> {
        self.factors.as_ref()
    }

    /// The observed policy series (length `n`) that entered the VAR as
    /// its last variable.
    pub fn policy(&self) -> &[f64] {
        &self.policy
    }

    /// Whether the slow/fast rotation was applied.
    pub fn is_slow_fast(&self) -> bool {
        self.slow_fast
    }

    /// Impulse response of an observed panel series to a VAR shock,
    /// mapped through the observation equation `X_t = L F_t`.
    ///
    /// The response of standardized series `series` at horizon `h` to a
    /// shock in VAR variable `shock` is
    ///
    /// ```text
    /// sum_{f = 0}^{r - 1} L[series, f] * Psi_h[f, shock],
    /// ```
    ///
    /// where `Psi_h` is the (non-orthogonalized) or Cholesky-orthogonalized
    /// MA coefficient of the factor VAR (`orthogonalized == true` gives
    /// the structural response to a one-standard-deviation recursive shock;
    /// with the policy variable ordered last, `shock == policy_index()`
    /// traces the monetary-policy IRF). Because the panel is standardized
    /// in step 1, the returned response is in *standard-deviation units of
    /// `series`*; multiply by `factor_model().scale()[series]` for the raw
    /// scale.
    ///
    /// The returned vector has `horizon + 1` entries (`h = 0..=horizon`).
    ///
    /// # Errors
    ///
    /// * [`FavarError::InvalidArgument`] if `series >= N` or
    ///   `shock >= r + 1`;
    /// * [`FavarError::Var`] if the MA representation cannot be formed.
    pub fn series_response(
        &self,
        series: usize,
        shock: usize,
        horizon: usize,
        orthogonalized: bool,
    ) -> Result<Vec<f64>, FavarError> {
        if series >= self.model.n_series() {
            return Err(FavarError::InvalidArgument {
                what: "series index out of range",
            });
        }
        if shock >= self.n_endog() {
            return Err(FavarError::InvalidArgument {
                what: "shock index out of range",
            });
        }
        let psi = if orthogonalized {
            self.var.orth_ma_rep(horizon)?
        } else {
            self.var.ma_rep(horizon)?
        };
        let loadings = self.model.loadings(self.n_factors)?;
        let resp = psi
            .iter()
            .map(|m| {
                (0..self.n_factors)
                    .map(|f| loadings[(series, f)] * m[(f, shock)])
                    .sum()
            })
            .collect();
        Ok(resp)
    }

    /// Impulse response of the observed policy variable itself to a VAR
    /// shock: `Psi_h[policy_index(), shock]` (non-orthogonalized) or the
    /// orthogonalized counterpart. Returned in the policy variable's own
    /// units (it is not standardized). `horizon + 1` entries.
    ///
    /// # Errors
    ///
    /// * [`FavarError::InvalidArgument`] if `shock >= r + 1`;
    /// * [`FavarError::Var`] if the MA representation cannot be formed.
    pub fn policy_response(
        &self,
        shock: usize,
        horizon: usize,
        orthogonalized: bool,
    ) -> Result<Vec<f64>, FavarError> {
        if shock >= self.n_endog() {
            return Err(FavarError::InvalidArgument {
                what: "shock index out of range",
            });
        }
        let psi = if orthogonalized {
            self.var.orth_ma_rep(horizon)?
        } else {
            self.var.ma_rep(horizon)?
        };
        let p = self.policy_index();
        Ok(psi.iter().map(|m| m[(p, shock)]).collect())
    }
}

/// Bernanke-Boivin-Eliasz slow/fast rotation: purge the contemporaneous
/// policy component from the `r` full-panel factors.
///
/// Extracts `r` slow-moving factors from the `slow_indices` subset of the
/// raw panel, regresses each full factor on `[slow factors, policy, 1]`,
/// and subtracts the fitted policy component.
fn slow_fast_rotate(
    model: &FactorModel,
    panel: MatRef<'_, f64>,
    slow_indices: &[usize],
    policy: &[f64],
    r: usize,
) -> Result<Mat<f64>, FavarError> {
    let n = model.n_obs();
    let n_series = model.n_series();
    if slow_indices.is_empty() {
        return Err(FavarError::InvalidArgument {
            what: "slow_indices must be non-empty",
        });
    }
    if slow_indices.len() < r {
        return Err(FavarError::InvalidArgument {
            what: "slow_indices must contain at least r series to admit r slow factors",
        });
    }
    // Distinctness and range.
    let mut seen = vec![false; n_series];
    for &idx in slow_indices {
        if idx >= n_series {
            return Err(FavarError::InvalidArgument {
                what: "slow index out of range",
            });
        }
        if seen[idx] {
            return Err(FavarError::InvalidArgument {
                what: "slow_indices must be distinct",
            });
        }
        seen[idx] = true;
    }

    // Slow subpanel and its r principal-component factors.
    let slow_panel = Mat::from_fn(n, slow_indices.len(), |i, j| panel[(i, slow_indices[j])]);
    let slow_model = FactorModel::fit(slow_panel.as_ref())?;
    let slow_factors = slow_model.factors(r)?;

    // Full factors C (n x r) to be cleaned.
    let c = model.factors(r)?;

    // Design Z = [slow factors (r), policy (1), constant (1)] (n x (r + 2)).
    let q = r + 2;
    let z = Mat::from_fn(n, q, |i, j| {
        if j < r {
            slow_factors[(i, j)]
        } else if j == r {
            policy[i]
        } else {
            1.0
        }
    });

    // Regress each full factor on Z; subtract the policy component.
    let policy_col = r; // index of the policy regressor in Z
    let mut cleaned = Mat::<f64>::zeros(n, r);
    for k in 0..r {
        let y: Vec<f64> = (0..n).map(|i| c[(i, k)]).collect();
        let beta = ols(z.as_ref(), &y)?;
        let b_policy = beta[policy_col];
        for i in 0..n {
            cleaned[(i, k)] = c[(i, k)] - b_policy * policy[i];
        }
    }
    Ok(cleaned)
}

/// Ordinary least squares via the thin SVD (pseudoinverse); robust to the
/// mild collinearity a `[factors, policy, constant]` design can carry.
/// Returns the coefficient vector (length `z.ncols()`).
fn ols(z: MatRef<'_, f64>, y: &[f64]) -> Result<Vec<f64>, FavarError> {
    let n = z.nrows();
    let q = z.ncols();
    let svd = z.thin_svd().map_err(|_| FavarError::SvdFailed)?;
    let u = svd.U();
    let v = svd.V();
    let s: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let rr = s.len();
    let s_max = s.iter().copied().fold(0.0f64, f64::max);
    let cutoff = s_max * (n.max(q) as f64) * f64::EPSILON;

    // d_k = (U' y)_k, filtered c_k = d_k / s_k, beta_j = sum_k V[j,k] c_k.
    let cvec: Vec<f64> = (0..rr)
        .map(|k| {
            let dk: f64 = (0..n).map(|i| u[(i, k)] * y[i]).sum();
            if s[k] > cutoff {
                dk / s[k]
            } else {
                0.0
            }
        })
        .collect();
    let beta = (0..q)
        .map(|j| (0..rr).map(|k| v[(j, k)] * cvec[k]).sum())
        .collect();
    Ok(beta)
}
