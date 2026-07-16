//! The linear-Gaussian state-space model: storage, builder, validation,
//! and initialization.
//!
//! Model form (Durbin & Koopman 2012, eq. 4.12, plus intercepts):
//!
//! ```text
//! y_t     = d + Z alpha_t + eps_t,        eps_t ~ N(0, H)
//! alpha_{t+1} = c + T alpha_t + R eta_t,  eta_t ~ N(0, Q)
//! alpha_1 ~ N(a_1, P_1),   P_1 = P_star + kappa P_inf  (kappa -> infinity)
//! ```
//!
//! with `y_t` of dimension `p`, `alpha_t` of dimension `m`, and `eta_t`
//! of dimension `r`. The system matrices are time-invariant in this pass;
//! they are stored behind the [`SystemMatrix`] accessor enum so a
//! time-varying variant can be added without breaking the API.

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::{jittered_cholesky, solve_discrete_lyapunov, symmetrize, LinalgError};

use crate::error::SsmError;

/// Storage for one system matrix.
///
/// Time-invariant systems store a single matrix; every consumer reads the
/// matrix through [`SystemMatrix::at`], so a `Varying` variant (per-period
/// matrices or a change-point tape) can be added later without an API
/// break. The enum is `#[non_exhaustive]` for exactly that reason.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SystemMatrix {
    /// The same matrix at every time period.
    Constant(Mat<f64>),
}

impl SystemMatrix {
    /// The matrix in effect at time period `t` (0-indexed).
    #[inline]
    pub fn at(&self, _t: usize) -> MatRef<'_, f64> {
        match self {
            Self::Constant(m) => m.as_ref(),
        }
    }

    /// Number of rows (constant across `t` by construction).
    #[inline]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Constant(m) => m.nrows(),
        }
    }

    /// Number of columns (constant across `t` by construction).
    #[inline]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Constant(m) => m.ncols(),
        }
    }
}

/// How the initial state `alpha_1 ~ N(a_1, P_1)` is specified.
#[derive(Debug, Clone)]
pub enum Initialization {
    /// Known initial mean and (finite) covariance.
    Known {
        /// Initial state mean `a_1` (length `m`).
        a1: Vec<f64>,
        /// Initial state covariance `P_1` (`m x m`, symmetric PSD).
        p1: Mat<f64>,
    },
    /// Stationary initialization: `a_1 = (I - T)^{-1} c` and `P_1` solves
    /// the discrete Lyapunov equation `P_1 = T P_1 T' + R Q R'`
    /// (the unconditional moments of the stationary state process,
    /// Lütkepohl 2005 sec. 2.1.4). Requires a stable `T`.
    Stationary,
    /// Exact diffuse initialization (Koopman 1997; Durbin & Koopman 2012
    /// ch. 5): `a_1 = 0`, `P_star = 0`, `P_inf = I`. The filter treats
    /// `P_1 = P_star + kappa P_inf` exactly in the limit `kappa -> inf`
    /// via the two-matrix `(P_inf, P_star)` recursions — never a
    /// large-kappa approximation.
    Diffuse,
    /// Mixed initialization with a per-state diffuse flag: flagged states
    /// get the exact-diffuse treatment (`P_inf` has a unit diagonal entry),
    /// unflagged states form a stationary block initialized by the
    /// Lyapunov equation of their sub-system.
    ///
    /// The stationary block must be autonomous: `T` must not feed diffuse
    /// states into stationary ones (`T[stationary, diffuse] = 0`),
    /// otherwise the "stationary" block has no unconditional distribution
    /// and `build` on the filter path returns an error.
    Mixed {
        /// `diffuse[i] == true` marks state `i` as diffuse; length `m`.
        diffuse: Vec<bool>,
    },
}

/// Resolved initial conditions for the filter: `alpha_1 ~ N(a_1, P_star +
/// kappa P_inf)` with `kappa -> infinity` handled exactly.
#[derive(Debug, Clone)]
pub struct InitialState {
    /// Initial state mean `a_1`.
    pub a1: Vec<f64>,
    /// Finite part `P_star` of the initial covariance.
    pub p_star: Mat<f64>,
    /// Diffuse part `P_inf` of the initial covariance (zero when the
    /// initialization is fully known/stationary).
    pub p_inf: Mat<f64>,
}

impl InitialState {
    /// True when any state has a diffuse prior (`P_inf != 0`).
    pub fn has_diffuse(&self) -> bool {
        let n = self.p_inf.nrows();
        for j in 0..n {
            for i in 0..n {
                if self.p_inf[(i, j)] != 0.0 {
                    return true;
                }
            }
        }
        false
    }
}

/// A validated time-invariant linear-Gaussian state-space model.
///
/// Construct through [`LinearGaussianSSM::builder`]; the builder's
/// [`SsmBuilder::build`] performs all dimension and PSD validation, so a
/// value of this type is always internally consistent.
#[derive(Debug, Clone)]
pub struct LinearGaussianSSM {
    p: usize,
    m: usize,
    r_dim: usize,
    z: SystemMatrix,
    h: SystemMatrix,
    t: SystemMatrix,
    r: SystemMatrix,
    q: SystemMatrix,
    state_intercept: Vec<f64>,
    obs_intercept: Vec<f64>,
    init: Initialization,
    h_is_diagonal: bool,
}

impl LinearGaussianSSM {
    /// Local level model (Durbin & Koopman 2012, ch. 2):
    ///
    /// ```text
    /// y_t = mu_t + eps_t,        eps_t ~ N(0, sigma2_eps)
    /// mu_{t+1} = mu_t + eta_t,   eta_t ~ N(0, sigma2_eta)
    /// ```
    ///
    /// with exact-diffuse initialization of the level (the level is a
    /// random walk, so it has no stationary distribution). Matches
    /// statsmodels `UnobservedComponents(level="llevel",
    /// use_exact_diffuse=True)`.
    ///
    /// # Errors
    ///
    /// [`SsmError::NotPsd`] when either variance is negative, and
    /// [`SsmError::NonFinite`] when either is NaN/infinite.
    pub fn local_level(sigma2_eps: f64, sigma2_eta: f64) -> Result<Self, SsmError> {
        let one = Mat::from_fn(1, 1, |_, _| 1.0);
        Self::builder(1, 1, 1)
            .z(one.clone())
            .h(Mat::from_fn(1, 1, |_, _| sigma2_eps))
            .t(one.clone())
            .r(one)
            .q(Mat::from_fn(1, 1, |_, _| sigma2_eta))
            .initialization(Initialization::Diffuse)
            .build()
    }

    /// AR(p) model with intercept in Harvey companion form, matching
    /// statsmodels `SARIMAX(order=(p, 0, 0), trend='c')`:
    ///
    /// ```text
    /// y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eta_t,
    /// eta_t ~ N(0, sigma2)
    /// ```
    ///
    /// as the state-space system with `y_t = alpha_{1,t}`, transition
    /// matrix `T` carrying the coefficients in its first column and ones
    /// on the superdiagonal (`T[i][0] = phi_{i+1}`, `T[i][i+1] = 1`), a
    /// zero observation covariance, and the intercept entering the state
    /// equation as `c e_1` — exactly SARIMAX's `trend='c'` placement, so
    /// the intercept is the regression constant, *not* the process mean
    /// (the mean is `c / (1 - phi_1 - ... - phi_p)`).
    ///
    /// Initialization is stationary (Lyapunov unconditional moments), the
    /// SARIMAX default for a stationary AR; the filter errors with a
    /// wrapped [`tsecon_linalg::LinalgError::Unstable`] if the
    /// coefficients are non-stationary.
    ///
    /// # Errors
    ///
    /// [`SsmError::InvalidArgument`] for empty `coeffs`,
    /// [`SsmError::NotPsd`] for negative `sigma2`, and
    /// [`SsmError::NonFinite`] for NaN/infinite inputs.
    pub fn ar(coeffs: &[f64], sigma2: f64, intercept: f64) -> Result<Self, SsmError> {
        let p_order = coeffs.len();
        if p_order == 0 {
            return Err(SsmError::InvalidArgument {
                what: "ar requires at least one autoregressive coefficient",
            });
        }
        let t = Mat::from_fn(p_order, p_order, |i, j| {
            if j == 0 {
                coeffs[i]
            } else if j == i + 1 {
                1.0
            } else {
                0.0
            }
        });
        let z = Mat::from_fn(1, p_order, |_, j| if j == 0 { 1.0 } else { 0.0 });
        let r = Mat::from_fn(p_order, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let mut c = vec![0.0; p_order];
        c[0] = intercept;
        Self::builder(1, p_order, 1)
            .z(z)
            .h(Mat::zeros(1, 1))
            .t(t)
            .r(r)
            .q(Mat::from_fn(1, 1, |_, _| sigma2))
            .state_intercept(c)
            .initialization(Initialization::Stationary)
            .build()
    }

    /// Starts a builder for a model with observation dimension `p`, state
    /// dimension `m`, and state-disturbance dimension `r`.
    pub fn builder(p: usize, m: usize, r: usize) -> SsmBuilder {
        SsmBuilder {
            p,
            m,
            r_dim: r,
            z: None,
            h: None,
            t: None,
            r: None,
            q: None,
            state_intercept: None,
            obs_intercept: None,
            init: Initialization::Diffuse,
        }
    }

    /// Observation dimension `p`.
    #[inline]
    pub fn obs_dim(&self) -> usize {
        self.p
    }

    /// State dimension `m`.
    #[inline]
    pub fn state_dim(&self) -> usize {
        self.m
    }

    /// State-disturbance dimension `r`.
    #[inline]
    pub fn disturbance_dim(&self) -> usize {
        self.r_dim
    }

    /// Design matrix `Z` (`p x m`).
    #[inline]
    pub fn z(&self) -> &SystemMatrix {
        &self.z
    }

    /// Observation covariance `H` (`p x p`).
    #[inline]
    pub fn h(&self) -> &SystemMatrix {
        &self.h
    }

    /// Transition matrix `T` (`m x m`).
    #[inline]
    pub fn t(&self) -> &SystemMatrix {
        &self.t
    }

    /// Disturbance selection matrix `R` (`m x r`).
    #[inline]
    pub fn r(&self) -> &SystemMatrix {
        &self.r
    }

    /// State-disturbance covariance `Q` (`r x r`).
    #[inline]
    pub fn q(&self) -> &SystemMatrix {
        &self.q
    }

    /// State intercept `c` (length `m`); enters as
    /// `alpha_{t+1} = c + T alpha_t + R eta_t`.
    #[inline]
    pub fn state_intercept(&self) -> &[f64] {
        &self.state_intercept
    }

    /// Observation intercept `d` (length `p`); enters as
    /// `y_t = d + Z alpha_t + eps_t`.
    #[inline]
    pub fn obs_intercept(&self) -> &[f64] {
        &self.obs_intercept
    }

    /// The initialization specification.
    #[inline]
    pub fn initialization(&self) -> &Initialization {
        &self.init
    }

    /// True when `H` is (exactly) diagonal — the requirement for the
    /// univariate filtering path.
    #[inline]
    pub fn h_is_diagonal(&self) -> bool {
        self.h_is_diagonal
    }

    /// `R Q R'` (`m x m`), the state-disturbance covariance in state
    /// coordinates, at time `t`.
    pub(crate) fn rqr(&self, t: usize) -> Result<Mat<f64>, SsmError> {
        let r = self.r.at(t);
        let q = self.q.at(t);
        let prod = r * q * r.transpose();
        Ok(symmetrize(prod.as_ref())?)
    }

    /// Resolves the initialization into explicit `(a_1, P_star, P_inf)`.
    ///
    /// * `Known`: `P_star = P_1`, `P_inf = 0`.
    /// * `Stationary`: `a_1 = (I - T)^{-1} c`, `P_star` from the discrete
    ///   Lyapunov equation `P = T P T' + R Q R'`, `P_inf = 0`. Errors with
    ///   [`LinalgError::Unstable`] (wrapped) when `rho(T) >= 1`.
    /// * `Diffuse`: `a_1 = 0`, `P_star = 0`, `P_inf = I`.
    /// * `Mixed`: unit `P_inf` diagonal on flagged states; the unflagged
    ///   (stationary) block gets its Lyapunov unconditional covariance and
    ///   mean; cross-covariances between the blocks are zero.
    pub fn initial_state(&self) -> Result<InitialState, SsmError> {
        let m = self.m;
        match &self.init {
            Initialization::Known { a1, p1 } => Ok(InitialState {
                a1: a1.clone(),
                p_star: p1.clone(),
                p_inf: Mat::zeros(m, m),
            }),
            Initialization::Stationary => {
                let t = self.t.at(0);
                let rqr = self.rqr(0)?;
                let p_star = solve_discrete_lyapunov(t, rqr.as_ref())?;
                let a1 = solve_i_minus_t(t, &self.state_intercept)?;
                Ok(InitialState {
                    a1,
                    p_star,
                    p_inf: Mat::zeros(m, m),
                })
            }
            Initialization::Diffuse => Ok(InitialState {
                a1: vec![0.0; m],
                p_star: Mat::zeros(m, m),
                p_inf: Mat::from_fn(m, m, |i, j| if i == j { 1.0 } else { 0.0 }),
            }),
            Initialization::Mixed { diffuse } => self.mixed_initial_state(diffuse),
        }
    }

    /// Mixed diffuse/stationary initialization (see [`Initialization::Mixed`]).
    fn mixed_initial_state(&self, diffuse: &[bool]) -> Result<InitialState, SsmError> {
        let m = self.m;
        if diffuse.len() != m {
            return Err(SsmError::Dimension {
                what: "Mixed initialization diffuse flags must have length m",
                expected: m,
                got: diffuse.len(),
            });
        }
        let stationary_idx: Vec<usize> = (0..m).filter(|&i| !diffuse[i]).collect();
        let t = self.t.at(0);
        // The stationary block must be autonomous: no feedback from
        // diffuse states into stationary ones.
        for &i in &stationary_idx {
            for j in 0..m {
                if diffuse[j] && t[(i, j)] != 0.0 {
                    return Err(SsmError::InvalidArgument {
                        what: "Mixed initialization requires \
                               T[stationary, diffuse] = 0: the stationary \
                               block must not be driven by diffuse states",
                    });
                }
            }
        }
        let mut a1 = vec![0.0; m];
        let mut p_star = Mat::<f64>::zeros(m, m);
        let p_inf = Mat::from_fn(m, m, |i, j| if i == j && diffuse[i] { 1.0 } else { 0.0 });
        let ms = stationary_idx.len();
        if ms > 0 {
            let t_ss = Mat::from_fn(ms, ms, |i, j| t[(stationary_idx[i], stationary_idx[j])]);
            let rqr = self.rqr(0)?;
            let rqr_ss = Mat::from_fn(ms, ms, |i, j| rqr[(stationary_idx[i], stationary_idx[j])]);
            let p_ss = solve_discrete_lyapunov(t_ss.as_ref(), rqr_ss.as_ref())?;
            let c_ss: Vec<f64> = stationary_idx
                .iter()
                .map(|&i| self.state_intercept[i])
                .collect();
            let a_ss = solve_i_minus_t(t_ss.as_ref(), &c_ss)?;
            for (bi, &i) in stationary_idx.iter().enumerate() {
                a1[i] = a_ss[bi];
                for (bj, &j) in stationary_idx.iter().enumerate() {
                    p_star[(i, j)] = p_ss[(bi, bj)];
                }
            }
        }
        Ok(InitialState { a1, p_star, p_inf })
    }
}

/// Solves `(I - T) x = c` by Gaussian elimination with partial pivoting
/// (the unconditional-mean solve for stationary initialization).
fn solve_i_minus_t(t: MatRef<'_, f64>, c: &[f64]) -> Result<Vec<f64>, SsmError> {
    let n = t.nrows();
    if c.iter().all(|&v| v == 0.0) {
        // Zero intercept: the unconditional mean is exactly zero, no
        // solve needed (and no spurious singularity worries).
        return Ok(vec![0.0; n]);
    }
    let mut a = Mat::from_fn(n, n, |i, j| {
        let eye = if i == j { 1.0 } else { 0.0 };
        eye - t[(i, j)]
    });
    let mut x: Vec<f64> = c.to_vec();
    for k in 0..n {
        // Partial pivot.
        let mut piv = k;
        let mut best = a[(k, k)].abs();
        for i in (k + 1)..n {
            if a[(i, k)].abs() > best {
                best = a[(i, k)].abs();
                piv = i;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return Err(SsmError::Linalg(LinalgError::NotPositiveDefinite {
                what: "I - T is singular: the state process has a unit root, \
                       so no unconditional mean exists (use diffuse \
                       initialization for the nonstationary states)",
            }));
        }
        if piv != k {
            for j in 0..n {
                let tmp = a[(k, j)];
                a[(k, j)] = a[(piv, j)];
                a[(piv, j)] = tmp;
            }
            x.swap(k, piv);
        }
        for i in (k + 1)..n {
            let factor = a[(i, k)] / a[(k, k)];
            if factor != 0.0 {
                for j in k..n {
                    let akj = a[(k, j)];
                    a[(i, j)] -= factor * akj;
                }
                x[i] -= factor * x[k];
            }
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for j in (k + 1)..n {
            s -= a[(k, j)] * x[j];
        }
        x[k] = s / a[(k, k)];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(SsmError::NonFinite {
            what: "unconditional state mean (I - T)^{-1} c",
        });
    }
    Ok(x)
}

/// Builder for [`LinearGaussianSSM`]; see [`LinearGaussianSSM::builder`].
///
/// Required matrices: `Z`, `H`, `T`, `R`, `Q`. Optional: the intercepts
/// (default zero) and the initialization (default [`Initialization::Diffuse`],
/// the exact-diffuse prior — this library never uses the approximate
/// big-kappa initialization).
#[derive(Debug, Clone)]
pub struct SsmBuilder {
    p: usize,
    m: usize,
    r_dim: usize,
    z: Option<Mat<f64>>,
    h: Option<Mat<f64>>,
    t: Option<Mat<f64>>,
    r: Option<Mat<f64>>,
    q: Option<Mat<f64>>,
    state_intercept: Option<Vec<f64>>,
    obs_intercept: Option<Vec<f64>>,
    init: Initialization,
}

impl SsmBuilder {
    /// Sets the design matrix `Z` (`p x m`).
    pub fn z(mut self, z: Mat<f64>) -> Self {
        self.z = Some(z);
        self
    }

    /// Sets the observation covariance `H` (`p x p`, symmetric PSD).
    pub fn h(mut self, h: Mat<f64>) -> Self {
        self.h = Some(h);
        self
    }

    /// Sets the transition matrix `T` (`m x m`).
    pub fn t(mut self, t: Mat<f64>) -> Self {
        self.t = Some(t);
        self
    }

    /// Sets the disturbance selection matrix `R` (`m x r`).
    pub fn r(mut self, r: Mat<f64>) -> Self {
        self.r = Some(r);
        self
    }

    /// Sets the state-disturbance covariance `Q` (`r x r`, symmetric PSD).
    pub fn q(mut self, q: Mat<f64>) -> Self {
        self.q = Some(q);
        self
    }

    /// Sets the state intercept `c` (length `m`; default zero).
    pub fn state_intercept(mut self, c: Vec<f64>) -> Self {
        self.state_intercept = Some(c);
        self
    }

    /// Sets the observation intercept `d` (length `p`; default zero).
    pub fn obs_intercept(mut self, d: Vec<f64>) -> Self {
        self.obs_intercept = Some(d);
        self
    }

    /// Sets the initialization (default: exact diffuse).
    pub fn initialization(mut self, init: Initialization) -> Self {
        self.init = init;
        self
    }

    /// Validates dimensions, finiteness, and covariance hygiene, and
    /// returns the model.
    ///
    /// `H` and `Q` are checked for symmetry (to a `1e-8` relative
    /// tolerance, then symmetrized exactly via the shared hygiene helper)
    /// and positive semidefiniteness (via the bounded jitter-ladder
    /// Cholesky: a matrix that cannot be factorized with at most a
    /// `1e-8`-relative diagonal jitter is rejected as indefinite).
    ///
    /// # Errors
    ///
    /// * [`SsmError::MissingMatrix`] when a required matrix was not set;
    /// * [`SsmError::Dimension`] on any shape violation;
    /// * [`SsmError::NonFinite`] on NaN/infinite entries;
    /// * [`SsmError::NotPsd`] when `H` or `Q` is asymmetric or indefinite;
    /// * [`SsmError::InvalidArgument`] on zero dimensions or bad `Mixed`
    ///   flags.
    pub fn build(self) -> Result<LinearGaussianSSM, SsmError> {
        let (p, m, r_dim) = (self.p, self.m, self.r_dim);
        if p == 0 || m == 0 || r_dim == 0 {
            return Err(SsmError::InvalidArgument {
                what: "model dimensions p, m, r must all be at least 1",
            });
        }
        let z = self.z.ok_or(SsmError::MissingMatrix { what: "Z" })?;
        let h = self.h.ok_or(SsmError::MissingMatrix { what: "H" })?;
        let t = self.t.ok_or(SsmError::MissingMatrix { what: "T" })?;
        let r = self.r.ok_or(SsmError::MissingMatrix { what: "R" })?;
        let q = self.q.ok_or(SsmError::MissingMatrix { what: "Q" })?;

        check_shape(&z, p, m, "Z must be p x m")?;
        check_shape(&h, p, p, "H must be p x p")?;
        check_shape(&t, m, m, "T must be m x m")?;
        check_shape(&r, m, r_dim, "R must be m x r")?;
        check_shape(&q, r_dim, r_dim, "Q must be r x r")?;
        check_finite(&z, "Z")?;
        check_finite(&h, "H")?;
        check_finite(&t, "T")?;
        check_finite(&r, "R")?;
        check_finite(&q, "Q")?;

        let h = check_psd(h, "H")?;
        let q = check_psd(q, "Q")?;

        let state_intercept = match self.state_intercept {
            Some(c) => {
                if c.len() != m {
                    return Err(SsmError::Dimension {
                        what: "state intercept c must have length m",
                        expected: m,
                        got: c.len(),
                    });
                }
                if c.iter().any(|v| !v.is_finite()) {
                    return Err(SsmError::NonFinite {
                        what: "state intercept c",
                    });
                }
                c
            }
            None => vec![0.0; m],
        };
        let obs_intercept = match self.obs_intercept {
            Some(d) => {
                if d.len() != p {
                    return Err(SsmError::Dimension {
                        what: "observation intercept d must have length p",
                        expected: p,
                        got: d.len(),
                    });
                }
                if d.iter().any(|v| !v.is_finite()) {
                    return Err(SsmError::NonFinite {
                        what: "observation intercept d",
                    });
                }
                d
            }
            None => vec![0.0; p],
        };

        let init = match self.init {
            Initialization::Known { a1, p1 } => {
                if a1.len() != m {
                    return Err(SsmError::Dimension {
                        what: "Known initialization a1 must have length m",
                        expected: m,
                        got: a1.len(),
                    });
                }
                if a1.iter().any(|v| !v.is_finite()) {
                    return Err(SsmError::NonFinite {
                        what: "Known initialization a1",
                    });
                }
                check_shape(&p1, m, m, "Known initialization P1 must be m x m")?;
                check_finite(&p1, "Known initialization P1")?;
                // PSD hygiene on the user-supplied initial covariance;
                // the exactly-symmetrized matrix is what gets stored.
                let p1 = check_psd(p1, "P1")?;
                Initialization::Known { a1, p1 }
            }
            Initialization::Mixed { diffuse } => {
                if diffuse.len() != m {
                    return Err(SsmError::Dimension {
                        what: "Mixed initialization diffuse flags must have length m",
                        expected: m,
                        got: diffuse.len(),
                    });
                }
                Initialization::Mixed { diffuse }
            }
            other => other,
        };

        let mut h_is_diagonal = true;
        'diag: for j in 0..p {
            for i in 0..p {
                if i != j && h[(i, j)] != 0.0 {
                    h_is_diagonal = false;
                    break 'diag;
                }
            }
        }

        Ok(LinearGaussianSSM {
            p,
            m,
            r_dim,
            z: SystemMatrix::Constant(z),
            h: SystemMatrix::Constant(h),
            t: SystemMatrix::Constant(t),
            r: SystemMatrix::Constant(r),
            q: SystemMatrix::Constant(q),
            state_intercept,
            obs_intercept,
            init,
            h_is_diagonal,
        })
    }
}

/// Shape check with a static description.
fn check_shape(
    mat: &Mat<f64>,
    rows: usize,
    cols: usize,
    what: &'static str,
) -> Result<(), SsmError> {
    if mat.nrows() != rows {
        return Err(SsmError::Dimension {
            what,
            expected: rows,
            got: mat.nrows(),
        });
    }
    if mat.ncols() != cols {
        return Err(SsmError::Dimension {
            what,
            expected: cols,
            got: mat.ncols(),
        });
    }
    Ok(())
}

/// Finiteness check.
fn check_finite(mat: &Mat<f64>, what: &'static str) -> Result<(), SsmError> {
    for j in 0..mat.ncols() {
        for i in 0..mat.nrows() {
            if !mat[(i, j)].is_finite() {
                return Err(SsmError::NonFinite { what });
            }
        }
    }
    Ok(())
}

/// Symmetric-PSD hygiene check for a covariance matrix: symmetry within a
/// `1e-8` relative tolerance, exact symmetrization (shared hygiene
/// helper), then positive semidefiniteness via the bounded jitter-ladder
/// Cholesky — a matrix the ladder cannot factorize is genuinely
/// indefinite. Returns the exactly-symmetrized matrix.
fn check_psd(mat: Mat<f64>, what: &'static str) -> Result<Mat<f64>, SsmError> {
    let n = mat.nrows();
    let mut scale = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            scale = scale.max(mat[(i, j)].abs());
        }
    }
    for j in 0..n {
        for i in 0..j {
            if (mat[(i, j)] - mat[(j, i)]).abs() > 1e-8 * scale.max(1.0) {
                return Err(SsmError::NotPsd { what });
            }
        }
    }
    let sym = symmetrize(mat.as_ref())?;
    match jittered_cholesky(sym.as_ref()) {
        Ok(_) => Ok(sym),
        Err(LinalgError::JitterExhausted { .. }) => Err(SsmError::NotPsd { what }),
        Err(e) => Err(SsmError::Linalg(e)),
    }
}
