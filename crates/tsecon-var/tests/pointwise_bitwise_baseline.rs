//! Bit-identity gate for the **pointwise** bands.
//!
//! The simultaneous (sup-t) band work is purely additive: it must not move a
//! single bit of the existing pointwise output, which is golden-gated against
//! statsmodels (`tests/irf_bands_golden.rs`) and published in the library's
//! interval-coverage audit. Rounding-level drift would invalidate both without
//! failing a `rtol = 1e-6` comparison.
//!
//! So this file fingerprints every pointwise surface at the *bit* level: an
//! FNV-1a hash over `f64::to_bits()` of every emitted number, in a fixed order.
//! The constants below were captured by running this file against the code as
//! it stood *before* the simultaneous-band work (commit `0989748`), and must
//! never be regenerated to make a test pass.
//!
//! # Read this before believing a failure
//!
//! These are exact double-precision fingerprints of `faer` matrix products.
//! `faer` dispatches different SIMD kernels on different CPUs, so a mismatch on
//! a machine other than the one that captured them (Apple silicon, macOS 25.5)
//! does **not** by itself prove the pointwise band changed. The way to tell is
//! to re-capture on the unmodified parent commit on the same machine and
//! compare. On the capture machine a mismatch is conclusive.
//!
//! The tolerance-based golden gate against statsmodels lives in
//! `tests/irf_bands_golden.rs`; this file is the finer sieve underneath it.

mod common;

use common::{as_mat, load_fixture};
use tsecon_linalg::faer::Mat;
use tsecon_var::irf_asymptotic::irf_asymptotic_se;
use tsecon_var::{bootstrap_irf_bands, Trend, VarSpec};

/// FNV-1a over the raw bit patterns of a stream of doubles.
struct BitHash(u64);

impl BitHash {
    fn new() -> Self {
        BitHash(0xcbf2_9ce4_8422_2325)
    }
    fn push(&mut self, x: f64) {
        for b in x.to_bits().to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn push_mat(&mut self, m: &Mat<f64>) {
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                self.push(m[(i, j)]);
            }
        }
    }
    fn push_cube(&mut self, c: &[Mat<f64>]) {
        for m in c {
            self.push_mat(m);
        }
    }
    fn finish(self) -> u64 {
        self.0
    }
}

const H: usize = 10;

fn irf_fit() -> tsecon_var::VarResults {
    let fx = load_fixture("var_irf_bands.json");
    let data = as_mat(&fx["data"]);
    VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

fn var_fit() -> tsecon_var::VarResults {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

fn asymptotic_se_hash() -> u64 {
    let res = irf_fit();
    let mut h = BitHash::new();
    for &orth in &[false, true] {
        for &cumulative in &[false, true] {
            let se = irf_asymptotic_se(&res, H, orth, cumulative).unwrap();
            h.push_cube(&se);
        }
    }
    h.finish()
}

fn forecast_interval_hash() -> u64 {
    let res = var_fit();
    let mut h = BitHash::new();
    for &alpha in &[0.05, 0.10, 0.32] {
        let fc = res.forecast_interval(12, alpha).unwrap();
        h.push_mat(&fc.point);
        h.push_mat(&fc.lower);
        h.push_mat(&fc.upper);
    }
    h.finish()
}

fn bootstrap_bands_hash() -> u64 {
    let fx = load_fixture("var_irf_bands.json");
    let data = as_mat(&fx["data"]);
    let mut h = BitHash::new();
    for &orth in &[false, true] {
        let b = bootstrap_irf_bands(
            data.as_ref(),
            2,
            Trend::Constant,
            6,
            orth,
            false,
            0.1,
            64,
            20260807,
            false,
        )
        .unwrap();
        h.push_cube(&b.point);
        h.push_cube(&b.se);
        h.push_cube(&b.lower);
        h.push_cube(&b.upper);
    }
    h.finish()
}

// ---------------------------------------------------------------------------
// The captured fingerprints. Do not regenerate to make a test pass.
// ---------------------------------------------------------------------------

/// `irf_asymptotic_se` over all four (orth, cumulative) combinations, H = 10,
/// on the statsmodels `var_irf_bands.json` VAR(2) fit.
const ASYMPTOTIC_SE: u64 = 0x2200_6d8d_c577_ef99;
/// `VarResults::forecast_interval(12, alpha)` point/lower/upper at
/// alpha = 0.05, 0.10, 0.32 on the `var.json` VAR(2) fit.
const FORECAST_INTERVAL: u64 = 0x5d1a_7c6d_9b65_c991;
/// `bootstrap_irf_bands` point/se/lower/upper, seed 20260807, n_boot = 64.
const BOOTSTRAP_BANDS: u64 = 0xfbc9_236f_5039_9b2b;

/// The delta-method standard errors that the `method="asymptotic"` arm of
/// `var_irf_bands` turns into `point ± z·se` are bit-for-bit unchanged.
#[test]
fn asymptotic_standard_errors_are_bit_identical() {
    assert_eq!(
        asymptotic_se_hash(),
        ASYMPTOTIC_SE,
        "irf_asymptotic_se moved: the pointwise asymptotic IRF band is not \
         bit-identical to the pre-change output"
    );
}

/// `var_forecast`'s point path and marginal interval are bit-for-bit unchanged.
#[test]
fn forecast_interval_is_bit_identical() {
    assert_eq!(
        forecast_interval_hash(),
        FORECAST_INTERVAL,
        "forecast_interval moved: the marginal var_forecast band is not \
         bit-identical to the pre-change output"
    );
}

/// The bootstrap percentile bands (point, se, lower, upper) are bit-for-bit
/// unchanged at a fixed seed — including the resampling stream itself, since a
/// changed draw sequence would change every number here.
#[test]
fn bootstrap_percentile_bands_are_bit_identical() {
    assert_eq!(
        bootstrap_bands_hash(),
        BOOTSTRAP_BANDS,
        "bootstrap_irf_bands moved: the pointwise percentile band is not \
         bit-identical to the pre-change output"
    );
}
