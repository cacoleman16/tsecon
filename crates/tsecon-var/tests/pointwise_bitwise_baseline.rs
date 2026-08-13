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
//! # Why the stored fingerprints are `#[ignore]`d
//!
//! These are exact double-precision fingerprints of `faer` matrix products,
//! captured on Apple silicon. `faer` dispatches different SIMD kernels on
//! different CPUs, so the last bits differ on x86-64 and the constants below
//! fail there for reasons that have nothing to do with this crate.
//!
//! That is not a quirk of this file, it is **library policy**: results are
//! bit-reproducible *per platform, not across* them, and a stored float
//! snapshot must therefore be compared with a tolerance, never with
//! `to_bits()`. This repository has been bitten by exactly this once before,
//! when a `to_bits()` snapshot captured on macOS failed Linux CI by one ulp.
//!
//! So the three hash tests are marked `#[ignore]`. They remain a genuine
//! developer check — run them with `--ignored` on the machine that captured
//! them, against the unmodified parent commit, and a mismatch is conclusive —
//! but they are the wrong instrument for CI.
//!
//! # What guards the invariant in CI instead
//!
//! The question that actually matters is *"does asking for a simultaneous band
//! change the pointwise output?"*, and that is a **same-run** comparison: both
//! sides are computed by the same binary on the same CPU with the same kernels,
//! so it can be bit-exact and is portable. Those tests are below and they run
//! everywhere.
//!
//! Cross-platform agreement of the pointwise numbers themselves is already
//! pinned, with a tolerance, by the statsmodels golden gate in
//! `tests/irf_bands_golden.rs`.

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
#[ignore = "stored f64 bit patterns are platform-specific; run with --ignored on the capture machine"]
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
#[ignore = "stored f64 bit patterns are platform-specific; run with --ignored on the capture machine"]
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
#[ignore = "stored f64 bit patterns are platform-specific; run with --ignored on the capture machine"]
fn bootstrap_percentile_bands_are_bit_identical() {
    assert_eq!(
        bootstrap_bands_hash(),
        BOOTSTRAP_BANDS,
        "bootstrap_irf_bands moved: the pointwise percentile band is not \
         bit-identical to the pre-change output"
    );
}

// ---------------------------------------------------------------------------
// Portable same-run invariants
//
// These answer the question the stored hashes were reaching for, without
// depending on which SIMD kernel the host dispatches: both sides are computed
// by this binary, on this CPU, in this run. A difference here is a real
// difference, on every platform.
// ---------------------------------------------------------------------------

/// Asking for a simultaneous band must not perturb the percentile band it sits
/// beside. The pointwise `lower`/`upper`/`point`/`se` must be bit-identical
/// whether or not a sup-t band was also requested, at the same seed.
#[test]
fn requesting_a_simultaneous_band_does_not_move_the_bootstrap_pointwise_output() {
    let fit = irf_fit();
    let endog = fit.endog.as_ref();
    let plain = bootstrap_irf_bands(
        endog,
        2,
        Trend::Constant,
        6,
        true,
        false,
        0.10,
        64,
        20260807,
        false,
    )
    .expect("plain bootstrap bands");
    let with_band = tsecon_var::irf_bootstrap::bootstrap_irf_bands_simultaneous(
        endog,
        2,
        Trend::Constant,
        6,
        true,
        false,
        0.10,
        64,
        20260807,
        false,
        tsecon_var::irf_asymptotic::BandMethod::SupT,
        tsecon_var::irf_asymptotic::IrfBandScope::Horizon,
    )
    .expect("simultaneous bootstrap bands");

    let same = |a: &[Mat<f64>], b: &[Mat<f64>], what: &str| {
        assert_eq!(a.len(), b.len(), "{what}: horizon count moved");
        for (h, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    assert_eq!(
                        x[(i, j)].to_bits(),
                        y[(i, j)].to_bits(),
                        "{what}: cell (h={h}, {i}, {j}) moved when a simultaneous \
                         band was requested"
                    );
                }
            }
        }
    };
    same(&plain.point, &with_band.point, "point");
    same(&plain.se, &with_band.se, "se");
    same(&plain.lower, &with_band.lower, "lower");
    same(&plain.upper, &with_band.upper, "upper");
}

/// The same invariant for `var_forecast`: the marginal interval is untouched by
/// the presence of a simultaneous one.
#[test]
fn requesting_a_simultaneous_band_does_not_move_the_forecast_marginal_interval() {
    let fit = var_fit();
    for &alpha in &[0.05_f64, 0.10, 0.32] {
        let plain = fit.forecast_interval(12, alpha).expect("marginal interval");
        let with_band = fit
            .forecast_interval_simultaneous(
                12,
                alpha,
                tsecon_var::irf_asymptotic::BandMethod::SupT,
                tsecon_var::forecast::ForecastBandScope::All,
                20260807,
                2_000,
            )
            .expect("simultaneous interval");
        let same_mat = |x: &Mat<f64>, y: &Mat<f64>, what: &str| {
            assert_eq!(x.nrows(), y.nrows(), "{what}: row count moved");
            assert_eq!(x.ncols(), y.ncols(), "{what}: col count moved");
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    assert_eq!(
                        x[(i, j)].to_bits(),
                        y[(i, j)].to_bits(),
                        "alpha={alpha}: {what} ({i}, {j}) moved when a \
                         simultaneous band was requested"
                    );
                }
            }
        };
        same_mat(&plain.lower, &with_band.lower, "marginal lower");
        same_mat(&plain.upper, &with_band.upper, "marginal upper");
    }
}
