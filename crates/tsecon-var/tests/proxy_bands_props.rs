//! Property and Monte-Carlo validation for the Jentsch-Lunsford moving-block
//! bootstrap bands (`tsecon_var::proxy_bands`).
//!
//! The golden test in `proxy_bands_golden.rs` pins the *arithmetic* against
//! an independent NumPy transcription. It cannot pin the *theory*, because
//! no external package implements these bands. This file carries that load
//! on four fronts, each aimed at one of the failure modes catalogued in
//! `docs/roadmap/15-proxy-svar-bands.md`:
//!
//! * **Reproducibility and structure** — a seed fixes the bands bit for bit;
//!   endpoints are ordered and finite; the `h = 0` cell of the normalizing
//!   variable is exactly degenerate (the free proof that the unit-effect
//!   normalization is re-imposed *inside* every draw).
//! * **Joint blocking** — the bootstrap identifying moment sits around the
//!   sample value rather than around zero, and the arithmetic of what
//!   independent block indices would do instead is measured here rather
//!   than asserted.
//! * **The wild bootstrap is invalid** — with a common Rademacher draw the
//!   identifying moment is bit-identical across draws, measured against the
//!   crate's own weight generator and its own resampling helper; the
//!   resulting bands are correspondingly too short.
//! * **Monte-Carlo coverage** — on a known-truth proxy-SVAR DGP the nominal
//!   90% Hall band covers the true impulse response near 90%, with left and
//!   right non-coverage reported separately (a total that looks fine can
//!   hide badly asymmetric one-sided coverage under skew), and the wild arm
//!   under-covers by comparison.
//!
//! The coverage numbers are Monte-Carlo estimates at a few hundred
//! replications, so the assertions are deliberately loose bands around the
//! nominal level: they are sized to catch the catastrophic failures (bands
//! orders of magnitude too wide under independent blocking, ~70-80% under
//! sign fixing, a systematic shortfall under the wild bootstrap), not to
//! certify the third decimal place.

mod common;

use common::Lcg;
use tsecon_bootstrap::{indices, BlockScheme, WildWeights};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;
use tsecon_var::proxy_bands::{
    default_block_length, proxy_svar_band_block_sensitivity, proxy_svar_bands,
    proxy_svar_bands_from_starts, wild_common_draw, ProxyBandMethod, ProxyBandSpec, ProxyBands,
};
use tsecon_var::{ma_rep, Trend, VarSpec};

// ------------------------------------------------------------------ the DGP

/// VAR(1) slope of the test DGP.
const A_TRUE: [[f64; 2]; 2] = [[0.60, 0.10], [0.20, 0.50]];
/// Structural impact matrix; column 0 is the shock the proxy identifies.
const H_TRUE: [[f64; 2]; 2] = [[1.00, 0.00], [0.50, 0.80]];
const NORM_VAR: usize = 0;
const UNIT: f64 = 1.0;

/// True unit-effect impact vector `b = unit * H[:, 0] / H[norm_var, 0]`.
fn true_impact() -> Vec<f64> {
    let scale = UNIT / H_TRUE[NORM_VAR][0];
    (0..2).map(|i| scale * H_TRUE[i][0]).collect()
}

/// True structural impulse response `theta_h = Psi_h(A_true) b`.
fn true_irf(horizon: usize) -> Vec<Vec<f64>> {
    let a = Mat::from_fn(2, 2, |i, j| A_TRUE[i][j]);
    let psi = ma_rep(std::slice::from_ref(&a), horizon).unwrap();
    let b = true_impact();
    psi.iter()
        .map(|p| {
            (0..2)
                .map(|i| (0..2).map(|k| p[(i, k)] * b[k]).sum::<f64>())
                .collect()
        })
        .collect()
}

/// Simulate the DGP: a stationary VAR(1) driven by `u_t = H eps_t`, plus a
/// proxy `m_t = phi * eps_{1t} + noise` that is relevant for the target
/// shock and orthogonal to the other by construction.
///
/// `nan_prefix` dates at the start of the residual sample are marked
/// unavailable (`NaN`), as a real narrative or high-frequency instrument
/// would be.
fn simulate(
    rng: &mut Lcg,
    n_obs: usize,
    phi: f64,
    noise: f64,
    nan_prefix: usize,
) -> (Mat<f64>, Vec<f64>) {
    let burn = 100usize;
    let total = n_obs + burn;
    let mut eps = vec![[0.0f64; 2]; total];
    for e in eps.iter_mut() {
        e[0] = rng.gaussian();
        e[1] = rng.gaussian();
    }
    let mut y = vec![[0.0f64; 2]; total];
    for t in 1..total {
        for i in 0..2 {
            let u = H_TRUE[i][0] * eps[t][0] + H_TRUE[i][1] * eps[t][1];
            y[t][i] = 0.1 + A_TRUE[i][0] * y[t - 1][0] + A_TRUE[i][1] * y[t - 1][1] + u;
        }
    }
    let data = Mat::from_fn(n_obs, 2, |t, j| y[burn + t][j]);
    // Aligned to the residual sample of a VAR(1): drop one presample row.
    let t_eff = n_obs - 1;
    let mut proxy = vec![0.0f64; t_eff];
    for (k, slot) in proxy.iter_mut().enumerate() {
        *slot = phi * eps[burn + 1 + k][0] + noise * rng.gaussian();
    }
    for slot in proxy.iter_mut().take(nan_prefix) {
        *slot = f64::NAN;
    }
    (data, proxy)
}

fn spec(n_boot: usize, seed: u64, horizon: usize) -> ProxyBandSpec {
    ProxyBandSpec {
        lags: 1,
        trend: Trend::Constant,
        horizon,
        norm_var: NORM_VAR,
        unit: UNIT,
        alpha: 0.10,
        n_boot,
        seed,
        method: ProxyBandMethod::MovingBlock,
        block_length: None,
        robust_f: true,
    }
}

// ------------------------------------------------- reproducibility, structure

/// The same seed produces bit-identical bands, and `par_replicate`'s
/// thread-count independence carries through.
#[test]
fn bands_are_bit_reproducible_from_the_seed() {
    let mut rng = Lcg::new(11);
    let (data, proxy) = simulate(&mut rng, 120, 0.8, 0.5, 4);
    let sp = spec(200, 20260805, 6);
    let a = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
    let b = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
    for h in 0..a.lower.len() {
        for i in 0..2 {
            assert_eq!(a.lower[h][i].to_bits(), b.lower[h][i].to_bits());
            assert_eq!(a.upper[h][i].to_bits(), b.upper[h][i].to_bits());
            assert_eq!(a.lower_efron[h][i].to_bits(), b.lower_efron[h][i].to_bits());
            assert_eq!(a.se[h][i].to_bits(), b.se[h][i].to_bits());
        }
    }
    assert_eq!(a.n_failed, b.n_failed);
}

/// Endpoints are ordered and finite, standard errors are non-negative, the
/// point estimate matches `proxy_svar`, and the failure accounting adds up.
#[test]
fn bands_are_structurally_sound() {
    let mut rng = Lcg::new(12);
    let (data, proxy) = simulate(&mut rng, 150, 0.8, 0.5, 5);
    let sp = spec(300, 7, 8);
    let bands = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();

    assert_eq!(bands.n_used + bands.n_failed, bands.n_boot);
    assert_eq!(bands.n_failed, bands.failures.total());
    assert!(bands.asymptotically_valid);
    assert!(bands.validity_note.contains("Jentsch-Lunsford"));
    assert_eq!(bands.block_length, default_block_length(149));

    for h in 0..=sp.horizon {
        for i in 0..2 {
            assert!(
                bands.lower[h][i] <= bands.upper[h][i],
                "Hall order at ({h},{i})"
            );
            assert!(
                bands.lower_efron[h][i] <= bands.upper_efron[h][i],
                "Efron order at ({h},{i})"
            );
            assert!(bands.se[h][i] >= 0.0 && bands.se[h][i].is_finite());
            assert!(bands.point[h][i].is_finite());
        }
    }

    // The point estimate is exactly `proxy_svar` on the same reduced form.
    let fit = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    let psi = fit.ma_rep(sp.horizon).unwrap();
    let direct = tsecon_ident::proxy_svar(
        fit.resid.as_ref(),
        &proxy,
        &psi,
        fit.sigma_u.as_ref(),
        NORM_VAR,
        UNIT,
        true,
    )
    .unwrap();
    for h in 0..=sp.horizon {
        for i in 0..2 {
            assert_eq!(bands.point[h][i], direct.irf[h][i], "point at ({h},{i})");
        }
    }
}

/// The `h = 0` band for the normalizing variable is degenerate at `unit`,
/// exactly, through the seeded path as well as the pinned-starts path.
///
/// If the normalization were hoisted out of the loop (`b* = (unit /
/// gammahat[norm_var]) * gamma*`) this cell would carry a small but nonzero
/// interval — the exact tell for that bug.
#[test]
fn per_draw_normalization_makes_the_impact_cell_degenerate() {
    let mut rng = Lcg::new(13);
    let (data, proxy) = simulate(&mut rng, 130, 0.8, 0.5, 3);
    for method in [ProxyBandMethod::MovingBlock, ProxyBandMethod::Wild] {
        let mut sp = spec(250, 99, 5);
        sp.method = method;
        let b = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
        assert_eq!(b.lower[0][NORM_VAR], UNIT, "{method:?} Hall lower");
        assert_eq!(b.upper[0][NORM_VAR], UNIT, "{method:?} Hall upper");
        assert_eq!(b.lower_efron[0][NORM_VAR], UNIT, "{method:?} Efron lower");
        assert_eq!(b.upper_efron[0][NORM_VAR], UNIT, "{method:?} Efron upper");
        assert_eq!(b.se[0][NORM_VAR], 0.0, "{method:?} SD");
        // The other impact cell is not degenerate, so this is the
        // normalization and not a dead bootstrap.
        assert!(b.upper[0][1] - b.lower[0][1] > 1e-6, "{method:?} free cell");
    }
}

/// Hall and Efron are genuinely different intervals here. They coincide only
/// when the bootstrap distribution is symmetric about the point estimate,
/// which a ratio estimand is not, so shipping only one of them would be a
/// silent choice.
#[test]
fn hall_and_efron_intervals_differ() {
    let mut rng = Lcg::new(14);
    let (data, proxy) = simulate(&mut rng, 140, 0.6, 0.7, 4);
    let bands = proxy_svar_bands(data.as_ref(), &proxy, &spec(400, 5, 6)).unwrap();
    let mut max_gap = 0.0f64;
    for h in 0..bands.lower.len() {
        for i in 0..2 {
            max_gap = max_gap.max((bands.lower[h][i] - bands.lower_efron[h][i]).abs());
            max_gap = max_gap.max((bands.upper[h][i] - bands.upper_efron[h][i]).abs());
        }
    }
    assert!(
        max_gap > 1e-6,
        "Hall and Efron endpoints are indistinguishable (max gap {max_gap:e}); one of them is \
         probably not being computed"
    );
}

// -------------------------------------------------------- the block scheme

/// The seeded path draws **moving** blocks, and this pins that byte for byte.
///
/// The golden test runs through `proxy_svar_bands_from_starts`, which takes
/// the starts as an argument and so never touches the scheme at all. Without
/// this test, swapping `BlockScheme::MovingBlock` for
/// `BlockScheme::CircularBlock` inside `proxy_svar_bands` is a one-token
/// change that the whole suite passes. That swap is failure mode 10 of
/// `docs/roadmap/15-proxy-svar-bands.md`: under the circular block every
/// observation appears in exactly `ell` blocks, which makes **grand-mean**
/// centering the correct fix, so a circular draw combined with this module's
/// position-wise centering re-introduces precisely the mean bias the
/// centering exists to remove.
///
/// The pin: rebuild the block starts outside the library from
/// `Stream::substreams(seed, n_boot)` — the same substreams `par_replicate`
/// hands each replication — under `BlockScheme::MovingBlock`, and require
/// the explicit-starts path fed those starts to reproduce the seeded path
/// exactly. A circular draw changes both the bound passed to the uniform
/// sampler (`T` rather than `T - ell + 1`, so a different bitmask consumes
/// the stream differently) and the block layout (wrap-around), so it cannot
/// survive a bitwise comparison.
#[test]
fn the_seeded_path_draws_moving_blocks_not_circular_ones() {
    let mut rng = Lcg::new(23);
    let (data, proxy) = simulate(&mut rng, 130, 0.8, 0.5, 4);
    let ell = 9usize;
    let mut sp = spec(64, 20260806, 4);
    sp.block_length = Some(ell);
    let t = data.nrows() - sp.lags;

    let starts: Vec<Vec<usize>> = Stream::substreams(sp.seed, sp.n_boot)
        .expect("substreams")
        .into_iter()
        .map(|mut s| {
            let idx = indices(BlockScheme::MovingBlock { block_length: ell }, t, &mut s)
                .expect("moving-block indices");
            // Blocks are laid end to end, so every `ell`-th position is a
            // block start.
            idx.iter().step_by(ell).copied().collect()
        })
        .collect();

    let seeded = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
    let pinned = proxy_svar_bands_from_starts(data.as_ref(), &proxy, &sp, &starts).unwrap();

    assert_eq!(seeded.n_failed, pinned.n_failed);
    for r in 0..sp.n_boot {
        assert_eq!(
            seeded.gamma_norm_draws[r].to_bits(),
            pinned.gamma_norm_draws[r].to_bits(),
            "draw {r}: the seeded scheme is not the moving block",
        );
    }
    for h in 0..seeded.lower.len() {
        for i in 0..2 {
            assert_eq!(seeded.lower[h][i].to_bits(), pinned.lower[h][i].to_bits());
            assert_eq!(seeded.upper[h][i].to_bits(), pinned.upper[h][i].to_bits());
        }
    }
}

/// Blocks come from the `T - ell + 1` **overlapping candidate starts** and
/// never wrap around the end of the sample.
///
/// A second, independent pin on the same thing, stated as a property of the
/// output rather than as a reconstruction of the RNG. Set `ell = T - 1`:
/// there are then exactly two candidate starts (`0` and `1`) and exactly two
/// blocks, so the moving-block rule admits exactly **four** index vectors.
/// Enumerate all four through the explicit-starts path and require every
/// seeded draw's `gamma*[norm_var]` to be one of them, bit for bit.
///
/// Under a circular scheme the second block's single observation is drawn
/// from all `T` positions rather than from `{0, 1}`, so all but ~5% of draws
/// land outside the admissible set immediately — and the first block wraps
/// besides.
#[test]
fn blocks_come_from_the_overlapping_candidates_and_never_wrap() {
    let mut rng = Lcg::new(24);
    let (data, proxy) = simulate(&mut rng, 41, 0.9, 0.4, 0);
    let t = data.nrows() - 1;
    let ell = t - 1;
    let mut sp = spec(80, 4242, 3);
    sp.block_length = Some(ell);

    let enumerated = proxy_svar_bands_from_starts(
        data.as_ref(),
        &proxy,
        &sp,
        &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
    )
    .unwrap();
    let admissible: Vec<u64> = enumerated
        .gamma_norm_draws
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(admissible.len(), 4);
    // The four must be distinct, or "is one of them" would be vacuous.
    let mut uniq = admissible.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), 4, "the enumerated index vectors must differ");

    let seeded = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
    for (r, g) in seeded.gamma_norm_draws.iter().enumerate() {
        assert!(
            admissible.contains(&g.to_bits()),
            "draw {r}: gamma*[norm_var] = {g} is not produced by any of the {} candidate-start \
             combinations the moving block admits — the blocks are being drawn from somewhere \
             else (a circular scheme draws starts from 0..T and wraps)",
            admissible.len(),
        );
    }
    let mut seen: Vec<u64> = seeded
        .gamma_norm_draws
        .iter()
        .map(|v| v.to_bits())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert!(
        seen.len() >= 2,
        "only {} distinct draw value(s) appeared, so 'is one of the four' has no teeth",
        seen.len()
    );
}

// ---------------------------------------------------------- joint blocking

/// The bootstrap identifying moment sits around the sample value and keeps
/// its sign, which is what joint blocking buys. The counterfactual is
/// measured in the same test rather than asserted: drawing *independent*
/// block indices for `u` and `m` collapses the moment toward zero and makes
/// the denominator change sign in about half the draws, so `rho*` becomes a
/// ratio of two mean-zero noise terms.
#[test]
fn joint_blocking_preserves_the_identifying_moment() {
    let mut rng = Lcg::new(15);
    let (data, proxy) = simulate(&mut rng, 160, 0.8, 0.5, 4);
    let bands = proxy_svar_bands(data.as_ref(), &proxy, &spec(400, 3, 4)).unwrap();

    let finite: Vec<f64> = bands
        .gamma_norm_draws
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let flips = finite
        .iter()
        .filter(|v| v.signum() != bands.point_gamma_norm.signum())
        .count();
    assert!(
        (mean - bands.point_gamma_norm).abs() < 0.4 * bands.point_gamma_norm.abs(),
        "mean gamma*[norm_var] = {mean:.4} against a sample value of {:.4}: a mean near zero is \
         the signature of independent block indices",
        bands.point_gamma_norm
    );
    assert!(
        flips * 20 < finite.len(),
        "{flips}/{} draws flipped the sign of gamma*[norm_var] on a strong instrument; under \
         independent blocking this runs near 50%",
        finite.len()
    );

    // The counterfactual, measured on the same residuals: pair the residual
    // blocks with an INDEPENDENTLY drawn set of proxy blocks and the moment
    // collapses toward zero.
    let fit = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    let t = fit.resid.nrows();
    let ell = default_block_length(t);
    let mut lcg = Lcg::new(777);
    let (mut joint_sum, mut indep_sum, mut indep_flips, mut n) = (0.0, 0.0, 0usize, 0usize);
    for _ in 0..200 {
        let ia = block_starts(&mut lcg, t, ell);
        let ib = block_starts(&mut lcg, t, ell);
        joint_sum += raw_moment(&fit.resid, &proxy, &ia, &ia, NORM_VAR);
        let g = raw_moment(&fit.resid, &proxy, &ia, &ib, NORM_VAR);
        indep_sum += g;
        if g.signum() != bands.point_gamma_norm.signum() {
            indep_flips += 1;
        }
        n += 1;
    }
    let joint_mean = joint_sum / n as f64;
    let indep_mean = indep_sum / n as f64;
    assert!(
        indep_mean.abs() < 0.25 * joint_mean.abs(),
        "independent indices should collapse the moment (joint {joint_mean:.4}, independent \
         {indep_mean:.4}) — if they do not, this test has no teeth"
    );
    assert!(
        indep_flips * 5 > n,
        "independent indices should flip the denominator's sign constantly, got \
         {indep_flips}/{n}"
    );
}

/// Moving-block index vector from a test-local RNG (the library's own engine
/// is exercised elsewhere; here the point is to build a *mismatched* pair,
/// which the library deliberately offers no way to do).
fn block_starts(rng: &mut Lcg, t: usize, ell: usize) -> Vec<usize> {
    let n_starts = t - ell + 1;
    let mut out = Vec::with_capacity(t);
    while out.len() < t {
        let s = (rng.next_u64() % n_starts as u64) as usize;
        let take = ell.min(t - out.len());
        out.extend(s..s + take);
    }
    out
}

/// `mean_O (m* - mbar)(u*_norm - ubar)` for a residual index vector and a
/// (possibly different) proxy index vector.
fn raw_moment(u: &Mat<f64>, proxy: &[f64], iu: &[usize], im: &[usize], norm_var: usize) -> f64 {
    let o: Vec<usize> = (0..iu.len())
        .filter(|&t| proxy[im[t]].is_finite())
        .collect();
    let no = o.len() as f64;
    let mbar = o.iter().map(|&t| proxy[im[t]]).sum::<f64>() / no;
    let ubar = o.iter().map(|&t| u[(iu[t], norm_var)]).sum::<f64>() / no;
    o.iter()
        .map(|&t| (proxy[im[t]] - mbar) * (u[(iu[t], norm_var)] - ubar))
        .sum::<f64>()
        / no
}

// ------------------------------------------------- the wild bootstrap arm

/// The argument against the wild bootstrap, measured against this crate's
/// own Rademacher generator and its own resampling helper: with a common
/// draw on residuals *and* proxy, `sum_t m*_t u*_t'` is **bit-identical** in
/// every one of 200 draws.
///
/// The identifying moment therefore carries no bootstrap variability, so the
/// sampling uncertainty of the identification step is missing from the
/// bands. That is a first-order failure: it does not shrink as `T` grows.
#[test]
fn wild_rademacher_freezes_the_identifying_moment() {
    let mut rng = Lcg::new(16);
    let (data, proxy) = simulate(&mut rng, 140, 0.8, 0.5, 4);
    let fit = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    let t = fit.resid.nrows();
    // NaN entries contribute nothing to the moment either way; zero them so
    // the comparison is over a finite quantity.
    let m: Vec<f64> = proxy
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();

    let base: Vec<f64> = (0..2)
        .map(|j| (0..t).map(|i| m[i] * fit.resid[(i, j)]).sum::<f64>())
        .collect();

    let mut stream = Stream::new(20260805);
    let mut worst = 0.0f64;
    let n_draws = 200;
    for _ in 0..n_draws {
        let w = WildWeights::Rademacher.sample(t, &mut stream);
        let (ustar, mstar) = wild_common_draw(fit.resid.as_ref(), &m, &w);
        for j in 0..2 {
            let got = (0..t).map(|i| mstar[i] * ustar[(i, j)]).sum::<f64>();
            assert_eq!(
                got.to_bits(),
                base[j].to_bits(),
                "moment moved at column {j}"
            );
            worst = worst.max((got - base[j]).abs());
        }
    }
    assert_eq!(
        worst, 0.0,
        "max deviation over {n_draws} draws must be exactly 0"
    );
}

/// The wild arm is reachable, is labelled, and produces materially shorter
/// bands than the moving-block arm at the horizons where the identification
/// step dominates the variance.
#[test]
fn wild_bands_are_labelled_and_too_short() {
    let mut rng = Lcg::new(17);
    let (data, proxy) = simulate(&mut rng, 150, 0.8, 0.5, 4);
    let mut sp = spec(400, 21, 6);
    let mbb = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
    sp.method = ProxyBandMethod::Wild;
    let wild = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();

    assert!(!wild.asymptotically_valid);
    assert!(wild.validity_note.contains("NOT ASYMPTOTICALLY VALID"));
    assert!(wild.validity_note.contains("Jentsch"));
    assert!(mbb.asymptotically_valid);

    // Impact response of the non-normalized variable: pure identification
    // uncertainty at h = 0, since Psi_0 = I carries no coefficient error.
    let w_wild = wild.upper[0][1] - wild.lower[0][1];
    let w_mbb = mbb.upper[0][1] - mbb.lower[0][1];
    assert!(
        w_wild < 0.6 * w_mbb,
        "wild impact band width {w_wild:.4} should be far below the moving-block width \
         {w_mbb:.4}: at h = 0 the only uncertainty is the identification step, which the wild \
         bootstrap freezes"
    );
}

// --------------------------------------------------- failures and NaN masks

/// Failed draws are counted by reason and reported, never dropped. A proxy
/// available on only a handful of dates makes the failure path reachable:
/// the availability pattern is itself resampled, so some draws retain fewer
/// than three finite entries.
#[test]
fn failed_draws_are_counted_and_reported() {
    let mut rng = Lcg::new(18);
    let (data, mut proxy) = simulate(&mut rng, 90, 0.9, 0.3, 0);
    // Keep only five scattered dates; everything else is unavailable.
    let keep = [7usize, 21, 33, 50, 66];
    for (k, slot) in proxy.iter_mut().enumerate() {
        if !keep.contains(&k) {
            *slot = f64::NAN;
        }
    }
    let bands = proxy_svar_bands(data.as_ref(), &proxy, &spec(500, 31, 4)).unwrap();

    assert_eq!(bands.n_proxy, keep.len());
    assert_eq!(bands.n_used + bands.n_failed, bands.n_boot);
    assert_eq!(bands.n_failed, bands.failures.total());
    assert!(
        bands.n_failed > 0,
        "a 5-observation proxy should make some draws fail; the counting path is then untested"
    );
    // Every failed draw carries NaN diagnostics rather than being removed
    // from the record.
    let nan_count = bands.gamma_norm_draws.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, bands.n_failed);
    assert_eq!(bands.first_stage_f_draws.len(), bands.n_boot);
    // Above 1% failures the result says so in words.
    if bands.n_failed * 100 > bands.n_boot {
        let w = bands.failure_warning.as_ref().expect("warning above 1%");
        assert!(w.contains("not trustworthy"));
    }
}

/// The `non_finite` guard fires, is counted under its own reason, and keeps
/// the overflowing draws out of the quantiles.
///
/// Reaching it needs a deliberate setup, and the setup is worth explaining
/// because it also documents what the guard is *not*. `BandFailures`
/// describes the classic trigger as an explosive `Ahat*` overflowing the MA
/// recursion — but that is nearly unreachable here, because `proxy_svar_bands`
/// computes the **point** estimate's `Psi_h` at the same horizon first, so a
/// system explosive enough to overflow a draw has already errored the whole
/// call out. (Measured while writing this test: on a near-unit-root sample
/// it took `horizon = 16000` to get even one draw in a hundred to overflow
/// before the point estimate did.)
///
/// The route that *is* reachable is the other one in the same guard: the
/// per-draw normalization. `b* = unit * rho*` with a heavy-tailed `rho*`
/// from a weak instrument, and a `unit` chosen large enough that a draw with
/// `|rho*| > 3` overflows. That is the same finite check on the same
/// `res.irf`, so the guard is exercised as written.
///
/// The teeth are quantitative: the overflowing draws must be **more than
/// `alpha/2`** of the total, because only then does admitting them reach the
/// order statistic the band endpoint reads — one non-finite draw buried in
/// the sorted array would not. Measured with the guard deleted: `non_finite`
/// drops to `0`, and 17 of the Efron endpoints and 18 of the Hall endpoints
/// come back `NaN` (the percentile interpolation differences `inf - inf`).
/// So this test fails twice over if the check on `res.irf` is removed.
#[test]
fn non_finite_draws_are_counted_and_kept_out_of_the_quantiles() {
    let mut rng = Lcg::new(19);
    // A deliberately weak instrument, so rho* = gamma*/gamma*[norm_var] is
    // genuinely heavy-tailed rather than tightly concentrated at 1.
    let (data, proxy) = simulate(&mut rng, 80, 0.05, 1.0, 0);
    let mut sp = spec(400, 9, 4);
    sp.unit = 6e307;
    let b = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();

    assert!(
        b.failures.non_finite > 0,
        "no draw overflowed, so the non_finite guard is still untested"
    );
    assert!(
        b.failures.non_finite as f64 > 0.5 * sp.alpha * sp.n_boot as f64,
        "only {}/{} draws overflowed ({:.1}%), at or below alpha/2 = {:.1}% — admitting them \
         would not move the {:.0}th percentile, so this test could not detect the guard's \
         removal",
        b.failures.non_finite,
        b.n_boot,
        100.0 * b.failures.non_finite as f64 / b.n_boot as f64,
        50.0 * sp.alpha,
        100.0 * (1.0 - sp.alpha / 2.0),
    );
    // The reason is specifically non_finite, not swept into another counter.
    assert_eq!(b.n_failed, b.failures.total());
    assert_eq!(b.n_failed, b.failures.non_finite);
    assert_eq!(b.n_used, b.n_boot - b.n_failed);
    assert!(b.n_used >= 2);

    // Nothing infinite reached the quantiles: the Efron endpoints are raw
    // order statistics of the surviving draws, so an admitted +inf would
    // appear there directly.
    for h in 0..b.lower.len() {
        for i in 0..2 {
            assert!(
                b.lower_efron[h][i].is_finite() && b.upper_efron[h][i].is_finite(),
                "non-finite Efron endpoint at ({h},{i}): an overflowing draw was admitted"
            );
            assert!(b.lower[h][i].is_finite() && b.upper[h][i].is_finite());
            assert!(b.lower[h][i] <= b.upper[h][i]);
        }
    }
    // Failed draws stay in the record as NaN rather than being removed.
    assert_eq!(b.gamma_norm_draws.len(), b.n_boot);
    assert_eq!(b.rho_draws.len(), b.n_boot);
    assert_eq!(
        b.rho_draws
            .iter()
            .filter(|r| r.iter().all(|v| v.is_nan()))
            .count(),
        b.n_failed
    );
}

/// `rho_draws` is the real per-draw ratio, not a decoration: it is exactly
/// normalized at `norm_var`, and it reproduces the `h = 0` Efron endpoints
/// through `theta*_0 = Psi_0 b* = unit * rho*`.
///
/// Spec Step 12 asks for `rho*` alongside `gamma*[norm_var]`, `F*` and
/// `reliability*`. It is the scale-free reading of the same fragility — and
/// the one quantity in which the unimplemented Jentsch-Lunsford proxy
/// rescaling provably cancels — so a stale or wrongly-scaled `rho_draws`
/// would silently mislead exactly the reader who went looking for it.
#[test]
fn rho_draws_are_the_per_draw_ratio_and_are_normalized_exactly() {
    let mut rng = Lcg::new(26);
    let (data, proxy) = simulate(&mut rng, 150, 0.7, 0.6, 5);
    let sp = spec(300, 77, 4);
    assert_eq!(sp.unit, 1.0, "the h = 0 cross-check below assumes unit = 1");
    let b = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();

    assert_eq!(b.rho_draws.len(), b.n_boot);
    let mut finite = 0usize;
    for (r, row) in b.rho_draws.iter().enumerate() {
        assert_eq!(row.len(), 2);
        if b.gamma_norm_draws[r].is_nan() {
            assert!(
                row.iter().all(|v| v.is_nan()),
                "draw {r}: failed but rho* is not NaN"
            );
            continue;
        }
        finite += 1;
        // rho*[norm_var] = gamma*[nv]/gamma*[nv] = 1 exactly, in every draw.
        assert_eq!(
            row[NORM_VAR], 1.0,
            "draw {r}: rho*[norm_var] is not exactly 1"
        );
        assert!(row[1].is_finite());
    }
    assert_eq!(finite, b.n_used);

    // theta*_{0,i} = unit * rho*_i, so the h = 0 Efron endpoints must be the
    // matching quantiles of rho_draws. This is what catches a rho* taken
    // from the wrong draw, the wrong variable, or the point estimate.
    let mut vals: Vec<f64> = b
        .rho_draws
        .iter()
        .map(|r| r[1])
        .filter(|v| v.is_finite())
        .collect();
    vals.sort_by(f64::total_cmp);
    let pct = |q: f64| -> f64 {
        let pos = q * (vals.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(vals.len() - 1);
        vals[lo] + (pos - lo as f64) * (vals[hi] - vals[lo])
    };
    for (q, got) in [
        (sp.alpha / 2.0, b.lower_efron[0][1]),
        (1.0 - sp.alpha / 2.0, b.upper_efron[0][1]),
    ] {
        let want = sp.unit * pct(q);
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "h = 0 Efron endpoint {got} disagrees with unit * quantile(rho*) = {want}"
        );
    }
}

/// No ad-hoc sign fixing: with a weak instrument the denominator
/// `gamma*[norm_var]` genuinely changes sign across draws, and those draws
/// are kept. Forcing the sign to agree with the point estimate would
/// truncate one lobe of a bimodal ratio and narrow the bands by fiat.
#[test]
fn sign_flipped_draws_are_retained() {
    let mut rng = Lcg::new(19);
    // A deliberately weak instrument: relevance barely above the noise.
    let (data, proxy) = simulate(&mut rng, 80, 0.05, 1.0, 0);
    let bands = proxy_svar_bands(data.as_ref(), &proxy, &spec(600, 41, 3)).unwrap();
    let pos = bands.gamma_norm_draws.iter().filter(|v| **v > 0.0).count();
    let neg = bands.gamma_norm_draws.iter().filter(|v| **v < 0.0).count();
    assert!(
        pos > 0 && neg > 0,
        "a weak instrument must produce gamma*[norm_var] of both signs ({pos} positive, \
         {neg} negative); a one-signed distribution means draws were being sign-fixed or \
         discarded"
    );
    // The consequence, which must also survive: the bands are wide.
    assert!(
        bands.upper[0][1] - bands.lower[0][1] > 0.5,
        "weak-instrument impact band is implausibly tight"
    );
}

/// The proxy may be supplied aligned to the residual sample or at full
/// length with the presample rows attached; the two give identical bands.
/// The `NaN` mask is never compacted — packing the finite entries into
/// positions `1..|O|` would destroy the date alignment with `uhat*` and
/// reproduce the independent-resampling failure exactly.
#[test]
fn proxy_alignment_conventions_agree_and_nans_are_not_compacted() {
    let mut rng = Lcg::new(20);
    let (data, proxy) = simulate(&mut rng, 120, 0.8, 0.5, 10);
    let sp = spec(200, 55, 4);
    let aligned = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();

    // Full-length form: one extra (presample) entry at the front.
    let mut full = vec![f64::NAN];
    full.extend_from_slice(&proxy);
    let from_full = proxy_svar_bands(data.as_ref(), &full, &sp).unwrap();
    for h in 0..aligned.lower.len() {
        for i in 0..2 {
            assert_eq!(
                aligned.lower[h][i].to_bits(),
                from_full.lower[h][i].to_bits()
            );
            assert_eq!(
                aligned.upper[h][i].to_bits(),
                from_full.upper[h][i].to_bits()
            );
        }
    }

    // With 10 of 119 dates unavailable the first stage must stay healthy.
    // Compaction would collapse it toward the null value of ~1.
    let mut f: Vec<f64> = aligned
        .first_stage_f_draws
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    f.sort_by(f64::total_cmp);
    let median = f[f.len() / 2];
    assert!(
        median > 5.0,
        "median bootstrap first-stage F = {median:.2}: a value near 1 is what proxy \
         compaction (or independent resampling) produces"
    );
}

/// Block-length sensitivity really does use **common random numbers**, not
/// merely a shared seed.
///
/// This is the load-bearing half of Step 13. Sharing `spec.seed` across three
/// calls to `proxy_svar_bands` would not share the draws: block starts come
/// from `uniform_index(stream, T - ell + 1)`, and a different bound changes
/// how much of the stream bitmask rejection consumes, so the three runs would
/// diverge into independent resampling noise. The diagnostic's whole value is
/// that a discontinuous jump means a bug — independent noise between the runs
/// confounds exactly that inference.
///
/// The construction is pinned rather than described: one `[0, 1)` uniform
/// matrix is drawn from `Stream::new(seed)` and mapped to starts by
/// `floor(u * (T - ell + 1))` for each `ell`, so the same uniform drives
/// block `j` of replication `r` in all three runs. Reproducing that here and
/// comparing bit for bit is the only way the claim in the doc comment is
/// worth anything.
#[test]
fn block_length_sensitivity_uses_common_random_numbers() {
    let mut rng = Lcg::new(27);
    let (data, proxy) = simulate(&mut rng, 130, 0.8, 0.5, 4);
    let sp = spec(120, 909, 4);
    let t = data.nrows() - sp.lags;
    let base = default_block_length(t);
    let ells = [base / 2, base, base * 2];

    let out = proxy_svar_band_block_sensitivity(data.as_ref(), &proxy, &sp).unwrap();

    // The same uniforms the library draws, in the same order.
    let n_max = t.div_ceil(ells[0]);
    let mut stream = Stream::new(sp.seed);
    let uniforms: Vec<Vec<f64>> = (0..sp.n_boot)
        .map(|_| (0..n_max).map(|_| stream.uniform_f64()).collect())
        .collect();

    for (k, ell) in ells.iter().copied().enumerate() {
        assert_eq!(out[k].block_length, ell);
        let n_starts = t - ell + 1;
        let starts: Vec<Vec<usize>> = uniforms
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&u| ((u * n_starts as f64) as usize).min(n_starts - 1))
                    .collect()
            })
            .collect();
        let mut s = sp;
        s.block_length = Some(ell);
        let want = proxy_svar_bands_from_starts(data.as_ref(), &proxy, &s, &starts).unwrap();
        for h in 0..want.lower.len() {
            for i in 0..2 {
                assert_eq!(
                    out[k].lower[h][i].to_bits(),
                    want.lower[h][i].to_bits(),
                    "ell = {ell}, cell ({h},{i}): the three runs do not share their draws"
                );
                assert_eq!(out[k].upper[h][i].to_bits(), want.upper[h][i].to_bits());
            }
        }
    }

    // Teeth: the uniforms really are shared, so the three runs are NOT
    // independent draws. A run at a different seed must differ.
    let mut other = sp;
    other.seed = sp.seed + 1;
    let alt = proxy_svar_band_block_sensitivity(data.as_ref(), &proxy, &other).unwrap();
    assert!(
        (0..alt[1].lower.len())
            .any(|h| alt[1].lower[h][1].to_bits() != out[1].lower[h][1].to_bits()),
        "changing the seed changed nothing, so the shared-uniform path is not being used"
    );
}

/// Block-length sensitivity: bands at `ell/2`, `ell`, `2*ell` move smoothly
/// and modestly. A discontinuous jump means the block construction or the
/// truncation to `T` is wrong; a strong monotone trend means `ell` is in the
/// wrong regime for this `T`.
///
/// The bound is a factor of `1.6` per doubling of `ell`, not the factor of
/// `3` a noise-confounded comparison would need. It is affordable **because**
/// the three runs share their uniforms (pinned by
/// [`block_length_sensitivity_uses_common_random_numbers`]): with the
/// resampling noise held fixed, the observed successive width ratios on this
/// and three neighbouring DGP draws run `0.78` to `1.17`, so `0.625..1.6`
/// leaves real margin without being vacuous.
#[test]
fn block_length_sensitivity_moves_smoothly() {
    let mut rng = Lcg::new(21);
    let (data, proxy) = simulate(&mut rng, 160, 0.8, 0.5, 4);
    let out = proxy_svar_band_block_sensitivity(data.as_ref(), &proxy, &spec(300, 63, 5)).unwrap();
    assert_eq!(out.len(), 3);
    let base = default_block_length(159);
    assert_eq!(out[0].block_length, base / 2);
    assert_eq!(out[1].block_length, base);
    assert_eq!(out[2].block_length, base * 2);

    let width = |b: &ProxyBands| -> f64 {
        let mut acc = 0.0;
        for h in 0..b.lower.len() {
            acc += b.upper[h][1] - b.lower[h][1];
        }
        acc / b.lower.len() as f64
    };
    let (w0, w1, w2) = (width(&out[0]), width(&out[1]), width(&out[2]));
    for (a, b) in [(w0, w1), (w1, w2)] {
        assert!(
            b < 1.6 * a && a < 1.6 * b,
            "band width jumps between block lengths ({w0:.4}, {w1:.4}, {w2:.4}); with common \
             random numbers, halving or doubling ell should move the bands modestly"
        );
    }
}

// -------------------------------------------------------- Monte-Carlo arm

/// Per-horizon coverage summary for one Monte-Carlo arm.
struct Coverage {
    /// Fraction of replications whose interval contained the truth, per
    /// horizon.
    cover: Vec<f64>,
    /// Fraction whose interval lay entirely **above** the truth.
    left_miss: Vec<f64>,
    /// Fraction whose interval lay entirely **below** the truth.
    right_miss: Vec<f64>,
    /// Mean interval width per horizon.
    width: Vec<f64>,
}

impl Coverage {
    fn mean_cover(&self) -> f64 {
        self.cover.iter().sum::<f64>() / self.cover.len() as f64
    }
    fn report(&self, name: &str) {
        println!("{name:>26}  cover {:?}", round3(&self.cover));
        println!("{:>26}  left  {:?}", "", round3(&self.left_miss));
        println!("{:>26}  right {:?}", "", round3(&self.right_miss));
        println!("{:>26}  width {:?}", "", round3(&self.width));
    }
}

fn round3(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1000.0).round() / 1000.0).collect()
}

/// Monte-Carlo configuration, shared by all three arms so the comparison is
/// on identical data.
const MC_REPS: usize = 150;
const MC_BOOT: usize = 199;
const MC_HORIZON: usize = 4;
/// `T = 299`, which puts the default block length (`ell = 21`) in the regime
/// Jentsch-Lunsford's rule was calibrated for. At `T = 99` the same rule
/// gives `ell = 16` — a sixth of the sample, only seven blocks — and
/// coverage is measurably worse for it (0.78 at impact against 0.83 with
/// `ell = 4`). That is a real property of the method at short samples, not a
/// defect, and it is why `proxy_svar_band_block_sensitivity` exists.
const MC_NOBS: usize = 300;

/// Band `MC_REPS` fresh draws of the DGP and measure how often the nominal
/// 90% Hall interval contains the true impulse response of variable 1.
///
/// Left and right non-coverage are tracked separately: under a skewed
/// bootstrap distribution the two-sided total can look near-nominal while
/// the truth falls outside on one side far more often than the other, and a
/// combined number hides it.
fn coverage(method: ProxyBandMethod) -> Coverage {
    let truth = true_irf(MC_HORIZON);
    let mut rng = Lcg::new(0xC0FFEE);
    let hh = MC_HORIZON + 1;
    let (mut hit, mut left, mut right) = (vec![0usize; hh], vec![0usize; hh], vec![0usize; hh]);
    let mut width = vec![0.0f64; hh];
    for r in 0..MC_REPS {
        let (data, proxy) = simulate(&mut rng, MC_NOBS, 0.8, 0.5, 3);
        let mut sp = spec(MC_BOOT, 1000 + r as u64, MC_HORIZON);
        sp.method = method;
        let b = proxy_svar_bands(data.as_ref(), &proxy, &sp).unwrap();
        for h in 0..hh {
            // Variable 1 only: variable 0 at h = 0 is degenerate by
            // construction, and counting it would flatter every arm.
            let (lo, hi, tv) = (b.lower[h][1], b.upper[h][1], truth[h][1]);
            width[h] += hi - lo;
            if tv < lo {
                left[h] += 1;
            } else if tv > hi {
                right[h] += 1;
            } else {
                hit[h] += 1;
            }
        }
    }
    let rf = MC_REPS as f64;
    Coverage {
        cover: hit.iter().map(|&c| c as f64 / rf).collect(),
        left_miss: left.iter().map(|&c| c as f64 / rf).collect(),
        right_miss: right.iter().map(|&c| c as f64 / rf).collect(),
        width: width.iter().map(|w| w / rf).collect(),
    }
}

/// The same DGP, banded by this crate's **already validated** Cholesky
/// residual bootstrap ([`tsecon_var::bootstrap_irf_bands`]).
///
/// The **estimand** is held exact: `H_TRUE` is lower triangular, so it *is*
/// the Cholesky factor of `Sigma_u`, and `H_TRUE[0][0] = 1`, so the recursive
/// identification's first column and the proxy's unit-effect response are the
/// same population object reached by two different identifying assumptions.
/// The **procedure** is not held fixed, and the difference must not be
/// papered over: this reference reports **Efron** percentile bands from an
/// **i.i.d. residual** bootstrap, while the proxy arm reports **Hall** bands
/// from a **moving block**. The gap between the two rows therefore mixes
/// three changes — identification layer, interval type, resampling scheme —
/// and cannot be read as the identification layer's cost. What it does
/// establish is a bound: a shortfall of this size at these horizons is
/// already present in a validated reduced-form bootstrap on the same data.
fn cholesky_baseline() -> Coverage {
    let a = Mat::from_fn(2, 2, |i, j| A_TRUE[i][j]);
    let psi = ma_rep(std::slice::from_ref(&a), MC_HORIZON).unwrap();
    let truth: Vec<f64> = (0..=MC_HORIZON)
        .map(|h| (0..2).map(|k| psi[h][(1, k)] * H_TRUE[k][0]).sum::<f64>())
        .collect();
    let hh = MC_HORIZON + 1;
    let mut rng = Lcg::new(0xC0FFEE);
    let (mut hit, mut left, mut right) = (vec![0usize; hh], vec![0usize; hh], vec![0usize; hh]);
    let mut width = vec![0.0f64; hh];
    for r in 0..MC_REPS {
        let (data, _proxy) = simulate(&mut rng, MC_NOBS, 0.8, 0.5, 3);
        let b = tsecon_var::bootstrap_irf_bands(
            data.as_ref(),
            1,
            Trend::Constant,
            MC_HORIZON,
            true,
            false,
            0.10,
            MC_BOOT,
            1000 + r as u64,
            false,
        )
        .unwrap();
        for h in 0..hh {
            let (lo, hi) = (b.lower[h][(1, 0)], b.upper[h][(1, 0)]);
            width[h] += hi - lo;
            if truth[h] < lo {
                left[h] += 1;
            } else if truth[h] > hi {
                right[h] += 1;
            } else {
                hit[h] += 1;
            }
        }
    }
    let rf = MC_REPS as f64;
    Coverage {
        cover: hit.iter().map(|&c| c as f64 / rf).collect(),
        left_miss: left.iter().map(|&c| c as f64 / rf).collect(),
        right_miss: right.iter().map(|&c| c as f64 / rf).collect(),
        width: width.iter().map(|w| w / rf).collect(),
    }
}

/// Monte-Carlo coverage of the nominal 90% Hall band on a known-truth
/// proxy-SVAR DGP, against two references measured on the identical data:
/// the wild arm, and this crate's already-validated Cholesky residual
/// bootstrap.
///
/// Measured here at `T = 299`, `ell = 21`, `B = 199`, 150 replications
/// (Monte-Carlo standard error ~0.025):
///
/// ```text
///              h = 0  h = 1  h = 2  h = 3  h = 4
/// moving block 0.860  0.847  0.793  0.780  0.807
/// wild         0.113  0.827  0.847  0.833  0.820
/// Cholesky ref 0.927  0.893  0.840  0.833  0.833
/// ```
///
/// Two things to read off that table, both of which the assertions below
/// pin:
///
/// 1. **The wild bootstrap collapses at impact** — 0.113 for a nominal 0.90,
///    with a mean interval width of 0.018 against the moving block's 0.173.
///    At `h = 0` the *only* uncertainty is the identification step, and the
///    common Rademacher draw freezes it, so the band is an order of
///    magnitude too short. This is the Jentsch-Lunsford result, measured.
/// 2. **At `h >= 1` the wild arm is not uniformly worse on this DGP.** Once
///    reduced-form coefficient uncertainty enters, it dominates, and the
///    wild arm's coverage is comparable (even slightly higher at `h = 2, 3`).
///    The direction of the horizon profile is DGP-dependent and is
///    deliberately not claimed to be universal — only the impact-horizon
///    collapse is, and that is the horizon where proxy SVARs are usually
///    read.
///
/// The moving block's own shortfall at longer horizons (0.78-0.81 for a
/// nominal 0.90) is **inherited from the reduced-form VAR bootstrap**, not
/// introduced by the proxy layer: the Cholesky reference lands within 0.07
/// of it at every horizon on the same replications, and this repository's
/// own coverage audit already records `var_irf_bands(method="bootstrap")` at
/// 0.848 for impact and 0.410 at `h = 12` on a persistent VAR without the
/// Kilian bias correction. The proxy bands do not currently offer a bias
/// correction; that is the honest cost and it is documented rather than
/// tuned away.
#[test]
fn monte_carlo_coverage_is_near_nominal_and_the_wild_arm_collapses_at_impact() {
    let mbb = coverage(ProxyBandMethod::MovingBlock);
    let wild = coverage(ProxyBandMethod::Wild);
    let chol = cholesky_baseline();
    mbb.report("moving block (Hall)");
    wild.report("wild (Hall)");
    chol.report("Cholesky reference");

    // 1. The moving-block arm is in the neighbourhood of nominal. Coverage
    //    far ABOVE nominal would be the independent-blocking bug (rho*
    //    Cauchy-like, bands orders of magnitude too wide, coverage ~1.0 and
    //    vacuous); far below would be sign fixing, dropped failures, or a
    //    fixed-design shortcut.
    assert!(
        mbb.cover[0] > 0.76 && mbb.cover[0] < 0.98,
        "impact coverage {:.3} for a nominal 0.90",
        mbb.cover[0]
    );
    assert!(
        mbb.mean_cover() > 0.72 && mbb.mean_cover() < 0.98,
        "mean coverage over h = 0..{MC_HORIZON} is {:.3}",
        mbb.mean_cover()
    );

    // 2. The wild arm collapses at impact, where the identification step is
    //    the whole variance.
    assert!(
        wild.cover[0] < 0.40,
        "the wild arm covered {:.3} at impact; the frozen identifying moment must cost it \
         most of its nominal level",
        wild.cover[0]
    );
    assert!(
        wild.width[0] < 0.4 * mbb.width[0],
        "wild impact width {:.4} against moving-block {:.4}",
        wild.width[0],
        mbb.width[0]
    );
    assert!(
        mbb.cover[0] > 2.0 * wild.cover[0],
        "moving block {:.3} vs wild {:.3} at impact",
        mbb.cover[0],
        wild.cover[0]
    );

    // 3. The longer-horizon shortfall is the reduced-form bootstrap's, not
    //    the proxy layer's: the validated Cholesky bands land close by on
    //    the same replications and the same population object.
    for h in 0..=MC_HORIZON {
        assert!(
            mbb.cover[h] > chol.cover[h] - 0.15,
            "at h = {h} the proxy band covered {:.3} against the Cholesky reference's {:.3}: \
             a gap that large is the proxy layer's own, not inherited",
            mbb.cover[h],
            chol.cover[h]
        );
    }

    // 4. Left and right non-coverage are reported separately because they
    //    are not symmetric — the ratio estimand is skewed, and the right
    //    miss grows with the horizon while the left miss does not.
    for h in 0..=MC_HORIZON {
        assert!(
            (mbb.left_miss[h] + mbb.right_miss[h] + mbb.cover[h] - 1.0).abs() < 1e-12,
            "coverage accounting at h = {h}"
        );
    }
}

// ------------------------------------------------------------------ guards

/// Domain guards fire with messages that say what to change.
#[test]
fn guards_fire() {
    let mut rng = Lcg::new(22);
    let (data, proxy) = simulate(&mut rng, 60, 0.8, 0.5, 2);

    let mut sp = spec(100, 1, 3);
    sp.lags = 0;
    assert!(proxy_svar_bands(data.as_ref(), &proxy, &sp).is_err());

    let mut sp = spec(1, 1, 3);
    sp.n_boot = 1;
    assert!(proxy_svar_bands(data.as_ref(), &proxy, &sp).is_err());

    let mut sp = spec(100, 1, 3);
    sp.alpha = 0.0;
    assert!(proxy_svar_bands(data.as_ref(), &proxy, &sp).is_err());

    let mut sp = spec(100, 1, 3);
    sp.norm_var = 7;
    assert!(proxy_svar_bands(data.as_ref(), &proxy, &sp).is_err());

    // A block length at or beyond T leaves fewer than two candidate blocks.
    let mut sp = spec(100, 1, 3);
    sp.block_length = Some(data.nrows());
    assert!(proxy_svar_bands(data.as_ref(), &proxy, &sp).is_err());

    // A proxy matching neither alignment convention.
    let sp = spec(100, 1, 3);
    assert!(proxy_svar_bands(data.as_ref(), &proxy[..5], &sp).is_err());
}
