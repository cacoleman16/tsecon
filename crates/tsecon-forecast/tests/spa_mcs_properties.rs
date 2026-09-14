//! Seeded Monte Carlo properties and guardrails for the multiple-comparison
//! tests: `spa_test` / `stepm_test` (Reality Check / SPA / StepM) and
//! `model_confidence_set` (MCS).
//!
//! What the goldens cannot prove is measured here with a stated seed. Two of
//! the three Monte Carlo studies are CROSS-CHECKED against the reference's
//! own rates on the same design — `fixtures/spa.json` and `fixtures/mcs.json`
//! carry them — so that behaviour tsecon shares with `arch` is reported as
//! the METHOD's while behaviour it does not share fails a test here:
//!
//! * the SIZE of the SPA test under equal predictive ability (every model
//!   exactly as good as the benchmark — the least favourable configuration),
//!   on iid and AR(1) loss differentials at four block lengths. The
//!   un-studentized rates are compared with arch's; the studentized ones
//!   (the crate default, which no package computes) are MEASURED and are
//!   over-sized at n = 200, which the test states and shows shrinking by
//!   n = 800 rather than papering over;
//! * its POWER against a dominated benchmark;
//! * the COVERAGE of the MCS. Hansen-Lunde-Nason's Theorem 1 is asymptotic
//!   and about the WHOLE best set; on a design with two exactly-equally-best
//!   models that set is contained about 0.87 of the time at a nominal 0.90,
//!   here and in `arch` alike, and the test pins the agreement rather than
//!   the theorem.
//!
//! Plus the reproducibility contract (bit-identical at any rayon thread
//! count; the seeded path equals an explicit replay of its substreams), the
//! resampling conventions shared with the reference, the structural
//! identities (p-value bracketing, nested sets, StepM subsets), the memory
//! budget on the one buffer sized by a product of user counts, and the
//! teaching refusals, each naming its parameter.
//!
//! Every measured rate is printed (`cargo test -- --nocapture`) and quoted
//! in the forecasting model card.

use serde_json::Value;
use tsecon_bootstrap::{indices, BlockScheme};
use tsecon_forecast::{
    model_confidence_set, model_confidence_set_with_indices, spa_test, spa_test_with_indices,
    stepm_test, ForecastError, McsMethod, McsOptions, ResampleScheme, SpaOptions, StepmOptions,
};
use tsecon_rng::Stream;

/// `fixtures/spa.json`, which carries the reference's own conventions probe
/// and its own size-under-the-null study (see the generator header).
fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/spa.json");
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

/// arch's own rejection frequencies under the least favourable null.
struct ArchSizeStudy {
    levels: [f64; 4],
    n: usize,
    mc: usize,
    reps: usize,
    /// `(label, rho, block_size, arch's rates at `levels`)`.
    configs: Vec<(String, f64, usize, Vec<f64>)>,
}

fn fixture_size_study() -> ArchSizeStudy {
    let fx = fixture();
    let s = &fx["_meta"]["size_study"];
    let levels: Vec<f64> = s["levels"]
        .as_array()
        .expect("levels")
        .iter()
        .map(|v| v.as_f64().expect("level"))
        .collect();
    let mut configs: Vec<(String, f64, usize, Vec<f64>)> = s["rates"]
        .as_object()
        .expect("rates")
        .iter()
        .map(|(label, v)| {
            (
                label.clone(),
                v["rho"].as_f64().expect("rho"),
                v["block_size"].as_u64().expect("block_size") as usize,
                v["rates"]
                    .as_array()
                    .expect("rates")
                    .iter()
                    .map(|x| x.as_f64().expect("rate"))
                    .collect(),
            )
        })
        .collect();
    configs.sort_by(|a, b| a.0.cmp(&b.0));
    ArchSizeStudy {
        levels: [levels[0], levels[1], levels[2], levels[3]],
        n: s["n"].as_u64().expect("n") as usize,
        mc: s["mc"].as_u64().expect("mc") as usize,
        reps: s["reps"].as_u64().expect("reps") as usize,
        configs,
    }
}

/// Standard normals from a Philox stream (Box-Muller).
struct Gauss(Stream);

impl Gauss {
    fn new(seed: u64) -> Self {
        Gauss(Stream::new(seed))
    }
    fn normal(&mut self) -> f64 {
        let u1 = (1.0 - self.0.uniform_f64()).max(1e-300);
        let u2 = self.0.uniform_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn ar1(&mut self, n: usize, rho: f64) -> Vec<f64> {
        let mut x = Vec::with_capacity(n);
        let mut prev = 0.0;
        let s = (1.0 - rho * rho).sqrt();
        for _ in 0..n {
            prev = rho * prev + s * self.normal();
            x.push(prev);
        }
        x
    }
}

/// Squared-error loss columns: errors share an AR(1) common component
/// (`common`) plus an idiosyncratic AR(1) part scaled per model, shifted by
/// `bias` — the fixture generator's design.
fn loss_panel(g: &mut Gauss, n: usize, scales: &[f64], bias: &[f64], rho: f64) -> Vec<Vec<f64>> {
    let u = g.ar1(n, rho);
    scales
        .iter()
        .zip(bias)
        .map(|(&s, &b)| {
            let v = g.ar1(n, rho);
            u.iter()
                .zip(&v)
                .map(|(&uu, &vv)| {
                    let e = 0.7 * uu + s * vv + b;
                    e * e
                })
                .collect()
        })
        .collect()
}

fn spa_opts(
    block_size: Option<usize>,
    reps: usize,
    scheme: ResampleScheme,
    seed: u64,
) -> SpaOptions {
    SpaOptions {
        block_size,
        reps,
        scheme,
        studentize: true,
        nested: false,
        seed,
    }
}

fn with_threads<T>(k: usize, f: impl FnOnce() -> T + Send) -> T
where
    T: Send,
{
    rayon::ThreadPoolBuilder::new()
        .num_threads(k)
        .build()
        .expect("pool")
        .install(f)
}

// ---------------------------------------------------------------- determinism

#[test]
fn spa_is_bit_identical_at_any_thread_count_and_seed_sensitive() {
    let mut g = Gauss::new(1);
    let cols = loss_panel(&mut g, 120, &[1.0, 0.9, 1.1, 1.0], &[0.0; 4], 0.4);
    let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
    for scheme in [
        ResampleScheme::Stationary,
        ResampleScheme::CircularBlock,
        ResampleScheme::MovingBlock,
    ] {
        let o = spa_opts(Some(6), 2500, scheme, 77);
        let base = spa_test(&bench, &models, &o).unwrap();
        for k in [1, 2, 3, 4] {
            let r = with_threads(k, || spa_test(&bench, &models, &o).unwrap());
            assert_eq!(r, base, "{scheme:?}: {k} threads");
            let s = with_threads(k, || {
                stepm_test(
                    &bench,
                    &models,
                    &StepmOptions {
                        size: 0.05,
                        spa: o.clone(),
                    },
                )
                .unwrap()
            });
            assert_eq!(s.spa, base, "{scheme:?}: StepM's SPA at {k} threads");
        }
        let other = spa_test(
            &bench,
            &models,
            &SpaOptions {
                seed: 78,
                ..o.clone()
            },
        )
        .unwrap();
        assert_ne!(
            other.boot_consistent, base.boot_consistent,
            "{scheme:?}: seed must matter"
        );
        assert_eq!(other.mean_loss_diff, base.mean_loss_diff);
    }
}

#[test]
fn mcs_is_bit_identical_at_any_thread_count() {
    let mut g = Gauss::new(2);
    let losses = loss_panel(
        &mut g,
        100,
        &[0.9, 0.9, 1.1, 1.3],
        &[0.0, 0.0, 0.3, 0.6],
        0.3,
    );
    for method in [McsMethod::Range, McsMethod::Max] {
        let o = McsOptions {
            size: 0.10,
            method,
            block_size: Some(5),
            reps: 2500,
            scheme: ResampleScheme::Stationary,
            seed: 5,
        };
        let base = model_confidence_set(&losses, &o).unwrap();
        for k in [1, 2, 3, 4] {
            let r = with_threads(k, || model_confidence_set(&losses, &o).unwrap());
            assert_eq!(r, base, "{method:?}: {k} threads");
        }
    }
}

#[test]
fn seeded_path_equals_an_explicit_replay_of_its_substreams() {
    // The documented contract: replication b resamples with substream b of
    // SeedSequence(seed) through tsecon_bootstrap::indices.
    let mut g = Gauss::new(3);
    let cols = loss_panel(&mut g, 90, &[1.0, 0.8, 1.2], &[0.0; 3], 0.5);
    let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
    let n = 90;
    for (scheme, block) in [
        (
            ResampleScheme::Stationary,
            tsecon_bootstrap::BlockScheme::Stationary { p: 0.25 },
        ),
        (
            ResampleScheme::CircularBlock,
            tsecon_bootstrap::BlockScheme::CircularBlock { block_length: 4 },
        ),
        (
            ResampleScheme::MovingBlock,
            tsecon_bootstrap::BlockScheme::MovingBlock { block_length: 4 },
        ),
    ] {
        let reps = 1500; // more than one 1024-substream chunk
        let replay: Vec<Vec<usize>> = Stream::substreams(11, reps)
            .unwrap()
            .iter_mut()
            .map(|s| indices(block, n, s).unwrap())
            .collect();
        for nested in [false, true] {
            let o = SpaOptions {
                nested,
                ..spa_opts(Some(4), reps, scheme, 11)
            };
            let seeded = spa_test(&bench, &models, &o).unwrap();
            let explicit = spa_test_with_indices(&bench, &models, &replay, &o).unwrap();
            assert_eq!(seeded, explicit, "{scheme:?} nested={nested}");
        }
        for method in [McsMethod::Range, McsMethod::Max] {
            let o = McsOptions {
                size: 0.1,
                method,
                block_size: Some(4),
                reps,
                scheme,
                seed: 11,
            };
            let seeded = model_confidence_set(&cols, &o).unwrap();
            let explicit = model_confidence_set_with_indices(&cols, &replay, &o).unwrap();
            assert_eq!(seeded, explicit, "{scheme:?} {method:?}");
        }
    }
}

// ---------------------------------------------------------- structural facts

#[test]
fn spa_pvalues_bracket_and_stepm_is_a_subset_of_the_winners() {
    let mut g = Gauss::new(4);
    for trial in 0..20 {
        let cols = loss_panel(
            &mut g,
            80,
            &[1.0, 0.8, 1.0, 1.2, 1.5],
            &[0.0, 0.0, 0.2, 0.0, 0.5],
            0.3,
        );
        let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
        for studentize in [true, false] {
            let o = SpaOptions {
                studentize,
                ..spa_opts(Some(4), 400, ResampleScheme::Stationary, trial)
            };
            let r = spa_test(&bench, &models, &o).unwrap();
            assert!(r.p_value_lower <= r.p_value_consistent, "trial {trial}");
            assert!(r.p_value_consistent <= r.p_value_upper, "trial {trial}");
            assert!((0.0..=1.0).contains(&r.p_value_upper));
            assert_eq!(r.boot_consistent.len(), 400);
            assert!(r.crit_lower[0] <= r.crit_lower[1] && r.crit_lower[1] <= r.crit_lower[2]);
            assert!(r.statistic.is_finite());
            assert_eq!(r.crit_levels, [0.90, 0.95, 0.99]);
            // Re-centred models are exactly those not significantly worse.
            for k in 0..4 {
                if r.mean_loss_diff[k] >= 0.0 {
                    assert!(r.recentered[k]);
                }
            }
            let s = stepm_test(
                &bench,
                &models,
                &StepmOptions {
                    size: 0.10,
                    spa: o.clone(),
                },
            )
            .unwrap();
            assert_eq!(s.spa, r);
            for &k in &s.superior_models {
                assert!(
                    r.mean_loss_diff[k] > 0.0,
                    "a superior model beats the benchmark on average"
                );
            }
            let mut sorted = s.superior_models.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted, s.superior_models);
            assert_eq!(s.steps.len(), s.step_crit_values.len());
            let union: usize = s.steps.iter().map(Vec::len).sum();
            assert_eq!(union, s.superior_models.len());
            // Later steps face a (weakly) lower bar: fewer models in the max.
            for w in s.step_crit_values.windows(2) {
                assert!(w[1] <= w[0] + 1e-12);
            }
        }
    }
}

#[test]
fn mcs_sets_are_nested_in_size_and_partition_the_models() {
    let mut g = Gauss::new(5);
    for trial in 0..15 {
        let losses = loss_panel(
            &mut g,
            100,
            &[0.9, 0.9, 1.0, 1.2, 1.5],
            &[0.0, 0.0, 0.3, 0.5, 0.9],
            0.3,
        );
        for method in [McsMethod::Range, McsMethod::Max] {
            let mut sets = Vec::new();
            for size in [0.01, 0.05, 0.10, 0.25, 0.50] {
                let o = McsOptions {
                    size,
                    method,
                    block_size: None,
                    reps: 400,
                    scheme: ResampleScheme::Stationary,
                    seed: trial,
                };
                let r = model_confidence_set(&losses, &o).unwrap();
                assert!(r.block_size_auto);
                assert!(r.block_size >= 1 && r.block_size < 100);
                let mut all = r.included.clone();
                all.extend(&r.excluded);
                all.sort_unstable();
                assert_eq!(all, (0..5).collect::<Vec<_>>());
                assert!(!r.included.is_empty());
                assert_eq!(r.elimination_order.len(), 5);
                assert_eq!(r.step_p_values.last(), Some(&1.0));
                assert!(r.mcs_p_values.iter().all(|p| (0.0..=1.0).contains(p)));
                // Same seed, same bootstrap: the p-values do not depend on size.
                sets.push((r.included.clone(), r.mcs_p_values.clone()));
            }
            for w in sets.windows(2) {
                assert_eq!(w[0].1, w[1].1, "{method:?}: p-values independent of size");
                // Larger size => (weakly) smaller set.
                assert!(
                    w[1].0.iter().all(|k| w[0].0.contains(k)),
                    "{method:?}: nested sets"
                );
            }
        }
    }
}

// ------------------------------------------------------------- Monte Carlo

/// Levels at which the p-value's CDF is measured.
const SIZE_LEVELS: [f64; 4] = [0.05, 0.10, 0.25, 0.50];

/// Rejection frequencies of the consistent SPA p-value under H0 — every model
/// EXACTLY as good as the benchmark (the least favourable configuration
/// `mu = 0`, six exchangeable squared-error loss columns) — at
/// [`SIZE_LEVELS`]. A uniform p-value would hit each level at its own rate.
/// Returns the rates and the mean block length used.
fn spa_size_study(
    n: usize,
    rho: f64,
    block_size: Option<usize>,
    studentize: bool,
    seed0: u64,
    mc: usize,
) -> (Vec<f64>, f64) {
    let mut hits = vec![0usize; SIZE_LEVELS.len()];
    let mut block_sum = 0.0;
    for r in 0..mc {
        let mut g = Gauss::new(seed0 + r as u64);
        let cols = loss_panel(&mut g, n, &[1.0; 6], &[0.0; 6], rho);
        let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
        let mut o = spa_opts(block_size, 300, ResampleScheme::Stationary, 1000 + r as u64);
        o.studentize = studentize;
        let res = spa_test(&bench, &models, &o).unwrap();
        block_sum += res.block_size as f64;
        for (h, &l) in hits.iter_mut().zip(&SIZE_LEVELS) {
            if res.p_value_consistent <= l {
                *h += 1;
            }
        }
    }
    (
        hits.iter().map(|&h| h as f64 / mc as f64).collect(),
        block_sum / mc as f64,
    )
}

/// Size of the UN-studentized statistic — White's Reality Check with Hansen's
/// three re-centrings, which is exactly what `arch.bootstrap.SPA` computes
/// (its `studentize` flag is inert; see `fixtures/generate_spa_fixtures.py`).
/// This is the only leg with a third-party rate to compare against, so the
/// comparison is made directly: `fixtures/spa.json`'s `_meta.size_study`
/// holds ARCH's own rejection frequencies on this design (the generator runs
/// them), and tsecon's own seeded draws must land within Monte Carlo distance
/// of them. A size distortion the two libraries SHARE is the method's and is
/// quoted as such in the model card; one they do not share is a bug here.
#[test]
fn spa_size_unstudentized_matches_arch_rejection_rates() {
    let study = fixture_size_study();
    assert_eq!(
        study.levels, SIZE_LEVELS,
        "the fixture's levels are this test's"
    );
    let mc = study.mc;
    for (label, rho, block, arch_rates) in &study.configs {
        let (rates, mean_block) = spa_size_study(study.n, *rho, Some(*block), false, 2026, mc);
        println!(
            "SPA size study, studentize=false ({label}: rho={rho}, block_size={block}; n={}, m=5, B={}, {mc} MC reps, mean block length {mean_block:.2}): P(p_consistent <= alpha) at alpha = {SIZE_LEVELS:?} -> {rates:?}  [arch on the same design: {arch_rates:?}]",
            study.n, study.reps
        );
        for ((&rate, &theirs), &alpha) in rates.iter().zip(arch_rates).zip(&SIZE_LEVELS) {
            // Two independent Monte Carlo runs of mc replications each.
            let se = (2.0 * alpha * (1.0 - alpha) / mc as f64).sqrt();
            assert!(
                (rate - theirs).abs() <= 4.0 * se,
                "{label} at alpha {alpha}: tsecon rejects at {rate}, arch at {theirs} — more than 4 MC se ({se:.4}) apart, which is a difference in the test, not in the draws"
            );
            // Guardrail on the shared behaviour: neither library may be
            // grossly over-sized on this design (both measure <= ~1.6 alpha).
            assert!(
                rate <= 2.0 * alpha + 3.0 * se,
                "{label}: rejection rate {rate} at alpha {alpha} is more than twice nominal"
            );
        }
    }
}

/// Do `arch`'s resample index arrays obey the conventions
/// `tsecon_bootstrap::indices` documents? The fixture generator settles this
/// on arch's side (it replays arch's own raw draws through tsecon's rule and
/// gets arch's arrays back, `_meta.conventions`); this test checks the other
/// half — that the stored arrays and tsecon's own draws satisfy the same
/// structural invariants, so a change to either library's layout breaks a
/// test rather than silently moving a golden.
#[test]
fn arch_resamples_and_tsecon_draws_share_the_block_conventions() {
    let fx = fixture();
    let conv = &fx["_meta"]["conventions"];
    assert!(
        conv["difference"]
            .as_str()
            .is_some_and(|s| s.contains("u <= p")),
        "the fixture must record the one documented difference between the schemes"
    );
    for scheme in ["stationary", "circular", "moving_block"] {
        for row in conv[scheme].as_array().expect("rows") {
            assert!(
                row["replayed_exactly"].as_bool() == Some(true),
                "{scheme}: the generator did not replay arch's draws exactly"
            );
        }
    }
    let cases = fx["cases"].as_array().expect("cases").to_vec();
    for case in &cases {
        let n = case["n"].as_u64().expect("n") as usize;
        let b = case["block_size"].as_u64().expect("block_size") as usize;
        let kind = case["bootstrap"].as_str().expect("bootstrap");
        let sch = match kind {
            "stationary" => BlockScheme::Stationary { p: 1.0 / b as f64 },
            "circular" => BlockScheme::CircularBlock { block_length: b },
            _ => BlockScheme::MovingBlock { block_length: b },
        };
        // arch's stored resamples...
        let theirs: Vec<Vec<usize>> = case["resamples"]
            .as_array()
            .expect("resamples")
            .iter()
            .map(|r| {
                r.as_array()
                    .expect("array")
                    .iter()
                    .map(|x| x.as_u64().expect("index") as usize)
                    .collect()
            })
            .collect();
        // ...and tsecon's own, at the same n and block length.
        let mut stream = Stream::new(20260911 + n as u64);
        let mine: Vec<Vec<usize>> = (0..theirs.len())
            .map(|_| indices(sch, n, &mut stream).unwrap())
            .collect();
        for (who, arrs) in [("arch", &theirs), ("tsecon", &mine)] {
            let mut wraps = 0usize;
            let mut restarts = 0usize;
            for arr in arrs.iter() {
                assert_eq!(arr.len(), n, "{who}/{kind}: resample length");
                assert!(
                    arr.iter().all(|&i| i < n),
                    "{who}/{kind}: index out of range"
                );
                match kind {
                    "stationary" => {
                        for w in arr.windows(2) {
                            if w[1] == (w[0] + 1) % n {
                                if w[0] == n - 1 {
                                    wraps += 1;
                                }
                            } else {
                                restarts += 1;
                            }
                        }
                    }
                    _ => {
                        // Every block of b consecutive positions must be b
                        // consecutive indices (mod n for circular), and the
                        // moving block must never wrap.
                        for (j, chunk) in arr.chunks(b).enumerate() {
                            let start = chunk[0];
                            if kind == "moving_block" {
                                assert!(
                                    start + b <= n,
                                    "{who}: moving-block start {start} would run past the sample"
                                );
                            }
                            for (t, &v) in chunk.iter().enumerate() {
                                let want = if kind == "circular" {
                                    (start + t) % n
                                } else {
                                    start + t
                                };
                                assert_eq!(
                                    v, want,
                                    "{who}/{kind}: block {j} is not {b} consecutive indices"
                                );
                                if kind == "circular" && start + t >= n {
                                    wraps += 1;
                                }
                            }
                        }
                    }
                }
            }
            if kind == "stationary" {
                let steps = (arrs.len() * (n - 1)) as f64;
                let freq = restarts as f64 / steps;
                assert!(
                    (freq - 1.0 / b as f64).abs() < 0.05,
                    "{who}/{kind}: restart frequency {freq} is not 1/{b}"
                );
                assert!(wraps > 0, "{who}/{kind}: the wrap at n never happened");
            } else if kind == "circular" {
                assert!(wraps > 0, "{who}/{kind}: no block ever wrapped");
            }
        }
    }
}

/// Size of the STUDENTIZED statistic (`studentize=true`, the default —
/// Hansen's own SPA). No package computes it (`arch`'s `studentize` flag is
/// inert), so there is no third-party rate: this test MEASURES it, and the
/// numbers it prints are the ones the model card quotes.
///
/// The distortion is real and has a mechanism. Hansen divides both the
/// observed statistic AND every bootstrap replicate by the SAME estimate
/// `omega_k` (2005, eqs. 5-8, which is what this crate implements and what
/// the golden pins). In the bootstrap world that makes each column's
/// re-centred resampled mean exactly `N(0, omega_k^2 / n)`, so dividing by
/// `omega_k` leaves no dispersion across columns; in the real world
/// `dbar_k / omega_k` still carries the sampling error of `omega_k` itself.
/// The maximum over columns therefore has more spread in the data than in
/// the bootstrap, and the consistent p-value is too small. It is a
/// finite-sample property of the published procedure, not of this
/// implementation — which is why the test also measures it at a larger `n`:
/// the rates must MOVE TOWARDS nominal as the variance estimate sharpens.
#[test]
fn spa_size_studentized_is_measured_and_shrinks_with_the_sample() {
    let mc = 400;
    let mut at_5pct = Vec::new();
    for (label, n, rho, block) in [
        (
            "iid losses, block_size=None (Politis-White)",
            200,
            0.0,
            None,
        ),
        (
            "AR(0.5) losses, block_size=None (Politis-White)",
            200,
            0.5,
            None,
        ),
        ("AR(0.5) losses, block_size=8", 200, 0.5, Some(8)),
        (
            "AR(0.5) losses, block_size=None (Politis-White)",
            800,
            0.5,
            None,
        ),
        ("AR(0.5) losses, block_size=8", 800, 0.5, Some(8)),
    ] {
        let (rates, mean_block) = spa_size_study(n, rho, block, true, 2026, mc);
        println!(
            "SPA size study, studentize=true ({label}; n={n}, m=5, B=300, {mc} MC reps, mean block length {mean_block:.2}): P(p_consistent <= alpha) at alpha = {SIZE_LEVELS:?} -> {rates:?}"
        );
        at_5pct.push((label, n, rates[0]));
        for (&rate, &alpha) in rates.iter().zip(&SIZE_LEVELS) {
            let se = (alpha * (1.0 - alpha) / mc as f64).sqrt();
            assert!(
                rate >= alpha - 4.0 * se - 0.01,
                "{label} (n={n}): rejection rate {rate} at alpha {alpha} is far below nominal"
            );
            assert!(
                rate <= 3.0 * alpha + 3.0 * se,
                "{label} (n={n}): studentized rejection rate {rate} at alpha {alpha} is worse than the measured distortion"
            );
        }
    }
    // The same two AR(0.5) configurations at n = 200 and n = 800: the
    // over-rejection at 5% must not GROW with the sample.
    for (small, large) in [(1usize, 3usize), (2, 4)] {
        let (ls, ns, rs) = at_5pct[small];
        let (_, nl, rl) = at_5pct[large];
        println!("  {ls}: 5% rejection {rs} at n={ns} -> {rl} at n={nl}");
        assert!(
            rl <= rs + 0.02,
            "{ls}: the studentized distortion at 5% grew from {rs} (n={ns}) to {rl} (n={nl})"
        );
    }
}

#[test]
fn spa_has_power_against_a_dominated_benchmark() {
    let mc = 200;
    let mut rejections_c = 0;
    let mut rejections_u = 0;
    let mut best_is_model0 = 0;
    for r in 0..mc {
        let mut g = Gauss::new(5000 + r as u64);
        // The benchmark's errors are the largest; model 0 is clearly better.
        let cols = loss_panel(&mut g, 200, &[1.0, 0.6, 1.0, 1.0], &[0.0; 4], 0.3);
        let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
        let o = spa_opts(None, 300, ResampleScheme::Stationary, 7000 + r as u64);
        let res = spa_test(&bench, &models, &o).unwrap();
        if res.p_value_consistent <= 0.05 {
            rejections_c += 1;
        }
        if res.p_value_upper <= 0.05 {
            rejections_u += 1;
        }
        if res.best_model == 0 {
            best_is_model0 += 1;
        }
    }
    let power_c = rejections_c as f64 / mc as f64;
    let power_u = rejections_u as f64 / mc as f64;
    println!("SPA power study (n=200, one model with 0.6x error scale, {mc} MC reps): reject at 5%: consistent {power_c}, upper {power_u}; best_model identified {}", best_is_model0 as f64 / mc as f64);
    assert!(power_c >= 0.90, "consistent power {power_c}");
    assert!(power_u >= 0.85, "upper (Reality Check) power {power_u}");
    assert!(best_is_model0 as f64 / mc as f64 >= 0.95);
}

/// Hansen-Lunde-Nason's Theorem 1 is ASYMPTOTIC and about the WHOLE set of
/// best models: `lim inf P(M* subset of M*_{1-alpha}) >= 1 - alpha`. The
/// design here gives two models EXACTLY equal expected loss, so `M*` has two
/// elements and containing it is the demanding event; `P(best model in set)`
/// is the easier one-model version.
///
/// On this design the two-element `M*` is contained about 0.86-0.87 of the
/// time at a nominal 0.90, at n = 150 and at n = 600 alike. That gap is NOT
/// this implementation's: `fixtures/mcs.json`'s `_meta.coverage_study` holds
/// `arch.bootstrap.MCS`'s own frequencies on the same design and the rates
/// below are asserted to agree with them. Two effects make the finite-sample
/// containment short of nominal — a step can eliminate one of the two best
/// models while the inferior ones are still in the set, and the final test
/// between two identical models rejects at its own size — and neither is a
/// defect of the code. The card states it plainly.
#[test]
fn mcs_covers_the_set_of_best_models_at_least_1_minus_size_of_the_time() {
    let fx = {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/mcs.json");
        let text = std::fs::read_to_string(path).expect("fixture file readable");
        serde_json::from_str::<Value>(&text).expect("fixture is valid JSON")
    };
    let study = &fx["_meta"]["coverage_study"];
    let mc = 1000usize;
    let arch_mc = study["mc"].as_u64().expect("mc") as usize;
    let size = study["size"].as_f64().expect("size");
    let reps = study["reps"].as_u64().expect("reps") as usize;
    let scales = f64s4(&study["scales"]);
    let bias = f64s4(&study["bias"]);
    let rho = study["rho"].as_f64().expect("rho");
    for (key, row) in study["rates"].as_object().expect("rates") {
        let n = row["n"].as_u64().expect("n") as usize;
        let block = row["block_size"].as_u64().expect("block_size") as usize;
        let method = match row["method"].as_str().expect("method") {
            "R" => McsMethod::Range,
            other => {
                assert_eq!(other, "max");
                McsMethod::Max
            }
        };
        let mut both_best_in = 0;
        let mut model0_in = 0;
        let mut worst_out = 0;
        let mut set_sizes = 0usize;
        for r in 0..mc {
            let mut g = Gauss::new(9000 + r as u64);
            let losses = loss_panel(&mut g, n, &scales, &bias, rho);
            let o = McsOptions {
                size,
                method,
                block_size: Some(block),
                reps,
                scheme: ResampleScheme::Stationary,
                seed: 100 + r as u64,
            };
            let res = model_confidence_set(&losses, &o).unwrap();
            if res.included.contains(&0) && res.included.contains(&1) {
                both_best_in += 1;
            }
            if res.included.contains(&0) {
                model0_in += 1;
            }
            if !res.included.contains(&3) {
                worst_out += 1;
            }
            set_sizes += res.included.len();
        }
        let cov = both_best_in as f64 / mc as f64;
        let cov0 = model0_in as f64 / mc as f64;
        let power = worst_out as f64 / mc as f64;
        let mean_set = set_sizes as f64 / mc as f64;
        let theirs = (
            row["both_best_in_set"].as_f64().expect("both"),
            row["best_model_in_set"].as_f64().expect("one"),
            row["worst_excluded"].as_f64().expect("worst"),
            row["mean_set_size"].as_f64().expect("size"),
        );
        println!(
            "MCS coverage study ({key}: {method:?}, n={n}, block_size={block}, m=4, two equally-best models, size={size}, B={reps}, {mc} MC reps): P(M* in set) = {cov}, P(best model in set) = {cov0}, P(dominated model excluded) = {power}, mean set size {mean_set:.2}  [arch on the same design, {arch_mc} reps: {:?}, {:?}, {:?}, {:.2}]",
            theirs.0, theirs.1, theirs.2, theirs.3
        );
        // Two independent Monte Carlo runs, of mc and arch_mc replications.
        let se = |p: f64| (p * (1.0 - p) * (1.0 / mc as f64 + 1.0 / arch_mc as f64)).sqrt();
        for (what, mine, ref_rate) in [
            ("P(M* in set)", cov, theirs.0),
            ("P(best model in set)", cov0, theirs.1),
            ("P(dominated excluded)", power, theirs.2),
        ] {
            assert!(
                (mine - ref_rate).abs() <= 4.0 * se(ref_rate).max(0.005),
                "{key}: {what} is {mine} here and {ref_rate} in arch — more than 4 MC se apart"
            );
        }
        assert!(
            (mean_set - theirs.3).abs() <= 0.15,
            "{key}: mean set size {mean_set} vs arch's {}",
            theirs.3
        );
        assert!(
            power >= 0.90,
            "{key}: the dominated model survives too often ({power})"
        );
        // The easy, one-model event does hold at the nominal level.
        assert!(
            cov0 >= 1.0 - size - 3.0 * (size * (1.0 - size) / mc as f64).sqrt(),
            "{key}: P(best model in set) = {cov0} is below the nominal {}",
            1.0 - size
        );
    }
}

/// The first four numbers of a JSON array.
fn f64s4(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

// ------------------------------------------------------------------ refusals

fn err_msg<T: std::fmt::Debug>(r: Result<T, ForecastError>) -> String {
    match r {
        Ok(v) => panic!("expected a refusal, got {v:?}"),
        Err(e) => e.to_string(),
    }
}

/// The only buffers whose size is a PRODUCT of user counts are the
/// `reps x m` resampled-mean matrices (and, for StepM, the `reps x m`
/// per-model replicate matrix). They are budgeted with `try_reserve`, so a
/// replication count no machine can serve must come back as a teaching
/// refusal naming `reps` — never a `capacity overflow` panic or an
/// allocator abort. The counts below are below the Python layer's own
/// `2**48` guard, so this is the Rust budget being exercised, not that one.
#[test]
fn an_impossible_replication_count_is_refused_not_allocated() {
    let mut g = Gauss::new(11);
    let cols = loss_panel(&mut g, 40, &[1.0, 0.9, 1.1, 1.0], &[0.0; 4], 0.0);
    let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
    for reps in [1usize << 44, 1usize << 47, 1_000_000_000_000] {
        let o = spa_opts(Some(4), reps, ResampleScheme::Stationary, 0);
        for m in [
            err_msg(spa_test(&bench, &models, &o)),
            err_msg(stepm_test(
                &bench,
                &models,
                &StepmOptions {
                    size: 0.05,
                    spa: o.clone(),
                },
            )),
            err_msg(model_confidence_set(
                &cols,
                &McsOptions {
                    size: 0.10,
                    method: McsMethod::Range,
                    block_size: Some(4),
                    reps,
                    scheme: ResampleScheme::Stationary,
                    seed: 0,
                },
            )),
        ] {
            assert!(
                m.contains("refusing to allocate") && m.contains("reduce reps"),
                "reps = {reps}: {m}"
            );
        }
    }
}

#[test]
fn spa_refusals_name_the_parameter() {
    let mut g = Gauss::new(6);
    let cols = loss_panel(&mut g, 40, &[1.0, 0.9, 1.1], &[0.0; 3], 0.0);
    let (bench, models) = (cols[0].clone(), cols[1..].to_vec());
    let o = spa_opts(Some(4), 50, ResampleScheme::Stationary, 0);

    let m = err_msg(spa_test(&bench[..2], &[models[0][..2].to_vec()], &o));
    assert!(
        m.contains("benchmark_losses") && m.contains("at least 3"),
        "{m}"
    );

    let m = err_msg(spa_test(&bench, &[], &o));
    assert!(
        m.contains("model_losses") && m.contains("at least one"),
        "{m}"
    );

    let ragged = vec![models[0].clone(), models[1][..30].to_vec()];
    let m = err_msg(spa_test(&bench, &ragged, &o));
    assert!(
        m.contains("model_losses") && m.contains("column 1") && m.contains("30"),
        "{m}"
    );

    let mut nan = models.clone();
    nan[1][7] = f64::NAN;
    let m = err_msg(spa_test(&bench, &nan, &o));
    assert!(
        m.contains("model_losses") && m.contains("period 7") && m.contains("column 1"),
        "{m}"
    );
    let mut bnan = bench.clone();
    bnan[3] = f64::INFINITY;
    let m = err_msg(spa_test(&bnan, &models, &o));
    assert!(
        m.contains("benchmark_losses") && m.contains("period 3"),
        "{m}"
    );

    let m = err_msg(spa_test(
        &bench,
        &models,
        &SpaOptions {
            reps: 0,
            ..o.clone()
        },
    ));
    assert!(m.contains("reps = 0"), "{m}");

    for bad in [0usize, 40, 41] {
        let m = err_msg(spa_test(
            &bench,
            &models,
            &SpaOptions {
                block_size: Some(bad),
                ..o.clone()
            },
        ));
        assert!(
            m.contains(&format!("block_size = {bad}")) && m.contains("n = 40"),
            "{m}"
        );
    }

    // A model identical to the benchmark: constant (zero) loss differential.
    let same = vec![bench.clone(), models[0].clone()];
    let m = err_msg(spa_test(&bench, &same, &o));
    assert!(
        m.contains("model_losses column 0") && m.contains("constant"),
        "{m}"
    );

    // Automatic block length needs enough data.
    let m = err_msg(spa_test(
        &bench[..8],
        &[models[0][..8].to_vec()],
        &SpaOptions {
            block_size: None,
            ..o.clone()
        },
    ));
    assert!(
        m.contains("block_size = None") && m.contains("Politis-White"),
        "{m}"
    );

    // Explicit resamples: wrong length, out-of-range index, none at all.
    let good: Vec<usize> = (0..40).collect();
    let m = err_msg(spa_test_with_indices(
        &bench,
        &models,
        &[good.clone(), good[..39].to_vec()],
        &o,
    ));
    assert!(m.contains("resamples[1]") && m.contains("39"), "{m}");
    let mut oob = good.clone();
    oob[5] = 40;
    let m = err_msg(spa_test_with_indices(&bench, &models, &[oob], &o));
    assert!(
        m.contains("resamples[0]") && m.contains("position 5"),
        "{m}"
    );
    let m = err_msg(spa_test_with_indices(&bench, &models, &[], &o));
    assert!(m.contains("resamples = 0"), "{m}");

    // StepM size.
    for bad in [0.0, 1.0, -0.1, f64::NAN] {
        let m = err_msg(stepm_test(
            &bench,
            &models,
            &StepmOptions {
                size: bad,
                spa: o.clone(),
            },
        ));
        assert!(m.contains("size = ") && m.contains("0 < size < 1"), "{m}");
    }
}

#[test]
fn mcs_refusals_name_the_parameter() {
    let mut g = Gauss::new(7);
    let losses = loss_panel(&mut g, 40, &[1.0, 0.9, 1.1], &[0.0; 3], 0.0);
    let o = McsOptions {
        size: 0.1,
        method: McsMethod::Range,
        block_size: Some(4),
        reps: 50,
        scheme: ResampleScheme::Stationary,
        seed: 0,
    };
    let m = err_msg(model_confidence_set(&losses[..1], &o));
    assert!(
        m.contains("losses = 1 column(s)") && m.contains("at least two"),
        "{m}"
    );
    let m = err_msg(model_confidence_set(&[vec![1.0], vec![2.0]], &o));
    assert!(m.contains("losses = 1 period(s)"), "{m}");
    let ragged = vec![losses[0].clone(), losses[1][..10].to_vec()];
    let m = err_msg(model_confidence_set(&ragged, &o));
    assert!(
        m.contains("losses") && m.contains("column 1") && m.contains("10"),
        "{m}"
    );
    for bad in [0.0, 1.0, 2.0] {
        let m = err_msg(model_confidence_set(
            &losses,
            &McsOptions {
                size: bad,
                ..o.clone()
            },
        ));
        assert!(m.contains(&format!("size = {bad}")), "{m}");
    }
    let m = err_msg(model_confidence_set(
        &losses,
        &McsOptions {
            reps: 0,
            ..o.clone()
        },
    ));
    assert!(m.contains("reps = 0"), "{m}");
    let m = err_msg(model_confidence_set(
        &losses,
        &McsOptions {
            block_size: Some(40),
            ..o.clone()
        },
    ));
    assert!(m.contains("block_size = 40"), "{m}");
    // Identical columns: zero pairwise bootstrap variance (range) / zero
    // standard deviation (max, with two models).
    let dup = vec![losses[0].clone(), losses[1].clone(), losses[1].clone()];
    let m = err_msg(model_confidence_set(&dup, &o));
    assert!(
        m.contains("models 1 and 2") && m.contains("identical"),
        "{m}"
    );
    let dup2 = vec![losses[1].clone(), losses[1].clone()];
    let m = err_msg(model_confidence_set(
        &dup2,
        &McsOptions {
            method: McsMethod::Max,
            ..o.clone()
        },
    ));
    assert!(
        m.contains("standard deviation") && m.contains("model 0"),
        "{m}"
    );
    let mut nan = losses.clone();
    nan[2][4] = f64::NAN;
    let m = err_msg(model_confidence_set(&nan, &o));
    assert!(
        m.contains("losses") && m.contains("period 4 of column 2"),
        "{m}"
    );
    let m = err_msg(model_confidence_set_with_indices(&losses, &[], &o));
    assert!(m.contains("resamples = 0"), "{m}");
}
