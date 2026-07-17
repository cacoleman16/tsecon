//! Property and distributional invariant tests: index ranges, block
//! structure, stationary block-length distribution, wild-weight moments,
//! and Politis-White behavior on a known AR(1) process.
//!
//! All statistical checks use fixed seeds, so they are deterministic; the
//! 3-standard-error tolerances describe how the thresholds were chosen,
//! not a flake probability.

use tsecon_bootstrap::{indices, optimal_block_length, replicate, BlockScheme, WildWeights};
use tsecon_rng::Stream;

// ------------------------------------------------------------- ranges

#[test]
fn all_indices_in_range_and_full_length() {
    let mut stream = Stream::new(20260716);
    for n in [1usize, 2, 3, 17, 100, 1024] {
        let mut schemes = vec![
            BlockScheme::Iid,
            BlockScheme::MovingBlock { block_length: 1 },
            BlockScheme::CircularBlock { block_length: 1 },
            BlockScheme::MovingBlock { block_length: n },
            BlockScheme::CircularBlock { block_length: n },
            BlockScheme::Stationary { p: 0.05 },
            BlockScheme::Stationary { p: 0.5 },
            BlockScheme::Stationary { p: 1.0 },
        ];
        if n >= 3 {
            schemes.push(BlockScheme::MovingBlock {
                block_length: n / 2,
            });
            schemes.push(BlockScheme::CircularBlock {
                block_length: n / 2,
            });
        }
        for scheme in schemes {
            for rep in 0..5 {
                let out = indices(scheme, n, &mut stream).unwrap();
                assert_eq!(out.len(), n, "{scheme:?} n={n} rep={rep}");
                assert!(
                    out.iter().all(|&i| i < n),
                    "{scheme:?} n={n} rep={rep}: index out of range"
                );
            }
        }
    }
}

// ---------------------------------------------------- block structure

#[test]
fn moving_blocks_are_consecutive_and_never_wrap() {
    let (n, l) = (40usize, 7usize);
    let mut stream = Stream::new(11);
    for _ in 0..200 {
        let out = indices(BlockScheme::MovingBlock { block_length: l }, n, &mut stream).unwrap();
        for chunk in out.chunks(l) {
            let start = chunk[0];
            // Starts are drawn from 0..=n-l, so even the truncated final
            // block would have fit entirely inside the sample.
            assert!(start <= n - l, "start {start} could wrap");
            for (j, &idx) in chunk.iter().enumerate() {
                assert_eq!(idx, start + j, "block must be consecutive, no modulo");
            }
        }
    }
}

#[test]
fn circular_blocks_are_consecutive_modulo_n_and_do_wrap() {
    let (n, l) = (10usize, 7usize);
    let mut stream = Stream::new(13);
    let mut saw_wrap = false;
    for _ in 0..200 {
        let out = indices(
            BlockScheme::CircularBlock { block_length: l },
            n,
            &mut stream,
        )
        .unwrap();
        for chunk in out.chunks(l) {
            let start = chunk[0];
            for (j, &idx) in chunk.iter().enumerate() {
                assert_eq!(idx, (start + j) % n, "block must be consecutive mod n");
            }
            if start + chunk.len() > n {
                saw_wrap = true;
            }
        }
    }
    // With starts uniform on 0..10 and l=7, non-wrapping blocks have
    // probability 4/10 each; 400 blocks without a single wrap is impossible
    // in a working implementation.
    assert!(saw_wrap, "circular blocks never wrapped");
}

// ------------------------------------- stationary block-length law

#[test]
fn stationary_mean_block_length_matches_one_over_p() {
    // Segment lengths are geometric(p) truncated by the series end. With
    // n*p >> 1 the truncation and coincidental-continuation (a restart that
    // happens to land on prev+1, prob 1/n) biases are far below one
    // standard error of the per-replication mean; the check below is a
    // 3-standard-error interval around 1/p using the empirical SE.
    let n = 10_000usize;
    let p = 0.05f64;
    let n_reps = 300usize;

    let mean_lengths: Vec<f64> = replicate(20260716, n_reps, |_, stream| {
        let out = indices(BlockScheme::Stationary { p }, n, stream).unwrap();
        let breaks = out.windows(2).filter(|w| w[1] != (w[0] + 1) % n).count();
        n as f64 / (breaks + 1) as f64
    })
    .unwrap();

    let r = n_reps as f64;
    let mean = mean_lengths.iter().sum::<f64>() / r;
    let var = mean_lengths
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f64>()
        / (r - 1.0);
    let se = (var / r).sqrt();

    let target = 1.0 / p;
    assert!(
        (mean - target).abs() <= 3.0 * se,
        "mean block length {mean} not within 3 SE ({se}) of {target}"
    );
    // The SE itself must be sane: geometric segments imply nontrivial
    // replication-to-replication spread, so se == 0 would flag a bug.
    assert!(se > 0.0);
}

#[test]
fn stationary_with_p_one_is_iid() {
    // p = 1 restarts every step: identical in law to the iid bootstrap, and
    // identical draw-for-draw to n bounded uniforms.
    let n = 50;
    let mut a = Stream::new(9);
    let mut b = Stream::new(9);
    let stat = indices(BlockScheme::Stationary { p: 1.0 }, n, &mut a).unwrap();
    let mut expected = Vec::with_capacity(n);
    // Mirror the documented draw order: first index is a bounded draw, then
    // each step consumes one uniform (always < 1) and one bounded draw.
    expected.push(draw_index(&mut b, n));
    for _ in 1..n {
        let _coin = b.uniform_f64();
        expected.push(draw_index(&mut b, n));
    }
    assert_eq!(stat, expected);
}

/// Reference bitmask-rejection bounded draw, duplicated from the crate's
/// documented algorithm to pin the stream-consumption contract.
fn draw_index(stream: &mut Stream, n: usize) -> usize {
    let bound = n as u64;
    let mask = u64::MAX >> (bound - 1).leading_zeros();
    loop {
        let v = stream.next_u64() & mask;
        if v < bound {
            return v as usize;
        }
    }
}

// -------------------------------------------------- wild-weight moments

/// Empirical raw moments 1..=3 of `n` draws.
fn empirical_moments(weights: WildWeights, n: usize, seed: u64) -> (f64, f64, f64) {
    let mut stream = Stream::new(seed);
    let (mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        let w = weights.draw(&mut stream);
        m1 += w;
        m2 += w * w;
        m3 += w * w * w;
    }
    let nf = n as f64;
    (m1 / nf, m2 / nf, m3 / nf)
}

#[test]
fn mammen_moments_are_zero_one_one() {
    // Exact moments of the two-point Mammen law: E[w]=0, E[w^2]=1,
    // E[w^3]=1, E[w^4]=2, E[w^6]=5 (golden-ratio arithmetic). Monte Carlo
    // standard errors: SE(m1)=sqrt(1/N), SE(m2)=sqrt((E[w^4]-1)/N)
    // =sqrt(1/N), SE(m3)=sqrt((E[w^6]-E[w^3]^2)/N)=sqrt(4/N).
    let n = 500_000;
    let (m1, m2, m3) = empirical_moments(WildWeights::Mammen, n, 20260716);
    let root_n = (n as f64).sqrt();
    assert!(m1.abs() <= 3.0 / root_n, "mean {m1}");
    assert!((m2 - 1.0).abs() <= 3.0 / root_n, "variance {m2}");
    assert!((m3 - 1.0).abs() <= 3.0 * 2.0 / root_n, "third moment {m3}");
}

#[test]
fn mammen_point_probabilities() {
    let n = 500_000usize;
    let sqrt5 = 5.0f64.sqrt();
    let low = (1.0 - sqrt5) / 2.0;
    let p_low = (sqrt5 + 1.0) / (2.0 * sqrt5);
    let mut stream = Stream::new(4);
    let hits = (0..n)
        .filter(|_| WildWeights::Mammen.draw(&mut stream) == low)
        .count();
    let phat = hits as f64 / n as f64;
    let se = (p_low * (1.0 - p_low) / n as f64).sqrt();
    assert!(
        (phat - p_low).abs() <= 3.0 * se,
        "P(low) {phat} vs {p_low} (se {se})"
    );
}

#[test]
fn rademacher_moments_and_support() {
    let n = 500_000;
    let mut stream = Stream::new(8);
    let w = WildWeights::Rademacher.sample(n, &mut stream);
    assert!(w.iter().all(|&x| x == 1.0 || x == -1.0));
    let mean = w.iter().sum::<f64>() / n as f64;
    // E=0, Var=1 => SE(mean) = 1/sqrt(N); third moment equals the mean for
    // +/-1 weights, so no separate check is needed.
    assert!(mean.abs() <= 3.0 / (n as f64).sqrt(), "mean {mean}");
}

#[test]
fn normal_weights_moments() {
    // SE(m1)=sqrt(1/N), SE(m2)=sqrt(Var(w^2)/N)=sqrt(2/N),
    // SE(m3)=sqrt(E[w^6]/N)=sqrt(15/N) for the standard normal.
    let n = 500_000;
    let (m1, m2, m3) = empirical_moments(WildWeights::Normal, n, 123);
    let nf = n as f64;
    assert!(m1.abs() <= 3.0 * (1.0 / nf).sqrt(), "mean {m1}");
    assert!((m2 - 1.0).abs() <= 3.0 * (2.0 / nf).sqrt(), "variance {m2}");
    assert!(m3.abs() <= 3.0 * (15.0 / nf).sqrt(), "third moment {m3}");
}

// -------------------------------------------- Politis-White on AR(1)

/// Simulate a mean-zero AR(1) `x_t = phi x_{t-1} + e_t`, `e_t ~ N(0,1)`,
/// discarding a burn-in.
fn ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let burn = 200;
    let mut stream = Stream::new(seed);
    let mut x = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + burn {
        x = phi * x + WildWeights::Normal.draw(&mut stream);
        if t >= burn {
            out.push(x);
        }
    }
    out
}

#[test]
fn politis_white_on_ar1_is_finite_positive_and_stable() {
    let n = 1_000usize;
    let series = ar1(n, 0.7, 20260716);
    let b = optimal_block_length(&series).unwrap();
    let b_max = (3.0 * (n as f64).sqrt()).min(n as f64 / 3.0).ceil();

    for len in [b.stationary, b.circular] {
        assert!(len.is_finite(), "block length must be finite");
        assert!(len > 0.0, "block length must be positive");
        assert!(len <= b_max, "block length {len} exceeds cap {b_max}");
    }
    // Persistent AR(1) with phi=0.7 needs materially more than trivial
    // blocks; the plug-in formula's population value is ~20 at n=1000.
    assert!(b.stationary > 3.0, "stationary {} too small", b.stationary);
    assert!(b.circular > 3.0, "circular {} too small", b.circular);
    // b_CB and b_SB differ only in the constant (4/3 vs 2), so
    // b_CB = (3/2)^(1/3) b_SB when neither is clamped.
    let ratio = b.circular / b.stationary;
    assert!(
        (ratio - 1.5f64.cbrt()).abs() < 1e-12,
        "constant ratio violated: {ratio}"
    );

    // Stable: a pure function of the data.
    assert_eq!(b, optimal_block_length(&series).unwrap());

    // Stable across seeds: different simulated paths of the same process
    // land in the same broad band.
    for seed in [1u64, 2, 3] {
        let b2 = optimal_block_length(&ar1(n, 0.7, seed)).unwrap();
        assert!(
            b2.stationary > 3.0 && b2.stationary <= b_max,
            "seed {seed}: stationary {} out of band",
            b2.stationary
        );
    }
}

#[test]
fn politis_white_white_noise_prefers_short_blocks() {
    // iid data has no dependence to preserve: the selected block length
    // must be far below the AR(1) answer.
    let series = ar1(2_000, 0.0, 42);
    let b = optimal_block_length(&series).unwrap();
    assert!(b.stationary >= 1.0);
    assert!(
        b.stationary < 6.0,
        "white noise stationary length {} suspiciously long",
        b.stationary
    );
}

#[test]
fn politis_white_increases_with_persistence() {
    // More persistence => longer optimal blocks, on the same innovation
    // draws.
    let weak = optimal_block_length(&ar1(2_000, 0.3, 7)).unwrap();
    let strong = optimal_block_length(&ar1(2_000, 0.9, 7)).unwrap();
    assert!(
        strong.stationary > weak.stationary,
        "phi=0.9 ({}) should need longer blocks than phi=0.3 ({})",
        strong.stationary,
        weak.stationary
    );
}
