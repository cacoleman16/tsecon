//! Reproducibility tests: the crate's headline contract is that every
//! result is a pure function of the user's seed — identical across calls,
//! identical between sequential and parallel drivers, and identical at any
//! thread count.

use tsecon_bootstrap::{indices, par_replicate, replicate, BlockScheme, WildWeights};
use tsecon_rng::Stream;

const N: usize = 64;

fn all_schemes() -> Vec<BlockScheme> {
    vec![
        BlockScheme::Iid,
        BlockScheme::MovingBlock { block_length: 7 },
        BlockScheme::CircularBlock { block_length: 7 },
        BlockScheme::Stationary { p: 0.15 },
    ]
}

#[test]
fn same_seed_same_indices() {
    for scheme in all_schemes() {
        let mut a = Stream::new(20260716);
        let mut b = Stream::new(20260716);
        assert_eq!(
            indices(scheme, N, &mut a).unwrap(),
            indices(scheme, N, &mut b).unwrap(),
            "{scheme:?}"
        );
    }
}

#[test]
fn different_seeds_differ() {
    // Not a hard mathematical guarantee, but a collision across all four
    // schemes at n=64 would indicate a seeding bug, not bad luck.
    let differs = all_schemes().iter().any(|&scheme| {
        let mut a = Stream::new(1);
        let mut b = Stream::new(2);
        indices(scheme, N, &mut a).unwrap() != indices(scheme, N, &mut b).unwrap()
    });
    assert!(differs);
}

#[test]
fn par_replicate_is_thread_count_invariant() {
    // The headline contract: bit-identical output vectors from explicit
    // 1-thread and 14-thread rayon pools, for every scheme plus wild
    // weights, and equal to the sequential driver.
    let seed = 987654321;
    let n_reps = 64;

    let task = |_rep: usize, stream: &mut Stream| {
        let mut out: Vec<(Vec<usize>, u64)> = Vec::new();
        for scheme in all_schemes() {
            let idx = indices(scheme, N, stream).unwrap();
            // Fold wild weights in so their draws are covered too; hash the
            // bits so the comparison is exact, not approximate.
            let w = WildWeights::Mammen.draw(stream).to_bits()
                ^ WildWeights::Normal.draw(stream).to_bits()
                ^ WildWeights::Rademacher.draw(stream).to_bits();
            out.push((idx, w));
        }
        out
    };

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool14 = rayon::ThreadPoolBuilder::new()
        .num_threads(14)
        .build()
        .unwrap();

    let r1 = pool1.install(|| par_replicate(seed, n_reps, task).unwrap());
    let r14 = pool14.install(|| par_replicate(seed, n_reps, task).unwrap());
    let seq = replicate(seed, n_reps, task).unwrap();

    assert_eq!(r1, r14, "1-thread vs 14-thread pools must be bit-identical");
    assert_eq!(r1, seq, "parallel vs sequential drivers must be identical");
}

#[test]
fn par_replicate_repeated_calls_are_identical() {
    let f = |_i: usize, s: &mut Stream| indices(BlockScheme::Stationary { p: 0.1 }, 100, s);
    let a = par_replicate(5, 32, f).unwrap();
    let b = par_replicate(5, 32, f).unwrap();
    assert_eq!(a, b);
}

#[test]
fn replications_are_independent_substreams() {
    let reps = replicate(3, 16, |_, s| indices(BlockScheme::Iid, N, s).unwrap()).unwrap();
    // No two replications share a substream (identical index vectors of
    // length 64 would be a spawn bug, not chance).
    for i in 0..reps.len() {
        for j in i + 1..reps.len() {
            assert_ne!(reps[i], reps[j], "replications {i} and {j} collided");
        }
    }
}

#[test]
fn wild_weights_same_seed_same_draws() {
    for weights in [
        WildWeights::Rademacher,
        WildWeights::Mammen,
        WildWeights::Normal,
    ] {
        let mut a = Stream::new(77);
        let mut b = Stream::new(77);
        let wa = weights.sample(256, &mut a);
        let wb = weights.sample(256, &mut b);
        assert_eq!(wa, wb, "{weights:?}");
    }
}
