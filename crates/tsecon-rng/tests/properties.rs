//! Property and invariant tests beyond the golden fixtures: advance
//! consistency, spawn discipline, output conventions, and statistical
//! smoke tests.

use tsecon_rng::{RngError, SeedSequence, Stream};

// ---------------------------------------------------------------- advance

/// `advance(k)` must land on exactly the state reached by `k` draws, for
/// deltas that exercise every buffer alignment and multi-block skips.
#[test]
fn advance_k_equals_k_draws_from_fresh_stream() {
    for k in [0u128, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100, 1001, 4096, 12345] {
        let mut drawn = Stream::new(20260716);
        for _ in 0..k {
            let _ = drawn.next_u64();
        }
        let mut jumped = Stream::new(20260716);
        jumped.advance(k);
        assert_eq!(jumped.counter(), drawn.counter(), "counter mismatch at k={k}");
        for i in 0..8 {
            assert_eq!(
                jumped.next_u64(),
                drawn.next_u64(),
                "draw {i} after advance({k}) diverged"
            );
        }
    }
}

/// Same invariant starting mid-buffer (after a partial block was consumed).
#[test]
fn advance_is_consistent_mid_buffer() {
    for warmup in 1usize..=6 {
        for k in [0u128, 1, 2, 3, 4, 5, 9, 100] {
            let mut drawn = Stream::new(7);
            let mut jumped = Stream::new(7);
            for _ in 0..warmup {
                let _ = drawn.next_u64();
                let _ = jumped.next_u64();
            }
            for _ in 0..k {
                let _ = drawn.next_u64();
            }
            jumped.advance(k);
            for _ in 0..8 {
                assert_eq!(
                    jumped.next_u64(),
                    drawn.next_u64(),
                    "diverged at warmup={warmup} k={k}"
                );
            }
        }
    }
}

/// Advancing in two hops equals advancing once by the sum.
#[test]
fn advance_composes_additively() {
    let mut once = Stream::from_key_counter(0xDEAD_BEEF, 0);
    once.advance(1_000_003);
    let mut twice = Stream::from_key_counter(0xDEAD_BEEF, 0);
    twice.advance(999_999);
    twice.advance(4);
    assert_eq!(once.next_u64(), twice.next_u64());
}

/// A huge advance must not wrap the low counter word incorrectly: it equals
/// draws across the u64 word boundary of the counter.
#[test]
fn advance_crosses_counter_word_boundary() {
    // Start the counter just below the first word's max so a few draws carry
    // into the second word.
    let mut drawn = Stream::from_raw_key_counter([1, 2], [u64::MAX - 1, 0, 0, 0]);
    let mut jumped = drawn.clone();
    for _ in 0..10 {
        let _ = drawn.next_u64();
    }
    jumped.advance(10);
    assert_eq!(jumped.counter(), drawn.counter());
    assert_eq!(jumped.next_u64(), drawn.next_u64());
    // The carry actually happened.
    assert_eq!(drawn.counter()[1], 1);
}

// ------------------------------------------------------------ conventions

/// Two `next_u32` calls consume exactly one `u64`: low half then high half.
#[test]
fn u32_draws_are_split_u64s() {
    let mut raw = Stream::new(3);
    let mut split = Stream::new(3);
    for _ in 0..16 {
        let v = raw.next_u64();
        assert_eq!(split.next_u32(), v as u32);
        assert_eq!(split.next_u32(), (v >> 32) as u32);
    }
}

/// Fill methods are equivalent to repeated single draws.
#[test]
fn fill_matches_repeated_draws() {
    let mut a = Stream::new(11);
    let mut b = Stream::new(11);
    let mut buf64 = [0u64; 13];
    a.fill_u64(&mut buf64);
    for (i, v) in buf64.iter().enumerate() {
        assert_eq!(*v, b.next_u64(), "u64 fill diverged at {i}");
    }

    let mut a = Stream::new(11);
    let mut b = Stream::new(11);
    let mut buf32 = [0u32; 13];
    a.fill_u32(&mut buf32);
    for (i, v) in buf32.iter().enumerate() {
        assert_eq!(*v, b.next_u32(), "u32 fill diverged at {i}");
    }

    let mut a = Stream::new(11);
    let mut b = Stream::new(11);
    let mut buff = [0f64; 13];
    a.fill_uniform_f64(&mut buff);
    for (i, v) in buff.iter().enumerate() {
        assert_eq!(v.to_bits(), b.uniform_f64().to_bits(), "f64 fill diverged at {i}");
    }
}

/// Uniforms are the documented deterministic transform of the raws and live
/// on the 53-bit grid in [0, 1).
#[test]
fn uniform_is_53_bit_transform_of_raw() {
    let mut raw = Stream::new(17);
    let mut uni = Stream::new(17);
    for _ in 0..1000 {
        let r = raw.next_u64();
        let u = uni.uniform_f64();
        let expect = (r >> 11) as f64 * (1.0 / 9007199254740992.0);
        assert_eq!(u.to_bits(), expect.to_bits());
        assert!((0.0..1.0).contains(&u));
    }
}

/// Clones replay the identical sequence; the original is unaffected by
/// drawing from the clone.
#[test]
fn clone_replays_identical_sequence() {
    let mut a = Stream::new(2024);
    let _ = a.next_u64();
    let mut b = a.clone();
    let from_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
    let from_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
    assert_eq!(from_a, from_b);
}

// ------------------------------------------------------------------ spawn

/// Incremental spawning matches batch spawning: spawn(2) then spawn(3)
/// yields the same children as spawn(5).
#[test]
fn incremental_spawn_matches_batch_spawn() {
    let mut batch_parent = SeedSequence::new(42);
    let batch = batch_parent.spawn(5).unwrap();

    let mut inc_parent = SeedSequence::new(42);
    let mut incremental = inc_parent.spawn(2).unwrap();
    incremental.extend(inc_parent.spawn(3).unwrap());
    assert_eq!(inc_parent.children_spawned(), 5);

    for (i, (a, b)) in batch.iter().zip(&incremental).enumerate() {
        assert_eq!(a.spawn_key(), b.spawn_key(), "spawn key mismatch at {i}");
        assert_eq!(
            a.generate_state_u32(4),
            b.generate_state_u32(4),
            "child {i} state mismatch"
        );
    }
}

/// Grandchildren extend the spawn key and differ from children and parent.
#[test]
fn spawn_key_propagates_to_grandchildren() {
    let mut root = SeedSequence::new(9);
    let mut child = root.spawn(1).unwrap().remove(0);
    let grandchild = child.spawn(1).unwrap().remove(0);
    assert_eq!(child.spawn_key(), &[0]);
    assert_eq!(grandchild.spawn_key(), &[0, 0]);
    let states = [
        root.generate_state_u32(4),
        child.generate_state_u32(4),
        grandchild.generate_state_u32(4),
    ];
    assert_ne!(states[0], states[1]);
    assert_ne!(states[0], states[2]);
    assert_ne!(states[1], states[2]);
}

/// The spawn limit surfaces as a typed error, not a panic.
#[test]
fn substream_spawn_limit_is_an_error() {
    let mut root = SeedSequence::new(1);
    // u32::MAX + 1 children are allowed in total; one more must fail.
    root.spawn(10).unwrap();
    let err = root.spawn(usize::MAX).unwrap_err();
    assert_eq!(
        err,
        RngError::SpawnLimitExceeded {
            requested: usize::MAX,
            available: u64::from(u32::MAX) + 1 - 10,
        }
    );
}

// -------------------------------------------------------------- substreams

/// Substreams are deterministic and stable across calls (the parallel
/// replication contract).
#[test]
fn substreams_are_reproducible() {
    let mut a = Stream::substreams(20260716, 8).unwrap();
    let mut b = Stream::substreams(20260716, 8).unwrap();
    for (i, (x, y)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
        for _ in 0..32 {
            assert_eq!(x.next_u64(), y.next_u64(), "substream {i} not reproducible");
        }
    }
}

/// Substream independence smoke test: distinct keys, no shared prefixes,
/// and negligible cross-correlation between the uniform sequences.
#[test]
fn substreams_are_pairwise_independent_smoke() {
    const N: usize = 10_000;
    let mut streams = Stream::substreams(0, 4).unwrap();

    // Distinct keys by construction.
    for i in 0..streams.len() {
        for j in (i + 1)..streams.len() {
            assert_ne!(streams[i].key(), streams[j].key(), "streams {i},{j} share a key");
        }
    }

    let samples: Vec<Vec<f64>> = streams
        .iter_mut()
        .map(|s| {
            let mut buf = vec![0.0; N];
            s.fill_uniform_f64(&mut buf);
            buf
        })
        .collect();

    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let (a, b) = (&samples[i], &samples[j]);
            assert_ne!(a[..64], b[..64], "streams {i},{j} share a prefix");
            // Sample correlation of U(0,1) draws: mean 0, sd ~ 1/sqrt(N).
            let ma = a.iter().sum::<f64>() / N as f64;
            let mb = b.iter().sum::<f64>() / N as f64;
            let mut cov = 0.0;
            let mut va = 0.0;
            let mut vb = 0.0;
            for k in 0..N {
                cov += (a[k] - ma) * (b[k] - mb);
                va += (a[k] - ma).powi(2);
                vb += (b[k] - mb).powi(2);
            }
            let corr = cov / (va.sqrt() * vb.sqrt());
            // 5 standard errors ~ 0.05 at N = 10_000.
            assert!(
                corr.abs() < 0.05,
                "streams {i},{j} correlated: r = {corr}"
            );
        }
    }
}

// -------------------------------------------------------------- statistics

/// Mean and variance of 1e5 uniforms match U(0,1) within 5 standard errors.
#[test]
fn uniform_moments_smoke() {
    const N: usize = 100_000;
    let mut stream = Stream::new(123_456_789);
    let mut buf = vec![0.0f64; N];
    stream.fill_uniform_f64(&mut buf);

    let mean = buf.iter().sum::<f64>() / N as f64;
    let var = buf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (N - 1) as f64;

    // sd(mean) = sqrt(1/12/N) ~ 0.00091; sd(var) ~ sqrt(1/180/N) ~ 0.00024.
    let mean_tol = 5.0 * (1.0f64 / 12.0 / N as f64).sqrt();
    let var_tol = 5.0 * (1.0f64 / 180.0 / N as f64).sqrt();
    assert!(
        (mean - 0.5).abs() < mean_tol,
        "mean {mean} outside 0.5 +/- {mean_tol}"
    );
    assert!(
        (var - 1.0 / 12.0).abs() < var_tol,
        "variance {var} outside 1/12 +/- {var_tol}"
    );

    // All draws in [0, 1).
    assert!(buf.iter().all(|u| (0.0..1.0).contains(u)));
}

/// Raw u32 draws hit both halves of the range about equally (catches
/// endianness/half-word mistakes that moment tests on f64 would miss).
#[test]
fn u32_high_bit_is_balanced() {
    const N: usize = 100_000;
    let mut stream = Stream::new(31_337);
    let mut high = 0usize;
    for _ in 0..N {
        if stream.next_u32() >= 1 << 31 {
            high += 1;
        }
    }
    let frac = high as f64 / N as f64;
    // 5 standard errors of a fair coin at N = 1e5 ~ 0.0079.
    assert!((frac - 0.5).abs() < 0.008, "high-bit fraction {frac}");
}

// ------------------------------------------------------------ concurrency

/// Streams can be moved into threads and produce the same results as when
/// run sequentially (thread-count invariance of the substream contract).
#[test]
fn substreams_match_across_scheduling() {
    let sequential: Vec<u64> = Stream::substreams(55, 4)
        .unwrap()
        .iter_mut()
        .map(|s| {
            let mut last = 0;
            for _ in 0..100 {
                last = s.next_u64();
            }
            last
        })
        .collect();

    let handles: Vec<_> = Stream::substreams(55, 4)
        .unwrap()
        .into_iter()
        .map(|mut s| {
            std::thread::spawn(move || {
                let mut last = 0;
                for _ in 0..100 {
                    last = s.next_u64();
                }
                last
            })
        })
        .collect();
    let parallel: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    assert_eq!(sequential, parallel);
}
