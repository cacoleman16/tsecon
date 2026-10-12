//! Golden-value tests against fixtures/philox.json (NumPy 1.26.4) and
//! fixtures/rng_substreams.json (NumPy 2.5, lazily derived children).
//!
//! Everything in this file must match NumPy bit for bit: raw u64 output,
//! Generator uniforms (a deterministic transform of the raws), SeedSequence
//! state words, and spawned or directly derived children.

use serde_json::Value;
use tsecon_rng::{SeedSequence, Stream};

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/philox.json");
    let text = std::fs::read_to_string(path).expect("fixture philox.json must exist");
    serde_json::from_str(&text).expect("fixture must be valid JSON")
}

/// Fixture integers are decimal strings (JSON numbers cannot hold u64).
fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|s| {
            s.as_str()
                .expect("expected decimal string")
                .parse::<u64>()
                .expect("expected u64 decimal string")
        })
        .collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|s| {
            s.as_str()
                .expect("expected decimal string")
                .parse::<u32>()
                .expect("expected u32 decimal string")
        })
        .collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected f64"))
        .collect()
}

#[test]
fn explicit_key_counter_raw_u64_bit_exact() {
    let fx = fixture();
    let cases = fx["explicit_key_counter"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let key: u128 = case["key"].as_str().unwrap().parse().unwrap();
        let counter: u128 = case["counter"].as_str().unwrap().parse().unwrap();
        let want = u64s(&case["raw_uint64"]);
        let mut stream = Stream::from_key_counter(key, counter);
        let mut got = vec![0u64; want.len()];
        stream.fill_u64(&mut got);
        assert_eq!(got, want, "raw mismatch for key={key} counter={counter}");
    }
}

#[test]
fn seeded_raw_u64_bit_exact() {
    let fx = fixture();
    for case in fx["seeded"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let want = u64s(&case["raw_uint64"]);
        let mut stream = Stream::new(seed);
        let got: Vec<u64> = (0..want.len()).map(|_| stream.next_u64()).collect();
        assert_eq!(got, want, "raw mismatch for seed={seed}");
    }
}

#[test]
fn seeded_generator_uniforms_bit_exact() {
    let fx = fixture();
    for case in fx["seeded"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let want = f64s(&case["uniform_f64"]);
        let mut stream = Stream::new(seed);
        let mut got = vec![0.0; want.len()];
        stream.fill_uniform_f64(&mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "uniform[{i}] mismatch for seed={seed}: got {g:?}, want {w:?}"
            );
        }
    }
}

#[test]
fn seed_sequence_state_u32_bit_exact() {
    let fx = fixture();
    for case in fx["seed_sequence"].as_array().unwrap() {
        let entropy = case["entropy"].as_u64().unwrap();
        let want = u32s(&case["state_uint32_8"]);
        let ss = SeedSequence::new(u128::from(entropy));
        assert_eq!(
            ss.generate_state_u32(want.len()),
            want,
            "u32 state mismatch for entropy={entropy}"
        );
    }
}

#[test]
fn seed_sequence_state_u64_bit_exact() {
    let fx = fixture();
    for case in fx["seed_sequence"].as_array().unwrap() {
        let entropy = case["entropy"].as_u64().unwrap();
        let want = u64s(&case["state_uint64_4"]);
        let ss = SeedSequence::new(u128::from(entropy));
        assert_eq!(
            ss.generate_state_u64(want.len()),
            want,
            "u64 state mismatch for entropy={entropy}"
        );
    }
}

#[test]
fn seed_sequence_spawned_children_bit_exact() {
    let fx = fixture();
    for case in fx["seed_sequence"].as_array().unwrap() {
        let entropy = case["entropy"].as_u64().unwrap();
        let want_children: Vec<Vec<u32>> = case["children_state_uint32_4"]
            .as_array()
            .unwrap()
            .iter()
            .map(u32s)
            .collect();
        let mut ss = SeedSequence::new(u128::from(entropy));
        let children = ss.spawn(want_children.len()).unwrap();
        for (i, (child, want)) in children.iter().zip(&want_children).enumerate() {
            assert_eq!(child.spawn_key(), &[i as u32]);
            assert_eq!(
                child.generate_state_u32(want.len()),
                *want,
                "child {i} state mismatch for entropy={entropy}"
            );
        }
    }
}

// ------------------------------------------- lazily derived substreams

fn substream_fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/rng_substreams.json"
    );
    let text = std::fs::read_to_string(path).expect("fixture rng_substreams.json must exist");
    serde_json::from_str(&text).expect("fixture must be valid JSON")
}

/// `Stream::substream(seed, i)` and `SeedSequence::child(i)` are NumPy's
/// `SeedSequence(seed, spawn_key=(i,))` bit for bit, at child indices up to
/// the last one the 32-bit spawn-key space admits (`2^32 - 1`) — the
/// identity the lazy replication drivers rest on: no vector of `n` streams
/// is needed to reach replication `i`.
#[test]
fn lazily_derived_substreams_bit_exact() {
    let fx = substream_fixture();
    let cases = fx["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 50);
    for case in cases {
        let seed: u64 = case["seed"].as_str().unwrap().parse().unwrap();
        let index = u32::try_from(case["index"].as_u64().unwrap()).unwrap();
        let ctx = format!("seed={seed} index={index}");

        let child = SeedSequence::new(u128::from(seed)).child(index);
        assert_eq!(child.spawn_key(), &[index], "{ctx}");
        assert_eq!(
            child.generate_state_u32(4),
            u32s(&case["state_uint32_4"]),
            "{ctx}: child state"
        );

        let want_raw = u64s(&case["raw_uint64"]);
        let mut s = Stream::substream(seed, index);
        let got: Vec<u64> = (0..want_raw.len()).map(|_| s.next_u64()).collect();
        assert_eq!(got, want_raw, "{ctx}: raw draws");
        let mut via_child = Stream::from_seed_sequence(&child);
        assert_eq!(via_child.next_u64(), want_raw[0], "{ctx}: via child()");

        let want_u = f64s(&case["uniform_f64"]);
        let mut s = Stream::substream(seed, index);
        for (i, w) in want_u.iter().enumerate() {
            assert_eq!(
                s.uniform_f64().to_bits(),
                w.to_bits(),
                "{ctx}: uniform[{i}]"
            );
        }
    }
}

/// The materialized `Stream::substreams(seed, n)` (now built straight from
/// each child sequence) still equals NumPy's
/// `[Philox(seed=c) for c in SeedSequence(seed).spawn(n)]`, across a
/// 1024-child boundary.
#[test]
fn materialized_substreams_match_numpy_spawn() {
    let fx = substream_fixture();
    let drv = &fx["driver"];
    let seed: u64 = drv["seed"].as_str().unwrap().parse().unwrap();
    let n = usize::try_from(drv["n"].as_u64().unwrap()).unwrap();
    let want: Vec<Vec<u64>> = drv["raw_uint64_2"]
        .as_array()
        .unwrap()
        .iter()
        .map(u64s)
        .collect();
    assert_eq!(want.len(), n);
    let mut streams = Stream::substreams(seed, n).unwrap();
    for (i, (s, w)) in streams.iter_mut().zip(&want).enumerate() {
        assert_eq!(vec![s.next_u64(), s.next_u64()], *w, "substream {i}");
    }
    let mut root = SeedSequence::new(u128::from(seed));
    for (i, (c, w)) in root.spawn(n).unwrap().iter().zip(&want).enumerate() {
        assert_eq!(
            Stream::from_seed_sequence(c).next_u64(),
            w[0],
            "spawned child {i}"
        );
    }
}
