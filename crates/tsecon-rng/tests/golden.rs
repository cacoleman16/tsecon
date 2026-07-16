//! Golden-value tests against fixtures/philox.json (NumPy 1.26.4).
//!
//! Everything in this file must match NumPy bit for bit: raw u64 output,
//! Generator uniforms (a deterministic transform of the raws), SeedSequence
//! state words, and spawned children.

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
