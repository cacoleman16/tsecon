//! Bit-identity regression for the pre-existing VECM deterministic cases.
//!
//! The 0.7.0 restricted-deterministic work rebuilt `fit_vecm_det`'s
//! internals (the lagged-levels block can now widen, the regressor block
//! can now carry seasonal dummies and a trend). The `"n"` and `"co"`
//! paths were required to stay **bit-identical** — same operations in the
//! same order, not merely close. These constants are the exact IEEE-754
//! bit patterns `fit_vecm_det` produced on the
//! `fixtures/vecm_deterministic.json` drifting draw at commit 6bd023c
//! (0.6.0), captured before the refactor; any drift here means the
//! shared code path changed arithmetic, which the 0.6.0 golden tests at
//! 1e-6 might not catch.

mod common;

use common::{as_endog, load_fixture};
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{fit_vecm_det, VecmDeterministic, VecmResult};

fn assert_bits(m: &Mat<f64>, expected: &[u64], what: &str) {
    assert_eq!(m.nrows() * m.ncols(), expected.len(), "{what}: shape");
    let mut idx = 0;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            assert_eq!(
                m[(i, j)].to_bits(),
                expected[idx],
                "{what}[({i},{j})]: {} != {}",
                m[(i, j)],
                f64::from_bits(expected[idx]),
            );
            idx += 1;
        }
    }
}

/// The exact 0.6.0 bit patterns of one deterministic case.
struct CaseBits {
    case: &'static str,
    alpha: &'static [u64],
    beta: &'static [u64],
    gamma: &'static [u64],
    det_coef: &'static [u64],
    sigma_u: &'static [u64],
    eig: &'static [u64],
    llf: u64,
}

fn assert_result_bits(r: &VecmResult, e: &CaseBits) {
    let case = e.case;
    assert_bits(&r.alpha, e.alpha, &format!("alpha ({case})"));
    assert_bits(&r.beta, e.beta, &format!("beta ({case})"));
    assert_bits(&r.gamma, e.gamma, &format!("gamma ({case})"));
    assert_bits(&r.det_coef, e.det_coef, &format!("det_coef ({case})"));
    assert_bits(&r.sigma_u, e.sigma_u, &format!("sigma_u ({case})"));
    assert_eq!(r.eig.len(), e.eig.len(), "eig length ({case})");
    for (i, (&a, &want)) in r.eig.iter().zip(e.eig).enumerate() {
        assert_eq!(a.to_bits(), want, "eig[{i}] ({case})");
    }
    assert_eq!(r.llf.to_bits(), e.llf, "llf ({case})");
    // The pre-0.7.0 result had no restricted deterministic terms at all.
    assert_eq!(r.det_coef_coint.nrows(), 0, "det_coef_coint rows ({case})");
    assert_eq!(r.seasons, 0, "seasons ({case})");
}

/// `deterministic = "n"` reproduces the 0.6.0 bits exactly.
#[test]
fn vecm_n_is_bit_identical_to_0_6_0() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let r = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::None).unwrap();
    assert_result_bits(
        &r,
        &CaseBits {
            case: "n",
            alpha: &[0xbfbf3cd67dff2c64, 0x3fa41c94a9375dc0, 0x3fa0b209a32014ee],
            beta: &[0x3ff0000000000000, 0x3f7548334e55b18a, 0xbfd4445a8c4276ae],
            gamma: &[
                0xbfddda0b8a57c7df,
                0x3fb27cd938a73d1d,
                0x3faf102234f95a3c,
                0xbfb27280f12a7316,
                0xbfd832c5b3daac58,
                0x3fc19e84accb03bb,
                0x3f838be5daa59ac5,
                0x3fa3c6c6e19623d7,
                0xbfdad2d1a57531ef,
            ],
            det_coef: &[],
            sigma_u: &[
                0x3f59c445854fa6b4,
                0x3ee3b45b8bd209dd,
                0x3f319a8bdb813564,
                0x3ee3b45b8bd209dd,
                0x3f60477ffb01ffc3,
                0x3f37f3e4996daa4d,
                0x3f319a8bdb813564,
                0x3f37f3e4996daa4d,
                0x3f5a954864f98533,
            ],
            eig: &[0x3fa5aa3e8bfaa3d6, 0x3f982880b5f6c92b, 0x3ed92300129eace8],
            llf: 0x40a091e6ce0548b6,
        },
    );
}

/// `deterministic = "co"` reproduces the 0.6.0 bits exactly.
#[test]
fn vecm_co_is_bit_identical_to_0_6_0() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let r = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    assert_result_bits(
        &r,
        &CaseBits {
            case: "co",
            alpha: &[0xbfc1d8c13d3f8497, 0xbfc27a325a25306d, 0x3fd2cd50e5ef736e],
            beta: &[0x3ff0000000000000, 0x3ff094b48c25a65e, 0xc00030bc28dec0e1],
            gamma: &[
                0xbfdd761e3016eceb,
                0x3fc1a6c262c9e0fa,
                0xbfb0ea9de63b18a7,
                0x3f929bed1a89d617,
                0xbfd3b8a1bd7c81c4,
                0xbf92baf4118231ac,
                0xbfbdfd4e40845723,
                0xbfba3790d02327a6,
                0xbfbe5da827a1bc36,
            ],
            det_coef: &[0x3ff9b8ab702e7196, 0x3ffa9f874e3e4d0c, 0xc00b1679348a36f3],
            sigma_u: &[
                0x3f59c03348ba6bef,
                0xbf11c6f2b9ae07b9,
                0x3f387be3aeee4c63,
                0xbf11c6f2b9ae07b9,
                0x3f5fa7151b50be6f,
                0x3f4035e27e8b2a8b,
                0x3f387be3aeee4c63,
                0x3f4035e27e8b2a8b,
                0x3f568028412663f7,
            ],
            eig: &[0x3fd2bc7d52a2bf41, 0x3fa26b9a6e2578ca, 0x3f833c190e8a5f9d],
            llf: 0x40a10a9f4e921a6e,
        },
    );
}
