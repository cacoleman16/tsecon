# Third-party licenses

`tsecon`'s compiled wheel statically links the Rust crates listed below. All
are permissive (MIT / Apache-2.0 / BSD / Zlib / Unicode-3.0) — there is no
copyleft (GPL/LGPL/AGPL/MPL) anywhere in the dependency tree. This file is the
license inventory; the full verbatim copyright notices for each crate are
reproduced in released wheels (generated with `cargo about` at release time,
per the packaging plan in docs/roadmap/14-packaging-distribution.md).

Generated from `cargo metadata` — 97 third-party packages.

| Crate | Version | License |
|---|---|---|
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| bitflags | 2.13.1 | MIT OR Apache-2.0 |
| bytemuck | 1.25.1 | Zlib OR Apache-2.0 OR MIT |
| bytemuck_derive | 1.11.0 | Zlib OR Apache-2.0 OR MIT |
| byteorder | 1.5.0 | Unlicense OR MIT |
| cfg-if | 1.0.4 | MIT OR Apache-2.0 |
| crossbeam-deque | 0.8.7 | MIT OR Apache-2.0 |
| crossbeam-epoch | 0.9.20 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8.22 | MIT OR Apache-2.0 |
| crunchy | 0.2.4 | MIT |
| defer | 0.2.1 | MIT/Apache-2.0 |
| dyn-stack | 0.13.2 | MIT |
| dyn-stack-macros | 0.1.3 | MIT |
| either | 1.16.0 | MIT OR Apache-2.0 |
| enum-as-inner | 0.6.1 | MIT/Apache-2.0 |
| equator | 0.2.2 | MIT |
| equator | 0.6.0 | MIT |
| equator-macro | 0.2.1 | MIT |
| equator-macro | 0.6.0 | MIT |
| faer | 0.24.4 | MIT |
| faer-traits | 0.24.0 | MIT |
| gemm | 0.19.0 | MIT |
| gemm-c32 | 0.19.0 | MIT |
| gemm-c64 | 0.19.0 | MIT |
| gemm-common | 0.19.0 | MIT |
| gemm-f16 | 0.19.0 | MIT |
| gemm-f32 | 0.19.0 | MIT |
| gemm-f64 | 0.19.0 | MIT |
| generativity | 1.2.1 | MIT OR Apache-2.0 |
| half | 2.7.1 | MIT OR Apache-2.0 |
| heck | 0.5.0 | MIT OR Apache-2.0 |
| hermit-abi | 0.5.2 | MIT OR Apache-2.0 |
| interpol | 0.2.1 | MIT |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| libc | 0.2.186 | MIT OR Apache-2.0 |
| libm | 0.2.16 | MIT |
| matrixmultiply | 0.3.11 | MIT/Apache-2.0 |
| memchr | 2.8.3 | Unlicense OR MIT |
| nano-gemm | 0.2.2 | MIT |
| nano-gemm-c32 | 0.2.1 | MIT |
| nano-gemm-c64 | 0.2.1 | MIT |
| nano-gemm-codegen | 0.2.1 | MIT |
| nano-gemm-core | 0.2.1 | MIT |
| nano-gemm-f32 | 0.2.1 | MIT |
| nano-gemm-f64 | 0.2.1 | MIT |
| ndarray | 0.17.2 | MIT OR Apache-2.0 |
| num-complex | 0.4.6 | MIT OR Apache-2.0 |
| num-integer | 0.1.46 | MIT OR Apache-2.0 |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 |
| numpy | 0.29.0 | BSD-2-Clause |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| paste | 1.0.15 | MIT OR Apache-2.0 |
| portable-atomic | 1.13.1 | Apache-2.0 OR MIT |
| portable-atomic-util | 0.2.7 | Apache-2.0 OR MIT |
| primal-check | 0.3.4 | MIT OR Apache-2.0 |
| private-gemm-x86 | 0.1.20 | MIT |
| proc-macro2 | 1.0.106 | MIT OR Apache-2.0 |
| pulp | 0.22.3 | MIT |
| pulp-wasm-simd-flag | 0.1.1 | MIT |
| pyo3 | 0.29.0 | MIT OR Apache-2.0 |
| pyo3-build-config | 0.29.0 | MIT OR Apache-2.0 |
| pyo3-ffi | 0.29.0 | MIT OR Apache-2.0 |
| pyo3-macros | 0.29.0 | MIT OR Apache-2.0 |
| pyo3-macros-backend | 0.29.0 | MIT OR Apache-2.0 |
| qd | 0.8.0 | MIT |
| quote | 1.0.46 | MIT OR Apache-2.0 |
| raw-cpuid | 11.6.0 | MIT |
| rawpointer | 0.2.1 | MIT/Apache-2.0 |
| rayon | 1.12.0 | MIT OR Apache-2.0 |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 |
| reborrow | 0.5.5 | MIT |
| rustc-hash | 2.1.3 | Apache-2.0 OR MIT |
| rustfft | 6.4.1 | MIT OR Apache-2.0 |
| same-file | 1.0.6 | Unlicense/MIT |
| seq-macro | 0.3.6 | MIT OR Apache-2.0 |
| serde | 1.0.228 | MIT OR Apache-2.0 |
| serde_core | 1.0.228 | MIT OR Apache-2.0 |
| serde_derive | 1.0.228 | MIT OR Apache-2.0 |
| serde_json | 1.0.150 | MIT OR Apache-2.0 |
| strength_reduce | 0.2.4 | MIT OR Apache-2.0 |
| syn | 1.0.109 | MIT OR Apache-2.0 |
| syn | 2.0.119 | MIT OR Apache-2.0 |
| sysctl | 0.6.0 | MIT |
| target-lexicon | 0.13.5 | Apache-2.0 WITH LLVM-exception |
| thiserror | 1.0.69 | MIT OR Apache-2.0 |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 |
| transpose | 0.2.3 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.24 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| version_check | 0.9.5 | MIT/Apache-2.0 |
| walkdir | 2.5.0 | Unlicense/MIT |
| winapi-util | 0.1.11 | Unlicense OR MIT |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| zerocopy | 0.8.54 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerocopy-derive | 0.8.54 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zmij | 1.0.23 | MIT |
