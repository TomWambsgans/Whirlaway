<h1 align="center">Whirlaway 🐎</h1>

A hash-based SNARK with lightweight proofs, powered by the [Whir](https://eprint.iacr.org/2024/1586) Polynomial Commitment Scheme.

## Specifications

- **Arithmetization**: AIR (Algebraic Intermediate Representation) with preprocessed columns
- **Security level**: 128 bits (without conjectures), presumably post-quantum (hash-based protocol)
- **Ingredients**: WHIR + Ring-Switching + Sumcheck + Univariate Skip

> **Note**: This library is under construction and not production-ready. Roadmap: Phase 1 = Perf. Phase 2 = Code quality.

## Proving System

The core argument builds upon [SuperSpartan](https://eprint.iacr.org/2023/552.pdf) (Srinath Setty, Justin Thaler, Riad Wahby), with AIR-specific optimizations developed by William Borgeaud in [A simple multivariate AIR argument inspired by SuperSpartan](https://solvable.group/posts/super-air/#fnref:1).

Key techniques:

- AIR table committed as a single multilinear polynomial
- Multiple column openings batched into a single PCS opening via the sumcheck protocol
- "Univariate Skip" from [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/108.pdf) (Angus Gruen) to skip the "embedding overhead" in the initial rounds of the sumcheck
- "Ring-switching" from [Binius' 2nd paper](https://eprint.iacr.org/2024/504.pdf) eliminates field "embedding overhead" in the PCS

## Poseidon2 Benchmark

Proving poseidon2 hashes on the koala-bear field. Note that we are limited by the two adicity (24) of the field (other fields will be included in the future).

### RTX 4090

| WHIR initial rate | 1/16    | 1/8     | 1/4     | 1/2     |
| ----------------- | ------- | ------- | ------- | ------- |
| poseidon2 count   | 2^15    | 2^16    | 2^17    | 2^18    |
| proving time      | 0.35 s  | 0.43 s  | 0.57 s  | 0.85 s  |
| hash / s          | 95K     | 152K    | 229K    | 307K    |
| proof size        | 153 KiB | 185 KiB | 229 KiB | 344 KiB |
| verification time | 6 ms    | 6 ms    | 7 ms    | 8 ms    |

### RTX 3060 mobile

| WHIR initial rate | 1/16    | 1/8     | 1/4     |
| ----------------- | ------- | ------- | ------- |
| poseidon2 count   | 2^15    | 2^16    | 2^17    |
| proving time      | 0.76 s  | 0.88 s  | 1.29 s  |
| hash / s          | 43K     | 74K     | 102K    |
| proof size        | 153 KiB | 185 KiB | 229 KiB |
| verification time | 6 ms    | 6 ms    | 7 ms    |

To reproduce the benchmark:

- Requires an Nvidia GPU (otherwise set `USE_CUDA` to false in `main.rs` but the current code is not optimized for CPU)
- Ensure CUDA Toolkit is installed, with a version matching the GPU driver
- `cargo run --release`

## Credits

- `crates/whir` and `crates/merkle-tree` derive primarily from the [original Whir implementation](https://github.com/WizardOfMenlo/whir) by the [paper](https://eprint.iacr.org/2024/1586)'s authors: Gal Arnon, Alessandro Chiesa, Giacomo Fenzi, and Eylon Yogev.
- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization (`src/examples/poseidon2_koala_bear`).
