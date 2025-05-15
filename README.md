<h1 align="center">Whirlaway 🐎</h1>

A hash-based SNARK with lightweight proofs, powered by the [Whir](https://eprint.iacr.org/2024/1586) Polynomial Commitment Scheme.

## Specifications

- **Arithmetization**: AIR (Algebraic Intermediate Representation) with preprocessed columns
- **Security level**: 128 bits, presumably post-quantum (hash-based protocol)
- **Ingredients**: WHIR + Ring-Switching + Sumcheck + Univariate Skip

> **Note**: This library is under construction and not production-ready. Roadmap: Phase 1 = Perf. Phase 2 = Code quality.

## Proving System

The protocol is detailed in [Whirlaway.pdf](Whirlaway.pdf)

The core argument builds upon [SuperSpartan](https://eprint.iacr.org/2023/552.pdf) (Srinath Setty, Justin Thaler, Riad Wahby), with AIR-specific optimizations developed by William Borgeaud in [A simple multivariate AIR argument inspired by SuperSpartan](https://solvable.group/posts/super-air/#fnref:1).

Key techniques:

- AIR table committed as a single multilinear polynomial
- Multiple column openings batched into a single PCS opening via the sumcheck protocol
- "Univariate Skip" from [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/108.pdf) (Angus Gruen) to skip the "embedding overhead" in the initial rounds of the sumcheck
- "Ring-switching" from [Binius' 2nd paper](https://eprint.iacr.org/2024/504.pdf) eliminates field "embedding overhead" in the PCS

## Poseidon2 Benchmark

Proving poseidon2 hashes on the koala-bear field, at **128 bits of security**, with an **RTX 4090**.

The performance (particularly the proof size) depends on the "mutual correlated agreement" assumptions (4.12 in the WHIR paper). Even though the "Johnson bound" is not formally proven in the context of WHIR, the authors are confident that the techniques of [Proximity Gaps for Reed–Solomon Codes](https://eprint.iacr.org/2020/654.pdf) (Eli Ben-Sasson, Dan Carmon, Yuval Ishai, Swastik Kopparty) can be adapted.

The other alternative, the "capacity bound", often used in practice, is conjectured.

### Johnson bound

| poseidon2 count     | 2^16    | 2^17    | 2^18    | 2^19    | 2^20        |
| ------------------- | ------- | ------- | ------- | ------- | ----------- |
| WHIR initial rate   | 1/16    | 1/8     | 1/4     | 1/2     | 1/2         |
| WHIR folding factor | 4       | 4       | 4       | 4       | 5           |
| proving time        | 0.25 s  | 0.28 s  | 0.39 s  | 0.60 s  | 1.02 s      |
| hash / s            | 258K    | 453K    | 670K    | 863K    | 🐎 1.02M 🐎 |
| proof size          | 140 KiB | 169 KiB | 228 KiB | 344 KiB | 464 KiB     |
| verification time   | 3 ms    | 4 ms    | 6 ms    | 11 ms   | 13 ms       |

### Capacity bound (conjecture)

| poseidon2 count     | 2^16   | 2^17   | 2^18    | 2^19    | 2^20        |
| ------------------- | ------ | ------ | ------- | ------- | ----------- |
| WHIR initial rate   | 1/16   | 1/8    | 1/4     | 1/2     | 1/2         |
| WHIR folding factor | 4      | 4      | 4       | 4       | 5           |
| proving time        | 0.23 s | 0.28 s | 0.37 s  | 0.57 s  | 0.99 s      |
| hash / s            | 273K   | 452K   | 693K    | 913K    | 🐎 1.05M 🐎 |
| proof size          | 82 KiB | 97 KiB | 126 KiB | 187 KiB | 247 KiB     |
| verification time   | 2 ms   | 3 ms   | 5 ms    | 7 ms    | 8 ms        |

To reproduce the benchmark:

- Requires an Nvidia GPU (otherwise set `USE_CUDA` to false in `main.rs` but the current code is not optimized for CPU)
- Ensure CUDA Toolkit is installed, with a version matching the GPU driver
- `cargo run --release`

## Credits

- `crates/whir` and `crates/merkle-tree` derive primarily from the [original Whir implementation](https://github.com/WizardOfMenlo/whir) by the [paper](https://eprint.iacr.org/2024/1586)'s authors: Gal Arnon, Alessandro Chiesa, Giacomo Fenzi, and Eylon Yogev.
- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization (`src/examples/poseidon2_koala_bear`).
