<h1 align="center">Whirlaway 🐎</h1>

A hash-based SNARK with lightweight proofs, powered by the [Whir](https://eprint.iacr.org/2024/1586) Polynomial Commitment Scheme.

## Proving System

The protocol is detailed in [Whirlaway.pdf](Whirlaway.pdf)

The core argument builds upon [SuperSpartan](https://eprint.iacr.org/2023/552.pdf) (Srinath Setty, Justin Thaler, Riad Wahby), with AIR-specific optimizations developed by William Borgeaud in [A simple multivariate AIR argument inspired by SuperSpartan](https://solvable.group/posts/super-air/#fnref:1).

Key techniques:

- AIR table committed as a single multilinear polynomial
- Sumcheck + "Univariate Skip" from [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/108.pdf) (Angus Gruen)

## Poseidon2 Benchmark

`RUSTFLAGS='-C target-cpu=native' cargo run --release`

CPU: 92K poseidon2 / s (i9-12900H) -> WIP, expect improvements in the future

GPU: 1M poseidon2 / s (RTX 4090) -> switch to branch [gpu](https://github.com/TomWambsgans/Whirlaway/tree/gpu)

## Credits

- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization (`src/examples/poseidon2_koala_bear`).
- [whir-p3](https://github.com/tcoratger/whir-p3): a Plonky3-compatible WHIR implementation
