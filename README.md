<h1 align="center">Whirlaway 🐎</h1>

A hash-based SNARK with lightweight proofs, powered by the [Whir](https://eprint.iacr.org/2024/1586) Polynomial Commitment Scheme.

## Specifications

- **Arithmetization**: AIR (Algebraic Intermediate Representation)
- **Proof size**: 150 / 200 KiB (with 128 bits of proven security)
- **Proving speed**: ~3000 poseidon2/s (field = koala-bear, on a standard laptop)

Being hash-based, the protocol is presumably post-quantum safe.

> **Note**: This library is experimental and not production-ready. The implementation is deliberately naive from both software and algorithmic perspectives. Known optimizations should yield >10× improvements in both aspects.
The open question is whether, once optimizations achieve ~100K poseidon2 hash/s, this approach can scale further to match Plonky3's impressive ~1M hash/s performance (at the expense of heavier proofs).

> Roadmap: Phase 1 = Speed. Phase 2 = Code quality.

## Proving System

The core argument builds upon [SuperSpartan](https://eprint.iacr.org/2023/552.pdf) (Srinath Setty, Justin Thaler, Riad Wahby), with AIR-specific optimizations developed by William Borgeaud in [A simple multivariate AIR argument inspired by SuperSpartan](https://solvable.group/posts/super-air/#fnref:1).

Key techniques:
- AIR table committed as a single multilinear polynomial
- Multiple column openings batched into a single PCS opening via the sumcheck protocol
- "Ring-switching" from [Binius' 2nd paper](https://eprint.iacr.org/2024/504.pdf) eliminates field "embedding overhead" in the PCS

Currently the main bottleneck is the sumcheck protocol, where the "embedding overhead" is substantial. The implementation uses the degree-8 extension of koala-bear, where multiplications are nearly 30× slower than in the prime field—presenting an immediate opportunity for optimization (see [this](https://eprint.iacr.org/2024/1046.pdf) and [this](https://eprint.iacr.org/2024/108.pdf)).

## Running the Poseidon2 Benchmark

```
cargo run --profile perf
```

## Credits

- `crates/whir` and `crates/merkle-tree` derive primarily from the [original Whir implementation](https://github.com/WizardOfMenlo/whir) by the [paper](https://eprint.iacr.org/2024/1586)'s authors: Gal Arnon, Alessandro Chiesa, Giacomo Fenzi, and Eylon Yogev.
- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization (`src/examples/poseidon2_koala_bear`).