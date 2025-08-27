<h1 align="center">Whirlaway 🐎</h1>

In this branch lives a simple [zkVM](minimal_zkVM.pdf).

## AIR Proving System

The protocol is detailed in [Whirlaway.pdf](Whirlaway.pdf)

The core argument builds upon [SuperSpartan](https://eprint.iacr.org/2023/552.pdf) (Srinath Setty, Justin Thaler, Riad Wahby), with AIR-specific optimizations developed by William Borgeaud in [A simple multivariate AIR argument inspired by SuperSpartan](https://solvable.group/posts/super-air/#fnref:1).

Key techniques:

- AIR table committed as a single multilinear polynomial
- Sumcheck + "Univariate Skip" from [Some Improvements for the PIOP for ZeroCheck](https://eprint.iacr.org/2024/108.pdf) (Angus Gruen)

## Benchmarks

cpu: i9-12900H, ram: 32 gb

> TLDR: Very slow, **but there is hope** (cf [TODO](TODO.md))

### Poseidon2

`RUSTFLAGS='-C target-cpu=native' cargo run --release`

CPU: 85K poseidon2 / s, GPU: 1M poseidon2 / s (RTX 4090) -> switch to branch [gpu](https://github.com/TomWambsgans/Whirlaway/tree/gpu)

### Recursion

`RUSTFLAGS='-C target-cpu=native' cargo test --release --package rec_aggregation --lib -- recursion::test_whir_recursion --nocapture`

The full recursion program is not finished yet. Instead, we prove validity of a WHIR opening, with 25 variables, and rate = 1/4.

![Alt text](docs/benchmark_graphs/graphs/recursive_whir_opening.svg)

### XMSS aggregation

`RUSTFLAGS='-C target-cpu=native' cargo test --release --package rec_aggregation --lib -- xmss_aggregate::test_xmss_aggregate --nocapture`


500 XMSS aggregated. "Trivial encoding". Very low lifetime (2^5, otherwise key generation is too long, but expect less than 10% of difference with longer liftetimes)

![Alt text](docs/benchmark_graphs/graphs/500_xmss_aggregated.svg)

## Credits

- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization.
- [whir-p3](https://github.com/tcoratger/whir-p3): a Plonky3-compatible WHIR implementation
