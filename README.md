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

CPU: 90K poseidon2 / s (i9-12900H), more soon

GPU: 1M poseidon2 / s (RTX 4090) -> switch to branch [gpu](https://github.com/TomWambsgans/Whirlaway/tree/gpu)

## Credits

- [Plonky3](https://github.com/Plonky3/Plonky3) for its finite field crates and poseidon2 AIR arithmetization (`src/examples/poseidon2_koala_bear`).
- [whir-p3](https://github.com/tcoratger/whir-p3): a Plonky3-compatible WHIR implementation

| StartFlag | Len | IndexA | IndexB | IndexRes | ValueA | ValueB | Res           | Computation                   |
| --------- | --- | ------ | ------ | -------- | ------ | ------ | ------------- | ----------------------------- |
| 1         | 4   | 90     | 211    | 74       | m[90]  | m[211] | m[74] = r3    | r3 = m[90] x m[211] + r2      |
| 0         | 3   | 91     | 212    | 74       | m[90]  | m[212] | m[74]         | r2 = m[91] x m[212] + r1      |
| 0         | 2   | 92     | 213    | 74       | m[90]  | m[213] | m[74]         | r1 = m[92] x m[213] + r0      |
| 0         | 1   | 93     | 214    | 74       | m[90]  | m[214] | m[74]         | r0 = m[93] x m[214]           |
| 1         | 10  | 1008   | 854    | 325      | m[90]  | m[854] | m[325] = r10' | r10' = m[1008] x m[854] + r9' |
| 0         | 9   | 1009   | 855    | 325      | m[90]  | m[855] | m[325]        | r9' = m[1009] x m[855] + r8'  |
| 0         | 8   | 1010   | 856    | 325      | m[90]  | m[856] | m[325]        | r8' = m[1010] x m[856] + r7'  |
| 0         | 7   | 1011   | 857    | 325      | m[90]  | m[857] | m[325]        | r7' = m[1011] x m[857] + r6'  |
| ...       | ... | ...    | ...    | ...      | ...    | ...    | ...           | ...                           |
