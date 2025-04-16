# TODO

- avoid repeated access to global memory in cuda sumcheck when there are enough registers
- cuda sumcheck: first round, if n_batching_scalars = 0 => sums are actually in F, not EF
- In whir::prove, the hypercube sum is already known, so no need to recompute it in sumcheck::prove
- Avoid variable reversing for whir sumchecks
- improve serialization / deserialization of field elements, improve fiat shamir
- https://eprint.iacr.org/2024/108.pdf section 3
- We can probably send less data in the first AIR sumcheck, with univariate skip, and the "current row" / "next row" reductions
- eval_mixed_tensor in cuda
- A_pol in cuda (in ring siwtch)
- matrix_up_folded / matrix_down_folded in cuda
- AIR inner sumcheck can bee accelerated (some factors have not all the variables + it's sparse)
- Neg in ArithmeticCircuitComposed

- Add ZK?