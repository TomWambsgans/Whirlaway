# TODO

- In whir::prove, the hypercube sum is already known, so no need to recompute it in sumcheck::prove
- Avoid variable reversing for whir sumchecks
- improve serialization / deserialization of field elements, improve fiat shamir
- https://eprint.iacr.org/2024/108.pdf section 3
- We can probably send less data in the first AIR sumcheck, with univariate skip, and the "current row" / "next row" reductions

- Univariate skip (done in rust (cf git history)), but the goal is to have it in cuda

- Add ZK?