# TODO

## Short Term

- In whir::prove, the hypercube sum is already known, so no need to recompute it in sumcheck::prove
- Fix NTT (fn new_from_field(), where 24 is hardcoded for koala-bear)
- Avoid variable reversing for whir sumchecks
- Use prefix to distinguish between leaf and internal nodes in Merkle tree (cf. RFC-6962) for soundness
- improve serialization / deserialization of field elements, improve fiat shamir
- https://eprint.iacr.org/2024/108.pdf section 3

## Long Term

- speedup ff multiplication with karatsuba (requires to tweak plonky3)
- 10x the prover speed by improving the sumcheck (cf https://eprint.iacr.org/2024/108.pdf section 5)
- 10x the prover speed with software optimizations (SIMD, reduce all the cloning...)
- Increase code quality
- Add ZK?