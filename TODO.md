# TODO

## Short Term

- In whir::prove, the hypercube sum is already known, so no need to recompute it in sumcheck::prove
- Fix NTT (fn new_from_field(), where 24 is hardcoded for koala-bear)
- Avoid variable reversing for whir sumchecks
- Multilinear polynomial evaluation can be done in 2^n multiplications instead of 2^(n+1)
- Use prefix to distinguish between leaf and internal nodes in Merkle tree (cf. RFC-6962) for soundness
- improve serialization / deserialization of field elements, improve fiat shamir

## Long Term

- 10x the prover speed by reducing embedding overhead in sumchecks
- 10x the prover speed with software optimizations (SIMD, reduce all the cloning...)
- Increase code quality
- Add ZK?