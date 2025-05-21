# TODO

- improve serialization / deserialization of field elements, improve fiat shamir
- We can probably send less data in the first AIR sumcheck, with univariate skip, and the "current row" / "next row" reductions
- AIR inner sumcheck can bee accelerated (some factors have not all the variables + it's sparse -> avoid "dummy variables")
- Neg in ArithmeticCircuitComposed
- sparse preprocessed columns
- MaybeUninit instead of allocating zeros when it's rewritten just after
- preprpcess twiddles

- Add ZK?
