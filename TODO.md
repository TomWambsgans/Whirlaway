# TODO

## Perf

- WHIR univariate skip
- Opti recursion bytecode
- inverse folding ordering in WHIR to enable Packing during sumcheck
- one can "move out" the variable of the eq(.) polynomials out of the sumcheck computation in WHIR (as done in the PIOP)
- Extension field: dim 5/6
- Structured AIR: often no all the columns use both up/down -> only handle the used ones to speed up the PIOP zerocheck
- use RowMAjorMatrix instead of Vec<Vec> for witness
- Fill Precompile tables during bytecode execution
- Use Univariate Skip to commit to tables with k.2^n rows (k small)
- increase density of multi commitments -> we can almost gain 2x for commitment costs of Poseidon16 + main table
- avoid field embedding in the initial sumcheck of logup*, when table / values are in base field
- incremental merkle paths in whir-p3

## Not Perf

- Whir batching: handle the case where the second polynomial is too small compared to the first one
- initial and final conditions on the execution table