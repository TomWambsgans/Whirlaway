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
- increase density of multi commitments -> we can almost reduce by 2x commitment costs
- avoid field embedding in the initial sumcheck of logup*, when table / values are in base field
- opti logup* GKR when the indexes are not a power of 2 (which is the case in the execution table)
- incremental merkle paths in whir-p3
- Experiment to increase degree, and reduce commitments, in Poseidon arithmetization
- Avoid embedding overhead on the flag, len, and index columns in the AIR table for dot products
- batch memory lookups (at least reduce to only 2 logup*)

## Not Perf

- Whir batching: handle the case where the second polynomial is too small compared to the first one
- bounddary condition on dot_product table: first flag = 1