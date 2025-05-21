use crate::CircuitComputation;

#[derive(Clone, Debug, Hash)]
pub struct SumcheckComputation<'a, F> {
    pub exprs: &'a [CircuitComputation<F>], // each one is multiplied by a 'batching scalar'. We assume the first batching scalar is always 1.
    pub n_multilinears: usize,              // including the eq_mle multiplier (if any)
    pub eq_mle_multiplier: bool,
}
