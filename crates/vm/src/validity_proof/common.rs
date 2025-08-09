use p3_field::Field;
use p3_util::log2_ceil_usize;
use utils::{Evaluation, from_end};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

pub fn poseidon_lookup_value<EF: Field>(
    n_poseidons_16: usize,
    n_poseidons_24: usize,
    poseidon16_evals: &[Evaluation<EF>],
    poseidon24_evals: &[Evaluation<EF>],
    poseidon_lookup_batching_chalenges: &MultilinearPoint<EF>,
) -> EF {
    let (point, diff) = if n_poseidons_16 > n_poseidons_24 {
        (
            &poseidon16_evals[0].point,
            log2_ceil_usize(n_poseidons_16) - log2_ceil_usize(n_poseidons_24),
        )
    } else {
        (
            &poseidon24_evals[0].point,
            log2_ceil_usize(n_poseidons_24) - log2_ceil_usize(n_poseidons_16),
        )
    };
    let factor: EF = from_end(point, diff).iter().map(|&f| EF::ONE - f).product();
    let (s16, s24) = if n_poseidons_16 > n_poseidons_24 {
        (EF::ONE, factor)
    } else {
        (factor, EF::ONE)
    };
    [
        poseidon16_evals[0].value * s16,
        poseidon16_evals[1].value * s16,
        poseidon16_evals[3].value * s16,
        poseidon16_evals[4].value * s16,
        poseidon24_evals[0].value * s24,
        poseidon24_evals[1].value * s24,
        poseidon24_evals[2].value * s24,
        poseidon24_evals[5].value * s24,
    ]
    .evaluate(&poseidon_lookup_batching_chalenges)
}
