use p3_air::BaseAir;
use p3_field::Field;
use p3_util::log2_ceil_usize;
use std::ops::Range;
use utils::{Evaluation, Poseidon16Air, Poseidon24Air, from_end};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

pub fn poseidon_16_column_groups(poseidon_16_air: &Poseidon16Air) -> Vec<Range<usize>> {
    vec![
        0..8,
        8..16,
        16..poseidon_16_air.width() - 16,
        poseidon_16_air.width() - 16..poseidon_16_air.width() - 8,
        poseidon_16_air.width() - 8..poseidon_16_air.width(),
    ]
}

pub fn poseidon_24_column_groups(poseidon_24_air: &Poseidon24Air) -> Vec<Range<usize>> {
    vec![
        0..8,
        8..16,
        16..24,
        24..poseidon_24_air.width() - 24,
        poseidon_24_air.width() - 24..poseidon_24_air.width() - 8, // TODO should we commit to this part ? Probably not, but careful here, we will not check evaluations for this part
        poseidon_24_air.width() - 8..poseidon_24_air.width(),
    ]
}

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
