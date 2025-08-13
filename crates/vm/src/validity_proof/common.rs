use p3_air::BaseAir;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_ceil_usize;
use rayon::prelude::*;
use std::ops::Range;
use utils::{
    Evaluation, Poseidon16Air, Poseidon24Air, from_end, padd_with_zero_to_next_power_of_two,
    remove_end,
};
use whir_p3::{
    fiat_shamir::errors::ProofError,
    poly::{
        evals::{EvaluationsList, fold_multilinear},
        multilinear::MultilinearPoint,
    },
};

use crate::{EF, N_INSTRUCTION_COLUMNS_IN_AIR, bytecode::bytecode::Bytecode};

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

pub fn poseidon_lookup_index_statements(
    poseidon_index_evals: &[EF],
    n_poseidons_16: usize,
    n_poseidons_24: usize,
    poseidon_logup_star_statements_indexes_point: &MultilinearPoint<EF>,
) -> Result<(Vec<Evaluation<EF>>, Vec<Evaluation<EF>>), ProofError> {
    let log_n_p16 = log2_ceil_usize(n_poseidons_16);
    let log_n_p24 = log2_ceil_usize(n_poseidons_24);
    let correcting_factor = from_end(
        poseidon_logup_star_statements_indexes_point,
        log_n_p16.abs_diff(log_n_p24),
    )
    .iter()
    .map(|&x| EF::ONE - x)
    .product::<EF>();
    let (correcting_factor_p16, correcting_factor_p24) = if n_poseidons_16 > n_poseidons_24 {
        (EF::ONE, correcting_factor)
    } else {
        (correcting_factor, EF::ONE)
    };
    let mut idx_point_right_p16 = poseidon_logup_star_statements_indexes_point[3..].to_vec();
    let mut idx_point_right_p24 = remove_end(
        &poseidon_logup_star_statements_indexes_point[3..],
        log_n_p16.abs_diff(log_n_p24),
    )
    .to_vec();
    if n_poseidons_16 < n_poseidons_24 {
        std::mem::swap(&mut idx_point_right_p16, &mut idx_point_right_p24);
    }
    let p16_indexes_statements = vec![
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ZERO, EF::ZERO], idx_point_right_p16.clone()].concat(),
            ),
            value: poseidon_index_evals[0] / correcting_factor_p16,
        },
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ZERO, EF::ONE], idx_point_right_p16.clone()].concat(),
            ),
            value: poseidon_index_evals[1] / correcting_factor_p16,
        },
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ONE, EF::ZERO], idx_point_right_p16.clone()].concat(),
            ),
            value: poseidon_index_evals[2] / correcting_factor_p16,
        },
    ];

    let p24_indexes_statements = vec![
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ZERO, EF::ZERO], idx_point_right_p24.clone()].concat(),
            ),
            value: poseidon_index_evals[4] / correcting_factor_p24,
        },
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ZERO, EF::ONE], idx_point_right_p24.clone()].concat(),
            ),
            value: poseidon_index_evals[6] / correcting_factor_p24,
        },
        Evaluation {
            point: MultilinearPoint(
                [vec![EF::ONE, EF::ZERO], idx_point_right_p24.clone()].concat(),
            ),
            value: poseidon_index_evals[7] / correcting_factor_p24,
        },
    ];

    if poseidon_index_evals[3] != poseidon_index_evals[2] + correcting_factor_p16 {
        return Err(ProofError::InvalidProof);
    }
    if poseidon_index_evals[5] != poseidon_index_evals[4] + correcting_factor_p24 {
        return Err(ProofError::InvalidProof);
    }
    Ok((p16_indexes_statements, p24_indexes_statements))
}

pub fn fold_bytecode(bytecode: &Bytecode, folding_challenges: &MultilinearPoint<EF>) -> Vec<EF> {
    let encoded_bytecode = padd_with_zero_to_next_power_of_two(
        &bytecode
            .instructions
            .par_iter()
            .flat_map(|i| padd_with_zero_to_next_power_of_two(&i.field_representation()))
            .collect::<Vec<_>>(),
    );
    fold_multilinear(&encoded_bytecode, &folding_challenges)
}

pub fn intitial_and_final_pc_conditions(
    bytecode: &Bytecode,
    log_n_cycles: usize,
) -> (Evaluation<EF>, Evaluation<EF>) {
    let initial_pc_statement = Evaluation {
        point: MultilinearPoint(EF::zero_vec(log_n_cycles)),
        value: EF::ZERO,
    };
    let final_pc_statement = Evaluation {
        point: MultilinearPoint(vec![EF::ONE; log_n_cycles]),
        value: EF::from_usize(bytecode.ending_pc),
    };
    (initial_pc_statement, final_pc_statement)
}
