use utils::{generate_trace_poseidon_16, generate_trace_poseidon_24};

use crate::{
    F,
    prove::{WitnessPoseidon16, WitnessPoseidon24},
};

pub fn build_poseidon_columns(
    poseidons_16: &[WitnessPoseidon16],
    poseidons_24: &[WitnessPoseidon24],
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let poseidon_16_data = poseidons_16.iter().map(|w| w.input).collect::<Vec<_>>();
    let poseidon_24_data = poseidons_24.iter().map(|w| w.input).collect::<Vec<_>>();
    let witness_matrix_poseidon_16 = generate_trace_poseidon_16(poseidon_16_data);
    let witness_matrix_poseidon_24 = generate_trace_poseidon_24(poseidon_24_data);
    let n_columns_poseidon_16 = witness_matrix_poseidon_16.width;
    let n_columns_poseidon_24 = witness_matrix_poseidon_24.width;

    let witness_matrix_poseidon_16_transposed = witness_matrix_poseidon_16.transpose();
    let witness_matrix_poseidon_24_transposed = witness_matrix_poseidon_24.transpose();

    assert_eq!(
        witness_matrix_poseidon_16_transposed.width,
        poseidons_16.len()
    );
    let witness_columns_poseidon_16 = (0..n_columns_poseidon_16)
        .map(|col| {
            witness_matrix_poseidon_16_transposed.values[col
                * witness_matrix_poseidon_16_transposed.width
                ..(col + 1) * witness_matrix_poseidon_16_transposed.width]
                .to_vec()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        witness_matrix_poseidon_24_transposed.width,
        poseidons_24.len()
    );
    let witness_columns_poseidon_24 = (0..n_columns_poseidon_24)
        .map(|col| {
            witness_matrix_poseidon_24_transposed.values[col
                * witness_matrix_poseidon_24_transposed.width
                ..(col + 1) * witness_matrix_poseidon_24_transposed.width]
                .to_vec()
        })
        .collect::<Vec<_>>();

    (witness_columns_poseidon_16, witness_columns_poseidon_24)
}
