use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{build_prover_state, build_verifier_state, padd_with_zero_to_next_power_of_two};
use whir_p3::poly::evals::EvaluationsList;

use crate::{table::AirTable, witness::AirWitness};

const N_PREPROCESSED_COLUMNS: usize = 3;
const N_COLUMNS: usize = 24;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

struct ExampleStructuredAir;

impl<F> BaseAir<F> for ExampleStructuredAir {
    fn width(&self) -> usize {
        N_COLUMNS
    }
    fn structured(&self) -> bool {
        true
    }
}

impl<AB: AirBuilder> Air<AB> for ExampleStructuredAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let up = main.row_slice(0).expect("The matrix is empty?");
        let up: &[AB::Var] = (*up).borrow();
        assert_eq!(up.len(), N_COLUMNS);
        let down = main.row_slice(1).expect("The matrix is empty?");
        let down: &[AB::Var] = (*down).borrow();
        assert_eq!(down.len(), N_COLUMNS);

        for j in N_PREPROCESSED_COLUMNS..N_COLUMNS {
            builder.assert_eq(
                down[j].clone(),
                up[j].clone()
                    + AB::I::from_usize(j)
                    + (0..N_PREPROCESSED_COLUMNS)
                        .map(|k| AB::Expr::from(down[k].clone()))
                        .product::<AB::Expr>(),
            );
        }
    }
}

struct ExampleUnstructuredAir;

impl<F> BaseAir<F> for ExampleUnstructuredAir {
    fn width(&self) -> usize {
        N_COLUMNS
    }
    fn structured(&self) -> bool {
        false
    }
}

impl<AB: AirBuilder> Air<AB> for ExampleUnstructuredAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let up = main.row_slice(0).expect("The matrix is empty?");
        let up: &[AB::Var] = (*up).borrow();
        assert_eq!(up.len(), N_COLUMNS);

        for j in N_PREPROCESSED_COLUMNS..N_COLUMNS {
            builder.assert_eq(
                up[j].clone(),
                (0..N_PREPROCESSED_COLUMNS)
                    .map(|k| AB::Expr::from(up[k].clone()))
                    .product::<AB::Expr>()
                    + AB::I::from_usize(j),
            );
        }
    }
}

fn generate_structured_trace(log_length: usize) -> Vec<Vec<F>> {
    let n_rows = 1 << log_length;
    let mut trace = vec![];
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..N_PREPROCESSED_COLUMNS {
        trace.push((0..n_rows).map(|_| rng.random()).collect::<Vec<F>>());
    }
    let mut witness_cols = vec![vec![F::ZERO]; N_COLUMNS - N_PREPROCESSED_COLUMNS];
    for i in 1..n_rows {
        for j in 0..N_COLUMNS - N_PREPROCESSED_COLUMNS {
            let witness_cols_j_i_min_1 = witness_cols[j][i - 1];
            witness_cols[j].push(
                witness_cols_j_i_min_1
                    + F::from_usize(j + N_PREPROCESSED_COLUMNS)
                    + (0..N_PREPROCESSED_COLUMNS)
                        .map(|k| trace[k][i])
                        .product::<F>(),
            );
        }
    }
    trace.extend(witness_cols);
    trace
}

fn generate_unstructured_trace(log_length: usize) -> Vec<Vec<F>> {
    let n_rows = 1 << log_length;
    let mut trace = vec![];
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..N_PREPROCESSED_COLUMNS {
        trace.push((0..n_rows).map(|_| rng.random()).collect::<Vec<F>>());
    }
    let mut witness_cols = vec![vec![]; N_COLUMNS - N_PREPROCESSED_COLUMNS];
    for i in 0..n_rows {
        for j in 0..N_COLUMNS - N_PREPROCESSED_COLUMNS {
            witness_cols[j].push(
                F::from_usize(j + N_PREPROCESSED_COLUMNS)
                    + (0..N_PREPROCESSED_COLUMNS)
                        .map(|k| trace[k][i])
                        .product::<F>(),
            );
        }
    }
    trace.extend(witness_cols);
    trace
}

#[test]
fn test_structured_air() {
    let log_n_rows = 12;
    let mut prover_state = build_prover_state::<EF>();

    let columns = generate_structured_trace(log_n_rows);
    let column_groups = vec![0..N_PREPROCESSED_COLUMNS, N_PREPROCESSED_COLUMNS..N_COLUMNS];
    let witness = AirWitness::new(&columns, &column_groups);

    let table = AirTable::<EF, _>::new(ExampleStructuredAir, 3);
    table.check_trace_validity(&witness).unwrap();
    let _evaluations_remaining_to_prove = table.prove(&mut prover_state, witness);
    let mut verifier_state = build_verifier_state(&prover_state);
    let evaluations_remaining_to_verify = table
        .verify(&mut verifier_state, log_n_rows, &column_groups)
        .unwrap();
    assert_eq!(
        padd_with_zero_to_next_power_of_two(&columns[..N_PREPROCESSED_COLUMNS].concat())
            .evaluate(&evaluations_remaining_to_verify[0].point),
        evaluations_remaining_to_verify[0].value
    );
    assert_eq!(
        padd_with_zero_to_next_power_of_two(&columns[N_PREPROCESSED_COLUMNS..N_COLUMNS].concat())
            .evaluate(&evaluations_remaining_to_verify[1].point),
        evaluations_remaining_to_verify[1].value
    );
}

#[test]
fn test_unstructured_air() {
    let log_n_rows = 12;
    let mut prover_state = build_prover_state::<EF>();

    let columns = generate_unstructured_trace(log_n_rows);
    let column_groups = vec![0..N_PREPROCESSED_COLUMNS, N_PREPROCESSED_COLUMNS..N_COLUMNS];
    let witness = AirWitness::new(&columns, &column_groups);

    let table = AirTable::<EF, _>::new(ExampleUnstructuredAir, 4);
    table.check_trace_validity(&witness).unwrap();
    let _evaluations_remaining_to_prove = table.prove(&mut prover_state, witness);
    let mut verifier_state = build_verifier_state(&prover_state);
    let evaluations_remaining_to_verify = table
        .verify(&mut verifier_state, log_n_rows, &column_groups)
        .unwrap();
    assert_eq!(
        padd_with_zero_to_next_power_of_two(&columns[..N_PREPROCESSED_COLUMNS].concat())
            .evaluate(&evaluations_remaining_to_verify[0].point),
        evaluations_remaining_to_verify[0].value
    );
    assert_eq!(
        padd_with_zero_to_next_power_of_two(&columns[N_PREPROCESSED_COLUMNS..N_COLUMNS].concat())
            .evaluate(&evaluations_remaining_to_verify[1].point),
        evaluations_remaining_to_verify[1].value
    );
}
