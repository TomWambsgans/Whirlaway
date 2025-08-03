use std::{borrow::Borrow, marker::PhantomData};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{build_merkle_compress, build_merkle_hash, build_prover_state, build_verifier_state};
use whir_p3::{
    dft::EvalsDft,
    whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder},
};

use crate::table::AirTable;

const N_PREPROCESSED_COLUMNS: usize = 3;
const N_COLUMNS: usize = 24;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

struct ExampleAir;

impl<F> BaseAir<F> for ExampleAir {
    fn width(&self) -> usize {
        N_COLUMNS
    }
    fn structured(&self) -> bool {
        true
    }
}

impl<AB: AirBuilder> Air<AB> for ExampleAir {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let up = main.row_slice(0).expect("The matrix is empty?");
        let up = (*up).borrow();
        assert_eq!(up.len(), N_COLUMNS);
        let down = main.row_slice(1).expect("The matrix is empty?");
        let down = (*down).borrow();
        assert_eq!(down.len(), N_COLUMNS);

        for j in N_PREPROCESSED_COLUMNS..N_COLUMNS {
            builder.assert_eq(
                down[j].clone(),
                up[j].clone()
                    + AB::I::from(AB::F::from_usize(j))
                    + (0..N_PREPROCESSED_COLUMNS)
                        .map(|k| AB::Expr::from(down[k].clone()))
                        .product::<AB::Expr>(),
            );
        }
    }
}

fn generate_trace(log_length: usize) -> Vec<Vec<F>> {
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
                    + (0..N_PREPROCESSED_COLUMNS).map(|k| trace[k][i]).product::<F>(),
            );
        }
    }
    trace.extend(witness_cols);
    trace
}

#[test]
fn test_example_air() {
    let log_n_rows = 12;
    let mut prover_state = build_prover_state::<EF>();

    let pcs = WhirConfigBuilder {
        folding_factor: FoldingFactor::Constant(4),
        soundness_type: SecurityAssumption::CapacityBound,
        merkle_hash: build_merkle_hash(),
        merkle_compress: build_merkle_compress(),
        pow_bits: 10,
        max_num_variables_to_send_coeffs: 6,
        rs_domain_initial_reduction_factor: 2,
        security_level: 100,
        starting_log_inv_rate: 1,
        base_field: PhantomData::<F>,
        extension_field: PhantomData::<EF>,
    };
    let dft = EvalsDft::default();

    let mut witness = generate_trace(log_n_rows);
    let preprocessed_columns = witness.drain(..N_PREPROCESSED_COLUMNS).collect::<Vec<_>>();

    let table = AirTable::<EF, _>::new(ExampleAir, log_n_rows, 3, preprocessed_columns);
    table.check_trace_validity(&witness).unwrap();
    table.prove(&mut prover_state, witness, &pcs, &dft);

    let mut verifier_state = build_verifier_state(&prover_state);

    table.verify(&mut verifier_state, &pcs).unwrap();
}
