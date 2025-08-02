use std::{borrow::Borrow, marker::PhantomData};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{build_merkle_compress, build_merkle_hash, build_prover_state, build_verifier_state};
use whir_p3::{
    dft::EvalsDft,
    whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder},
};

use crate::{AirSettings, table::AirTable};

const N_PREPROCESSED_COLUMNS: usize = 7;
const N_WITNESS_COLUMNS: usize = 24;
const TOTAL_COLUMNS: usize = N_PREPROCESSED_COLUMNS + N_WITNESS_COLUMNS;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

struct ExampleAir;

impl<F> BaseAir<F> for ExampleAir {
    fn width(&self) -> usize {
        N_PREPROCESSED_COLUMNS + N_WITNESS_COLUMNS
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
        let down = main.row_slice(1).expect("The matrix is empty?");
        let down = (*down).borrow();
        assert_eq!(up.len(), TOTAL_COLUMNS);
        assert_eq!(down.len(), TOTAL_COLUMNS);

        for i in N_PREPROCESSED_COLUMNS..TOTAL_COLUMNS {
            builder.assert_eq(
                down[i].clone(),
                up[(i * 2 + 3) % TOTAL_COLUMNS].clone() * up[(i * 6 + 1) % TOTAL_COLUMNS].clone()
                    + down[(i * 9 + 7) % N_PREPROCESSED_COLUMNS].clone(),
            );
        }
    }
}

fn generate_trace(log_length: usize) -> Vec<Vec<F>> {
    let n_rows = 1 << log_length;
    let mut rng = StdRng::seed_from_u64(0);
    let mut trace = vec![];
    for _ in 0..N_PREPROCESSED_COLUMNS {
        trace.push((0..n_rows).map(|_| rng.random()).collect());
    }
    for _ in N_PREPROCESSED_COLUMNS..N_WITNESS_COLUMNS {
        trace.push(F::zero_vec(n_rows));
    }
    for i in 1..n_rows {
        for j in N_PREPROCESSED_COLUMNS..TOTAL_COLUMNS {
            trace[j][i] = trace[(j * 2 + 3) % TOTAL_COLUMNS][i - 1]
                * trace[(j * 6 + 1) % TOTAL_COLUMNS][i - 1]
                + trace[(j * 9 + 7) % N_PREPROCESSED_COLUMNS][i - 1]
        }
    }
    trace
}

#[test]
fn test_example_air() {
    let log_n_rows = 10;
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

    let mut witness = generate_trace(log_n_rows);
    let preprocessed_columns = witness.drain(..N_PREPROCESSED_COLUMNS).collect::<Vec<_>>();

    let table = AirTable::<EF, _>::new(ExampleAir, true, log_n_rows, 3, preprocessed_columns, 2);
    let dft = EvalsDft::default();
    let settings = AirSettings {
        univariate_skips: 3,
    };
    table.prove(&settings, &mut prover_state, witness, &pcs, &dft);

    let mut verifier_state = build_verifier_state(&prover_state);

    table
        .verify(&settings, &mut verifier_state, log_n_rows, &pcs)
        .unwrap();
}
