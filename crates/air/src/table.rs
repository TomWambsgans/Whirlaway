use arithmetic_circuit::CircuitComputation;
use p3_field::{ExtensionField, Field, PrimeField, TwoAdicField};
use tracing::instrument;

use algebra::pols::{Multilinear, UnivariatePolynomial};
use utils::{HypercubePoint, log2_up};
use whir_p3::{
    fiat_shamir::pow::blake3::Blake3PoW,
    parameters::{MultivariateParameters, ProtocolParameters},
    whir::parameters::WhirConfig,
};

use crate::{AirSettings, ByteHash, FieldHash, MyCompress, WHIR_POW_BITS};

pub struct AirTable<F: Field> {
    pub log_length: usize,
    pub n_columns: usize,
    pub constraints: Vec<CircuitComputation<F>>, // n_vars = 2 * n_columns. First half = columns of row i, second half = columns of row i + 1
    pub preprocessed_columns: Vec<Multilinear<F>>, // TODO 'sparse' preprocessed columns (with non zero values at cylic shifts)
    // below are the data which is common to all proofs / verifications
    pub(crate) univariate_selectors: Vec<UnivariatePolynomial<F>>,
    pub(crate) lde_matrix_up: CircuitComputation<F>,
    pub(crate) lde_matrix_down: CircuitComputation<F>,
}

impl<F: PrimeField + TwoAdicField> AirTable<F> {
    pub fn n_witness_columns(&self) -> usize {
        self.n_columns - self.preprocessed_columns.len()
    }

    pub fn log_n_witness_columns(&self) -> usize {
        // rounded up
        log2_up(self.n_witness_columns())
    }

    pub fn n_preprocessed_columns(&self) -> usize {
        self.preprocessed_columns.len()
    }

    #[instrument(name = "check_validity", skip_all)]
    pub fn check_validity(&self, witness: &[Multilinear<F>]) {
        let log_length = witness[0].n_vars;
        assert_eq!(self.n_witness_columns(), witness.len());
        assert!(witness.iter().all(|w| w.n_vars == log_length));

        for constraint in &self.constraints {
            for (up, down) in
                HypercubePoint::iter(log_length).zip(HypercubePoint::iter(log_length).skip(1))
            {
                let mut point = self
                    .preprocessed_columns
                    .iter()
                    .chain(witness)
                    .map(|col| col.eval_hypercube(&up))
                    .collect::<Vec<_>>();
                point.extend(
                    self.preprocessed_columns
                        .iter()
                        .chain(witness)
                        .map(|col| col.eval_hypercube(&down))
                        .collect::<Vec<_>>(),
                );
                assert!(
                    constraint.eval(&point).is_zero(),
                    "Constraint is not satisfied",
                );
            }
        }
    }

    pub fn constraint_degree(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| c.composition_degree)
            .max_by_key(|d| *d)
            .unwrap()
    }

    pub fn build_whir_params<EF: ExtensionField<F> + TwoAdicField>(
        &self,
        settings: &AirSettings,
    ) -> WhirConfig<EF, F, FieldHash, MyCompress, Blake3PoW> {
        let num_variables = self.log_length + self.log_n_witness_columns();
        let mv_params = MultivariateParameters::<EF>::new(num_variables);

        let byte_hash = ByteHash {};
        let merkle_hash = FieldHash::new(byte_hash);
        let merkle_compress = MyCompress::new(byte_hash);

        let whir_params = ProtocolParameters::<_, _> {
            initial_statement: true,
            security_level: settings.security_bits,
            pow_bits: WHIR_POW_BITS,
            folding_factor: settings.whir_folding_factor,
            merkle_hash,
            merkle_compress,
            soundness_type: settings.whir_soudness_type,
            starting_log_inv_rate: settings.whir_log_inv_rate,
        };

        WhirConfig::<EF, F, _, _, _>::new(mv_params, whir_params)
    }
}
