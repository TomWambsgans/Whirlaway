use arithmetic_circuit::CircuitComputation;
use p3_field::{Field, PrimeField};
use tracing::instrument;

use algebra::pols::{MultilinearHost, UnivariatePolynomial};
use utils::{HypercubePoint, log2_up};

pub struct AirTable<F: Field> {
    pub log_length: usize,
    pub n_columns: usize,
    pub constraints: Vec<CircuitComputation<F>>, // n_vars = 2 * n_columns. First half = columns of row i, second half = columns of row i + 1
    pub preprocessed_columns: Vec<MultilinearHost<F>>, // TODO 'sparse' preprocessed columns (with non zero values at cylic shifts)
    // below are the data which is common to all proofs / verifications
    pub(crate) univariate_selectors: Vec<UnivariatePolynomial<F>>,
    pub(crate) lde_matrix_up: CircuitComputation<F>,
    pub(crate) lde_matrix_down: CircuitComputation<F>,
}

impl<F: PrimeField> AirTable<F> {
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
    pub fn check_validity(&self, witness: &[MultilinearHost<F>]) {
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
}
