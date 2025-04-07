use p3_field::{ExtensionField, Field};
use tracing::instrument;

use algebra::pols::{CircuitComputation, Evaluation, HypercubePoint, MultilinearPolynomial};

#[derive(Clone, Debug)]
pub struct BoundaryCondition<F: Field> {
    pub col: usize,
    pub row: usize,
    pub value: F,
}

impl<F: Field> BoundaryCondition<F> {
    pub(crate) fn encode<EF: ExtensionField<F>>(&self, log_length: usize) -> Evaluation<EF> {
        Evaluation {
            point: HypercubePoint {
                n_vars: log_length,
                val: self.row,
            }
            .to_vec::<EF>(), // TODO avoid embedding ?
            value: EF::from(self.value), // TODO avoid embedding ?
        }
    }
}

pub struct AirTable<F: Field> {
    pub n_columns: usize,
    pub constraints: Vec<CircuitComputation<F>>, // n_vars = 2 * n_columns. First half = columns of row i, second half = columns of row i + 1
    pub boundary_conditions: Vec<BoundaryCondition<F>>,
}

impl<F: Field> AirTable<F> {
    #[instrument(name = "check_validity", skip_all)]
    pub fn check_validity(&self, witness: &[MultilinearPolynomial<F>]) {
        let log_length = witness[0].n_vars;
        assert_eq!(self.n_columns, witness.len());
        assert!(witness.iter().all(|w| w.n_vars == log_length));

        for constraint in &self.constraints {
            for (up, down) in
                HypercubePoint::iter(log_length).zip(HypercubePoint::iter(log_length).skip(1))
            {
                let mut point = witness
                    .iter()
                    .map(|col| col.eval_hypercube(&up))
                    .collect::<Vec<_>>();
                point.extend(
                    witness
                        .iter()
                        .map(|col| col.eval_hypercube(&down))
                        .collect::<Vec<_>>(),
                );
                assert!(
                    constraint.eval(&point).is_zero(),
                    "Constraint is not satisfied",
                );
            }
        }
        for bound_condition in &self.boundary_conditions {
            assert_eq!(
                witness[bound_condition.col].eval_hypercube(&HypercubePoint {
                    n_vars: log_length,
                    val: bound_condition.row
                }),
                bound_condition.value
            );
        }
    }

    pub fn constraint_degree(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| c.composition_degree)
            .max()
            .unwrap()
    }
}
