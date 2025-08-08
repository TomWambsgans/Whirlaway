use p3_field::{ExtensionField, Field};

use p3_matrix::dense::RowMajorMatrixView;
use p3_uni_stark::get_symbolic_constraints;
use rand::distr::{Distribution, StandardUniform};
use utils::{ConstraintChecker, PF};

use crate::{MyAir, witness::AirWitness};

pub struct AirTable<EF: Field, A> {
    pub air: A,
    pub n_constraints: usize,
    pub univariate_skips: usize,

    _phantom: std::marker::PhantomData<EF>,
}

impl<EF: ExtensionField<PF<EF>>, A: MyAir<EF>> AirTable<EF, A> {
    pub fn new(air: A, univariate_skips: usize) -> Self {
        let symbolic_constraints = get_symbolic_constraints(&air, 0, 0);
        let n_constraints = symbolic_constraints.len();
        let constraint_degree =
            Iterator::max(symbolic_constraints.iter().map(|c| c.degree_multiple())).unwrap();
        assert_eq!(constraint_degree, air.degree());
        Self {
            air,
            n_constraints,
            univariate_skips,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn n_columns(&self) -> usize {
        self.air.width()
    }

    pub fn check_trace_validity(&self, witness: &AirWitness<PF<EF>>) -> Result<(), String>
    where
        A: MyAir<EF>,
        StandardUniform: Distribution<EF>,
    {
        if witness.n_columns() != self.n_columns() {
            return Err(format!("Invalid number of columns",));
        }
        let handle_errors = |row: usize, constraint_checker: &mut ConstraintChecker<PF<EF>>| {
            if constraint_checker.errors.len() > 0 {
                return Err(format!(
                    "Trace is not valid at row {}: contraints not respected: {}",
                    row,
                    constraint_checker
                        .errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            Ok(())
        };
        if self.air.structured() {
            for row in 0..witness.n_rows() - 1 {
                let up = (0..self.n_columns())
                    .map(|j| witness[j][row])
                    .collect::<Vec<_>>();
                let down = (0..self.n_columns())
                    .map(|j| witness[j][row + 1])
                    .collect::<Vec<_>>();
                let up_and_down = [up, down].concat();
                let mut constraints_checker = ConstraintChecker {
                    main: RowMajorMatrixView::new(&up_and_down, self.air.width()),
                    constraint_index: 0,
                    errors: Vec::new(),
                };
                self.air.eval(&mut constraints_checker);
                handle_errors(row, &mut constraints_checker)?;
            }
        } else {
            for row in 0..witness.n_rows() {
                let up = (0..self.n_columns())
                    .map(|j| witness[j][row])
                    .collect::<Vec<_>>();
                let mut constraints_checker = ConstraintChecker {
                    main: RowMajorMatrixView::new(&up, self.air.width()),
                    constraint_index: 0,
                    errors: Vec::new(),
                };
                self.air.eval(&mut constraints_checker);
                handle_errors(row, &mut constraints_checker)?;
            }
        }
        Ok(())
    }
}
