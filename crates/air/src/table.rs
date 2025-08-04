use p3_air::Air;
use p3_field::{ExtensionField, Field, TwoAdicField};

use p3_matrix::dense::RowMajorMatrixView;
use p3_uni_stark::{SymbolicAirBuilder, get_symbolic_constraints};
use rand::distr::{Distribution, StandardUniform};
use utils::{ConstraintChecker, PF, log2_up, univariate_selectors};
use whir_p3::poly::dense::WhirDensePolynomial;

pub struct AirTable<EF: Field, A> {
    pub air: A,
    pub log_length: usize,
    pub preprocessed_columns: Vec<Vec<PF<EF>>>,
    pub n_constraints: usize,
    pub constraint_degree: usize,
    pub univariate_skips: usize,
    pub univariate_selectors: Vec<WhirDensePolynomial<PF<EF>>>,

    _phantom: std::marker::PhantomData<EF>,
}

impl<EF, A: Air<SymbolicAirBuilder<PF<EF>>>> AirTable<EF, A>
where
    EF: ExtensionField<PF<EF>> + TwoAdicField,
    PF<EF>: TwoAdicField,
{
    pub fn new(
        air: A,
        log_length: usize,
        univariate_skips: usize,
        preprocessed_columns: Vec<Vec<PF<EF>>>,
    ) -> Self {
        let symbolic_constraints = get_symbolic_constraints(&air, 0, 0);
        let n_constraints = symbolic_constraints.len();
        let constraint_degree =
            Iterator::max(symbolic_constraints.iter().map(|c| c.degree_multiple())).unwrap();
        Self {
            air,
            log_length,
            preprocessed_columns,
            n_constraints,
            constraint_degree,
            univariate_skips,
            univariate_selectors: univariate_selectors(univariate_skips),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn n_columns(&self) -> usize {
        self.air.width()
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn n_witness_columns(&self) -> usize {
        self.n_columns() - self.preprocessed_columns.len()
    }

    /// rounded up
    pub fn log_n_witness_columns(&self) -> usize {
        log2_up(self.n_witness_columns())
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn n_preprocessed_columns(&self) -> usize {
        self.preprocessed_columns.len()
    }

    pub fn check_trace_validity(&self, witness: &[Vec<PF<EF>>]) -> Result<(), String>
    where
        A: for<'a> Air<ConstraintChecker<'a, PF<EF>>>,
        StandardUniform: Distribution<EF>,
    {
        let mut trace = self.preprocessed_columns.clone();
        trace.extend_from_slice(witness);
        if trace.len() != self.n_columns() {
            return Err(format!(
                "Trace has {} columns, expected {}",
                trace.len(),
                self.n_columns()
            ));
        }
        for i in 0..self.n_columns() {
            if trace[i].len() != (1 << self.log_length) {
                return Err(format!(
                    "Column {} has {} rows, expected {}",
                    i,
                    trace[i].len(),
                    1 << self.log_length
                ));
            }
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
            for row in 0..(1 << self.log_length) - 1 {
                let up = (0..self.n_columns())
                    .map(|j| trace[j][row])
                    .collect::<Vec<_>>();
                let down = (0..self.n_columns())
                    .map(|j| trace[j][row + 1])
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
            for row in 0..1 << self.log_length {
                let up = (0..self.n_columns())
                    .map(|j| trace[j][row])
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
