use p3_air::Air;
use p3_field::{ExtensionField, Field, TwoAdicField, cyclic_subgroup_known_order};

use p3_uni_stark::{SymbolicAirBuilder, get_symbolic_constraints};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
};
use sumcheck::SumcheckComputation;
use utils::{ConstraintFolder, PF, log2_up, univariate_selectors};
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
        A: for<'a> Air<ConstraintFolder<'a, EF, EF>>,
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
        if trace[0].len() != (1 << self.log_length) {
            return Err(format!(
                "Trace has {} rows, expected {}",
                trace[0].len(),
                1 << self.log_length
            ));
        }
        let alpha: EF = StdRng::seed_from_u64(0).random();
        if self.air.structured() {
            for i in 0..1 << self.log_length - 1 {
                let up = (0..self.n_columns())
                    .map(|j| EF::from(trace[j][i]))
                    .collect::<Vec<_>>();
                let down = (0..self.n_columns())
                    .map(|j| EF::from(trace[j][i + 1]))
                    .collect::<Vec<_>>();
                let up_and_down = [up, down].concat();
                if SumcheckComputation::eval(
                    &self.air,
                    &up_and_down,
                    &cyclic_subgroup_known_order(alpha, self.n_constraints).collect::<Vec<_>>(),
                ) != EF::ZERO
                {
                    return Err(format!("Trace is not valid at row {}", i));
                }
            }
        } else {
            for i in 0..1 << self.log_length - 1 {
                let up = (0..self.n_columns())
                    .map(|j| EF::from(trace[j][i]))
                    .collect::<Vec<_>>();
                if SumcheckComputation::eval(
                    &self.air,
                    &up,
                    &cyclic_subgroup_known_order(alpha, self.n_constraints).collect::<Vec<_>>(),
                ) != EF::ZERO
                {
                    return Err(format!("Trace is not valid at row {}", i));
                }
            }
        }
        Ok(())
    }
}
