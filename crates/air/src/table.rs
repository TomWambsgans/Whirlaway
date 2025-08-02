use p3_air::Air;
use p3_field::{ExtensionField, Field, TwoAdicField};

use p3_uni_stark::{SymbolicAirBuilder, get_symbolic_constraints};
use utils::{PF, log2_up, univariate_selectors};
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
        constraint_degree: usize,
    ) -> Self {
        let symbolic_constraints = get_symbolic_constraints(&air, 0, 0);
        let n_constraints = symbolic_constraints.len();

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
}
