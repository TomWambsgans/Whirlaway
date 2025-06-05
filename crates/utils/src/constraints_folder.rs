use p3_air::AirBuilder;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

#[derive(Debug)]
pub struct ConstraintFolder<'a, F, NF, EF>
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
{
    pub main: RowMajorMatrixView<'a, NF>,
    pub alpha_powers: &'a [EF],
    pub accumulator: EF,
    pub constraint_index: usize,
    pub _phantom: std::marker::PhantomData<F>,
}

impl<'a, F, NF, EF> AirBuilder for ConstraintFolder<'a, F, NF, EF>
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
{
    type F = F;
    type Expr = NF;
    type Var = NF;
    type M = RowMajorMatrixView<'a, NF>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: NF = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += alpha_power * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _: [I; N]) {
        unreachable!()
    }
}
