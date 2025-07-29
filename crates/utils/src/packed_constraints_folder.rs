use p3_air::AirBuilder;
use p3_field::BasedVectorSpace;
use p3_field::PackedField;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

use crate::EFPackingOf;

#[derive(Debug)]
pub struct ConstraintFolderPacked<'a, F: Field, EF: ExtensionField<F>> {
    pub main: RowMajorMatrixView<'a, F::Packing>,
    pub alpha_powers: &'a [EF],
    pub decomposed_alpha_powers: &'a [Vec<F>],
    pub accumulator: EFPackingOf<EF, F>,
    pub constraint_index: usize,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for ConstraintFolderPacked<'a, F, EF> {
    type F = F;
    type Expr = F::Packing;
    type Var = F::Packing;
    type M = RowMajorMatrixView<'a, F::Packing>;

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

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator +=
            Into::<<EF as ExtensionField<F>>::ExtensionPacking>::into(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.accumulator +=
            <EF as ExtensionField<F>>::ExtensionPacking::from_basis_coefficients_fn(|i| {
                let alpha_powers = &self.decomposed_alpha_powers[i]
                    [self.constraint_index..(self.constraint_index + N)];
                F::Packing::packed_linear_combination::<N>(alpha_powers, &expr_array)
            });
        self.constraint_index += N;
    }
}
