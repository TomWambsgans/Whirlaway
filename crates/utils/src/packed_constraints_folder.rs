use crate::EFPacking;
use crate::PF;
use crate::PFPacking;
use p3_air::AirBuilder;
use p3_field::BasedVectorSpace;
use p3_field::PackedField;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

#[derive(Debug)]
pub struct ConstraintFolderPacked<'a, EF: Field + ExtensionField<PF<EF>>> {
    pub main: RowMajorMatrixView<'a, PFPacking<EF>>,
    pub alpha_powers: &'a [EF],
    pub decomposed_alpha_powers: &'a [Vec<PF<EF>>],
    pub accumulator: EFPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: Field + ExtensionField<PF<EF>>> AirBuilder for ConstraintFolderPacked<'a, EF> {
    type F = PF<EF>;
    type Expr = PFPacking<EF>;
    type Var = PFPacking<EF>;
    type M = RowMajorMatrixView<'a, PFPacking<EF>>;

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
        self.accumulator += Into::<EFPacking<EF>>::into(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.accumulator += EFPacking::<EF>::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PFPacking::<EF>::packed_linear_combination::<N>(alpha_powers, &expr_array)
        });
        self.constraint_index += N;
    }
}
