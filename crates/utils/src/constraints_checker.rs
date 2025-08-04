use p3_air::AirBuilder;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrixView;

/*
Debug purpose
*/

#[derive(Debug)]
pub struct ConstraintChecker<'a, F> {
    pub main: RowMajorMatrixView<'a, F>,
    pub constraint_index: usize,
    pub errors: Vec<usize>
}

impl<'a, F: Field> AirBuilder for ConstraintChecker<'a, F> {
    type F = F;
    type I = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrixView<'a, F>;

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
        let x: F = x.into();
        if !x.is_zero() {
            self.errors.push(self.constraint_index);
        }
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _: [I; N]) {
        unreachable!()
    }
}
