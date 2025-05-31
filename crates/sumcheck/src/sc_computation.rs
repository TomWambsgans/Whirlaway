use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use utils::ConstraintFolder;

pub trait SumcheckComputation<F, NF, EF>: Sync
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF;
}

impl<'a, F: Field, NF, EF, A> SumcheckComputation<F, NF, EF> for A
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
    A: Air<ConstraintFolder<'a, F, NF, EF>>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF {
        let point: &'a [NF] = unsafe { std::mem::transmute::<&[NF], &'a [NF]>(point) };
        let alpha_powers: &'a [EF] =
            unsafe { std::mem::transmute::<&[EF], &'a [EF]>(alpha_powers) };
        assert_eq!(<A as BaseAir<F>>::width(self) * 2, point.len());
        let mut folder = ConstraintFolder {
            main: RowMajorMatrixView::new(point, point.len() / 2),
            alpha_powers,
            accumulator: EF::ZERO,
            constraint_index: 0,
            _phantom: std::marker::PhantomData,
        };
        self.eval(&mut folder);
        folder.accumulator
    }
}
