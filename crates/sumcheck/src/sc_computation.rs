use p3_air::Air;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use utils::{
    ConstraintFolder, ConstraintFolderPackedBase, ConstraintFolderPackedExtension, EFPacking, PF,
    PFPacking,
};

pub trait SumcheckComputation<NF, EF>: Sync {
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF;
}

impl<NF, EF, A> SumcheckComputation<NF, EF> for A
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF> + ExtensionField<PF<EF>>,
    A: for<'a> Air<ConstraintFolder<'a, NF, EF>>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolder {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers,
            accumulator: EF::ZERO,
            constraint_index: 0,
        };
        self.eval(&mut folder);
        folder.accumulator
    }
}

pub trait SumcheckComputationPacked<EF>: Sync
where
    EF: Field + ExtensionField<PF<EF>>,
{
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;

    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF>;
}

impl<EF: Field, A> SumcheckComputationPacked<EF> for A
where
    EF: ExtensionField<PF<EF>>,
    A: for<'a> Air<ConstraintFolderPackedBase<'a, EF>>
        + for<'a> Air<ConstraintFolderPackedExtension<'a, EF>>,
{
    fn eval_packed_base(&self, point: &[PFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolderPackedBase {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers: alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        self.eval(&mut folder);

        folder.accumulator
    }

    fn eval_packed_extension(&self, point: &[EFPacking<EF>], alpha_powers: &[EF]) -> EFPacking<EF> {
        if self.structured() {
            assert_eq!(point.len(), A::width(self) * 2);
        } else {
            assert_eq!(point.len(), A::width(self));
        }
        let mut folder = ConstraintFolderPackedExtension {
            main: RowMajorMatrixView::new(point, A::width(self)),
            alpha_powers: alpha_powers,
            accumulator: Default::default(),
            constraint_index: 0,
        };
        self.eval(&mut folder);

        folder.accumulator
    }
}

pub struct ProductComputation;

impl<EF: Field> SumcheckComputation<EF, EF> for ProductComputation {
    fn eval(&self, point: &[EF], _: &[EF]) -> EF {
        unsafe { *point.get_unchecked(0) * *point.get_unchecked(1) }
    }
}

impl<EF: Field + ExtensionField<PF<EF>>> SumcheckComputationPacked<EF> for ProductComputation {
    fn eval_packed_base(&self, _: &[PFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        unreachable!()
    }
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &[EF]) -> EFPacking<EF> {
        unsafe { *point.get_unchecked(0) * *point.get_unchecked(1) }
    }
}
