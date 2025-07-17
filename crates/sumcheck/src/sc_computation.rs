use p3_air::Air;
use p3_field::PackedValue;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use utils::{ConstraintFolder, ConstraintFolderPacked};

pub trait SumcheckComputation<F, NF, EF>: Sync
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF;
}

impl<F, NF, EF, A> SumcheckComputation<F, NF, EF> for A
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    A: for<'a> Air<ConstraintFolder<'a, F, NF, EF>>,
{
    fn eval(&self, point: &[NF], alpha_powers: &[EF]) -> EF {
        assert_eq!(A::width(self) * 2, point.len());
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

pub trait SumcheckComputationPacked<F, EF>: Sync
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Currently we only support NF = F, TODO make it work for NF = EF
    fn eval_packed(
        &self,
        point: &[F::Packing],
        alpha_powers: &[EF],
        decomposed_alpha_powers: &[Vec<F>],
    ) -> impl Iterator<Item = EF> + Send + Sync;
}

impl<F, EF, A> SumcheckComputationPacked<F, EF> for A
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'a> Air<ConstraintFolderPacked<'a, F, EF>>,
{
    fn eval_packed(
        &self,
        point: &[F::Packing],
        alpha_powers: &[EF],
        decomposed_alpha_powers: &[Vec<F>],
    ) -> impl Iterator<Item = EF> {
        let mut folder = ConstraintFolderPacked {
            main: RowMajorMatrixView::new(point, point.len() / 2),
            alpha_powers: alpha_powers,
            decomposed_alpha_powers: decomposed_alpha_powers,
            accumulator: <EF as ExtensionField<F>>::ExtensionPacking::ZERO,
            constraint_index: 0,
        };
        self.eval(&mut folder);

        (0..F::Packing::WIDTH).map(move |idx_in_packing| {
            EF::from_basis_coefficients_fn(|coeff_idx| {
                BasedVectorSpace::<F::Packing>::as_basis_coefficients_slice(&folder.accumulator)
                    [coeff_idx]
                    .as_slice()[idx_in_packing]
            })
        })
    }
}
