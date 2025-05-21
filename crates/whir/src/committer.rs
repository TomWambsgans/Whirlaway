use super::parameters::WhirConfig;
use algebra::pols::{CoefficientList, Multilinear};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use rand::distr::{Distribution, StandardUniform};
use tracing::instrument;

pub struct Witness<F: Field, EF: ExtensionField<F>> {
    pub(crate) polynomial: CoefficientList<F>,
    pub lagrange_polynomial: Multilinear<F>,
    pub(crate) merkle_tree: MerkleTree<F>,
    pub(crate) merkle_leaves: Vec<F>,
    pub(crate) ood_points: Vec<Vec<EF>>,
    pub(crate) ood_answers: Vec<EF>,
}

impl<F: Field + TwoAdicField + Ord, EF: ExtensionField<F>> WhirConfig<F, EF>
where
    F::PrimeSubfield: TwoAdicField,
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
{
    #[instrument(name = "whir: commit", skip_all)]
    pub fn commit(
        &self,
        lagrange_polynomial: Multilinear<F>,
        fs_prover: &mut FsProver,
    ) -> Witness<F, EF>
    where
        StandardUniform: Distribution<EF>,
    {
        let _span = tracing::info_span!("lagrange -> monomial convertion").entered();
        let polynomial = lagrange_polynomial.clone().to_monomial_basis();

        std::mem::drop(_span);

        let expansion = 1 << self.starting_log_inv_rate;

        let folded_evals = polynomial
            .expand_from_coeff_and_restructure(expansion, self.folding_factor.at_round(0));

        // Group folds together as a leaf.
        let fold_size = 1 << self.folding_factor.at_round(0);

        let merkle_tree = MerkleTree::new(&folded_evals, fold_size);

        let root = merkle_tree.root();
        fs_prover.add_bytes(&root.0);

        let (mut ood_points, mut ood_answers) = (Vec::new(), Vec::new());
        if self.committment_ood_samples > 0 {
            ood_points = (0..self.committment_ood_samples)
                .map(|_| fs_prover.challenge_scalars::<EF>(self.num_variables)) // OOD point in EF because the coeffs are in F, to ensure a result in EF
                .collect::<Vec<_>>();
            ood_answers = ood_points
                .iter()
                .map(|ood_point| lagrange_polynomial.evaluate_in_large_field(ood_point))
                .collect::<Vec<_>>();

            fs_prover.add_scalars(&ood_answers);
        }

        Witness {
            polynomial,
            lagrange_polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        }
    }
}
