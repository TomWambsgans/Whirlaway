use super::parameters::WhirConfig;
use algebra::pols::{CoefficientList, Multilinear};
use cuda_engine::{HostOrDeviceBuffer, cuda_sync};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use tracing::instrument;

pub struct Witness<F: Field> {
    pub(crate) polynomial: CoefficientList<F>,
    pub lagrange_polynomial: Multilinear<F>,
    pub(crate) merkle_tree: MerkleTree<F>,
    pub(crate) merkle_leaves: HostOrDeviceBuffer<F>,
    pub(crate) ood_points: Vec<Vec<F::PrimeSubfield>>,
    pub(crate) ood_answers: Vec<F>,
}

pub struct Committer<F, RCF>(WhirConfig<F, RCF>);

impl<F: Field + TwoAdicField + Ord, RCF: Field> Committer<F, RCF>
where
    F::PrimeSubfield: TwoAdicField,
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
{
    pub fn new(config: WhirConfig<F, RCF>) -> Self {
        Self(config)
    }

    #[instrument(name = "whir: commit", skip_all)]
    pub fn commit(
        &self,
        fs_prover: &mut FsProver,
        lagrange_polynomial: Multilinear<F>,
    ) -> Witness<F> {
        let _span = tracing::info_span!("lagrange -> monomial convertion").entered();
        let polynomial = lagrange_polynomial.to_monomial_basis();
        cuda_sync();
        std::mem::drop(_span);

        let expansion = 1 << self.0.starting_log_inv_rate;

        let folded_evals = polynomial
            .expand_from_coeff_and_restructure(expansion, self.0.folding_factor.at_round(0));

        // Group folds together as a leaf.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        let merkle_tree = MerkleTree::new(&folded_evals, fold_size);

        let root = merkle_tree.root();
        fs_prover.add_bytes(&root.0);

        let (mut ood_points, mut ood_answers) = (Vec::new(), Vec::new());
        if self.0.committment_ood_samples > 0 {
            ood_points = (0..self.0.committment_ood_samples)
                .map(|_| fs_prover.challenge_scalars::<F::PrimeSubfield>(self.0.num_variables))
                .collect::<Vec<_>>();
            ood_answers = ood_points
                .iter()
                .map(|ood_point| lagrange_polynomial.evaluate_in_small_field(ood_point))
                .collect::<Vec<_>>();
            cuda_sync();
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
