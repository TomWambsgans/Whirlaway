use super::parameters::WhirConfig;
use algebra::pols::{CoefficientList, Multilinear};
use cuda_engine::{HostOrDeviceBuffer, cuda_sync};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;

use p3_field::{ExtensionField, Field, TwoAdicField};
use tracing::instrument;
use utils::multilinear_point_from_univariate;

pub struct Witness<F: Field, EF: ExtensionField<F>> {
    pub(crate) polynomial: CoefficientList<F>,
    pub lagrange_polynomial: Multilinear<F>,
    pub(crate) merkle_tree: MerkleTree<F>,
    pub(crate) merkle_leaves: HostOrDeviceBuffer<F>,
    pub(crate) ood_points: Vec<EF>,
    pub(crate) ood_answers: Vec<EF>,
}

pub struct Committer<F: Field, EF: ExtensionField<F>>(WhirConfig<F, EF>);

impl<F: Field + TwoAdicField + Ord, EF: ExtensionField<F>> Committer<F, EF>
where
    F: TwoAdicField + Ord,
    F::PrimeSubfield: TwoAdicField,
{
    pub fn new(config: WhirConfig<F, EF>) -> Self {
        Self(config)
    }

    #[instrument(name = "whir: commit", skip_all)]
    pub fn commit(
        &self,
        fs_prover: &mut FsProver,
        lagrange_polynomial: Multilinear<F>,
    ) -> Option<Witness<F, EF>> {
        let polynomial = lagrange_polynomial.to_monomial_basis();

        let expansion = 1 << self.0.starting_log_inv_rate;

        let folded_evals = polynomial
            .expand_from_coeff_and_restructure(expansion, self.0.folding_factor.at_round(0));

        // Group folds together as a leaf.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        let merkle_tree = MerkleTree::new(&folded_evals, fold_size);

        let root = merkle_tree.root();
        fs_prover.add_bytes(&root.0);

        let mut ood_points = vec![EF::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = Vec::with_capacity(self.0.committment_ood_samples);
        if self.0.committment_ood_samples > 0 {
            ood_points = fs_prover.challenge_scalars(self.0.committment_ood_samples);
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                lagrange_polynomial.evaluate(&multilinear_point_from_univariate(
                    *ood_point,
                    self.0.num_variables,
                ))
            }));
            cuda_sync();
            fs_prover.add_scalars(&ood_answers);
        }

        Some(Witness {
            polynomial,
            lagrange_polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        })
    }
}
