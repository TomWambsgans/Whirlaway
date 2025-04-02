use super::parameters::WhirConfig;
use crate::{
    poly_utils::{coeffs::CoefficientList, fold::restructure_evaluations},
    utils::{self, expand_from_coeff_maybe_with_cuda},
};
use algebra::field_utils::multilinear_point_from_univariate;
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;

use p3_field::TwoAdicField;
use tracing::instrument;

pub struct Witness<F: TwoAdicField> {
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: MerkleTree<F>,
    pub(crate) merkle_leaves: Vec<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
}

pub struct Committer<F: TwoAdicField>(WhirConfig<F>);

impl<F: TwoAdicField> Committer<F> {
    pub fn new(config: WhirConfig<F>) -> Self {
        Self(config)
    }

    #[instrument(name = "whir: commit", skip_all)]
    pub fn commit(
        &self,
        fs_prover: &mut FsProver,
        polynomial: CoefficientList<F>,
    ) -> Option<Witness<F>> {
        let base_domain = self.0.starting_domain.base_domain.as_ref().unwrap();
        let expansion = base_domain.size() / polynomial.num_coeffs();
        let evals = expand_from_coeff_maybe_with_cuda(polynomial.coeffs(), expansion, self.0.cuda);
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor.at_round(0));
        let folded_evals = restructure_evaluations(
            folded_evals,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor.at_round(0),
        );

        // Group folds together as a leaf.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        let merkle_tree = MerkleTree::new(&folded_evals, fold_size);

        let root = merkle_tree.root();

        fs_prover.add_bytes(&root);

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = Vec::with_capacity(self.0.committment_ood_samples);
        if self.0.committment_ood_samples > 0 {
            ood_points = fs_prover.challenge_scalars(self.0.committment_ood_samples);
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                polynomial.evaluate(&multilinear_point_from_univariate(
                    *ood_point,
                    self.0.mv_parameters.num_variables,
                ))
            }));
            fs_prover.add_scalars(&ood_answers);
        }

        Some(Witness {
            polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        })
    }
}
