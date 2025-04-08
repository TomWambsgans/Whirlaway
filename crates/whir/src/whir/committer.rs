use crate::utils::expand_from_coeff_and_restructure;

use super::parameters::WhirConfig;
use algebra::{pols::CoefficientList, utils::multilinear_point_from_univariate};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;

use p3_field::{ExtensionField, Field, TwoAdicField};
use tracing::instrument;

pub struct Witness<EF: Field> {
    pub(crate) polynomial: CoefficientList<EF>,
    pub(crate) merkle_tree: MerkleTree<EF>,
    pub(crate) merkle_leaves: Vec<EF>,
    pub(crate) ood_points: Vec<EF>,
    pub(crate) ood_answers: Vec<EF>,
}

pub struct Committer<F: TwoAdicField, EF: ExtensionField<F>>(WhirConfig<F, EF>);

impl<F: TwoAdicField, EF: ExtensionField<F>> Committer<F, EF> {
    pub fn new(config: WhirConfig<F, EF>) -> Self {
        Self(config)
    }

    #[instrument(name = "whir: commit", skip_all)]
    pub fn commit(
        &self,
        fs_prover: &mut FsProver,
        polynomial: CoefficientList<EF>,
    ) -> Option<Witness<EF>> {
        let base_domain = self.0.starting_domain.base_domain.as_ref().unwrap();
        let expansion = base_domain.size() / polynomial.num_coeffs();

        let folded_evals = expand_from_coeff_and_restructure(
            polynomial.coeffs(),
            expansion,
            base_domain.group_gen_inv(),
            self.0.folding_factor.at_round(0),
            self.0.cuda,
        );

        // Group folds together as a leaf.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        let merkle_tree = MerkleTree::new(&folded_evals, fold_size);

        let root = merkle_tree.root();

        fs_prover.add_bytes(&root);

        let mut ood_points = vec![EF::ZERO; self.0.committment_ood_samples];
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
