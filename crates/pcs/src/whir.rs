use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use utils::{Evaluation, MyExtensionField, Statement};
use whir::{
    committer::Committer,
    parameters::{WhirConfig, WhirConfigBuilder},
    prover::Prover,
    verifier::{ParsedCommitment, Verifier, WhirError},
};

use whir::committer::Witness as WhirWitness;

use crate::PcsParams;

use super::{PCS, PcsWitness};

impl<'a, F: Field> PcsWitness<F> for WhirWitness<F> {
    fn pol(&self) -> &Multilinear<F> {
        &self.lagrange_polynomial
    }
}

impl PcsParams for WhirConfigBuilder {
    fn security_bits(&self) -> usize {
        self.security_level
    }
}

impl<F: Field + TwoAdicField + Ord, RCF: Field> PCS<F, F> for WhirConfig<F, RCF>
where
    F::PrimeSubfield: TwoAdicField,
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + MyExtensionField<RCF>,
    RCF: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
{
    type Witness = WhirWitness<F>;
    type ParsedCommitment = ParsedCommitment<F>;
    type VerifError = WhirError;
    type Params = WhirConfigBuilder;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        params.build(n_vars)
    }

    fn commit(&self, pol: Multilinear<F>, fs_prover: &mut FsProver) -> Self::Witness {
        Committer::new(self.clone()).commit(fs_prover, pol)
    }

    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Self::ParsedCommitment, Self::VerifError> {
        Verifier(self.clone()).parse_commitment(fs_verifier)
    }

    fn open(&self, witness: Self::Witness, eval: &Evaluation<F>, fs_prover: &mut FsProver) {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Prover(self.clone()).prove(fs_prover, statement, witness);
    }

    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        eval: &Evaluation<F>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), Self::VerifError> {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Verifier(self.clone()).verify(fs_verifier, parsed_commitment, statement)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::pols::{MultilinearDevice, MultilinearHost};
    use arithmetic_circuit::ArithmeticCircuit;
    use cuda_engine::{
        CudaFunctionInfo, SumcheckComputation, cuda_init, cuda_load_function,
        cuda_preprocess_many_sumcheck_computations, cuda_preprocess_twiddles, cuda_sync,
        memcpy_htod,
    };
    use p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, PrimeField32, extension::BinomialExtensionField,
    };
    use p3_koala_bear::KoalaBear;
    use tracing_forest::{ForestLayer, util::LevelFilter};
    use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
    use whir::parameters::{FoldingFactor, SoundnessType};

    #[test]
    fn test_whir_debug() {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();

        type F = BinomialExtensionField<KoalaBear, 8>;
        type RCF = BinomialExtensionField<KoalaBear, 4>;

        let num_vars = 15;
        let folding = FoldingFactor::Constant(4);
        let log_inv_rate = 3;
        let cuda = true;
        let params = WhirConfigBuilder::standard(
            SoundnessType::ProvableList,
            128,
            log_inv_rate,
            folding,
            cuda,
        );

        test_whir_pcs_helper::<F, RCF>(&params, num_vars);
    }

    #[test]
    fn bench_whir() {
        type F = BinomialExtensionField<KoalaBear, 8>;
        type EF = F;
        let log_inv_rate = 2;
        let folding = FoldingFactor::Constant(4);
        let cuda = true;
        let params = WhirConfigBuilder::standard(
            SoundnessType::ProvableList,
            128,
            log_inv_rate,
            folding,
            cuda,
        );
        for num_vars in 17..24 {
            println!("num_vars: {}", num_vars);
            test_whir_pcs_helper::<F, EF>(&params, num_vars);
        }
    }

    #[test]
    #[ignore]
    fn test_whir_pcs_long() {
        type F = BinomialExtensionField<KoalaBear, 8>;
        type RCF = BinomialExtensionField<KoalaBear, 4>;

        for cuda in [true, false] {
            for num_vars in 5..20 {
                for folding in [
                    FoldingFactor::Constant(3),
                    FoldingFactor::Constant(4),
                    FoldingFactor::ConstantFromSecondRound(5, 4),
                    FoldingFactor::ConstantFromSecondRound(3, 4),
                ] {
                    for log_inv_rate in [1, 2, 3] {
                        let params = WhirConfigBuilder::standard(
                            SoundnessType::ProvableList,
                            70,
                            log_inv_rate,
                            folding,
                            cuda,
                        );
                        test_whir_pcs_helper::<F, RCF>(&params, num_vars);
                    }
                }
            }
        }
    }

    fn test_whir_pcs_helper<F: Field + TwoAdicField + Ord, RCF: Field>(
        params: &WhirConfigBuilder,
        num_vars: usize,
    ) where
        F::PrimeSubfield: TwoAdicField,
        F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + MyExtensionField<RCF>,
        RCF: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
        <F as PrimeCharacteristicRing>::PrimeSubfield: PrimeField32,
    {
        let config = WhirConfig::<F, RCF>::new(num_vars, params);

        dbg!(&config);

        if config.cuda {
            cuda_init();
            let prod_sumcheck = SumcheckComputation::<KoalaBear> {
                exprs: &[(ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1))
                    .fix_computation(false)],
                n_multilinears: 2,
                eq_mle_multiplier: false,
            };
            cuda_load_function(CudaFunctionInfo::one_field::<F::PrimeSubfield>(
                "multilinear.cu",
                "eq_mle_start",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F::PrimeSubfield>(
                "multilinear.cu",
                "eq_mle_steps",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "eq_mle_start",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "eq_mle_steps",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear_evaluations.cu",
                "eval_multilinear_in_lagrange_basis_steps",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear_evaluations.cu",
                "eval_multilinear_in_lagrange_basis_shared_memory",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F::PrimeSubfield>(
                "multilinear_evaluations.cu",
                "eval_multilinear_in_lagrange_basis_steps",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F::PrimeSubfield>(
                "multilinear_evaluations.cu",
                "eval_multilinear_in_lagrange_basis_shared_memory",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "lagrange_to_monomial_basis_steps",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "lagrange_to_monomial_basis_end",
            ));
            cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
            cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "scale_in_place",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "add_slices",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F::PrimeSubfield>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F::PrimeSubfield, F>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F::PrimeSubfield, RCF>(
                "multilinear.cu",
                "linear_combination",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, RCF>(
                "multilinear.cu",
                "linear_combination",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear.cu",
                "linear_combination",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "sum_in_place",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear.cu",
                "linear_combination_at_row_level",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear.cu",
                "linear_combination_at_row_level",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
                "multilinear.cu",
                "add_assign_slices",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<RCF, F>(
                "multilinear.cu",
                "add_assign_slices",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F::PrimeSubfield, RCF>(
                "multilinear.cu",
                "add_assign_slices",
            ));
            cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));
            cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
            cuda_preprocess_twiddles::<F::PrimeSubfield>(
                num_vars + config.starting_log_inv_rate - config.folding_factor.maximum(),
            );
            let dim_ef = <F as BasedVectorSpace<F::PrimeSubfield>>::DIMENSION;
            cuda_preprocess_many_sumcheck_computations(&prod_sumcheck, &[(1, dim_ef, dim_ef)]);
            cuda_sync();
        }

        let mut fs_prover = FsProver::new(config.cuda);

        let evals = (0..1 << config.num_variables)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let pol = if config.cuda {
            Multilinear::Device(MultilinearDevice::new(memcpy_htod(&evals)))
        } else {
            Multilinear::Host(MultilinearHost::new(evals))
        };
        let point = (0..config.num_variables)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let value = pol.evaluate_in_large_field(&point);
        let eval = Evaluation { point, value };

        let time = std::time::Instant::now();
        let witness = config.commit(pol, &mut fs_prover);
        println!("Commit: {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        config.open(witness, &eval, &mut fs_prover);
        println!("Open: {} ms", time.elapsed().as_millis());

        let transcript = fs_prover.transcript();
        println!("Proof size: {:.1} KiB\n", transcript.len() as f64 / 1024.0);
        let mut fs_verifier = FsVerifier::new(transcript);
        let parsed_commitment = config.parse_commitment(&mut fs_verifier).unwrap();
        config
            .verify(&parsed_commitment, &eval, &mut fs_verifier)
            .unwrap();
    }
}
