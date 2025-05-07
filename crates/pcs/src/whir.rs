use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use utils::{Evaluation, KeccakDigest, Statement};
use whir::{
    committer::Committer,
    parameters::{WhirConfig, WhirConfigBuilder},
    prover::Prover,
    verifier::{ParsedCommitment, Verifier, WhirError},
};

use whir::committer::Witness as WhirWitness;

use crate::PcsParams;

use super::{PCS, PcsWitness};

impl<'a, F: Field, EF: ExtensionField<F>> PcsWitness<F> for WhirWitness<F, EF> {
    fn pol(&self) -> &Multilinear<F> {
        &self.lagrange_polynomial
    }
}

impl PcsParams for WhirConfigBuilder {
    fn security_bits(&self) -> usize {
        self.security_level
    }
}

impl<
    F: TwoAdicField + Ord + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    EF: ExtensionField<F>
        + TwoAdicField
        + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>
        + Ord,
> PCS<F, EF> for WhirConfig<F, EF>
where
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    type Witness = WhirWitness<F, EF>;
    type ParsedCommitment = ParsedCommitment<EF, KeccakDigest>;
    type VerifError = WhirError;
    type Params = WhirConfigBuilder;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        params.build(n_vars)
    }

    fn commit(&self, pol: Multilinear<F>, fs_prover: &mut FsProver) -> Self::Witness {
        Committer::new(self.clone()).commit(fs_prover, pol).unwrap()
    }

    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Self::ParsedCommitment, Self::VerifError> {
        Verifier::<F, EF>::new(self.clone()).parse_commitment(fs_verifier)
    }

    fn open(&self, witness: Self::Witness, eval: &Evaluation<EF>, fs_prover: &mut FsProver) {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Prover(self.clone())
            .prove(fs_prover, statement, witness)
            .unwrap();
    }

    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        eval: &Evaluation<EF>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), Self::VerifError> {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Verifier::new(self.clone()).verify(fs_verifier, parsed_commitment, statement)
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
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
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

        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 8>;

        let num_vars = 20;
        let folding = FoldingFactor::Constant(4);
        let log_inv_rate = 1;
        let cuda = true;
        let params = WhirConfigBuilder::standard(
            SoundnessType::ProvableList,
            100,
            log_inv_rate,
            folding,
            cuda,
        );

        test_whir_pcs_helper::<F, EF>(&params, num_vars);
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
        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 4>;

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
                        test_whir_pcs_helper::<F, EF>(&params, num_vars);
                        test_whir_pcs_helper::<EF, EF>(&params, num_vars);
                    }
                }
            }
        }
    }

    fn test_whir_pcs_helper<
        F: TwoAdicField + Ord + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
        EF: TwoAdicField
            + Ord
            + ExtensionField<F>
            + BasedVectorSpace<<F as PrimeCharacteristicRing>::PrimeSubfield>
            + p3_field::ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    >(
        params: &WhirConfigBuilder,
        num_vars: usize,
    ) where
        <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    {
        let config = WhirConfig::<F, EF>::new(num_vars, params);

        dbg!(&config);

        if config.cuda {
            cuda_init();
            let prod_sumcheck = SumcheckComputation::<KoalaBear> {
                exprs: &[(ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1))
                    .fix_computation(false)],
                n_multilinears: 2,
                eq_mle_multiplier: false,
            };
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "eq_mle",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
                "multilinear.cu",
                "eval_multilinear_in_lagrange_basis",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
                "multilinear.cu",
                "eval_multilinear_in_lagrange_basis",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "lagrange_to_monomial_basis",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "lagrange_to_monomial_basis",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "ntt/transpose.cu",
                "transpose",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "ntt/transpose.cu",
                "transpose",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "ntt/bit_reverse.cu",
                "reverse_bit_order_for_ntt",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "ntt/bit_reverse.cu",
                "reverse_bit_order_for_ntt",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<KoalaBear, EF>(
                "ntt/ntt.cu",
                "ntt_step",
            ));
            cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<EF>());
            cuda_load_function(CudaFunctionInfo::two_fields::<KoalaBear, F>(
                "ntt/ntt.cu",
                "ntt_step",
            ));
            cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "scale_in_place",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "add_slices",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, KoalaBear>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<KoalaBear, EF>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
                "multilinear.cu",
                "fold_rectangular",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "sum_in_place",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
                "multilinear.cu",
                "linear_combination_at_row_level",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
                "multilinear.cu",
                "linear_combination_at_row_level",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "add_assign_slices",
            ));
            cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));
            cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
            cuda_preprocess_twiddles::<KoalaBear>();
            let dim_ef = <EF as BasedVectorSpace<F::PrimeSubfield>>::DIMENSION;
            cuda_preprocess_many_sumcheck_computations(&prod_sumcheck, &[(1, dim_ef, dim_ef)]);
            cuda_sync();
        }

        let mut fs_prover = FsProver::new();

        let evals = (0..1 << config.num_variables)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let pol = if config.cuda {
            Multilinear::Device(MultilinearDevice::new(memcpy_htod(&evals)))
        } else {
            Multilinear::Host(MultilinearHost::new(evals))
        };
        let point = (0..config.num_variables)
            .map(|x| EF::from_u64(x as u64))
            .collect::<Vec<_>>();
        let value = pol.evaluate(&point);
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
