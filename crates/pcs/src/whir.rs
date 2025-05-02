use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use utils::{Evaluation, KeccakDigest};
use whir::whir::{
    Statement,
    committer::Committer,
    parameters::WhirConfig,
    prover::Prover,
    verifier::{ParsedCommitment, Verifier, WhirError},
};

use whir::whir::committer::Witness as WhirWitness;

use crate::PcsParams;

use super::{PCS, PcsWitness};

pub use whir::parameters::WhirParameters;

impl<'a, F: Field, EF: ExtensionField<F>> PcsWitness<F> for WhirWitness<F, EF> {
    fn pol(&self) -> &Multilinear<F> {
        &self.lagrange_polynomial
    }
}

impl PcsParams for WhirParameters {
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
    type Params = WhirParameters;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        WhirConfig::<F, EF>::new(n_vars, params)
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
    fn test_whir_pcs() {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();

        let n_vars = 17;
        let cuda = true;
        // type F = KoalaBear;
        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 8>;

        let pcs = WhirConfig::<F, EF>::new(
            n_vars,
            &WhirParameters::standard(
                SoundnessType::ProvableList,
                128,
                2,
                FoldingFactor::Constant(4),
                cuda,
            ),
        );

        if cuda {
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
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "multilinear.cu",
                "lagrange_to_monomial_basis_rev",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<F>(
                "multilinear.cu",
                "lagrange_to_monomial_basis_rev",
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
                "reverse_bit_order_global",
            ));
            cuda_load_function(CudaFunctionInfo::one_field::<EF>(
                "ntt/bit_reverse.cu",
                "reverse_bit_order_global",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
                "multilinear.cu",
                "dot_product",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
                "multilinear.cu",
                "dot_product",
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
            cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
                "multilinear.cu",
                "eval_multilinear_in_monomial_basis",
            ));
            cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
                "multilinear.cu",
                "eval_multilinear_in_monomial_basis",
            ));
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
            let dim_ef = <EF as BasedVectorSpace<KoalaBear>>::DIMENSION;
            cuda_preprocess_many_sumcheck_computations(&prod_sumcheck, &[(1, dim_ef, dim_ef)]);
            cuda_sync();
        }

        let mut fs_prover = FsProver::new();

        let evals = (0..1 << n_vars)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let pol = if cuda {
            Multilinear::Device(MultilinearDevice::new(memcpy_htod(&evals)))
        } else {
            Multilinear::Host(MultilinearHost::new(evals))
        };
        let point = (0..n_vars)
            .map(|x| EF::from_u64(x as u64))
            .collect::<Vec<_>>();
        let value = pol.evaluate(&point);
        let eval = Evaluation { point, value };

        let time = std::time::Instant::now();
        let witness = pcs.commit(pol, &mut fs_prover);
        println!("Commit: {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        pcs.open(witness, &eval, &mut fs_prover);
        println!("Open: {} ms", time.elapsed().as_millis());

        let transcript = fs_prover.transcript();
        println!("Proof size: {:.1} KiB\n", transcript.len() as f64 / 1024.0);
        let mut fs_verifier = FsVerifier::new(transcript);
        let parsed_commitment = pcs.parse_commitment(&mut fs_verifier).unwrap();
        pcs.verify(&parsed_commitment, &eval, &mut fs_verifier)
            .unwrap();
    }
}
