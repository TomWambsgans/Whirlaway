use algebra::pols::{Multilinear, MultilinearDevice, MultilinearHost};
use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{
    CudaFunctionInfo, SumcheckComputation, cuda_init, cuda_load_function,
    cuda_preprocess_many_sumcheck_computations, cuda_preprocess_twiddles, cuda_sync, memcpy_htod,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{
    BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
    extension::BinomialExtensionField,
};
use p3_koala_bear::KoalaBear;
use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use utils::Evaluation;

use crate::parameters::{FoldingFactor, SoundnessType, WhirConfigBuilder};

#[test]
fn test_whir_debug() {
    type F = KoalaBear;
    type EF = KoalaBear;

    let num_vars = 12;
    let folding = FoldingFactor::Constant(3);
    let log_inv_rate = 1;
    let cuda = false;
    let params = WhirConfigBuilder::standard(
        SoundnessType::ConjectureList,
        20,
        log_inv_rate,
        folding,
        2,
        cuda,
    );

    test_whir_pcs_helper::<F, EF>(&params, num_vars);
}

#[test]
fn bench_whir() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type F = BinomialExtensionField<KoalaBear, 4>;
    type EF = F;
    let log_inv_rate = 2;
    let folding = FoldingFactor::Constant(4);
    let cuda = true;
    let params = WhirConfigBuilder::standard(
        SoundnessType::ConjectureList,
        110,
        log_inv_rate,
        folding,
        3,
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
                for soudness_type in [
                    SoundnessType::ConjectureList,
                    SoundnessType::ProvableList,
                    SoundnessType::UniqueDecoding,
                ] {
                    for log_inv_rate in [1, 2, 3] {
                        for innitial_domain_reduction_factor in [1, 2] {
                            let params = WhirConfigBuilder::standard(
                                soudness_type,
                                70,
                                log_inv_rate,
                                folding,
                                innitial_domain_reduction_factor,
                                cuda,
                            );
                            test_whir_pcs_helper::<F, EF>(&params, num_vars);
                            test_whir_pcs_helper::<EF, EF>(&params, num_vars);
                        }
                    }
                }
            }
        }
    }
}

fn test_whir_pcs_helper<F: TwoAdicField + Ord, EF: ExtensionField<F>>(
    params: &WhirConfigBuilder,
    num_vars: usize,
) where
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    EF: ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield> + TwoAdicField + Ord,
    F::PrimeSubfield: PrimeField32 + TwoAdicField,
{
    let config = params.build::<F, EF>(num_vars);

    // dbg!(&config);

    if config.cuda {
        cuda_init();
        let prod_sumcheck = SumcheckComputation::<KoalaBear> {
            exprs: &[
                (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1)).fix_computation(false)
            ],
            n_multilinears: 2,
            eq_mle_multiplier: false,
        };
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "eq_mle_steps",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_shared_memory",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_shared_memory",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
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
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<EF>());
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "scale_in_place",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "add_slices",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "linear_combination",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "linear_combination",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "linear_combination",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "sum_in_place",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "add_assign_slices",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
            "multilinear.cu",
            "add_assign_slices",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "add_assign_slices",
        ));
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
        cuda_preprocess_twiddles::<F::PrimeSubfield>(
            num_vars + config.starting_log_inv_rate - config.folding_factor.maximum(),
        );
        let dim_f = <F as BasedVectorSpace<F::PrimeSubfield>>::DIMENSION;
        let dim_ef = <EF as BasedVectorSpace<EF::PrimeSubfield>>::DIMENSION;
        cuda_preprocess_many_sumcheck_computations(&prod_sumcheck, &[(dim_f, dim_f, dim_ef)]);
        cuda_preprocess_many_sumcheck_computations(&prod_sumcheck, &[(dim_f, dim_ef, dim_ef)]);
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
        .map(|x| EF::from_u64(x as u64))
        .collect::<Vec<_>>();
    let value = pol.evaluate_in_large_field(&point);
    let eval = Evaluation { point, value };

    let time = std::time::Instant::now();
    let witness = config.commit(pol, &mut fs_prover);
    println!("Commit: {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    config.open(witness, vec![eval.clone()], &mut fs_prover);
    println!("Open: {} ms", time.elapsed().as_millis());

    let transcript = fs_prover.transcript();
    println!("Proof size: {:.1} KiB\n", transcript.len() as f64 / 1024.0);
    let mut fs_verifier = FsVerifier::new(transcript);
    let parsed_commitment = config.parse_commitment(&mut fs_verifier).unwrap();
    config
        .verify(&parsed_commitment, vec![eval], &mut fs_verifier)
        .unwrap();
}
