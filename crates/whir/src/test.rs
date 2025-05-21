use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{
    ExtensionField, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
    extension::BinomialExtensionField,
};
use p3_koala_bear::KoalaBear;
use rand::distr::{Distribution, StandardUniform};
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
    let params =
        WhirConfigBuilder::standard(SoundnessType::ConjectureList, 20, log_inv_rate, folding, 2);

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
    let params =
        WhirConfigBuilder::standard(SoundnessType::ConjectureList, 110, log_inv_rate, folding, 3);
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
                        );
                        test_whir_pcs_helper::<F, EF>(&params, num_vars);
                        test_whir_pcs_helper::<EF, EF>(&params, num_vars);
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
    StandardUniform: Distribution<F>,
    StandardUniform: Distribution<EF>,
{
    let config = params.build::<F, EF>(num_vars);

    // dbg!(&config);

    let mut fs_prover = FsProver::new();

    let evals = (0..1 << config.num_variables)
        .map(|x| F::from_u64(x as u64))
        .collect::<Vec<_>>();
    let pol = Multilinear::new(evals);
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
