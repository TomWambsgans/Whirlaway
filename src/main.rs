use cuda_engine::{
    CudaFunctionInfo, cuda_init, cuda_load_function, cuda_preprocess_twiddles, cuda_sync,
    memcpy_htod,
};
use icicle_core::ntt::get_root_of_unity;
use icicle_core::ntt::{NTTConfig, NTTDir, NTTInitDomainConfig, initialize_domain};
use icicle_core::traits::{FieldImpl, GenerateRandom};
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::{Device, runtime};
use p3_field::extension::BinomialExtensionField;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    runtime::load_backend_from_env_or_default().unwrap();
    let device = Device::new("CUDA", 0);
    icicle_runtime::set_default_device(&device).unwrap();
    icicle_runtime::set_device(&device).unwrap();

    {
        // ICICLE BabyBear^4 NTT

        use icicle_babybear::field::ExtensionField as BabyBearExtensionField;
        use icicle_babybear::field::ScalarField as BabyBearScalarField;

        let log_size = 26;
        let size = 1 << log_size;
        let scalars_host = <BabyBearExtensionField as FieldImpl>::Config::generate_random(size);
        let mut scalars_dev = DeviceVec::<BabyBearExtensionField>::device_malloc(size).unwrap();
        scalars_dev
            .copy_from_host(HostSlice::from_slice(&scalars_host))
            .unwrap();
        let mut ntt_results = DeviceVec::<BabyBearExtensionField>::device_malloc(size).unwrap();
        initialize_domain(
            get_root_of_unity::<BabyBearScalarField>(size.try_into().unwrap()),
            &NTTInitDomainConfig::default(),
        )
        .unwrap();
        let cfg = NTTConfig::<BabyBearScalarField>::default();
        let start = Instant::now();
        icicle_core::ntt::ntt(&scalars_dev, NTTDir::kForward, &cfg, &mut ntt_results[..]).unwrap();
        println!(
            "ICICLE NTT, for BabyBear^4 (approx 128 bits), on size 2^{log_size} took: {} ms",
            start.elapsed().as_millis()
        );
    }

    {
        // ICICLE Bn254 NTT, 2^25

        use icicle_bn254::curve::ScalarField as Bn254;

        let log_size = 25;
        let size = 1 << log_size;
        let scalars_host = <Bn254 as FieldImpl>::Config::generate_random(size);
        let mut scalars_dev = DeviceVec::<Bn254>::device_malloc(size).unwrap();
        scalars_dev
            .copy_from_host(HostSlice::from_slice(&scalars_host))
            .unwrap();
        let mut ntt_results = DeviceVec::<Bn254>::device_malloc(size).unwrap();
        initialize_domain(
            get_root_of_unity::<Bn254>(size.try_into().unwrap()),
            &NTTInitDomainConfig::default(),
        )
        .unwrap();
        let cfg = NTTConfig::<Bn254>::default();
        let start = Instant::now();
        icicle_core::ntt::ntt(&scalars_dev, NTTDir::kForward, &cfg, &mut ntt_results[..]).unwrap();
        println!(
            "ICICLE NTT, for Bn254, on size 2^{log_size} took: {} ms",
            start.elapsed().as_millis()
        );
    }

    {
        // Whirlaway BanyBear^4 NTT
        use p3_baby_bear::BabyBear;

        type BabyBear4 = BinomialExtensionField<BabyBear, 4>;

        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<BabyBear, BabyBear4>(
            "ntt/ntt.cu",
            "ntt_step",
        ));
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<BabyBear4>());

        cuda_preprocess_twiddles::<BabyBear>();

        let log_size = 26;
        let mut rng = StdRng::seed_from_u64(0);
        let scalars_host = (0..(1 << log_size))
            .map(|_| rng.random())
            .collect::<Vec<BabyBear4>>();
        let mut scalars_dev = memcpy_htod(&scalars_host);
        cuda_sync();

        let time = Instant::now();
        cuda_bindings::cuda_ntt(&mut scalars_dev, log_size);
        cuda_sync();
        println!(
            "Whirlaway NTT, for BabyBear^4 (approx 128 bits), on size 2^{log_size} took: {} ms",
            time.elapsed().as_millis()
        );
    }

    {
        // Whirlaway BanyBear^4 NTT
        use p3_baby_bear::BabyBear;

        type BabyBear8 = BinomialExtensionField<BabyBear, 8>;

        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<BabyBear, BabyBear8>(
            "ntt/ntt.cu",
            "ntt_step",
        ));
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<BabyBear8>());
        cuda_preprocess_twiddles::<BabyBear>();

        let log_size = 25;
        let mut rng = StdRng::seed_from_u64(0);
        let scalars_host = (0..(1 << log_size))
            .map(|_| rng.random())
            .collect::<Vec<BabyBear8>>();
        let mut scalars_dev = memcpy_htod(&scalars_host);
        cuda_sync();

        let time = Instant::now();
        cuda_bindings::cuda_ntt(&mut scalars_dev, log_size);
        cuda_sync();
        println!(
            "Whirlaway NTT, for BabyBear^8 (approx 256 bits), on size 2^{log_size} took: {} ms",
            time.elapsed().as_millis()
        );
    }
}
