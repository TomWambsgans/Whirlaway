#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use p3_koala_bear::KoalaBear;
use pcs::WhirParameters;
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

const USE_CUDA: bool = true;

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    if USE_CUDA {
        cuda_bindings::init::<KoalaBear>(&[]);
    }
    prove_poseidon2(13, WhirParameters::standard(128, 3, USE_CUDA));
}
