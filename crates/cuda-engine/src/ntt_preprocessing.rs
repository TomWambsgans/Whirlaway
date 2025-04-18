use std::any::TypeId;

use p3_field::{PrimeField32, TwoAdicField};
use rayon::prelude::*;
use utils::powers_parallel;

use crate::*;

pub fn cuda_preprocess_all_twiddles<F: TwoAdicField + PrimeField32>(whir_folding_factor: usize) {
    cuda_preprocess_twiddles::<F>();
    cuda_preprocess_correction_twiddles::<F>(whir_folding_factor);
}

pub fn cuda_preprocess_twiddles<F: TwoAdicField + PrimeField32>() {
    let mut guard = cuda_engine().twiddles.write().unwrap();
    if guard.contains_key(&TypeId::of::<F>()) {
        return;
    }
    let _span = tracing::info_span!("Preprocessing cuda twiddles").entered();
    let mut all_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        all_twiddles.extend(powers_parallel(F::two_adic_generator(i), 1 << i));
    }
    let all_twiddles_dev = memcpy_htod(&all_twiddles);
    cuda_sync();
    let all_twiddles_u32 = unsafe { std::mem::transmute(all_twiddles_dev) };
    guard.insert(TypeId::of::<F>(), all_twiddles_u32);
}

pub fn cuda_preprocess_correction_twiddles<F: TwoAdicField + PrimeField32>(
    whir_folding_factor: usize,
) {
    let mut guard = cuda_engine().correction_twiddles.write().unwrap();
    if guard.contains_key(&(TypeId::of::<F>(), whir_folding_factor)) {
        return;
    }
    let _span = tracing::info_span!(
        "Preprocessing cuda correction twiddles",
        whir_folding_factor
    )
    .entered();
    let folding_size = 1 << whir_folding_factor;
    let size_inv = F::from_u64(folding_size).inverse();
    // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
    let mut all_correction_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY - whir_folding_factor {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        let inv_root = F::two_adic_generator(i + whir_folding_factor).inverse();
        let inv_powers = powers_parallel(inv_root, 1 << (i + whir_folding_factor));
        let correction_twiddles = (0..1 << (i + whir_folding_factor))
            .into_par_iter()
            .map(|j| size_inv * inv_powers[((j % folding_size) * (j / folding_size)) as usize])
            .collect::<Vec<_>>();
        all_correction_twiddles.extend(correction_twiddles);
    }
    let all_correction_twiddles_dev = memcpy_htod(&all_correction_twiddles);
    cuda_sync();
    let all_correction_twiddles_u32 = unsafe { std::mem::transmute(all_correction_twiddles_dev) };
    guard.insert(
        (TypeId::of::<F>(), whir_folding_factor),
        all_correction_twiddles_u32,
    );
}
