use std::any::TypeId;

use p3_field::{PrimeField32, TwoAdicField};
use utils::powers_parallel;

use crate::*;

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
