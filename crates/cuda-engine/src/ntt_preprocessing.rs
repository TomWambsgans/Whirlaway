use std::any::TypeId;

use cudarc::driver::CudaSlice;
use p3_field::{Field, PrimeField32, TwoAdicField};
use utils::powers_parallel;

use crate::*;

const MAX_DOMAIN_SIZE: usize = 64;

pub fn cuda_preprocess_twiddles<F: TwoAdicField + PrimeField32>(log_max_size: usize) {
    assert!(log_max_size <= F::TWO_ADICITY && log_max_size <= MAX_DOMAIN_SIZE);
    let engine = cuda_engine();
    let mut guard = engine.twiddles.write().unwrap();
    let (twiddles, ptrs) = guard
        .entry(TypeId::of::<F::PrimeSubfield>())
        .or_insert_with(|| {
            (
                Vec::with_capacity(MAX_DOMAIN_SIZE),
                cuda_alloc_zeros(MAX_DOMAIN_SIZE),
            )
        });
    if log_max_size <= twiddles.len() {
        return;
    }
    let _span = tracing::info_span!("Preprocessing cuda twiddles", log_max_size).entered();
    for i in twiddles.len() + 1..=log_max_size {
        let twiddles_host = powers_parallel(F::two_adic_generator(i), 1 << (i - 1));
        let twiddles_dev = memcpy_htod(&twiddles_host);
        let twiddles_u32 = unsafe { std::mem::transmute::<_, CudaSlice<u32>>(twiddles_dev) };
        cuda_set_at_index(ptrs, i - 1, CudaPtr::from(&twiddles_u32, engine));
        twiddles.push(twiddles_u32);
    }
    cuda_sync();
}

pub fn cuda_twiddles<F: Field>(
    log_domain_size: usize,
) -> &'static CudaSlice<CudaPtr<F::PrimeSubfield>> {
    let guard = cuda_engine().twiddles.read().unwrap();
    let (twiddles, ptrs) = guard
        .get(&TypeId::of::<F::PrimeSubfield>())
        .unwrap_or_else(|| {
            panic!(
                "twiddles have not been preprocessed for : {}",
                std::any::type_name::<F::PrimeSubfield>()
            )
        });
    assert!(
        log_domain_size <= twiddles.len(),
        "{} > {}",
        log_domain_size,
        twiddles.len()
    );
    unsafe { std::mem::transmute(ptrs) }
}
