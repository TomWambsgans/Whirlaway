use std::any::TypeId;

use cudarc::driver::CudaSlice;
use p3_field::{PrimeField32, TwoAdicField};
use utils::powers_parallel;

use crate::*;

pub fn cuda_preprocess_twiddles<F: TwoAdicField + PrimeField32>(log_max_size: usize) {
    assert!(log_max_size <= F::TWO_ADICITY);
    let mut guard = cuda_engine().twiddles.write().unwrap();
    let twiddles = guard
        .entry(TypeId::of::<F::PrimeSubfield>())
        .or_insert_with(|| Vec::with_capacity(64));
    if log_max_size <= twiddles.len() {
        return;
    }
    let _span =
        tracing::info_span!("Preprocessing cuda twiddles, size = {}", log_max_size).entered();
    for i in twiddles.len() + 1..=log_max_size {
        let twiddles_host = powers_parallel(F::two_adic_generator(i), 1 << (i - 1));
        let twiddles_dev = memcpy_htod(&twiddles_host);
        let itwiddles_u32 = unsafe { std::mem::transmute::<_, CudaSlice<u32>>(twiddles_dev) };
        twiddles.push(itwiddles_u32);
    }
    cuda_sync();
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    #[test]
    fn test_cuda_preprocess_twiddles() {
        type F = KoalaBear;
        cuda_init();
        cuda_preprocess_twiddles::<F>(5);
        cuda_preprocess_twiddles::<F>(3);
        cuda_preprocess_twiddles::<F>(15);
        let root = F::two_adic_generator(10);
        let twiddles_dev = cuda_twiddles::<F>(10);
        let twiddles = memcpy_dtoh(twiddles_dev);
        cuda_sync();
        for i in 0..(1 << 9) {
            let expected = root.exp_u64(i as u64);
            assert_eq!(expected, twiddles[i]);
        }
    }
}
