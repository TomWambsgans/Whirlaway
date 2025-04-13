use std::{any::TypeId, borrow::Borrow};

use cudarc::driver::{CudaSlice, PushKernelArg};

use cuda_engine::{CudaCall, SumcheckComputation, concat_pointers, cuda_alloc, memcpy_htod};
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use crate::{MAX_N_BLOCKS, cuda_dot_product, cuda_fold_sum, cuda_sum};

// TODO avoid hardcoding
const SUMCHECK_MAX_LOG_N_THREADS_PER_BLOCK: u32 = 9;
const SUMCHECK_MAX_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_MAX_LOG_N_THREADS_PER_BLOCK;

/// Async
pub fn cuda_sum_over_hypercube_of_computation<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Borrow<CudaSlice<NF>>,
>(
    comp: &SumcheckComputation<F>,
    multilinears: &[ML], // in lagrange basis
    batching_scalars: &[EF],
    eq_mle: Option<&CudaSlice<EF>>,
) -> EF {
    if TypeId::of::<EF>() != TypeId::of::<NF>() {
        unimplemented!()
    }
    let multilinears: Vec<&CudaSlice<NF>> =
        multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert_eq!(batching_scalars.len(), comp.exprs.len());
    assert!(multilinears[0].len().is_power_of_two());
    let n_vars = multilinears[0].len().ilog2() as u32;
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars as usize));
    assert_eq!(eq_mle.is_some(), comp.eq_mle_multiplier);

    let n_compute_units = comp.n_cuda_compute_units() as u32;
    let n_threads_per_block = (n_compute_units << n_vars).min(SUMCHECK_MAX_N_THREADS_PER_BLOCK);
    let n_blocks = ((n_compute_units << n_vars).div_ceil(n_threads_per_block)).min(MAX_N_BLOCKS);
    let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION as u32;

    let multilinears_ptrs_dev = concat_pointers(&multilinears);

    let batching_scalars_dev = memcpy_htod(batching_scalars);
    let mut sums_dev = cuda_alloc::<EF>((n_compute_units as usize) << n_vars);

    let module_name = format!("sumcheck_{:x}", comp.uuid());
    let mut call = CudaCall::new(&module_name, "sum_over_hypercube_ext")
        .threads_per_block(n_threads_per_block)
        .blocks(n_blocks)
        .shared_mem_bytes(batching_scalars.len() as u32 * ext_degree * 4); // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];;
    call.arg(&multilinears_ptrs_dev);
    call.arg(&mut sums_dev);
    call.arg(&batching_scalars_dev);
    call.arg(&n_vars);
    call.arg(&n_compute_units);
    call.launch();

    let hypercube_evals = if n_compute_units == 1 {
        sums_dev
    } else {
        cuda_fold_sum(&sums_dev, n_compute_units as usize)
    };

    if comp.eq_mle_multiplier {
        let eq_mle = eq_mle.unwrap();
        cuda_dot_product(eq_mle, &hypercube_evals)
    } else {
        cuda_sum(hypercube_evals)
    }
}

#[cfg(test)]
mod test {
    use crate::cuda_sum_over_hypercube_of_computation;
    use algebra::pols::{MultilinearDevice, MultilinearHost};
    use arithmetic_circuit::ArithmeticCircuit;
    use cuda_engine::{
        SumcheckComputation, cuda_init, cuda_preprocess_sumcheck_computation, memcpy_htod,
    };
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rayon::prelude::*;

    #[test]
    fn test_cuda_sum_over_hypercube_of_computation() {
        type F = KoalaBear;
        const EXT_DEGREE: usize = 8;
        type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

        let n_multilinears = 11;
        let n_vars = 12;
        let depth = 1;
        let n_batching_scalars = 17;

        let rng = &mut StdRng::seed_from_u64(0);
        let exprs = (0..n_batching_scalars)
            .map(|_| ArithmeticCircuit::random(rng, n_multilinears, depth).fix_computation(true))
            .collect::<Vec<_>>();

        let sumcheck_computation = SumcheckComputation {
            n_multilinears,
            exprs: &exprs,
            eq_mle_multiplier: false,
        };
        let time = std::time::Instant::now();
        cuda_init();
        cuda_preprocess_sumcheck_computation(&sumcheck_computation);
        println!("CUDA initialized in {} ms", time.elapsed().as_millis());

        let rng = &mut StdRng::seed_from_u64(0);

        let multilinears = (0..n_multilinears)
            .map(|_| MultilinearHost::<EF>::random(rng, n_vars))
            .collect::<Vec<_>>();

        let batching_scalar: EF = rng.random();
        let batching_scalars = (0..n_batching_scalars)
            .map(|e| batching_scalar.exp_u64(e))
            .collect::<Vec<_>>();

        let time = std::time::Instant::now();
        let expected_sum = (0..1 << n_vars)
            .into_par_iter()
            .map(|i| {
                exprs.iter().zip(&batching_scalars).map(|(comp, b)| {
                comp.eval(
                    &(0..n_multilinears)
                        .map(|j| multilinears[j].evals[i])
                        .collect::<Vec<_>>(),
                )
            } * *b).sum::<EF>()
            })
            .sum::<EF>();
        println!("CPU hypercube sum took {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        let multilinears_dev = multilinears
            .iter()
            .map(|multilinear| MultilinearDevice::new(memcpy_htod(&multilinear.evals)))
            .collect::<Vec<_>>();
        let copy_duration = time.elapsed();

        let time = std::time::Instant::now();
        let cuda_sum = cuda_sum_over_hypercube_of_computation::<F, EF, _, _>(
            &sumcheck_computation,
            &multilinears_dev,
            &batching_scalars,
            None,
        );

        println!(
            "CUDA hypercube sum took {} ms (copy duration: {} ms)",
            time.elapsed().as_millis(),
            copy_duration.as_millis()
        );

        assert_eq!(cuda_sum, expected_sum);
    }
}
