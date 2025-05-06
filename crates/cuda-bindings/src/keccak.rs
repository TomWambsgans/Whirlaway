use cuda_engine::{
    CudaCall, CudaFunctionInfo, cuda_alloc_zeros, cuda_sync, memcpy_dtoh, memcpy_htod,
};
use cudarc::driver::{CudaView, CudaViewMut, DeviceRepr, PushKernelArg};
use utils::KeccakDigest;

pub fn cuda_keccak256<T: DeviceRepr>(
    input: &CudaView<T>,
    batch_size: usize,
    output: &mut CudaViewMut<KeccakDigest>,
) {
    assert!(input.len() % batch_size == 0);
    let n_inputs = (input.len() / batch_size) as u32;
    assert_eq!(n_inputs, output.len() as u32);
    let input_length = (batch_size * std::mem::size_of::<T>()) as u32;
    let mut launch_args = CudaCall::new(
        CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"),
        n_inputs,
    );
    launch_args.arg(input);
    launch_args.arg(&n_inputs);
    launch_args.arg(&input_length);
    launch_args.arg(output);
    launch_args.launch();
}

// Sync
// Non deterministic
pub fn cuda_pow_grinding(seed: &KeccakDigest, ending_zeros_count: usize) -> u64 {
    let seed_dev = memcpy_htod(&seed.0);
    let mut solution_increment_dev = cuda_alloc_zeros::<u64>(1); // TODO it's size_t, not u64
    let ending_zeros_count_u32 = ending_zeros_count as u32;

    let mut starting_nonce = 0u64;
    loop {
        let mut launch_args = CudaCall::new(
            CudaFunctionInfo::basic("keccak.cu", "pow_grinding"),
            1u32.checked_shl(ending_zeros_count as u32).unwrap(),
        );

        let total_n_threads = launch_args.total_n_threads(false) as u64;
        let n_iters = (1_u64 << ending_zeros_count).div_ceil(10 * total_n_threads) as u32; // each kernel has 10 % chance of finding a correct nonce

        launch_args.arg(&seed_dev);
        launch_args.arg(&ending_zeros_count_u32);
        launch_args.arg(&starting_nonce);
        launch_args.arg(&n_iters);
        launch_args.arg(&mut solution_increment_dev);
        launch_args.launch();
        let solution_increment = memcpy_dtoh(&solution_increment_dev)[0];
        cuda_sync();
        if solution_increment != 0 {
            return solution_increment + starting_nonce;
        }
        starting_nonce += n_iters as u64 * total_n_threads;
    }
}

#[cfg(test)]
mod tests {
    use cuda_engine::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rayon::prelude::*;
    use utils::{KeccakDigest, count_ending_zero_bits, keccak256};

    use super::*;

    #[test]
    fn test_cuda_pow_grinding() {
        let ending_zeros_count = 18;

        cuda_init();
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
        let mut rng = StdRng::seed_from_u64(0);
        let seed: [u8; 32] = rng.random();
        cuda_sync();
        let time = std::time::Instant::now();
        let nonce = cuda_pow_grinding(&KeccakDigest(seed), ending_zeros_count);
        cuda_sync();
        println!(
            "CUDA pow grinding took {} ms for {} bits (nonce = {})",
            time.elapsed().as_millis(),
            ending_zeros_count,
            nonce
        );
        assert!(
            count_ending_zero_bits(&keccak256(&[&seed[..], &nonce.to_be_bytes()].concat()).0)
                >= ending_zeros_count as usize
        );
    }

    #[test]
    fn test_cuda_keccak() {
        let t = std::time::Instant::now();
        cuda_init();
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));

        println!("CUDA initialized in {} ms", t.elapsed().as_millis());

        let n_inputs = 4;
        let batch_size = 16;
        let input = (0..n_inputs * batch_size)
            .map(|i| (i % 256) as u8)
            .collect::<Vec<u8>>();

        let time = std::time::Instant::now();
        let expected_result = (0..n_inputs)
            .into_par_iter()
            .map(|i| keccak256(&input[i * batch_size..(i + 1) * batch_size]))
            .collect::<Vec<KeccakDigest>>();
        println!("CPU keccak took {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        let input_dev = memcpy_htod(&input);
        cuda_sync();
        println!("CUDA memcpy_htod took {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        let output_dev = cuda_alloc::<KeccakDigest>(n_inputs as usize);
        cuda_keccak256(
            &input_dev.as_view(),
            batch_size,
            &mut output_dev.as_view_mut(),
        );
        cuda_sync();
        println!("CUDA keccak took {} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        let dest = memcpy_dtoh(&output_dev);
        cuda_sync();
        println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());
        assert!(dest == expected_result);

        let n_particular_hashes = 300;
        let particular_indexes = (0..n_particular_hashes)
            .map(|i| (i * i * 3 + i + 78) % n_inputs)
            .collect::<Vec<_>>();
        let mut particular_hashes = (0..n_particular_hashes)
            .map(|_| [KeccakDigest::default()])
            .collect::<Vec<_>>();

        let time = std::time::Instant::now();
        for i in 0..n_particular_hashes {
            particular_hashes[i] =
                memcpy_dtoh(&output_dev.slice(particular_indexes[i]..1 + particular_indexes[i]))
                    .try_into()
                    .unwrap();
        }
        cuda_sync();
        println!(
            "CUDA transfer of {} hashes from gpu to cpu took {} ms",
            n_particular_hashes,
            time.elapsed().as_millis()
        );

        for i in 0..n_particular_hashes {
            assert_eq!(
                particular_hashes[i][0],
                expected_result[particular_indexes[i]]
            );
        }
    }
}
