use cuda_engine::cuda_info;
use cudarc::driver::{CudaView, CudaViewMut, DeviceRepr, LaunchConfig, PushKernelArg};
use utils::KeccakDigest;

const NUM_THREADS: u32 = 256;

pub fn cuda_keccak256<T: DeviceRepr>(
    input: &CudaView<T>,
    batch_size: usize,
    output: &mut CudaViewMut<KeccakDigest>,
) {
    assert!(input.len() % batch_size == 0);
    let n_inputs = input.len() / batch_size;
    assert_eq!(n_inputs, output.len());
    let input_length = (batch_size * std::mem::size_of::<T>()) as u32;
    let input_packed_length = input_length;

    let cuda = &cuda_info();
    let f = cuda.get_function("keccak", "batch_keccak256");

    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(input);
    launch_args.arg(&n_inputs);
    launch_args.arg(&input_length);
    launch_args.arg(&input_packed_length);
    launch_args.arg(output);
    unsafe { launch_args.launch(cfg) }.unwrap();
}

#[cfg(test)]
mod tests {
    use cuda_engine::{cuda_alloc, cuda_info, cuda_sync, memcpy_dtoh, memcpy_htod};
    use p3_koala_bear::KoalaBear;
    use rayon::prelude::*;
    use utils::{KeccakDigest, keccak256};

    use crate::cuda_keccak256;

    #[test]
    #[ignore]
    fn test_cuda_keccak() {
        let t = std::time::Instant::now();
        cuda_engine::init::<KoalaBear>(&[], 0);
        println!("CUDA initialized in {} ms", t.elapsed().as_millis());

        let n_inputs = 10_000;
        let batch_size = 501;
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
            cuda_info()
                .stream
                .memcpy_dtoh(
                    &output_dev.slice(particular_indexes[i]..1 + particular_indexes[i]),
                    &mut particular_hashes[i],
                )
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
