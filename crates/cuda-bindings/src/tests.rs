use crate::{cuda_batch_keccak, init_cuda};

#[test]
fn test_cuda_keccak() {
    let t = std::time::Instant::now();
    init_cuda().unwrap();
    println!("CUDA initialized in {} ms", t.elapsed().as_millis());

    let n_inputs = 10_000;
    let input_length = 1000;
    let input_packed_length = 1111;
    let src_bytes = (0..n_inputs * input_packed_length)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();

    let t = std::time::Instant::now();
    let dest = cuda_batch_keccak(&src_bytes, input_length, input_packed_length).unwrap();
    println!("CUDA took {} ms", t.elapsed().as_millis());
    for i in 0..n_inputs {
        assert_eq!(
            hash_keccak256(
                &src_bytes[i * input_packed_length..i * input_packed_length + input_length]
            ),
            dest[i]
        );
    }
}

fn hash_keccak256(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}
