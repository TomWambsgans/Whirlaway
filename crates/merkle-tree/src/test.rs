use std::collections::HashSet;

use cuda_engine::memcpy_htod;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::MerkleTree;

#[test]
#[ignore]
fn test_merkle_tree() {
    cuda_engine::init::<KoalaBear>(&[], 0);
    let rand = &mut StdRng::seed_from_u64(0);
    let height = 9;
    let batch_size = 3;
    let n_indexes = 51;
    let leaves = (0..batch_size << height).map(|i| i).collect::<Vec<usize>>();
    let leaves_dev = memcpy_htod(&leaves);
    let cpu_merkle_tree = MerkleTree::new_cpu(&leaves, batch_size);
    let gpu_merkle_tree = MerkleTree::new_gpu(&leaves_dev, batch_size);
    assert_eq!(cpu_merkle_tree.root(), gpu_merkle_tree.root());
    for merkle_tree in [cpu_merkle_tree, gpu_merkle_tree] {
        let root_hash = merkle_tree.root();
        let mut indexes = HashSet::new();
        while indexes.len() < n_indexes {
            indexes.insert(rand.random_range(0..1 << height));
        }
        let indexes = indexes.into_iter().collect::<Vec<_>>();
        let proof = merkle_tree.generate_multi_proof(indexes.clone());
        let leaves_to_verify = indexes
            .iter()
            .map(|&i| leaves[i * batch_size..(i + 1) * batch_size].to_vec())
            .collect::<Vec<_>>();
        assert!(proof.verify(&root_hash, &leaves_to_verify, height));
    }
}
