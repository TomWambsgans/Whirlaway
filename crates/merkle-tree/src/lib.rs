#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use rayon::prelude::*;
use sha3::Digest;
use std::collections::BTreeSet;
use std::fmt::Debug;
use tracing::instrument;
use utils::{KeccakDigest, keccak256};

fn leaf_hash<F>(input: &[F]) -> KeccakDigest {
    // TODO this is ugly
    let buff = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, std::mem::size_of_val(input))
    };
    keccak256(buff)
}

fn two_to_one_hash(left_input: &KeccakDigest, right_input: &KeccakDigest) -> KeccakDigest {
    let mut h = sha3::Keccak256::new();
    h.update(left_input.0);
    h.update(right_input.0);
    let mut output = [0; 32];
    output.copy_from_slice(&h.finalize()[..]);
    KeccakDigest(output)
}

#[derive(Debug, Clone, Default)]
pub struct MultiPath {
    /// For node i path, stores at index i the suffix of the path for Incremental Encoding
    pub auth_paths_suffixes: Vec<Vec<KeccakDigest>>,
    /// stores the leaf indexes of the nodes to prove
    pub leaf_indexes: Vec<usize>,
}

impl MultiPath {
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.auth_paths_suffixes.len();
        assert_eq!(n, self.leaf_indexes.len());

        let mut res = (n as u32).to_le_bytes().to_vec();
        for i in 0..n {
            assert!(self.auth_paths_suffixes[i].len() <= u8::MAX as usize);
            res.push(self.auth_paths_suffixes[i].len() as u8);
            for j in 0..self.auth_paths_suffixes[i].len() {
                res.extend_from_slice(&self.auth_paths_suffixes[i][j].0);
            }
            assert!(self.leaf_indexes[i] <= u32::MAX as usize);
            res.extend_from_slice(&(self.leaf_indexes[i] as u32).to_le_bytes());
        }
        res
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let n = u32::from_le_bytes(bytes.get(0..4)?.try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut res = Self::default();
        for _ in 0..n {
            if offset > bytes.len() - 1 {
                return None;
            }
            let suffix_len = bytes[offset] as usize;
            offset += 1;
            let mut paths_suffixes = Vec::new();
            for _ in 0..suffix_len {
                paths_suffixes.push(KeccakDigest(
                    bytes.get(offset..offset + 32)?.try_into().unwrap(),
                ));
                offset += 32;
            }
            res.auth_paths_suffixes.push(paths_suffixes);
            res.leaf_indexes.push(u32::from_le_bytes(
                bytes.get(offset..offset + 4)?.try_into().unwrap(),
            ) as usize);
            offset += 4;
        }

        if offset != bytes.len() {
            return None;
        }

        Some(res)
    }

    pub fn verify<F>(
        &self,
        root_hash: &KeccakDigest,
        leaves: &[Vec<F>],
        tree_height: usize,
    ) -> bool {
        let n = leaves.len();
        if n != self.leaf_indexes.len() || n != self.auth_paths_suffixes.len() {
            return false;
        }

        let mut leafs_with_index = self
            .leaf_indexes
            .iter()
            .zip(leaves.iter())
            .map(|(i, l)| (*i, l))
            .collect::<Vec<_>>();
        leafs_with_index.sort_by_key(|(i, _)| *i);

        let mut prev_auth_path: Vec<KeccakDigest> = vec![];
        let mut prev_hash_chain: Vec<KeccakDigest> = vec![];

        for i in 0..n {
            // decode i-th auth path

            let (leaf_index, leaf) = leafs_with_index[i];

            let suffix = self.auth_paths_suffixes[i].clone();

            let mut auth_path = if i == 0 {
                if suffix.len() != tree_height {
                    return false;
                }
                suffix
            } else {
                if suffix.len() > tree_height - 1 {
                    return false;
                }
                let prefix_len = tree_height - 1 - suffix.len();
                prev_auth_path.reverse();
                let mut path = prev_auth_path[0..prefix_len].to_vec();
                path.push(prev_hash_chain[tree_height - 1 - prefix_len].clone());
                path.extend_from_slice(&suffix);
                path
            };

            auth_path.reverse();

            let mut curr_hash = leaf_hash(leaf);
            prev_hash_chain.clear();

            let mut index = leaf_index;
            // Check levels between leaf level and root
            for level in 0..tree_height {
                prev_hash_chain.push(curr_hash.clone());

                let (left, right) = if index & 1 == 0 {
                    // left child
                    (&curr_hash, &auth_path[level])
                } else {
                    // right child
                    (&auth_path[level], &curr_hash)
                };
                index /= 2;
                curr_hash = two_to_one_hash(left, right);
            }

            // check if final hash is root
            if &curr_hash != root_hash {
                return false;
            }

            prev_auth_path = auth_path;
        }
        true
    }
}

/// Defines a merkle tree data structure.
/// This merkle tree has runtime fixed height, and assumes number of leaves is 2^height.
pub struct MerkleTree<F> {
    /// The ith nodes (starting at 1st) children are at indices `2*i`, `2*i+1`
    /// (The first element is the root, and the last elements are the leaves)
    nodes: Vec<KeccakDigest>,
    /// Stores the height of the MerkleTree
    height: usize,
    root: KeccakDigest,

    _field: std::marker::PhantomData<F>,
}

impl<F: Sync> MerkleTree<F> {
    /// `leaves.len()` should be power of two.
    #[instrument(name = "merkle tree creation", skip_all)]
    pub fn new(leaves: &[F], batch_size: usize) -> Self {
        assert!(leaves.len() % batch_size == 0);
        let leaf_digests = leaves
            .par_chunks_exact(batch_size)
            .map(|input| leaf_hash(input))
            .collect::<Vec<_>>();

        let leaf_nodes_size = leaf_digests.len();
        assert!(leaf_nodes_size.is_power_of_two() && leaf_nodes_size > 1);
        let height = leaf_nodes_size.ilog2() as usize;

        // initialize the merkle tree as array of nodes in level order
        let mut nodes = vec![KeccakDigest::default(); (1 << (height + 1)) - 1];
        nodes[(1 << height) - 1..].clone_from_slice(&leaf_digests);

        for level in (1..=height).rev() {
            let start = (1 << level) - 1;
            let (left, right) = nodes.split_at_mut(start);
            let left = &mut left[(1 << (level - 1)) - 1..];
            left.par_iter_mut().enumerate().for_each(|(i, l)| {
                *l = two_to_one_hash(&right[2 * i], &right[2 * i + 1]);
            });
        }
        let root = nodes[0].clone();
        MerkleTree {
            nodes,
            height,
            root,
            _field: std::marker::PhantomData,
        }
    }

    pub fn root(&self) -> &KeccakDigest {
        &self.root
    }

    /// Returns the authentication path from leaf at `index` to root, as a Vec of digests
    fn compute_auth_path(&self, index: usize) -> Vec<KeccakDigest> {
        // Get Leaf hash, and leaf sibling hash,
        let leaf_index_in_tree = index + (1 << self.height) - 1;

        let mut path = Vec::with_capacity(self.height);
        // Iterate from the bottom layer after the leaves, to the top, storing all sibling node's hash values.
        let mut current_node = leaf_index_in_tree;
        while current_node > 0 {
            let sibling_node = if current_node % 2 == 1 {
                current_node + 1
            } else {
                current_node - 1
            };
            path.push(self.nodes[sibling_node].clone());
            current_node = (current_node - 1) >> 1;
        }
        assert_eq!(path.len(), self.height);

        // we want to make path from root to bottom
        path.reverse();
        path
    }

    pub fn generate_multi_proof(&self, mut indexes: Vec<usize>) -> MultiPath {
        // pruned and sorted for encoding efficiency

        assert_eq!(
            indexes.len(),
            BTreeSet::from_iter(indexes.clone()).len(),
            "duplicates"
        );
        assert!(indexes.iter().all(|&i| i < 1 << self.height));

        let mut permutation = (0..indexes.len()).collect::<Vec<_>>();
        permutation.sort_by_key(|&i| indexes[i]);

        indexes.sort();

        let mut auth_paths_suffixes: Vec<Vec<KeccakDigest>> = Vec::with_capacity(indexes.len());
        let mut prev_path = Vec::new();

        for &index in &indexes {
            let path = self.compute_auth_path(index);
            if prev_path.is_empty() {
                // first index
                auth_paths_suffixes.push(path.clone());
            } else {
                // prefix_length is the biggest integer such that prev_path[..i] == path[..i]
                let prefix_length = path
                    .iter()
                    .zip(&prev_path)
                    .take_while(|(a, b)| a == b)
                    .count();
                auth_paths_suffixes.push(path[(prefix_length + 1).min(path.len())..].to_vec());
            }
            prev_path = path;
        }

        // permute back
        let mut leaf_indexes = vec![0; indexes.len()];
        for i in 0..indexes.len() {
            leaf_indexes[permutation[i]] = indexes[i];
        }

        MultiPath {
            leaf_indexes,
            auth_paths_suffixes,
        }
    }
}
