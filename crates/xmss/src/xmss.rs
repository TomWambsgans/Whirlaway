use rand::Rng;

use crate::*;

pub struct XmssSecretKey {
    pub wots_secret_keys: Vec<WotsSecretKey>,
    pub merkle_tree: Vec<Vec<Digest>>,
}

pub struct XmssSignature {
    pub wots_signature: WotsSignature,
    pub merkle_proof: Vec<(bool, Digest)>,
}

pub struct XmssPublicKey(pub Digest);

impl XmssSecretKey {
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut wots_secret_keys = Vec::new();
        for _ in 0..1 << XMSS_MERKLE_HEIGHT {
            wots_secret_keys.push(WotsSecretKey::random(rng));
        }
        let leaves = wots_secret_keys
            .iter()
            .map(|w| w.public_key().hash())
            .collect::<Vec<_>>();
        let mut merkle_tree = vec![leaves];
        for _ in 0..XMSS_MERKLE_HEIGHT {
            let mut next_level = Vec::new();
            let current_level = merkle_tree.last().unwrap();
            for (left, right) in current_level.chunks(2).map(|chunk| (chunk[0], chunk[1])) {
                next_level.push(poseidon16_compress(&left, &right));
            }
            merkle_tree.push(next_level);
        }
        Self {
            wots_secret_keys,
            merkle_tree,
        }
    }

    pub fn sign(&self, message_hash: &Digest, index: usize, rng: &mut impl Rng) -> XmssSignature {
        assert!(
            index < (1 << XMSS_MERKLE_HEIGHT),
            "Index out of bounds for XMSS signature"
        );
        let wots_signature = self.wots_secret_keys[index].sign(message_hash, rng);
        let mut merkle_proof = Vec::new();
        let mut current_index = index;
        for level in 0..XMSS_MERKLE_HEIGHT {
            let is_left = current_index % 2 == 0;
            let neighbour_index = if is_left {
                current_index + 1
            } else {
                current_index - 1
            };
            let neighbour = self.merkle_tree[level][neighbour_index];
            merkle_proof.push((is_left, neighbour));
            current_index /= 2;
        }
        XmssSignature {
            wots_signature,
            merkle_proof,
        }
    }

    pub fn public_key(&self) -> XmssPublicKey {
        XmssPublicKey(self.merkle_tree.last().unwrap()[0])
    }
}

impl XmssPublicKey {
    pub fn verify(&self, message_hash: &Digest, signature: &XmssSignature) -> bool {
        let Some(wots_public_key) = signature
            .wots_signature
            .recover_public_key(message_hash, &signature.wots_signature)
        else {
            return false;
        };
        // merkle root verification
        let mut current_hash = wots_public_key.hash();
        if signature.merkle_proof.len() != XMSS_MERKLE_HEIGHT {
            return false;
        }
        for (is_left, neighbour) in &signature.merkle_proof {
            if *is_left {
                current_hash = poseidon16_compress(&current_hash, neighbour)
            } else {
                current_hash = poseidon16_compress(neighbour, &current_hash);
            }
        }
        current_hash == self.0
    }
}
