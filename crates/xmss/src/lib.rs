#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use p3_koala_bear::KoalaBear;
use rand::Rng;
use utils::{poseidon16_kb, poseidon24_kb};

type F = KoalaBear;
pub type Digest = [F; 8];
pub type Message = [u8; N_CHAINS]; // each value is < CHAIN_LOG_LENGTH

pub const N_CHAINS: usize = 64;
pub const CHAIN_LOG_LENGTH: usize = 3;
pub const CHAIN_LENGTH: usize = 1 << CHAIN_LOG_LENGTH;

pub const XMSS_MERKLE_HEIGHT: usize = 5;

pub struct WotsSecretKey {
    pre_images: [Digest; N_CHAINS],
    public_key: WotsPublicKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WotsPublicKey(pub [Digest; N_CHAINS]);
pub struct WotsSignature(pub [Digest; N_CHAINS]);

impl WotsSecretKey {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut pre_images = [Default::default(); N_CHAINS];
        for i in 0..N_CHAINS {
            let mut pre_image = [F::default(); 8];
            for j in 0..8 {
                pre_image[j] = rng.random();
            }
            pre_images[i] = pre_image;
        }
        Self::new(pre_images)
    }

    pub fn new(pre_images: [Digest; N_CHAINS]) -> Self {
        let mut public_key = [Default::default(); N_CHAINS];
        for i in 0..N_CHAINS {
            public_key[i] = iterate_hash(&pre_images[i], CHAIN_LENGTH);
        }
        Self {
            pre_images,
            public_key: WotsPublicKey(public_key),
        }
    }

    pub fn public_key(&self) -> &WotsPublicKey {
        &self.public_key
    }

    pub fn sign(&self, message: &Message) -> WotsSignature {
        let mut signature = [Default::default(); N_CHAINS];
        for i in 0..N_CHAINS {
            assert!(
                (message[i] as usize) < CHAIN_LENGTH,
                "Message value out of bounds"
            );
            signature[i] = iterate_hash(&self.pre_images[i], message[i] as usize);
        }
        WotsSignature(signature)
    }
}

impl WotsSignature {
    pub fn recover_public_key(
        &self,
        message: &Message,
        signature: &WotsSignature,
    ) -> WotsPublicKey {
        let mut public_key = [Default::default(); N_CHAINS];
        for i in 0..N_CHAINS {
            assert!(
                (message[i] as usize) < CHAIN_LENGTH,
                "Message value out of bounds"
            );
            public_key[i] = iterate_hash(&signature.0[i], CHAIN_LENGTH - message[i] as usize);
        }
        WotsPublicKey(public_key)
    }
}

impl WotsPublicKey {
    pub fn hash(&self) -> Digest {
        assert!(N_CHAINS % 2 == 0, "TODO");
        let mut digest = Default::default();
        for (a, b) in self.0.chunks(2).map(|chunk| (chunk[0], chunk[1])) {
            digest = poseidon24_kb(a, b, digest).0;
        }
        digest
    }
}

pub struct XmssSecretKey {
    pub wots_secret_keys: Vec<WotsSecretKey>,
    pub merkle_tree: Vec<Vec<Digest>>,
}

pub struct XmssSignature {
    pub wots_signature: WotsSignature,
    pub merkle_proof: Vec<(bool, Digest)>,
}

pub struct XmssPublicKey {
    pub root: Digest,
}

impl XmssSecretKey {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
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
                next_level.push(poseidon16_kb(left, right).0);
            }
            merkle_tree.push(next_level);
        }
        Self {
            wots_secret_keys,
            merkle_tree,
        }
    }

    pub fn sign(&self, message: &Message, index: usize) -> XmssSignature {
        assert!(
            index < (1 << XMSS_MERKLE_HEIGHT),
            "Index out of bounds for XMSS signature"
        );
        let wots_signature = self.wots_secret_keys[index].sign(message);
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
        XmssPublicKey {
            root: self.merkle_tree.last().unwrap()[0],
        }
    }
}

impl XmssPublicKey {
    pub fn verify(&self, message: &Message, signature: &XmssSignature) -> bool {
        let wots_public_key = signature
            .wots_signature
            .recover_public_key(message, &signature.wots_signature);
        // merkle root verification
        let mut current_hash = wots_public_key.hash();
        if signature.merkle_proof.len() != XMSS_MERKLE_HEIGHT {
            return false;
        }
        for (is_left, neighbour) in &signature.merkle_proof {
            if *is_left {
                current_hash = poseidon16_kb(current_hash, *neighbour).0;
            } else {
                current_hash = poseidon16_kb(*neighbour, current_hash).0;
            }
        }
        current_hash == self.root
    }
}

fn iterate_hash(a: &Digest, n: usize) -> Digest {
    let mut res = *a;
    for _ in 0..n {
        res = poseidon16_kb(res, Default::default()).0;
    }
    res
}

pub fn random_message<R: Rng>(rng: &mut R) -> Message {
    let mut message = [0u8; N_CHAINS];
    for i in 0..N_CHAINS {
        message[i] = rng.random_range(0..CHAIN_LENGTH) as u8;
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_wots_signature() {
        let mut rng = StdRng::seed_from_u64(0);
        let sk = WotsSecretKey::random(&mut rng);
        let message = random_message(&mut rng);
        let signature = sk.sign(&message);
        assert_eq!(
            signature.recover_public_key(&message, &signature),
            *sk.public_key()
        );
    }

    #[test]
    fn test_xmss_signature() {
        let mut rng = StdRng::seed_from_u64(0);
        let sk = XmssSecretKey::random(&mut rng);
        let message = random_message(&mut rng);
        let index = rng.random_range(0..(1 << XMSS_MERKLE_HEIGHT));
        let signature = sk.sign(&message, index);
        let public_key = sk.public_key();
        assert!(public_key.verify(&message, &signature));
    }
}
