use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeccakDigest(pub [u8; 32]);

pub fn keccak256(data: &[u8]) -> KeccakDigest {
    KeccakDigest(Keccak256Hash.hash_slice(data))
}
