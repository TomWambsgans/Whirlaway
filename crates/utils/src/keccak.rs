use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;
use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeccakDigest(pub [u8; 32]);

impl fmt::Display for KeccakDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

pub fn keccak256(data: &[u8]) -> KeccakDigest {
    KeccakDigest(Keccak256Hash.hash_slice(data))
}
