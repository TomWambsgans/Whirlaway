use sha3::{Digest, Keccak256};
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
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    KeccakDigest(output)
}
