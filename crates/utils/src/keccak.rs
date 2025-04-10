use cudarc::driver::DeviceRepr;
use sha3::{Digest, Keccak256};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeccakDigest(pub [u8; 32]);

unsafe impl DeviceRepr for KeccakDigest {}

impl KeccakDigest {
    pub fn to_string(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
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
