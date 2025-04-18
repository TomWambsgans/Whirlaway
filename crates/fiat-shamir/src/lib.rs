#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use tracing::instrument;
use utils::{deserialize_field, serialize_field};

use p3_field::Field;
use sha3::{Digest, Keccak256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsError;

pub struct FsProver {
    state: [u8; 32],
    transcript: Vec<u8>,
}

pub struct FsVerifier {
    state: [u8; 32],
    transcript: Vec<u8>,
    cursor: usize,
}

impl FsProver {
    pub fn new() -> Self {
        FsProver {
            state: [0u8; 32],
            transcript: Vec::new(),
        }
    }

    pub fn state_hex(&self) -> String {
        self.state.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn transcript_len(&self) -> usize {
        self.transcript.len()
    }

    fn update_state(&mut self, data: &[u8]) {
        self.state = hash_sha3(&[&self.state[..], data].concat());
    }

    pub fn add_bytes(&mut self, bytes: &[u8]) {
        self.transcript.extend_from_slice(bytes);
        self.update_state(bytes);
    }

    pub fn add_variable_bytes(&mut self, bytes: &[u8]) {
        self.add_bytes(&(bytes.len() as u32).to_le_bytes());
        self.add_bytes(bytes);
    }

    pub fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        let challenge = generate_pseudo_random(&self.state, len);
        self.update_state(len.to_be_bytes().as_ref());
        challenge
    }

    pub fn add_scalars<F: Field>(&mut self, scalars: &[F]) {
        for scalar in scalars {
            self.add_bytes(&serialize_field(scalar));
        }
    }

    pub fn add_scalar_matrix<F: Field>(&mut self, scalars: &[Vec<F>], fixed_dims: bool) {
        let n = scalars.len();
        let m = scalars[0].len();
        assert!(scalars.iter().all(|v| v.len() == m));
        if !fixed_dims {
            self.add_bytes(&(n as u32).to_le_bytes());
            self.add_bytes(&(m as u32).to_le_bytes());
        }
        for row in scalars {
            for scalar in row {
                self.add_bytes(&serialize_field(scalar));
            }
        }
    }

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        let mut rng = StdRng::from_seed(self.challenge_bytes(32).try_into().unwrap());
        (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>()
    }

    #[instrument(name = "Fiat SHamir pow", skip(self))]
    pub fn challenge_pow(&mut self, bits: usize) {
        let nonce = (0..u64::MAX)
            .into_par_iter()
            .find_any(|&nonce| {
                let hash = hash_sha3(&[&self.state[..], &nonce.to_le_bytes()].concat());
                count_ending_zero_bits(&hash) >= bits
            })
            .expect("Failed to find a nonce");
        self.add_bytes(&nonce.to_le_bytes())
    }

    pub fn transcript(self) -> Vec<u8> {
        self.transcript
    }
}

impl FsVerifier {
    pub fn new(transcript: Vec<u8>) -> Self {
        FsVerifier {
            state: [0u8; 32],
            transcript,
            cursor: 0,
        }
    }
    fn update_state(&mut self, data: &[u8]) {
        self.state = hash_sha3(&[&self.state[..], data].concat());
    }

    pub fn state_hex(&self) -> String {
        self.state.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn next_bytes(&mut self, len: usize) -> Result<Vec<u8>, FsError> {
        // take the len last bytes from the transcript
        if len + self.cursor > self.transcript.len() {
            return Err(FsError {});
        }
        let bytes = self.transcript[self.cursor..self.cursor + len].to_vec();
        self.cursor += len;
        self.update_state(&bytes);
        Ok(bytes)
    }

    pub fn next_variable_bytes(&mut self) -> Result<Vec<u8>, FsError> {
        let len = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
        self.next_bytes(len)
    }

    pub fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        let challenge = generate_pseudo_random(&self.state, len);
        self.update_state(len.to_be_bytes().as_ref());
        challenge
    }

    pub fn next_scalars<F: Field>(&mut self, len: usize) -> Result<Vec<F>, FsError> {
        let mut res = Vec::new();
        for _ in 0..len {
            let bytes = self.next_bytes(std::mem::size_of::<F>())?;
            res.push(deserialize_field(&bytes).ok_or(FsError {})?);
        }
        Ok(res)
    }

    pub fn next_scalar_matrix<F: Field>(
        &mut self,
        dims: Option<(usize, usize)>,
    ) -> Result<Vec<Vec<F>>, FsError> {
        let (n, m) = match dims {
            Some((n, m)) => (n, m),
            None => {
                let n = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
                let m = u32::from_le_bytes(self.next_bytes(4)?.try_into().unwrap()) as usize;
                (n, m)
            }
        };
        let mut res = Vec::new();
        for _ in 0..n {
            let mut row = Vec::new();
            for _ in 0..m {
                let bytes = self.next_bytes(std::mem::size_of::<F>())?;
                row.push(deserialize_field(&bytes).ok_or(FsError {})?);
            }
            res.push(row);
        }
        Ok(res)
    }

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        let mut rng = StdRng::from_seed(self.challenge_bytes(32).try_into().unwrap());
        (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>()
    }

    pub fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError> {
        let initial_state = self.state;
        let nonce = u64::from_le_bytes(self.next_bytes(8).unwrap().try_into().unwrap());
        if count_ending_zero_bits(&hash_sha3(
            &[&initial_state[..], &nonce.to_le_bytes()].concat(),
        )) >= bits
        {
            Ok(())
        } else {
            Err(FsError {})
        }
    }
}

pub trait FsParticipant {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8>;
    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>;
}

impl FsParticipant for FsProver {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        FsProver::challenge_bytes(self, len)
    }

    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        FsProver::challenge_scalars(self, len)
    }
}

impl FsParticipant for FsVerifier {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8> {
        FsVerifier::challenge_bytes(self, len)
    }

    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F> {
        FsVerifier::challenge_scalars(self, len)
    }
}

fn hash_sha3(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

pub fn generate_pseudo_random(seed: &[u8; 32], len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(len);
    let mut counter = 0u32;

    while result.len() < len {
        let mut hasher = Keccak256::default();
        hasher.update(seed);
        hasher.update(&counter.to_be_bytes());
        let hash_result = hasher.finalize();
        let bytes_to_take = std::cmp::min(hash_result.len(), len - result.len());
        result.extend_from_slice(&hash_result[..bytes_to_take]);
        counter += 1;
    }

    result
}

fn count_ending_zero_bits(buff: &[u8]) -> usize {
    let mut count = 0;
    'outer: for byte in buff {
        for i in 0..8 {
            if byte & (1 << i) == 0 {
                count += 1;
            } else {
                break 'outer;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_pow() {
        let mut prover = FsProver::new();
        let time = std::time::Instant::now();
        prover.challenge_pow(16);
        println!("Time: {:?}", time.elapsed());
    }
}
