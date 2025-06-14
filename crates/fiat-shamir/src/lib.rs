#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use std::time::Duration;

use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
};
use rayon::prelude::*;

use utils::{KeccakDigest, count_ending_zero_bits, deserialize_field, keccak256, serialize_field};

use p3_field::Field;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsError;

#[derive(Default)]
pub struct FsProver {
    state: KeccakDigest,
    transcript: Vec<u8>,
}

pub struct FsVerifier {
    state: KeccakDigest,
    transcript: Vec<u8>,
    cursor: usize,
}

impl FsProver {
    pub fn new() -> Self {
        Self {
            state: KeccakDigest::default(),
            transcript: Vec::new(),
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn transcript_len(&self) -> usize {
        self.transcript.len()
    }

    fn update_state(&mut self, data: &[u8]) {
        self.state = keccak256(&[&self.state.0, data].concat());
    }

    pub fn add_bytes(&mut self, bytes: &[u8]) {
        self.transcript.extend_from_slice(bytes);
        self.update_state(bytes);
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

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = StdRng::from_seed(self.challenge_bytes(32).try_into().unwrap());
        (0..len).map(|_| rng.random()).collect::<Vec<_>>()
    }

    pub fn challenge_pow(&mut self, bits: usize) {
        assert!(bits < 30, "too much grinding: {bits} bits");
        if bits == 0 {
            return;
        }
        let time = std::time::Instant::now();
        let nonce = (0..u64::MAX)
            .into_par_iter()
            .find_any(|&nonce| {
                let hash = keccak256(&[&self.state.0[..], &nonce.to_be_bytes()].concat());
                count_ending_zero_bits(&hash.0) >= bits
            })
            .expect("Failed to find a nonce");
        let grinding_time = time.elapsed();
        if grinding_time > Duration::from_millis(10) {
            tracing::warn!("long PoW grinding: {} ms", grinding_time.as_millis());
        }

        self.add_bytes(&nonce.to_be_bytes());
    }

    pub fn transcript(self) -> Vec<u8> {
        self.transcript
    }
}

impl FsVerifier {
    pub fn new(transcript: Vec<u8>) -> Self {
        Self {
            state: KeccakDigest::default(),
            transcript,
            cursor: 0,
        }
    }
    fn update_state(&mut self, data: &[u8]) {
        self.state = keccak256(&[&self.state.0, data].concat());
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

    pub fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = StdRng::from_seed(self.challenge_bytes(32).try_into().unwrap());
        (0..len).map(|_| rng.random()).collect::<Vec<_>>()
    }

    pub fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError> {
        if bits == 0 {
            return Ok(());
        }
        let initial_state = self.state.clone();
        let nonce = u64::from_be_bytes(self.next_bytes(8).unwrap().try_into().unwrap());
        if count_ending_zero_bits(
            &keccak256(&[&initial_state.0[..], &nonce.to_be_bytes()].concat()).0,
        ) >= bits
        {
            Ok(())
        } else {
            Err(FsError {})
        }
    }
}

pub trait FsParticipant {
    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>;
    fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError>;
}

impl FsParticipant for FsProver {
    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        Self::challenge_scalars(self, len)
    }

    fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError> {
        Self::challenge_pow(self, bits);
        Ok(())
    }
}

impl FsParticipant for FsVerifier {
    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        Self::challenge_scalars(self, len)
    }

    fn challenge_pow(&mut self, bits: usize) -> Result<(), FsError> {
        Self::challenge_pow(self, bits)
    }
}

fn generate_pseudo_random(seed: &KeccakDigest, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(len);
    let mut counter = 0u32;

    while result.len() < len {
        let hash_result = keccak256(&[&seed.0[..], &counter.to_be_bytes()].concat()).0;
        let bytes_to_take = std::cmp::min(hash_result.len(), len - result.len());
        result.extend_from_slice(&hash_result[..bytes_to_take]);
        counter += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_pow() {
        let mut prover = FsProver::new();
        let time = std::time::Instant::now();
        prover.challenge_pow(12);
        println!("Time: {:?}", time.elapsed());
    }
}
