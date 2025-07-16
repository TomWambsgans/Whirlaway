use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::Permutation;
use rand::SeedableRng;

type F = KoalaBear;
pub type Digest = [F; 8];

pub fn poseidon16_kb(a: Digest, b: Digest) -> (Digest, Digest) {
    let poseidon_16 =
        Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rand::rngs::StdRng::seed_from_u64(0));
    let mut buff = [F::default(); 16];
    buff[..8].copy_from_slice(&a);
    buff[8..].copy_from_slice(&b);
    poseidon_16.permute_mut(&mut buff);
    let (mut res_a, mut res_b) = ([F::default(); 8], [F::default(); 8]);
    res_a.copy_from_slice(&buff[..8]);
    res_b.copy_from_slice(&buff[8..]);
    (res_a, res_b)
}

pub fn poseidon24_kb(a: Digest, b: Digest, c: Digest) -> (Digest, Digest, Digest) {
    let poseidon_24 =
        Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rand::rngs::StdRng::seed_from_u64(0));
    let mut buff = [F::default(); 24];
    buff[..8].copy_from_slice(&a);
    buff[8..16].copy_from_slice(&b);
    buff[16..].copy_from_slice(&c);
    poseidon_24.permute_mut(&mut buff);
    let (mut res_a, mut res_b, mut res_c) =
        ([F::default(); 8], [F::default(); 8], [F::default(); 8]);
    res_a.copy_from_slice(&buff[..8]);
    res_b.copy_from_slice(&buff[8..16]);
    res_c.copy_from_slice(&buff[16..]);
    (res_a, res_b, res_c)
}
