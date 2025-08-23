#![cfg_attr(not(test), warn(unused_crate_dependencies))]
use p3_koala_bear::KoalaBear;
use p3_symmetric::Permutation;
use utils::{get_poseidon16, get_poseidon24};

mod wots;
pub use wots::*;
mod xmss;
pub use xmss::*;

type F = KoalaBear;
type Digest = [F; 8];

// WOTS
pub const V: usize = 68;
pub const W: usize = 4;
pub const CHAIN_LENGTH: usize = 1 << W;
pub const D: usize = 90;
pub const TARGET_SUM: usize = V * (W - 1) - D;

pub const XMSS_MERKLE_HEIGHT: usize = 5;

fn poseidon16_compress(a: &Digest, b: &Digest) -> Digest {
    get_poseidon16().permute([*a, *b].concat().try_into().unwrap())[0..8]
        .try_into()
        .unwrap()
}

fn poseidon24_compress(a: &Digest, b: &Digest, c: &Digest) -> Digest {
    get_poseidon24().permute([*a, *b, *c].concat().try_into().unwrap())[16..24]
        .try_into()
        .unwrap()
}
