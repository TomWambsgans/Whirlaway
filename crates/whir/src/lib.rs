#![cfg_attr(not(test), warn(unused_crate_dependencies))]

pub mod domain; // Domain that we are evaluating over
pub mod parameters;
pub mod poly_utils; // Utils for polynomials
pub mod utils; // Utils in general
pub mod whir; // The real prover
