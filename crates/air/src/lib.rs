#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod builder;
mod prove;
mod setup;
mod table;
mod utils;
mod verify;

#[cfg(test)]
mod test;

pub use builder::*;

const UNIVARIATE_SKIPS: usize = 3;
