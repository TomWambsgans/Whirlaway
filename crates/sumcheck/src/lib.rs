#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod prove;
pub use prove::*;

mod verify;
pub use verify::*;

#[cfg(test)]
mod test;
