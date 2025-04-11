#![cfg_attr(not(test), warn(unused_crate_dependencies))]

// mod prove;
// pub use prove::*;

mod verify;
pub use verify::*;

// mod cuda;
// pub use cuda::*;

mod prove;
pub use prove::*;

#[cfg(test)]
mod test;
