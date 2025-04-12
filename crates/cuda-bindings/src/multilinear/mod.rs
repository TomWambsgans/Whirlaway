mod basis_convertion;
pub use basis_convertion::*;

mod evaluation;
pub use evaluation::*;

mod custom;
pub use custom::*;

mod operations;
pub use operations::*;

const MULTILINEAR_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const MULTILINEAR_N_THREADS_PER_BLOCK: u32 = 1 << MULTILINEAR_LOG_N_THREADS_PER_BLOCK;
