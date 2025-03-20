pub mod committer;
pub mod fs_utils;
pub mod parameters;
pub mod prover;
pub mod verifier;

#[derive(Debug, Clone, Default)]
pub struct Statement<F> {
    pub points: Vec<Vec<F>>,
    pub evaluations: Vec<F>,
}
