pub mod committer;
pub mod fs_utils;
pub mod parameters;
pub mod prover;
pub mod verifier;

#[derive(Debug, Clone, Default)]
pub struct Statement<EF> {
    pub points: Vec<Vec<EF>>,
    pub evaluations: Vec<EF>,
}
