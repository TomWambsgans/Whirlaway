use std::fmt::Debug;

use whir_p3::poly::multilinear::MultilinearPoint;

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    pub point: MultilinearPoint<F>,
    pub value: F,
}
