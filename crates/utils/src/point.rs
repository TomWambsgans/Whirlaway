use std::fmt::{ Debug, };

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    pub point: Vec<F>,
    pub value: F,
}
