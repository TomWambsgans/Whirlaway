use std::fmt::Debug;

use p3_field::{ExtensionField, Field};

use super::{ArithmeticCircuit, CircuitComputation};

mod custom;
mod generic;

pub use custom::*;
pub use generic::*;

#[derive(Clone, Debug)]
pub enum TransparentMultivariatePolynomial<F: Field, EF: ExtensionField<F>> {
    Generic(GenericTransparentMultivariatePolynomial<F>),
    Custom(CustomTransparentMultivariatePolynomial<F, EF>), // For performance reasons
}

impl<F: Field, EF: ExtensionField<F>> From<GenericTransparentMultivariatePolynomial<F>>
    for TransparentMultivariatePolynomial<F, EF>
{
    fn from(p: GenericTransparentMultivariatePolynomial<F>) -> Self {
        TransparentMultivariatePolynomial::Generic(p)
    }
}

impl<F: Field, EF: ExtensionField<F>> From<CustomTransparentMultivariatePolynomial<F, EF>>
    for TransparentMultivariatePolynomial<F, EF>
{
    fn from(p: CustomTransparentMultivariatePolynomial<F, EF>) -> Self {
        TransparentMultivariatePolynomial::Custom(p)
    }
}

#[derive(Clone, Debug)]
pub enum TransparentComputation<F: Field, EF: ExtensionField<F>> {
    Generic(CircuitComputation<F, usize>),
    Custom(CustomComputation<F, EF>),
}

impl<F: Field, EF: ExtensionField<F>> TransparentMultivariatePolynomial<F, EF> {
    pub fn fix_computation(&self) -> TransparentComputation<F, EF> {
        match self {
            TransparentMultivariatePolynomial::Generic(p) => {
                TransparentComputation::Generic(p.fix_computation())
            }
            TransparentMultivariatePolynomial::Custom(p) => {
                TransparentComputation::Custom(p.fix_computation())
            }
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            TransparentMultivariatePolynomial::Generic(p) => p.n_vars,
            TransparentMultivariatePolynomial::Custom(p) => p.n_vars,
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> TransparentComputation<F, EF> {
    pub fn eval<NF: ExtensionField<F>>(&self, point: &[NF]) -> EF
    where
        EF: ExtensionField<NF>,
    {
        match self {
            TransparentComputation::Generic(c) => EF::from(c.eval(point)),
            TransparentComputation::Custom(c) => c.eval(point),
        }
    }
}
