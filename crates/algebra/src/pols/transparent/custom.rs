use std::fmt::Debug;

use p3_field::{ExtensionField, Field};

use crate::pols::{CircuitComputation, utils::max_degree_per_vars_sum};

use super::{ArithmeticCircuit, GenericTransparentMultivariatePolynomial};

/// Product of 2 factors
/// Left = one variable
/// Right = a sum of expressions (with scalars in F), each multiplied by a scalar in EF
#[derive(Clone, Debug)]
pub struct CustomTransparentMultivariatePolynomial<F: Field, EF: ExtensionField<F>> {
    pub n_vars: usize,
    pub left: usize,
    pub right: Vec<(EF, ArithmeticCircuit<F, usize>)>,
}

#[derive(Clone, Debug)]
pub struct CustomComputation<F: Field, EF: ExtensionField<F>> {
    left: usize,
    right: Vec<(EF, CircuitComputation<F, usize>)>,
}

impl<F: Field, EF: ExtensionField<F>> CustomTransparentMultivariatePolynomial<F, EF> {
    pub fn new(n_vars: usize, left: usize, right: Vec<(EF, ArithmeticCircuit<F, usize>)>) -> Self {
        assert!(left < n_vars);
        Self {
            n_vars,
            left,
            right,
        }
    }

    pub fn eval(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        point[self.left]
            * self
                .right
                .iter()
                .map(|(s, expr)| *s * expr.eval_field(&|i| point[*i]))
                .sum::<EF>()
    }

    pub fn fix_computation(&self) -> CustomComputation<F, EF> {
        CustomComputation {
            left: self.left,
            right: self
                .right
                .iter()
                .map(|(s, expr)| (*s, expr.fix_computation()))
                .collect(),
        }
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        let mut res = max_degree_per_vars_sum(
            &self
                .right
                .iter()
                .map(|(_, expr)| {
                    GenericTransparentMultivariatePolynomial::new(expr.clone(), self.n_vars)
                        .max_degree_per_vars()
                })
                .collect::<Vec<_>>(),
        );
        res[self.left] += 1;
        res
    }
}

impl<F: Field, EF: ExtensionField<F>> CustomComputation<F, EF> {
    pub fn eval(&self, point: &[EF]) -> EF {
        point[self.left]
            * self
                .right
                .iter()
                .map(|(s, expr)| *s * expr.eval(point))
                .sum::<EF>()
    }
}
