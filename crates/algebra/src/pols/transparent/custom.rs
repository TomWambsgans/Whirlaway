use std::fmt::Debug;

use p3_field::{ExtensionField, Field};

use crate::pols::{CircuitComputation, utils::max_degree_per_vars_sum};

use super::{ArithmeticCircuit, GenericTransparentMultivariatePolynomial};

/// a sum of expressions (with scalars in F), each multiplied by a scalar in EF
#[derive(Clone, Debug)]
pub struct CustomTransparentMultivariatePolynomial<F: Field, EF: ExtensionField<F>> {
    pub n_vars: usize,
    pub linear_comb: Vec<(EF, ArithmeticCircuit<F, usize>)>,
}

#[derive(Clone, Debug)]
pub struct CustomComputation<F: Field, EF: ExtensionField<F>>(
    Vec<(EF, CircuitComputation<F, usize>)>,
);

impl<F: Field, EF: ExtensionField<F>> CustomTransparentMultivariatePolynomial<F, EF> {
    pub fn new(n_vars: usize, linear_comb: Vec<(EF, ArithmeticCircuit<F, usize>)>) -> Self {
        Self {
            n_vars,
            linear_comb,
        }
    }

    pub fn eval(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        self.linear_comb
            .iter()
            .map(|(s, expr)| *s * expr.eval_field(&|i| point[*i]))
            .sum::<EF>()
    }

    pub fn fix_computation(&self) -> CustomComputation<F, EF> {
        CustomComputation(
            self.linear_comb
                .iter()
                .map(|(s, expr)| (*s, expr.fix_computation()))
                .collect(),
        )
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        max_degree_per_vars_sum(
            &self
                .linear_comb
                .iter()
                .map(|(_, expr)| {
                    GenericTransparentMultivariatePolynomial::new(expr.clone(), self.n_vars)
                        .max_degree_per_vars()
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<F: Field, EF: ExtensionField<F>> CustomComputation<F, EF> {
    pub fn eval<NF: ExtensionField<F>>(&self, point: &[NF]) -> EF
    where
        EF: ExtensionField<NF>,
    {
        self.0
            .iter()
            .map(|(s, expr)| *s * expr.eval(point))
            .sum::<EF>()
    }
}
