use std::ops::Range;

use p3_field::Field;

use crate::pols::utils::{max_degree_per_vars_prod, max_degree_per_vars_sum};

use super::{
    ArithmeticCircuit, HypercubePoint, MultilinearPolynomial, PartialHypercubePoint,
    TransparentMultivariatePolynomial,
};

#[derive(Clone, Debug)]
pub struct ComposedPolynomial<F: Field> {
    pub n_vars: usize,
    pub nodes: Vec<MultilinearPolynomial<F>>,
    pub vars_shift: Vec<Range<usize>>,
    pub structure: TransparentMultivariatePolynomial<F>, // each var represents a polynomial (stored in "nodes")
}

impl<F: Field> ComposedPolynomial<F> {
    pub fn new(
        n_vars: usize,
        nodes: Vec<MultilinearPolynomial<F>>,
        vars_shift: Vec<Range<usize>>,
        structure: TransparentMultivariatePolynomial<F>,
    ) -> Self {
        assert_eq!(nodes.len(), vars_shift.len());
        for i in 0..nodes.len() {
            assert_eq!(nodes[i].n_vars(), vars_shift[i].len());
            assert!(nodes[i].n_vars() <= n_vars);
        }
        Self {
            n_vars,
            nodes,
            vars_shift,
            structure,
        }
    }

    pub fn new_without_shift(
        n_vars: usize,
        nodes: Vec<MultilinearPolynomial<F>>,
        structure: TransparentMultivariatePolynomial<F>,
    ) -> Self {
        let vars_shift = vec![0..n_vars; nodes.len()];
        Self {
            n_vars,
            nodes,
            vars_shift,
            structure,
        }
    }

    pub fn new_product(n_vars: usize, nodes: Vec<MultilinearPolynomial<F>>) -> Self {
        let circuit = ArithmeticCircuit::new_product(
            (0..nodes.len())
                .map(|i| ArithmeticCircuit::Node(i))
                .collect(),
        );
        let structure = TransparentMultivariatePolynomial::new(circuit, nodes.len());
        ComposedPolynomial::new_without_shift(n_vars, nodes, structure)
    }

    pub fn fix_variable(&mut self, z: F) {
        // computes f'(Y, Z, ...) := f(z, Y, Z, ...)
        // TODO clean the cirsuit (structure) ?

        assert!(self.n_vars >= 1);
        for i in 0..self.nodes.len() {
            let var_shift = &self.vars_shift[i];
            if var_shift.start == 0 && var_shift.end > 0 {
                self.nodes[i].fix_variable(z);
            }
            self.vars_shift[i] = var_shift.start.saturating_sub(1)..var_shift.end.saturating_sub(1);
        }
        self.n_vars -= 1;
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        let mut max_degree_per_vars_per_nodes = vec![];
        for i in 0..self.nodes.len() {
            max_degree_per_vars_per_nodes.push(vec![0; self.n_vars]);
            max_degree_per_vars_per_nodes[i][self.vars_shift[i].clone()]
                .copy_from_slice(&self.nodes[i].max_degree_per_vars());
        }
        self.structure.coefs.parse(
            &|_| vec![],
            &|i| max_degree_per_vars_per_nodes[*i].clone(),
            &|subs| max_degree_per_vars_prod(&subs),
            &|subs| max_degree_per_vars_sum(&subs),
        )
    }

    pub fn eval(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval(&point[self.vars_shift[i].clone()]));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> F {
        assert_eq!(point.n_vars, self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval_hypercube(&point.crop(self.vars_shift[i].clone())));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint<F>) -> F {
        assert_eq!(point.n_vars(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            let var_shift = self.vars_shift[i].clone();
            if var_shift.is_empty() {
                nodes_evals.push(self.nodes[i].eval(&[]));
            } else if var_shift.start == 0 {
                nodes_evals.push(
                    self.nodes[i].eval_partial_hypercube(&PartialHypercubePoint {
                        left: point.left,
                        right: point.right.crop(0..var_shift.end - 1),
                    }),
                );
            } else {
                nodes_evals.push(self.nodes[i].eval_hypercube(&HypercubePoint::from_vec(
                    &point.right.to_vec::<F>()[var_shift.start - 1..var_shift.end - 1],
                )));
            }
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn sum_over_hypercube(&self) -> F {
        self.sum_over_partial_hypercube(F::ZERO) + self.sum_over_partial_hypercube(F::ONE)
    }

    pub fn sum_over_partial_hypercube(&self, left: F) -> F {
        HypercubePoint::iter(self.n_vars - 1)
            .map(|right| self.eval_partial_hypercube(&PartialHypercubePoint { left, right }))
            .sum()
    }

    pub fn as_product_mut(&mut self) -> Option<&mut Vec<MultilinearPolynomial<F>>> {
        if self.structure.coefs.is_product() {
            Some(&mut self.nodes)
        } else {
            None
        }
    }
}
