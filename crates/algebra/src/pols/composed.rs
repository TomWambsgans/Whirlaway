use std::ops::Range;

use p3_field::{ExtensionField, Field};

use crate::pols::utils::{max_degree_per_vars_prod, max_degree_per_vars_sum};

use super::{
    ArithmeticCircuit, GenericTransparentMultivariatePolynomial, HypercubePoint,
    MultilinearPolynomial, PartialHypercubePoint, TransparentComputation,
    TransparentMultivariatePolynomial,
};

#[derive(Clone, Debug)]
pub struct ComposedPolynomial<F: Field, EF: ExtensionField<F>> {
    pub n_vars: usize,
    pub nodes: Vec<MultilinearPolynomial<EF>>,
    pub vars_shift: Vec<Range<usize>>,
    pub structure: TransparentComputation<F, EF>, // each var represents a polynomial (stored in "nodes")
    max_degree_per_vars: Vec<usize>,
}

impl<F: Field, EF: ExtensionField<F>> ComposedPolynomial<F, EF> {
    pub fn new(
        n_vars: usize,
        nodes: Vec<MultilinearPolynomial<EF>>,
        vars_shift: Vec<Range<usize>>,
        structure: impl Into<TransparentMultivariatePolynomial<F, EF>>,
    ) -> Self {
        assert_eq!(nodes.len(), vars_shift.len());
        for i in 0..nodes.len() {
            assert_eq!(nodes[i].n_vars(), vars_shift[i].len());
            assert!(nodes[i].n_vars() <= n_vars);
        }
        let structure: TransparentMultivariatePolynomial<F, EF> = structure.into();
        let max_degree_per_vars = max_degree_per_vars(n_vars, &nodes, &vars_shift, &structure);
        let fixed_structure = structure.fix_computation();
        Self {
            n_vars,
            nodes,
            vars_shift,
            structure: fixed_structure,
            max_degree_per_vars,
        }
    }

    pub fn new_without_shift(
        n_vars: usize,
        nodes: Vec<MultilinearPolynomial<EF>>,
        structure: impl Into<TransparentMultivariatePolynomial<F, EF>>,
    ) -> Self {
        let vars_shift = vec![0..n_vars; nodes.len()];
        Self::new(n_vars, nodes, vars_shift, structure)
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        self.max_degree_per_vars.clone()
    }

    pub fn new_product(n_vars: usize, nodes: Vec<MultilinearPolynomial<EF>>) -> Self {
        let circuit = ArithmeticCircuit::new_product(
            (0..nodes.len())
                .map(|i| ArithmeticCircuit::Node(i))
                .collect(),
        );
        let structure = GenericTransparentMultivariatePolynomial::new(circuit, nodes.len());
        ComposedPolynomial::new_without_shift(n_vars, nodes, structure)
    }

    pub fn fix_variable(&mut self, z: EF) {
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

    pub fn eval(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval(&point[self.vars_shift[i].clone()]));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> EF {
        assert_eq!(point.n_vars, self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval_hypercube(&point.crop(self.vars_shift[i].clone())));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint<EF>) -> EF {
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

    pub fn sum_over_hypercube(&self) -> EF {
        self.sum_over_partial_hypercube(EF::ZERO) + self.sum_over_partial_hypercube(EF::ONE)
    }

    pub fn sum_over_partial_hypercube(&self, left: EF) -> EF {
        HypercubePoint::iter(self.n_vars - 1)
            .map(|right| self.eval_partial_hypercube(&PartialHypercubePoint { left, right }))
            .sum()
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<MultilinearPolynomial<EF>> {
        &mut self.nodes
    }
}

pub fn max_degree_per_vars<F: Field, EF: ExtensionField<F>>(
    n_vars: usize,
    nodes: &[MultilinearPolynomial<EF>],
    vars_shift: &[Range<usize>],
    structure: &TransparentMultivariatePolynomial<F, EF>,
) -> Vec<usize> {
    let mut max_degree_per_vars_per_nodes = vec![];
    for i in 0..nodes.len() {
        max_degree_per_vars_per_nodes.push(vec![0; n_vars]);
        max_degree_per_vars_per_nodes[i][vars_shift[i].clone()]
            .copy_from_slice(&nodes[i].max_degree_per_vars());
    }
    match structure {
        TransparentMultivariatePolynomial::Generic(generic) => generic.coefs.parse(
            &|_| vec![],
            &|i| max_degree_per_vars_per_nodes[*i].clone(),
            &|subs| max_degree_per_vars_prod(&subs),
            &|subs| max_degree_per_vars_sum(&subs),
        ),
        TransparentMultivariatePolynomial::Custom(custom) => {
            let mut res = max_degree_per_vars_sum(
                &custom
                    .right
                    .iter()
                    .map(|(_, expr)| {
                        expr.parse(
                            &|_| vec![],
                            &|i| max_degree_per_vars_per_nodes[*i].clone(),
                            &|subs| max_degree_per_vars_prod(&subs),
                            &|subs| max_degree_per_vars_sum(&subs),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            for i in 0..n_vars {
                res[i] += max_degree_per_vars_per_nodes[custom.left][i];
            }
            res
        }
    }
}
