use p3_field::{ExtensionField, Field};

use crate::pols::utils::{max_degree_per_vars_prod, max_degree_per_vars_sum};

use super::{
    ArithmeticCircuit, CircuitComputation, HypercubePoint, MixedFieldElement,
    MultilinearPolynomial, PartialHypercubePoint, TransparentPolynomial,
};

#[derive(Clone, Debug)]
pub struct ComposedPolynomial<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<F> + ExtensionField<NF>,
> {
    pub n_vars: usize,
    pub nodes: Vec<MultilinearPolynomial<NF>>,
    pub structure: TransparentPolynomial<F>, // each var represents a polynomial (stored in "nodes")
    pub max_degree_per_vars: Vec<usize>,
}

impl<F: Field, EF: ExtensionField<F>> ComposedPolynomial<F, EF> {
    pub fn new(
        n_vars: usize,
        small_nodes: Vec<MultilinearPolynomial<F>>,
        large_nodes: Vec<MultilinearPolynomial<EF>>,
        structure: TransparentPolynomial<F, EF>,
    ) -> Self {
        for i in 0..small_nodes.len() {
            assert_eq!(small_nodes[i].n_vars, n_vars);
        }
        for i in 0..large_nodes.len() {
            assert_eq!(large_nodes[i].n_vars, n_vars);
        }
        let n_nodes = small_nodes.len() + large_nodes.len();
        let max_degree_per_vars = max_degree_per_vars(n_vars, n_nodes, &structure);
        let fixed_structure = structure.fix_computation(true);
        Self {
            n_vars,
            small_nodes,
            large_nodes,
            structure: fixed_structure,
            max_degree_per_vars,
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.small_nodes.len() + self.large_nodes.len()
    }

    pub fn max_degree_per_vars(&self) -> &[usize] {
        &self.max_degree_per_vars
    }

    pub fn new_large_product(n_vars: usize, nodes: Vec<MultilinearPolynomial<EF>>) -> Self {
        let structure = (0..nodes.len())
            .map(|i| ArithmeticCircuit::Node(i))
            .product();
        ComposedPolynomial::new(n_vars, vec![], nodes, structure)
    }

    pub fn fix_variable(self, z: EF) -> Self {
        // computes f'(Y, Z, ...) := f(z, Y, Z, ...)
        assert!(self.n_vars >= 1);
        let mut nodes = Vec::with_capacity(self.n_nodes());
        for node in self.small_nodes {
            nodes.push(node.fix_variable(z));
        }
        for node in self.large_nodes {
            nodes.push(node.fix_variable(z));
        }
        ComposedPolynomial {
            n_vars: self.n_vars - 1,
            small_nodes: vec![],
            large_nodes: nodes,
            max_degree_per_vars: self.max_degree_per_vars[1..].to_vec(),
            structure: self.structure,
        }
    }

    pub fn eval(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for small_node in &self.small_nodes {
            nodes_evals.push(small_node.eval(&point));
        }
        for large_node in &self.large_nodes {
            nodes_evals.push(large_node.eval(&point));
        }
        self.structure.eval_at_large(&nodes_evals)
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> EF {
        assert_eq!(point.n_vars, self.n_vars);
        let mut nodes_evals = Vec::new();
        for small_node in &self.small_nodes {
            nodes_evals.push(MixedFieldElement::Small(small_node.eval_hypercube(point)));
        }
        for large_node in &self.large_nodes {
            nodes_evals.push(MixedFieldElement::Large(large_node.eval_hypercube(point)));
        }
        self.structure.eval(&nodes_evals).to_large()
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> EF {
        assert_eq!(point.n_vars(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for small_node in &self.small_nodes {
            nodes_evals.push(MixedFieldElement::Small(
                small_node.eval_partial_hypercube(point),
            ));
        }
        for large_node in &self.large_nodes {
            nodes_evals.push(MixedFieldElement::Large(
                large_node.eval_partial_hypercube(point),
            ));
        }
        self.structure.eval(&nodes_evals).to_large()
    }

    pub fn sum_over_hypercube(&self) -> EF {
        HypercubePoint::iter(self.n_vars)
            .map(|point| self.eval_hypercube(&point))
            .sum()
    }

    pub fn sum_over_partial_hypercube(&self, left: u32) -> EF {
        HypercubePoint::iter(self.n_vars - 1)
            .map(|right| self.eval_partial_hypercube(&PartialHypercubePoint { left, right }))
            .sum()
    }
}

pub fn max_degree_per_vars<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    n_vars: usize,
    n_nodes: usize,
    structure: &TransparentPolynomial<F>,
) -> Vec<usize> {
    let max_degree_per_vars_per_nodes = (0..n_nodes).map(|_| vec![1; n_vars]).collect::<Vec<_>>();
    structure.parse(
        &|_| vec![],
        &|i| max_degree_per_vars_per_nodes[*i].clone(),
        &|left, right| max_degree_per_vars_prod(&vec![left, right]),
        &|left, right| max_degree_per_vars_sum(&vec![left, right]),
    )
}
