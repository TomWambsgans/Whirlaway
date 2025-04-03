use p3_field::{ExtensionField, Field};

use crate::pols::utils::{max_degree_per_vars_prod, max_degree_per_vars_sum};

use super::{
    ArithmeticCircuit, GenericTransparentMultivariatePolynomial, HypercubePoint,
    MultilinearPolynomial, PartialHypercubePoint, TransparentComputation,
    TransparentMultivariatePolynomial,
};

#[derive(Clone, Debug)]
pub struct ComposedPolynomial<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> = NF,
> {
    pub n_vars: usize,
    pub nodes: Vec<MultilinearPolynomial<NF>>,
    pub structure: TransparentComputation<F, EF>, // each var represents a polynomial (stored in "nodes")
    pub max_degree_per_vars: Vec<usize>,
}

impl<F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>
    ComposedPolynomial<F, NF, EF>
{
    pub fn new(
        n_vars: usize,
        nodes: Vec<MultilinearPolynomial<NF>>,
        structure: impl Into<TransparentMultivariatePolynomial<F, EF>>,
    ) -> Self {
        for i in 0..nodes.len() {
            assert_eq!(nodes[i].n_vars, n_vars);
        }
        let structure: TransparentMultivariatePolynomial<F, EF> = structure.into();
        let max_degree_per_vars = max_degree_per_vars(&nodes, &structure);
        let fixed_structure = structure.fix_computation();
        Self {
            n_vars,
            nodes,
            structure: fixed_structure,
            max_degree_per_vars,
        }
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        self.max_degree_per_vars.clone()
    }

    pub fn new_product(n_vars: usize, nodes: Vec<MultilinearPolynomial<NF>>) -> Self {
        let circuit = ArithmeticCircuit::new_product(
            (0..nodes.len())
                .map(|i| ArithmeticCircuit::Node(i))
                .collect(),
        );
        let structure = GenericTransparentMultivariatePolynomial::new(circuit, nodes.len());
        ComposedPolynomial::new(n_vars, nodes, structure)
    }

    pub fn fix_variable(self, z: EF) -> ComposedPolynomial<F, EF, EF> {
        // computes f'(Y, Z, ...) := f(z, Y, Z, ...)
        assert!(self.n_vars >= 1);
        let mut nodes = Vec::<MultilinearPolynomial<EF>>::with_capacity(self.nodes.len());
        for node in self.nodes.into_iter() {
            nodes.push(node.fix_variable(z));
        }
        ComposedPolynomial {
            n_vars: self.n_vars - 1,
            max_degree_per_vars: self.max_degree_per_vars[1..].to_vec(),
            nodes,
            structure: self.structure,
        }
    }

    pub fn eval(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        let mut nodes_evals = Vec::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval(&point));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> EF {
        assert_eq!(point.n_vars, self.n_vars);
        let mut nodes_evals = Vec::<NF>::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval_hypercube(&point));
        }
        self.structure.eval(&nodes_evals)
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> EF {
        assert_eq!(point.n_vars(), self.n_vars);
        let mut nodes_evals = Vec::<NF>::new();
        for i in 0..self.nodes.len() {
            nodes_evals.push(self.nodes[i].eval_partial_hypercube(point));
        }
        self.structure.eval(&nodes_evals)
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

    pub fn nodes_mut(&mut self) -> &mut Vec<MultilinearPolynomial<NF>> {
        &mut self.nodes
    }
}

pub fn max_degree_per_vars<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    nodes: &[MultilinearPolynomial<NF>],
    structure: &TransparentMultivariatePolynomial<F, EF>,
) -> Vec<usize> {
    let mut max_degree_per_vars_per_nodes = vec![];
    for i in 0..nodes.len() {
        max_degree_per_vars_per_nodes.push(nodes[i].max_degree_per_vars());
    }
    match structure {
        TransparentMultivariatePolynomial::Generic(generic) => generic.coefs.parse(
            &|_| vec![],
            &|i| max_degree_per_vars_per_nodes[*i].clone(),
            &|left, right| max_degree_per_vars_prod(&vec![left, right]),
            &|left, right| max_degree_per_vars_sum(&vec![left, right]),
        ),
        TransparentMultivariatePolynomial::Custom(custom) => max_degree_per_vars_sum(
            &custom
                .linear_comb
                .iter()
                .map(|(_, expr)| {
                    expr.parse(
                        &|_| vec![],
                        &|i| max_degree_per_vars_per_nodes[*i].clone(),
                        &|left, right| max_degree_per_vars_prod(&vec![left, right]),
                        &|left, right| max_degree_per_vars_sum(&vec![left, right]),
                    )
                })
                .collect::<Vec<_>>(),
        ),
    }
}
