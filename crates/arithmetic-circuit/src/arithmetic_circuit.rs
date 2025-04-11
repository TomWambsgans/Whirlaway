use std::{
    collections::HashMap,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

#[derive(Debug, Clone, Hash)]
pub enum ArithmeticCircuitComposed<F, N> {
    // TODO Add a Subtraction operation
    Sum((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
    Product((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
}

#[derive(Debug, Clone, Hash)]
pub enum ArithmeticCircuit<F, N> {
    Scalar(F),
    Node(N),
    Composed(Arc<ArithmeticCircuitComposed<F, N>>),
}

impl<F, N> ArithmeticCircuit<F, N> {
    pub fn new_sum(mut pols: Vec<Self>) -> Self {
        let mut sum = pols.pop().unwrap();
        while !pols.is_empty() {
            sum = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Sum((
                sum,
                pols.pop().unwrap(),
            ))));
        }
        sum
    }

    pub fn new_product(mut pols: Vec<Self>) -> Self {
        let mut sum = pols.pop().unwrap();
        while !pols.is_empty() {
            sum = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Product((
                sum,
                pols.pop().unwrap(),
            ))));
        }
        sum
    }

    // all functions should be pure, and deterministic
    // TODO optimize (idealy the values in the hashmap should be references)
    // a function calling parse should not call itself recursively in the closures
    pub fn parse<
        V: Clone,
        FScalar: for<'a> Fn(&'a F) -> V,
        FNode: for<'a> Fn(&'a N) -> V,
        FProd: Fn(V, V) -> V,
        FSum: Fn(V, V) -> V,
    >(
        &self,
        f_scalar: &FScalar,
        f_node: &FNode,
        f_prod: &FProd,
        f_sum: &FSum,
    ) -> V {
        fn parse_inner<
            F,
            N,
            V: Clone,
            FScalar: for<'a> Fn(&'a F) -> V,
            FNode: for<'a> Fn(&'a N) -> V,
            FProd: Fn(V, V) -> V,
            FSum: Fn(V, V) -> V,
        >(
            c: &ArithmeticCircuit<F, N>,
            f_scalar: &FScalar,
            f_node: &FNode,
            f_prod: &FProd,
            f_sum: &FSum,
            prev_computation: &mut HashMap<*const ArithmeticCircuitComposed<F, N>, V>,
        ) -> V {
            match c {
                ArithmeticCircuit::Scalar(scalar) => f_scalar(scalar),
                ArithmeticCircuit::Node(node) => f_node(node),
                ArithmeticCircuit::Composed(composed) => {
                    if let Some(val) = prev_computation.get(&Arc::as_ptr(composed)) {
                        val.clone()
                    } else {
                        let (ArithmeticCircuitComposed::Product((left, right))
                        | ArithmeticCircuitComposed::Sum((left, right))) = &**composed;
                        let eval_left =
                            parse_inner(left, f_scalar, f_node, f_prod, f_sum, prev_computation);
                        let eval_right =
                            parse_inner(right, f_scalar, f_node, f_prod, f_sum, prev_computation);
                        let eval = match &**composed {
                            ArithmeticCircuitComposed::Sum(_) => f_sum(eval_left, eval_right),
                            ArithmeticCircuitComposed::Product(_) => f_prod(eval_left, eval_right),
                        };
                        prev_computation.insert(Arc::as_ptr(composed), eval.clone());
                        eval
                    }
                }
            }
        }

        parse_inner(self, f_scalar, f_node, f_prod, f_sum, &mut HashMap::new())
    }

    pub fn map_node<N2: Clone, FNode: Fn(&N) -> ArithmeticCircuit<F, N2>>(
        &self,
        f_node: &FNode,
    ) -> ArithmeticCircuit<F, N2>
    where
        F: Clone,
    {
        self.parse(
            &|scalar| ArithmeticCircuit::Scalar(scalar.clone()),
            f_node,
            &|left, right| ArithmeticCircuit::new_product(vec![left, right]),
            &|left, right| ArithmeticCircuit::new_sum(vec![left, right]),
        )
    }
}

impl<F: Default, N> Default for ArithmeticCircuit<F, N> {
    fn default() -> Self {
        ArithmeticCircuit::Scalar(F::default())
    }
}

impl<F, N> From<F> for ArithmeticCircuit<F, N> {
    fn from(scalar: F) -> Self {
        ArithmeticCircuit::Scalar(scalar)
    }
}

impl<F, N> Add for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ArithmeticCircuit::new_sum(vec![self, other])
    }
}

impl<F, N> Mul for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        ArithmeticCircuit::new_product(vec![self, other])
    }
}

impl<F: Default, N> AddAssign for ArithmeticCircuit<F, N> {
    fn add_assign(&mut self, other: Self) {
        *self = Self::new_sum(vec![std::mem::take(self), other]);
    }
}

impl<F: Default, N> MulAssign for ArithmeticCircuit<F, N> {
    fn mul_assign(&mut self, other: Self) {
        *self = Self::new_product(vec![std::mem::take(self), other]);
    }
}

impl<F, N> Product for ArithmeticCircuit<F, N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_product(iter.collect())
    }
}

impl<F, N> Sum for ArithmeticCircuit<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_sum(iter.collect())
    }
}
