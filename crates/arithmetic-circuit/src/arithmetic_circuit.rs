use std::{
    collections::HashMap,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
    sync::Arc,
};

#[derive(Debug, Clone, Hash)]
pub enum ArithmeticCircuitComposed<F, N> {
    // TODO Add a Subtraction operation
    Sum((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
    Sub((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
    Product((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
    // Neg(ArithmeticCircuit<F, N>), // TODO
}

#[derive(Debug, Clone, Hash)]
pub enum ArithmeticCircuit<F, N> {
    Scalar(F),
    Node(N),
    Composed(Arc<ArithmeticCircuitComposed<F, N>>),
}

impl<F, N> ArithmeticCircuit<F, N> {
    pub fn new_sum(mut pols: Vec<Self>) -> Self
    where
        F: Sum,
    {
        if pols.iter().all(|x| x.is_scalar()) {
            return ArithmeticCircuit::Scalar(pols.into_iter().map(|x| x.as_scalar()).sum());
        }
        let mut sum = pols.pop().unwrap();
        while !pols.is_empty() {
            sum = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Sum((
                sum,
                pols.pop().unwrap(),
            ))));
        }
        sum
    }

    pub fn new_product(mut pols: Vec<Self>) -> Self
    where
        F: Product,
    {
        if pols.iter().all(|x| x.is_scalar()) {
            return ArithmeticCircuit::Scalar(pols.into_iter().map(|x| x.as_scalar()).product());
        }
        let mut sum = pols.pop().unwrap();
        while !pols.is_empty() {
            sum = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Product((
                sum,
                pols.pop().unwrap(),
            ))));
        }
        sum
    }

    pub fn new_substraction(a: Self, b: Self) -> Self
    where
        F: Sub<Output = F>,
    {
        if a.is_scalar() && b.is_scalar() {
            return ArithmeticCircuit::Scalar(a.as_scalar() - b.as_scalar());
        }
        ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Sub((a, b))))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, ArithmeticCircuit::Scalar(_))
    }

    pub fn as_scalar(self) -> F {
        match self {
            ArithmeticCircuit::Scalar(scalar) => scalar,
            _ => panic!("Expected a scalar"),
        }
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
        FSub: Fn(V, V) -> V,
    >(
        &self,
        f_scalar: &FScalar,
        f_node: &FNode,
        f_prod: &FProd,
        f_sum: &FSum,
        f_sub: &FSub,
    ) -> V {
        fn parse_inner<
            F,
            N,
            V: Clone,
            FScalar: for<'a> Fn(&'a F) -> V,
            FNode: for<'a> Fn(&'a N) -> V,
            FProd: Fn(V, V) -> V,
            FSum: Fn(V, V) -> V,
            FSub: Fn(V, V) -> V,
        >(
            c: &ArithmeticCircuit<F, N>,
            f_scalar: &FScalar,
            f_node: &FNode,
            f_prod: &FProd,
            f_sum: &FSum,
            f_sub: &FSub,
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
                        | ArithmeticCircuitComposed::Sum((left, right))
                        | ArithmeticCircuitComposed::Sub((left, right))) = &**composed;
                        let eval_left = parse_inner(
                            left,
                            f_scalar,
                            f_node,
                            f_prod,
                            f_sum,
                            f_sub,
                            prev_computation,
                        );
                        let eval_right = parse_inner(
                            right,
                            f_scalar,
                            f_node,
                            f_prod,
                            f_sum,
                            f_sub,
                            prev_computation,
                        );
                        let eval = match &**composed {
                            ArithmeticCircuitComposed::Sum(_) => f_sum(eval_left, eval_right),
                            ArithmeticCircuitComposed::Product(_) => f_prod(eval_left, eval_right),
                            ArithmeticCircuitComposed::Sub(_) => f_sub(eval_left, eval_right),
                        };
                        prev_computation.insert(Arc::as_ptr(composed), eval.clone());
                        eval
                    }
                }
            }
        }

        parse_inner(
            self,
            f_scalar,
            f_node,
            f_prod,
            f_sum,
            f_sub,
            &mut HashMap::new(),
        )
    }

    pub fn map_node<N2: Clone, FNode: Fn(&N) -> ArithmeticCircuit<F, N2>>(
        &self,
        f_node: &FNode,
    ) -> ArithmeticCircuit<F, N2>
    where
        F: Clone + Product + Sum + Sub<Output = F>,
    {
        self.parse(
            &|scalar| ArithmeticCircuit::Scalar(scalar.clone()),
            f_node,
            &|left, right| ArithmeticCircuit::new_product(vec![left, right]),
            &|left, right| ArithmeticCircuit::new_sum(vec![left, right]),
            &|left, right| ArithmeticCircuit::new_substraction(left, right),
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

impl<F: Sum, N> Add for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ArithmeticCircuit::new_sum(vec![self, other])
    }
}

impl<F: Product, N> Mul for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        ArithmeticCircuit::new_product(vec![self, other])
    }
}

impl<F: Sub<Output = F>, N> Sub for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new_substraction(self, other)
    }
}

impl<F: Default + Sum, N> AddAssign for ArithmeticCircuit<F, N> {
    fn add_assign(&mut self, other: Self) {
        *self = Self::new_sum(vec![std::mem::take(self), other]);
    }
}

impl<F: Default + Product, N> MulAssign for ArithmeticCircuit<F, N> {
    fn mul_assign(&mut self, other: Self) {
        *self = Self::new_product(vec![std::mem::take(self), other]);
    }
}

impl<F: Default + Sub<Output = F>, N> SubAssign for ArithmeticCircuit<F, N> {
    fn sub_assign(&mut self, other: Self) {
        *self = Self::new_substraction(std::mem::take(self), other);
    }
}

impl<F: Product, N> Product for ArithmeticCircuit<F, N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_product(iter.collect())
    }
}

impl<F: Sum, N> Sum for ArithmeticCircuit<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_sum(iter.collect())
    }
}
