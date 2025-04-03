use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

#[derive(Debug, Clone)]
pub enum ArithmeticCircuitComposed<F, N> {
    Sum((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
    Product((ArithmeticCircuit<F, N>, ArithmeticCircuit<F, N>)),
}

#[derive(Debug, Clone)]
pub enum ArithmeticCircuit<F, N> {
    Scalar(F),
    Node(N),
    Composed(Arc<ArithmeticCircuitComposed<F, N>>),
}

impl<F: Clone, N: Clone> ArithmeticCircuit<F, N> {
    pub fn new_sum(pols: Vec<Self>) -> Self {
        assert!(pols.len() >= 1);
        if pols.len() == 1 {
            return pols[0].clone();
        }
        let mut sum = pols[1].clone();
        for i in 2..pols.len() {
            sum = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Sum((
                sum,
                pols[i].clone(),
            ))));
        }
        Self::Composed(Arc::new(ArithmeticCircuitComposed::Sum((
            pols[0].clone(),
            sum,
        ))))
    }

    pub fn new_product(pols: Vec<Self>) -> Self {
        assert!(pols.len() >= 1);
        if pols.len() == 1 {
            return pols[0].clone();
        }
        let mut product = pols[1].clone();
        for i in 2..pols.len() {
            product = ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Product((
                product,
                pols[i].clone(),
            ))));
        }
        Self::Composed(Arc::new(ArithmeticCircuitComposed::Product((
            pols[0].clone(),
            product,
        ))))
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

    pub fn map_node<EF: ExtensionField<F>, N2: Clone, FNode: Fn(&N) -> ArithmeticCircuit<EF, N2>>(
        &self,
        f_node: &FNode,
    ) -> ArithmeticCircuit<EF, N2>
    where
        F: Field,
    {
        self.parse(
            &|scalar| ArithmeticCircuit::Scalar(EF::from(*scalar)),
            f_node,
            &|left, right| {
                ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Product((
                    left, right,
                ))))
            },
            &|left, right| {
                ArithmeticCircuit::Composed(Arc::new(ArithmeticCircuitComposed::Sum((left, right))))
            },
        )
    }

    pub fn eval_field<EF: ExtensionField<F>, NodeToField: for<'a> Fn(&'a N) -> EF>(
        &self,
        node_to_field: &NodeToField,
    ) -> EF
    where
        F: Field,
    {
        self.parse(
            &|scalar| EF::from(*scalar),
            &|node| node_to_field(node),
            &|left, right| left * right,
            &|left, right| left + right,
        )
    }

    pub fn to_string(&self) -> String
    where
        F: Field,
        N: Debug,
    {
        // Use a hashmap to track unique composed circuits and assign them indices
        let mut composed_indices = HashMap::new();
        let mut next_index = 0;

        // First pass: index all composed circuits
        fn index_composed<F: Field, N: Debug>(
            circuit: &ArithmeticCircuit<F, N>,
            indices: &mut HashMap<*const ArithmeticCircuitComposed<F, N>, usize>,
            next_index: &mut usize,
        ) {
            if let ArithmeticCircuit::Composed(composed) = circuit {
                let ptr = Arc::as_ptr(composed);
                if let Entry::Vacant(e) = indices.entry(ptr) {
                    e.insert(*next_index);
                    *next_index += 1;

                    // Recursively index subcircuits
                    match &**composed {
                        ArithmeticCircuitComposed::Sum((left, right))
                        | ArithmeticCircuitComposed::Product((left, right)) => {
                            for subcircuit in [left, right] {
                                index_composed(subcircuit, indices, next_index);
                            }
                        }
                    }
                }
            }
        }

        index_composed(self, &mut composed_indices, &mut next_index);

        // Second pass: build the string representation
        fn to_string_inner<F: Field, N: Debug>(
            circuit: &ArithmeticCircuit<F, N>,
            indices: &HashMap<*const ArithmeticCircuitComposed<F, N>, usize>,
        ) -> String {
            match circuit {
                ArithmeticCircuit::Scalar(scalar) => format!("scalar({:?})", scalar),
                ArithmeticCircuit::Node(node) => format!("node({:?})", node),
                ArithmeticCircuit::Composed(composed) => {
                    let ptr = Arc::as_ptr(composed);
                    let index = indices.get(&ptr).unwrap();

                    let op_type = match &**composed {
                        ArithmeticCircuitComposed::Sum(_) => "Sum",
                        ArithmeticCircuitComposed::Product(_) => "Product",
                    };

                    // Represent a composed circuit by its index
                    format!("{}#{}", op_type, index)
                }
            }
        }

        // Build the full representation with definitions
        let mut result = String::new();

        // First add the main circuit representation
        result.push_str(&format!(
            "Result: {}\n\n",
            to_string_inner(self, &composed_indices)
        ));

        // Then add definitions for all composed circuits
        if !composed_indices.is_empty() {
            result.push_str("Definitions:\n");

            // Create a map from indices to pointers for sorting
            let mut indices_to_ptr: Vec<(usize, *const ArithmeticCircuitComposed<F, N>)> =
                composed_indices.iter().map(|(k, v)| (*v, *k)).collect();
            indices_to_ptr.sort_by_key(|(idx, _)| *idx);

            for (idx, ptr) in indices_to_ptr {
                // Safety: We know these pointers are valid because they came from Arc
                let composed = unsafe { &*ptr };

                match composed {
                    ArithmeticCircuitComposed::Sum((left, right)) => {
                        result.push_str(&format!("Sum#{} = ", idx));
                        result.push_str(&to_string_inner(left, &composed_indices));
                        result.push_str(" + ");
                        result.push_str(&to_string_inner(right, &composed_indices));
                        result.push('\n');
                    }
                    ArithmeticCircuitComposed::Product((left, right)) => {
                        result.push_str(&format!("Product#{} = ", idx));
                        result.push_str(&to_string_inner(left, &composed_indices));
                        result.push_str(" * ");
                        result.push_str(&to_string_inner(right, &composed_indices));
                        result.push('\n');
                    }
                }
            }
        }

        result
    }
}

impl<F: Field, N: Clone> ArithmeticCircuit<F, N> {
    pub fn embed<EF: ExtensionField<F>>(self) -> ArithmeticCircuit<EF, N> {
        // TODO avoid embed
        self.map_node(&|n| ArithmeticCircuit::Node(n.clone()))
    }
}

impl<F: Field, N> Default for ArithmeticCircuit<F, N> {
    fn default() -> Self {
        Self::from(F::ZERO)
    }
}

impl<F, N> From<F> for ArithmeticCircuit<F, N> {
    fn from(f: F) -> Self {
        ArithmeticCircuit::Scalar(f)
    }
}

impl<F: Clone, N: Clone> Add for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ArithmeticCircuit::new_sum(vec![self, other])
    }
}

impl<F: Clone, N: Clone> Mul for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        ArithmeticCircuit::new_product(vec![self, other])
    }
}

impl<F: Field, N: Clone> Sub for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (other * F::NEG_ONE)
    }
}

impl<F: Field, N: Clone> Add<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn add(self, other: F) -> Self {
        self + Self::from(other)
    }
}

impl<F: Field, N: Clone> Mul<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn mul(self, other: F) -> Self {
        self * Self::from(other)
    }
}

impl<F: Field, N: Clone> Sub<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn sub(self, other: F) -> Self {
        self + (-other)
    }
}

impl<F: Field, N: Clone> Neg for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn neg(self) -> Self {
        self * F::NEG_ONE
    }
}

macro_rules! impl_assign_ops {
    ($($op_trait:ident, $op_fn:ident, $op:tt);*) => {
        $(
            impl<F: Field, N: Clone> $op_trait<ArithmeticCircuit<F, N>> for ArithmeticCircuit<F, N> {
                fn $op_fn(&mut self, other: Self) {
                    *self = std::mem::take(self) $op other;
                }
            }
        )*
    };
}

impl_assign_ops!(
    AddAssign, add_assign, +;
    SubAssign, sub_assign, -;
    MulAssign, mul_assign, *
);

macro_rules! impl_field_assign_ops {
    ($($op_trait:ident, $op_fn:ident, $op:tt);*) => {
        $(
            impl<F: Field, N: Clone> $op_trait<F> for ArithmeticCircuit<F, N> {
                fn $op_fn(&mut self, other: F) {
                    *self = std::mem::take(self) $op other;
                }
            }
        )*
    };
}

impl_field_assign_ops!(
    AddAssign, add_assign, +;
    MulAssign, mul_assign, *;
    SubAssign, sub_assign, -
);

impl<F: Field, N: Clone> Product for ArithmeticCircuit<F, N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_product(iter.collect())
    }
}

impl<F: Field, N: Clone> Sum for ArithmeticCircuit<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new_sum(iter.collect())
    }
}

impl<F: Field, N: Clone + Debug> PrimeCharacteristicRing for ArithmeticCircuit<F, N> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::Scalar(F::ZERO);
    const ONE: Self = Self::Scalar(F::ONE);
    const NEG_ONE: Self = Self::Scalar(F::NEG_ONE);
    const TWO: Self = Self::Scalar(F::TWO);

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        ArithmeticCircuit::Scalar(F::from_prime_subfield(f))
    }
}

impl<F: Field, N: Clone + Debug> Algebra<F> for ArithmeticCircuit<F, N> {}
