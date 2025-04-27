use super::ArithmeticCircuit;
use p3_field::{Algebra, Field, PrimeCharacteristicRing};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub type TransparentPolynomial<F> = ArithmeticCircuit<F, usize>;

impl<F: Field> TransparentPolynomial<F> {
    pub fn eq_extension_n_scalars(scalars: &[F]) -> Self {
        // eq(scalars, Xs) = ((scalars[0] X0 + (scalars[0] - 1) (X0 - 1)) * ((scalars[1] X1 + (scalars[1] - 1) (X1 - 1)) ...
        let left = (0..scalars.len())
            .map(|i| ArithmeticCircuit::Node(i))
            .collect::<Vec<_>>();
        let right = (0..scalars.len())
            .map(|i| ArithmeticCircuit::Scalar(scalars[i]))
            .collect::<Vec<_>>();
        Self::_eq_extension(&left, &right)
    }

    pub fn eq_extension_2n_vars(n: usize) -> Self {
        // eq(Xs, Ys) = ((Y0 X0 + (Y0 - 1) (X0 - 1)) * ((Y1 X1 + (Y1 - 1) (X1 - 1)) ...
        let left = (0..n)
            .map(|i| ArithmeticCircuit::Node(i))
            .collect::<Vec<_>>();
        let right = (0..n)
            .map(|i| ArithmeticCircuit::Node(i + n))
            .collect::<Vec<_>>();
        Self::_eq_extension(&left, &right)
    }

    pub fn next(n: usize) -> Self {
        // returns a polynomial P in 2n vars, where P(x, y) = 1 iif y = x + 1 in big indian (both numbers are n bits)
        let factor = |l, r| ArithmeticCircuit::Node(l) * (-ArithmeticCircuit::Node(r) + F::ONE);
        let g = |k| {
            let mut factors = vec![];
            for i in (n - k)..n {
                factors.push(factor(i, i + n));
            }
            factors.push(factor(2 * n - 1 - k, n - 1 - k));
            if k < n - 1 {
                factors.push(Self::_eq_extension(
                    &(0..n - k - 1)
                        .map(|i| ArithmeticCircuit::Node(i))
                        .collect::<Vec<_>>(),
                    &(0..n - k - 1)
                        .map(|i| ArithmeticCircuit::Node(i + n))
                        .collect::<Vec<_>>(),
                ));
            }
            ArithmeticCircuit::new_product(factors)
        };
        (0..n).map(g).sum()
    }

    fn _eq_extension(left: &[Self], right: &[Self]) -> Self {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(l, r)| (l.clone() * r.clone()) + ((-l.clone() + F::ONE) * (-r.clone() + F::ONE)))
            .product()
    }

    pub fn max_degree_per_vars(&self, n_vars: usize) -> Vec<usize> {
        self.parse(
            &|_| vec![0; n_vars],
            &|i| {
                let mut res = vec![0; n_vars];
                res[*i] = 1;
                res
            },
            &|left, right| max_degree_per_vars_prod(&vec![left, right]),
            &|left, right| max_degree_per_vars_sum(&vec![left, right]),
            &|left, right| max_degree_per_vars_sum(&vec![left, right]),
        )
    }

    // usefull for tests
    pub fn random<R: Rng>(rng: &mut R, n_nodes: usize, depth: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        let mut circuit = ArithmeticCircuit::Node(0);
        let rand_f = |rng: &mut R| -> F { rng.random() };
        let rand_n = |rng: &mut R| ArithmeticCircuit::Node(rng.random_range(0..n_nodes));

        for _ in 0..depth {
            let a = ((circuit.clone() * rand_f(rng)) + rand_f(rng)) * rand_n(rng);
            let b = (circuit.clone() + a.clone() * rand_f(rng)) * rand_n(rng);
            let c =
                (a.clone() + b.clone() + circuit.clone() * rand_f(rng)) * rand_n(rng) + rand_n(rng);
            let d = b.clone() * c.clone() + (c.clone() * rand_f(rng)) * rand_n(rng);
            let e = (c.clone() + d.clone()) * rand_n(rng);
            circuit = e.clone() + (d.clone() * rand_f(rng)) * rand_n(rng);
            circuit = circuit.clone() + (circuit.clone() * rand_f(rng)) * rand_n(rng);
            circuit = circuit.clone() * rand_n(rng) + (circuit.clone() * rand_f(rng)) * rand_n(rng);
            circuit = circuit.clone() + (circuit.clone() * rand_f(rng)) * rand_n(rng) + rand_f(rng);
        }

        circuit
    }

    // pub fn to_string(&self) -> String
    // where
    //     F: Field,
    //     N: Debug,
    // {
    //     // Use a hashmap to track unique composed circuits and assign them indices
    //     let mut composed_indices = HashMap::new();
    //     let mut next_index = 0;

    //     // First pass: index all composed circuits
    //     fn index_composed<F: Field, N: Debug>(
    //         circuit: &ArithmeticCircuit<F, N>,
    //         indices: &mut HashMap<*const ArithmeticCircuitComposed<F, N>, usize>,
    //         next_index: &mut usize,
    //     ) {
    //         if let ArithmeticCircuit::Composed(composed) = circuit {
    //             let ptr = Arc::as_ptr(composed);
    //             if let Entry::Vacant(e) = indices.entry(ptr) {
    //                 e.insert(*next_index);
    //                 *next_index += 1;

    //                 // Recursively index subcircuits
    //                 match &**composed {
    //                     ArithmeticCircuitComposed::Sum((left, right))
    //                     | ArithmeticCircuitComposed::Product((left, right)) => {
    //                         for subcircuit in [left, right] {
    //                             index_composed(subcircuit, indices, next_index);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     index_composed(self, &mut composed_indices, &mut next_index);

    //     // Second pass: build the string representation
    //     fn to_string_inner<F: Field, N: Debug>(
    //         circuit: &ArithmeticCircuit<F, N>,
    //         indices: &HashMap<*const ArithmeticCircuitComposed<F, N>, usize>,
    //     ) -> String {
    //         match circuit {
    //             ArithmeticCircuit::Scalar(scalar) => format!("scalar({:?})", scalar),
    //             ArithmeticCircuit::Node(node) => format!("node({:?})", node),
    //             ArithmeticCircuit::Composed(composed) => {
    //                 let ptr = Arc::as_ptr(composed);
    //                 let index = indices.get(&ptr).unwrap();

    //                 let op_type = match &**composed {
    //                     ArithmeticCircuitComposed::Sum(_) => "Sum",
    //                     ArithmeticCircuitComposed::Product(_) => "Product",
    //                 };

    //                 // Represent a composed circuit by its index
    //                 format!("{}#{}", op_type, index)
    //             }
    //         }
    //     }

    //     // Build the full representation with definitions
    //     let mut result = String::new();

    //     // First add the main circuit representation
    //     result.push_str(&format!(
    //         "Result: {}\n\n",
    //         to_string_inner(self, &composed_indices)
    //     ));

    //     // Then add definitions for all composed circuits
    //     if !composed_indices.is_empty() {
    //         result.push_str("Definitions:\n");

    //         // Create a map from indices to pointers for sorting
    //         let mut indices_to_ptr: Vec<(usize, *const ArithmeticCircuitComposed<F, N>)> =
    //             composed_indices.iter().map(|(k, v)| (*v, *k)).collect();
    //         indices_to_ptr.sort_by_key(|(idx, _)| *idx);

    //         for (idx, ptr) in indices_to_ptr {
    //             // Safety: We know these pointers are valid because they came from Arc
    //             let composed = unsafe { &*ptr };

    //             match composed {
    //                 ArithmeticCircuitComposed::Sum((left, right)) => {
    //                     result.push_str(&format!("Sum#{} = ", idx));
    //                     result.push_str(&to_string_inner(left, &composed_indices));
    //                     result.push_str(" + ");
    //                     result.push_str(&to_string_inner(right, &composed_indices));
    //                     result.push('\n');
    //                 }
    //                 ArithmeticCircuitComposed::Product((left, right)) => {
    //                     result.push_str(&format!("Product#{} = ", idx));
    //                     result.push_str(&to_string_inner(left, &composed_indices));
    //                     result.push_str(" * ");
    //                     result.push_str(&to_string_inner(right, &composed_indices));
    //                     result.push('\n');
    //                 }
    //             }
    //         }
    //     }

    //     result
    // }
}

impl<F: Field, N> Add<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn add(self, other: F) -> Self {
        Self::new_sum(vec![self, Self::from(other)])
    }
}

impl<F: Field, N> Mul<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn mul(self, other: F) -> Self {
        Self::new_product(vec![self, Self::from(other)])
    }
}

impl<F: Field, N> Sub<F> for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn sub(self, other: F) -> Self {
        self + (-other)
    }
}

impl<F: Field, N> Neg for ArithmeticCircuit<F, N> {
    type Output = Self;

    fn neg(self) -> Self {
        self * F::NEG_ONE
    }
}

macro_rules! impl_field_assign_ops {
    ($($op_trait:ident, $op_fn:ident, $op:tt);*) => {
        $(
            impl<F: Field, N> $op_trait<F> for ArithmeticCircuit<F, N> {
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use p3_koala_bear::KoalaBear;
    use utils::HypercubePoint;

    use super::*;
    use p3_field::PrimeCharacteristicRing;
    type F = KoalaBear;

    #[test]
    fn test_next() {
        let n = 5;
        let next = TransparentPolynomial::<F>::next(n);
        let mut one_points = HashSet::new();
        for x in 0..(1 << n) - 1 {
            let y = x + 1;
            one_points.insert(
                [
                    HypercubePoint { n_vars: n, val: x }.to_vec::<F>(),
                    HypercubePoint { n_vars: n, val: y }.to_vec::<F>(),
                ]
                .concat(),
            );
        }
        for x in 0..1 << (n * 2) {
            let point = HypercubePoint {
                n_vars: n * 2,
                val: x,
            }
            .to_vec();
            if one_points.contains(&point) {
                assert_eq!(next.fix_computation(false).eval(&point), F::ONE);
            } else {
                assert_eq!(next.fix_computation(false).eval(&point), F::ZERO);
            }
        }
    }
}

pub fn max_degree_per_vars_prod(subs: &[Vec<usize>]) -> Vec<usize> {
    let n_vars = subs.iter().map(|s| s.len()).max().unwrap_or_default();
    let mut res = vec![0; n_vars];
    for i in 0..subs.len() {
        for j in 0..subs[i].len() {
            res[j] += subs[i][j];
        }
    }
    res
}

pub fn max_degree_per_vars_sum(subs: &[Vec<usize>]) -> Vec<usize> {
    let n_vars = subs.iter().map(|s| s.len()).max().unwrap();
    let mut res = vec![0; n_vars];
    for i in 0..subs.len() {
        for j in 0..subs[i].len() {
            res[j] = res[j].max(subs[i][j]);
        }
    }
    res
}
