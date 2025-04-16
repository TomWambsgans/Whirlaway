use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

use p3_field::{BasedVectorSpace, ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

#[derive(Clone, PartialEq, Eq)]
pub struct TensorAlgebra<F: Field, EF: ExtensionField<F>> {
    pub data: Vec<Vec<F>>,
    _phantom: std::marker::PhantomData<EF>,
}

impl<F: Field, EF: ExtensionField<F>> TensorAlgebra<F, EF> {
    pub fn new(data: Vec<Vec<F>>) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn kappa() -> usize {
        assert!(<EF as BasedVectorSpace<F>>::DIMENSION.is_power_of_two());
        <EF as BasedVectorSpace<F>>::DIMENSION.trailing_zeros() as usize
    }

    pub fn two_pow_kappa() -> usize {
        1 << Self::kappa()
    }

    pub fn scale_columns(&self, scalar: EF) -> Self {
        let mut columns = self.columns();
        for col in &mut columns {
            *col *= scalar;
        }
        Self::from_columns(&columns)
    }

    pub fn scale_rows(&self, scalar: EF) -> Self {
        let mut rows = self.rows();
        for row in &mut rows {
            *row *= scalar;
        }
        Self::from_rows(&rows)
    }

    pub fn phi_0_times_phi_1(p0: &EF, p1: &EF) -> Self {
        let p0_split: &[F] = p0.as_basis_coefficients_slice();
        let p1_split: &[F] = p1.as_basis_coefficients_slice();
        let two_pow_kappa = Self::two_pow_kappa();
        let mut data = vec![vec![F::ZERO; two_pow_kappa]; two_pow_kappa];
        for i in 0..two_pow_kappa {
            for (j, p1_split) in p1_split.iter().enumerate() {
                data[i][j] = p0_split[i] * *p1_split;
            }
        }
        Self::new(data)
    }

    pub fn rows(&self) -> Vec<EF> {
        self.data
            .iter()
            .map(|row| EF::from_basis_coefficients_slice(row))
            .collect()
    }

    pub fn columns(&self) -> Vec<EF> {
        let rotated = switch_lines_columns(self.data.clone());
        rotated
            .iter()
            .map(|row| EF::from_basis_coefficients_slice(row))
            .collect()
    }

    pub fn zero() -> Self {
        let data = vec![vec![F::ZERO; Self::two_pow_kappa()]; Self::two_pow_kappa()];
        Self::new(data)
    }

    pub fn one() -> Self {
        let mut res = Self::zero();
        res.data[0][0] = F::ONE;
        res
    }

    pub fn random<R: Rng>(rng: &mut R) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        let mut res = Self::zero();
        for i in 0..Self::two_pow_kappa() {
            for j in 0..Self::two_pow_kappa() {
                res.data[i][j] = rng.random();
            }
        }
        res
    }

    pub fn from_rows(rows: &[EF]) -> Self {
        assert_eq!(rows.len(), Self::two_pow_kappa());
        let data = rows
            .iter()
            .map(|row| row.as_basis_coefficients_slice().to_vec())
            .collect();
        Self::new(data)
    }

    pub fn from_columns(columns: &[EF]) -> Self {
        assert_eq!(columns.len(), Self::two_pow_kappa());
        let data = switch_lines_columns(
            columns
                .iter()
                .map(|col| col.as_basis_coefficients_slice().to_vec())
                .collect::<Vec<_>>(),
        );
        Self::new(data)
    }
}

impl<F: Field, EF: ExtensionField<F>> Add for TensorAlgebra<F, EF> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = Self::zero();
        for i in 0..Self::two_pow_kappa() {
            for j in 0..Self::two_pow_kappa() {
                res.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        res
    }
}

impl<F: Field, EF: ExtensionField<F>> AddAssign for TensorAlgebra<F, EF> {
    fn add_assign(&mut self, rhs: Self) {
        *self = std::mem::replace(self, TensorAlgebra::zero()) + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Sum for TensorAlgebra<F, EF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

pub fn switch_lines_columns<D: Default + Clone>(mut matrix: Vec<Vec<D>>) -> Vec<Vec<D>> {
    let n = matrix.len();
    let m = matrix[0].len();
    let mut switched = vec![vec![D::default(); n]; m];
    for i in 0..n {
        for (j, switch) in switched.iter_mut().enumerate() {
            std::mem::swap(&mut switch[i], &mut matrix[i][j]);
        }
    }
    switched
}

impl<F: Field, EF: ExtensionField<F>> std::fmt::Debug for TensorAlgebra<F, EF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..Self::two_pow_kappa() {
            for j in 0..Self::two_pow_kappa() {
                write!(f, "{:?} ", self.data[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
