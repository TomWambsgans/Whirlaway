use p3_field::{ExtensionField, Field};

use super::{DenseMultilinearPolynomial, HypercubePoint, PartialHypercubePoint};

#[derive(Clone, Debug)]
pub struct SparseMultilinearPolynomial<F: Field> {
    pub n_vars: usize,
    pub n_rows: usize,                                               // in log
    pub n_cols: usize,                                               // in log
    pub n_final_bits_per_row: usize,                                 // in log
    pub evals: Vec<(DenseMultilinearPolynomial<F>, HypercubePoint)>, // last vars = eq extension
}

impl<F: Field> SparseMultilinearPolynomial<F> {
    pub fn n_coefs(&self) -> usize {
        1 << self.n_vars
    }

    pub fn new(evals: Vec<(DenseMultilinearPolynomial<F>, HypercubePoint)>) -> Self {
        let n_rows = evals.len().next_power_of_two().trailing_zeros() as usize;
        let n_final_bits_per_row = evals[0].1.n_vars;
        for i in 0..evals.len() {
            assert_eq!(
                (evals[0].0.n_vars, evals[0].1.n_vars),
                (evals[i].0.n_vars, evals[i].1.n_vars)
            );
        }
        let n_vars = n_rows + evals[0].0.n_vars + evals[0].1.n_vars;
        let n_cols = n_vars - n_rows;
        Self {
            n_vars,
            n_rows,
            n_cols,
            n_final_bits_per_row,
            evals,
        }
    }

    pub fn eval<EF: ExtensionField<F>>(&self, _point: &[EF]) -> EF {
        todo!()
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> F {
        assert_eq!(self.n_vars, point.n_vars);
        let row = point.val & ((1 << self.n_rows) - 1);
        if row >= self.evals.len() {
            return F::ZERO;
        }
        let (row_pol, final_bits) = &self.evals[row];
        let col = point.val >> self.n_rows;
        let col_left = col >> self.n_final_bits_per_row;
        let col_right = col & ((1 << self.n_final_bits_per_row) - 1);
        if col_right == final_bits.val {
            row_pol.eval_hypercube(&HypercubePoint {
                n_vars: self.n_cols - self.n_final_bits_per_row,
                val: col_left,
            })
        } else {
            F::ZERO
        }
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> F {
        self.eval_hypercube(&HypercubePoint {
            n_vars: point.n_vars(),
            val: point.right.val + (1 << point.right.n_vars),
        }) * F::from_u32(point.left)
            + self.eval_hypercube(&HypercubePoint {
                n_vars: point.n_vars(),
                val: point.right.val,
            }) * (F::ONE - F::from_u32(point.left))
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        vec![1; self.n_vars]
    }

    pub fn densify(&self) -> DenseMultilinearPolynomial<F> {
        let mut evals = vec![F::ZERO; 1 << self.n_vars];
        for (row_index, (row_pol, final_bits)) in self.evals.iter().enumerate() {
            for (j, v) in row_pol.evals.iter().enumerate() {
                let mut index = j << (self.n_final_bits_per_row + self.n_rows); // first vars
                index += final_bits.val << self.n_rows;
                index += row_index;
                evals[index] = *v;
            }
        }
        DenseMultilinearPolynomial::new(evals)
    }
}
