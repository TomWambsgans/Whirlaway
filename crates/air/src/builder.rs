use algebra::pols::{MultilinearHost, univariate_selectors};
use arithmetic_circuit::ArithmeticCircuit;
use p3_field::Field;

use crate::{
    UNIVARIATE_SKIPS,
    utils::{matrix_down_lde, matrix_up_lde},
};

use super::table::AirTable;

pub type ColIndex = usize;
pub type RowIndex = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintVariable {
    col_index: ColIndex,
    alignment: Alignment,
}

pub type AirExpr<F> = ArithmeticCircuit<F, ConstraintVariable>;

pub struct AirBuilder<F: Field, const COLS: usize> {
    log_length: usize,
    constraints: Vec<AirExpr<F>>, // every expr should equal to zero
    preprocessed_columns: Vec<Vec<F>>,
}

impl<F: Field, const COLS: usize> AirBuilder<F, COLS> {
    pub fn new(log_length: usize) -> Self {
        Self {
            log_length,
            constraints: Vec::new(),
            preprocessed_columns: Vec::new(),
        }
    }

    pub fn assert_zero(&mut self, expr: AirExpr<F>) {
        self.constraints.push(expr);
    }

    pub fn assert_eq(&mut self, expr_1: AirExpr<F>, expr_2: AirExpr<F>) {
        self.assert_zero(expr_1 - expr_2);
    }

    /// Assumes condition is 0 or 1
    pub fn assert_eq_if(&mut self, expr_1: AirExpr<F>, expr_2: AirExpr<F>, condition: AirExpr<F>) {
        self.assert_zero_if(expr_1 - expr_2, condition);
    }

    /// Assumes condition is 0 or 1
    pub fn assert_zero_if(&mut self, expr: AirExpr<F>, condition: AirExpr<F>) {
        self.assert_zero(expr * condition);
    }

    pub fn add_preprocess_column(&mut self, col: Vec<F>) {
        assert_eq!(col.len(), 1 << self.log_length);
        self.preprocessed_columns.push(col);
    }

    pub fn vars(
        &self,
    ) -> (
        [ArithmeticCircuit<F, ConstraintVariable>; COLS],
        [ArithmeticCircuit<F, ConstraintVariable>; COLS],
    ) {
        let up = (0..COLS)
            .map(|col_index| {
                ArithmeticCircuit::Node(ConstraintVariable {
                    col_index,
                    alignment: Alignment::Up,
                })
            })
            .collect::<Vec<_>>();
        let down = (0..COLS)
            .map(|col_index| {
                ArithmeticCircuit::Node(ConstraintVariable {
                    col_index,
                    alignment: Alignment::Down,
                })
            })
            .collect::<Vec<_>>();
        (up.try_into().unwrap(), down.try_into().unwrap())
    }

    pub fn build(mut self) -> AirTable<F> {
        let constraints = std::mem::take(&mut self.constraints)
            .into_iter()
            .map(|expr| {
                expr.map_node(&|var| match var.alignment {
                    Alignment::Up => ArithmeticCircuit::Node(var.col_index),
                    Alignment::Down => ArithmeticCircuit::Node(var.col_index + COLS),
                })
                .fix_computation(true)
            })
            .collect();

        AirTable {
            log_length: self.log_length,
            n_columns: COLS,
            constraints,
            preprocessed_columns: self
                .preprocessed_columns
                .into_iter()
                .map(MultilinearHost::new)
                .collect(),
            univariate_selectors: univariate_selectors(UNIVARIATE_SKIPS),
            lde_matrix_up: matrix_up_lde(self.log_length).fix_computation(true),
            lde_matrix_down: matrix_down_lde(self.log_length).fix_computation(true),
        }
    }
}
