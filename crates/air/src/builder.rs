use std::{borrow::Borrow, collections::BTreeMap};

use algebra::pols::{ArithmeticCircuit, TransparentMultivariatePolynomial};
use p3_field::Field;

use super::table::{AirConstraint, AirTable, BoundaryCondition};

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
    constraints: Vec<(String, AirExpr<F>)>, // every expr should equal to zero
    fixd_values: BTreeMap<(ColIndex, RowIndex), F>,
}

impl<F: Field, const COLS: usize> AirBuilder<F, COLS> {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            fixd_values: BTreeMap::new(),
        }
    }

    pub fn assert_zero(&mut self, name: &str, expr: AirExpr<F>) {
        self.constraints
            .push((name.to_string(), expr.borrow().clone()));
    }

    pub fn assert_eq(&mut self, name: &str, expr_1: AirExpr<F>, expr_2: AirExpr<F>) {
        self.assert_zero(name, expr_1 - expr_2);
    }

    pub fn set_fixed_value(&mut self, col_index: ColIndex, row_index: RowIndex, value: F) {
        self.fixd_values.insert((col_index, row_index), value);
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
            .map(|(name, expr)| AirConstraint {
                name,
                expr: TransparentMultivariatePolynomial::new(
                    expr.map_node(&|var| match var.alignment {
                        Alignment::Up => ArithmeticCircuit::Node(var.col_index),
                        Alignment::Down => ArithmeticCircuit::Node(var.col_index + COLS),
                    }),
                    COLS * 2,
                ),
            })
            .collect();
        let boundary_conditions = std::mem::take(&mut self.fixd_values)
            .into_iter()
            .map(|((col_index, row_index), value)| BoundaryCondition {
                col: col_index,
                row: row_index,
                value,
            })
            .collect();
        AirTable {
            n_columns: COLS,
            constraints,
            boundary_conditions,
        }
    }
}
