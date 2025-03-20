use std::marker::PhantomData;

use air::AirBuilder;
use air::AirExpr;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::GenericPoseidon2LinearLayers;

use super::columns::{FullRound, PartialRound, Poseidon2Cols, SBox};
use super::constants::RoundConstants;

/// Assumes the field size is at least 16 bits.
#[derive(Debug)]
pub struct Poseidon2Air<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    _phantom: PhantomData<LinearLayers>,
}

impl<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>
    Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >
{
    pub const fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        Self {
            constants,
            _phantom: PhantomData,
        }
    }
}

pub(crate) fn write_constraints<
    F: Field,
    LinearLayers: GenericPoseidon2LinearLayers<AirExpr<F>, WIDTH>,
    const COLS: usize, // total number of columns
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    air: &Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    builder: &mut AirBuilder<F, COLS>,
    local: &Poseidon2Cols<
        AirExpr<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
) {
    let mut state: [_; WIDTH] = local.inputs.clone().map(|x| x.into());

    LinearLayers::external_linear_layer(&mut state);

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<F, LinearLayers, COLS, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.constants.beginning_full_round_constants[round],
            builder,
        );
    }

    for round in 0..PARTIAL_ROUNDS {
        eval_partial_round::<F, LinearLayers, COLS, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.partial_rounds[round],
            air.constants.partial_round_constants[round],
            builder,
        );
    }

    for round in 0..HALF_FULL_ROUNDS {
        eval_full_round::<F, LinearLayers, COLS, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &local.ending_full_rounds[round],
            &air.constants.ending_full_round_constants[round],
            builder,
        );
    }
}

#[inline]
fn eval_full_round<
    F: Field,
    LinearLayers: GenericPoseidon2LinearLayers<AirExpr<F>, WIDTH>,
    const COLS: usize,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AirExpr<F>; WIDTH],
    full_round: &FullRound<AirExpr<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
    builder: &mut AirBuilder<F, COLS>,
) {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s += *r;
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    LinearLayers::external_linear_layer(state);
    for (state_i, post_i) in state.iter_mut().zip(&full_round.post) {
        builder.assert_eq("eval_full_round", state_i.clone(), post_i.clone());
        *state_i = post_i.clone();
    }
}

#[inline]
fn eval_partial_round<
    F: Field,
    LinearLayers: GenericPoseidon2LinearLayers<AirExpr<F>, WIDTH>,
    const COLS: usize,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [AirExpr<F>; WIDTH],
    partial_round: &PartialRound<AirExpr<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: F,
    builder: &mut AirBuilder<F, COLS>,
) {
    state[0] += round_constant;
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    builder.assert_eq(
        "eval_partial_round",
        state[0].clone(),
        partial_round.post_sbox.clone(),
    );
    state[0] = partial_round.post_sbox.clone();

    LinearLayers::internal_linear_layer(state);
}

/// Evaluates the S-box over a degree-1 expression `x`.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-box. The supported degrees are
/// `3`, `5`, `7`, and `11`.
#[inline]
fn eval_sbox<F: Field, const COLS: usize, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &SBox<AirExpr<F>, DEGREE, REGISTERS>,
    x: &mut AirExpr<F>,
    builder: &mut AirBuilder<F, COLS>,
) {
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let committed_x3 = sbox.0[0].clone();
            let x2 = x.square();
            builder.assert_eq("eval_sbox", committed_x3.clone(), x2.clone() * x.clone());
            committed_x3 * x2
        }
        (7, 1) => {
            let committed_x3 = sbox.0[0].clone();
            builder.assert_eq("eval_sbox", committed_x3.clone(), x.cube());
            committed_x3.square() * x.clone()
        }
        (11, 2) => {
            let committed_x3 = sbox.0[0].clone();
            let committed_x9 = sbox.0[1].clone();
            let x2 = x.square();
            builder.assert_eq("eval_sbox 1", committed_x3.clone(), x2.clone() * x.clone());
            builder.assert_eq("eval_sbox 2", committed_x9.clone(), committed_x3.cube());
            committed_x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
