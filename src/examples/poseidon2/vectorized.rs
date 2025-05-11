use core::borrow::{Borrow, BorrowMut};

use air::{AirBuilder, AirExpr};
use p3_field::Field;
use p3_poseidon2::GenericPoseidon2LinearLayers;

use super::air::{Poseidon2Air, write_constraints};
use super::columns::Poseidon2Cols;
use super::constants::RoundConstants;

/// A "vectorized" version of Poseidon2Cols, for computing multiple Poseidon2 permutations per row.
#[repr(C)]
pub struct VectorizedPoseidon2Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) cols:
        [Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;
            VECTOR_LEN],
}

impl<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    Borrow<
        VectorizedPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    > for [T]
{
    fn borrow(
        &self,
    ) -> &VectorizedPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<VectorizedPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
                VECTOR_LEN,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    BorrowMut<
        VectorizedPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut VectorizedPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<VectorizedPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
                VECTOR_LEN,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

/// A "vectorized" version of Poseidon2Air, for computing multiple Poseidon2 permutations per row.
pub struct VectorizedPoseidon2Air<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) air: Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
}

impl<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    VectorizedPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    pub const fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        Self {
            air: Poseidon2Air::new(constants),
        }
    }
}

pub(crate) fn write_vectorized_constraints<
    F: Field,
    LinearLayers: GenericPoseidon2LinearLayers<AirExpr<F>, WIDTH>,
    const COLS: usize, // total number of columns
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    air: &VectorizedPoseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >,
    builder: &mut AirBuilder<F, COLS>,
) {
    let (up, _down) = builder.vars();
    let columns_up = Borrow::<
        VectorizedPoseidon2Cols<
            AirExpr<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    >::borrow(&up[..]);
    for perm in &columns_up.cols {
        write_constraints(&air.air, builder, perm);
    }
}
