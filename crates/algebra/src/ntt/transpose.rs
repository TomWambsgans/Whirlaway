use super::super::utils::is_power_of_two;
use super::{MatrixMut, utils::workload_size};
use std::mem::swap;

use rayon::join;

// NOTE: The assumption that rows and cols are a power of two are actually only relevant for the square matrix case.
// (This is because the algorithm recurses into 4 sub-matrices of half dimension; we assume those to be square matrices as well, which only works for powers of two).

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
/// This algorithm assumes that both rows and cols are powers of two.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len() % (rows * cols), 0);
    debug_assert!(is_power_of_two(rows));
    debug_assert!(is_power_of_two(cols));
    // eprintln!(
    //     "Transpose {} x {rows} x {cols} matrix.",
    //     matrix.len() / (rows * cols)
    // );
    if rows == cols {
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            let matrix = MatrixMut::from_mut_slice(matrix, rows, cols);
            transpose_square(matrix);
        }
    } else {
        // TODO: Special case for rows = 2 * cols and cols = 2 * rows.
        // TODO: Special case for very wide matrices (e.g. n x 16).
        let mut scratch = vec![matrix[0]; rows * cols];
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            scratch.copy_from_slice(matrix);
            let src = MatrixMut::from_mut_slice(scratch.as_mut_slice(), rows, cols);
            let dst = MatrixMut::from_mut_slice(matrix, cols, rows);
            transpose_copy(src, dst);
        }
    }
}

// The following function have both a parallel and a non-parallel implementation.
// We fuly split those in a parallel and a non-parallel functions (rather than using #[cfg] within a single function)
// and have main entry point fun that just calls the appropriate version (either fun_parallel or fun_not_parallel).
// The sole reason is that this simplifies unit tests: We otherwise would need to build twice to cover both cases.
// For effiency, we assume the compiler inlines away the extra "indirection" that we add to the entry point function.

// NOTE: We could lift the Send constraints on non-parallel build.

fn transpose_copy<F: Sized + Copy + Send>(src: MatrixMut<F>, dst: MatrixMut<F>) {
    transpose_copy_parallel(src, dst);
}

/// Sets `dst` to the transpose of `src`. This will panic if the sizes of `src` and `dst` are not compatible.
fn transpose_copy_parallel<F: Sized + Copy + Send>(
    src: MatrixMut<'_, F>,
    mut dst: MatrixMut<'_, F>,
) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());
    if src.rows() * src.cols() > workload_size::<F>() {
        // Split along longest axis and recurse.
        // This results in a cache-oblivious algorithm.
        let ((a, b), (x, y)) = if src.rows() > src.cols() {
            let n = src.rows() / 2;
            (src.split_vertical(n), dst.split_horizontal(n))
        } else {
            let n = src.cols() / 2;
            (src.split_horizontal(n), dst.split_vertical(n))
        };
        join(
            || transpose_copy_parallel(a, x),
            || transpose_copy_parallel(b, y),
        );
    } else {
        for i in 0..src.rows() {
            for j in 0..src.cols() {
                dst[(j, i)] = src[(i, j)];
            }
        }
    }
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
fn transpose_square<F: Sized + Send>(m: MatrixMut<F>) {
    transpose_square_parallel(m);
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
/// This is the parallel version.
fn transpose_square_parallel<F: Sized + Send>(mut m: MatrixMut<F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        join(
            || transpose_square_swap_parallel(b, c),
            || {
                join(
                    || transpose_square_parallel(a),
                    || transpose_square_parallel(d),
                )
            },
        );
    } else {
        for i in 0..size {
            for j in (i + 1)..size {
                // unsafe needed due to lack of bounds-check by swap. We are guaranteed that (i,j) and (j,i) are within the bounds.
                unsafe {
                    m.swap((i, j), (j, i));
                }
            }
        }
    }
}

/// Transpose and swap two square size matrices (parallel version). The size must be a power of two.
fn transpose_square_swap_parallel<F: Sized + Send>(mut a: MatrixMut<F>, mut b: MatrixMut<F>) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(is_power_of_two(a.rows()));
    debug_assert!(workload_size::<F>() >= 2); // otherwise, we would recurse even if size == 1.
    let size = a.rows();
    if 2 * size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (aa, ab, ac, ad) = a.split_quadrants(n, n);
        let (ba, bb, bc, bd) = b.split_quadrants(n, n);

        join(
            || {
                join(
                    || transpose_square_swap_parallel(aa, ba),
                    || transpose_square_swap_parallel(ab, bc),
                )
            },
            || {
                join(
                    || transpose_square_swap_parallel(ac, bb),
                    || transpose_square_swap_parallel(ad, bd),
                )
            },
        );
    } else {
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)])
            }
        }
    }
}
