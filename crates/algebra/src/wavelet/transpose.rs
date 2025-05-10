use crate::wavelet::workload_size;

use super::MatrixMut;
use std::mem::swap;

use rayon::join;

// NOTE: The assumption that rows and cols are a power of two are actually only relevant for the square matrix case.
// (This is because the algorithm recurses into 4 sub-matrices of half dimension; we assume those to be square matrices as well, which only works for powers of two).

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
/// This algorithm assumes that both rows and cols are powers of two.
///
/// Example:
/// transpose([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 2, 4)
/// gives:[1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    assert_eq!(matrix.len() % (rows * cols), 0);
    assert!(rows.is_power_of_two());
    assert!(cols.is_power_of_two());
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

/// Transposes a rectangular matrix into another matrix.
fn transpose_copy<F: Sized + Copy + Send>(src: MatrixMut<'_, F>, mut dst: MatrixMut<'_, F>) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());

    let (rows, cols) = (src.rows(), src.cols());

    // Direct element-wise transposition for small matrices (avoids recursion overhead)
    if rows * cols <= 64 {
        unsafe {
            for i in 0..rows {
                for j in 0..cols {
                    *dst.ptr_at_mut(j, i) = *src.ptr_at(i, j);
                }
            }
        }
        return;
    }

    // Determine optimal split axis
    let (src_a, src_b, dst_a, dst_b) = if rows > cols {
        let split_size = rows / 2;
        let (s1, s2) = src.split_vertical(split_size);
        let (d1, d2) = dst.split_horizontal(split_size);
        (s1, s2, d1, d2)
    } else {
        let split_size = cols / 2;
        let (s1, s2) = src.split_horizontal(split_size);
        let (d1, d2) = dst.split_vertical(split_size);
        (s1, s2, d1, d2)
    };

    rayon::join(
        || transpose_copy(src_a, dst_a),
        || transpose_copy(src_b, dst_b),
    );
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
fn transpose_square<F: Sized + Send>(mut m: MatrixMut<F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        join(
            || transpose_square_swap(b, c),
            || join(|| transpose_square(a), || transpose_square(d)),
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

/// Swaps two square sub-matrices in-place, transposing them simultaneously.
fn transpose_square_swap<F: Sized + Send>(mut a: MatrixMut<'_, F>, mut b: MatrixMut<'_, F>) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(a.rows().is_power_of_two());
    debug_assert!(workload_size::<F>() >= 2);

    let size = a.rows();

    // Direct swaps for small matrices (≤8x8)
    // - Avoids recursion overhead
    // - Uses basic element-wise swaps
    if size <= 8 {
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
            }
        }
        return;
    }

    // If the matrix is large, use recursive subdivision:
    // - Improves cache efficiency by working on smaller blocks
    // - Enables parallel execution
    if 2 * size * size > workload_size::<F>() {
        let n = size / 2;
        let (aa, ab, ac, ad) = a.split_quadrants(n, n);
        let (ba, bb, bc, bd) = b.split_quadrants(n, n);

        rayon::join(
            || {
                rayon::join(
                    || transpose_square_swap(aa, ba),
                    || transpose_square_swap(ab, bc),
                )
            },
            || {
                rayon::join(
                    || transpose_square_swap(ac, bb),
                    || transpose_square_swap(ad, bd),
                )
            },
        );
    } else {
        // Optimized 2×2 loop unrolling for larger blocks
        // - Reduces loop overhead
        // - Increases memory access efficiency
        for i in (0..size).step_by(2) {
            for j in (0..size).step_by(2) {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
                swap(&mut a[(i + 1, j)], &mut b[(j, i + 1)]);
                swap(&mut a[(i, j + 1)], &mut b[(j + 1, i)]);
                swap(&mut a[(i + 1, j + 1)], &mut b[(j + 1, i + 1)]);
            }
        }
    }
}
