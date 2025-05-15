__device__ int index_transpose(int i, int log_rows, int log_cols)
{
    /*
    Interpret i as an index in a slice contaning concatenated row-major matrices (each matrix is of size 2^log_rows x 2^log_cols).
    Each matrix is transposed to column-major order.
    The ordering of the matrices is preserved.
    The function returns the index of i in slice containing concatenated column-major matrices.
    */
    int cols = 1 << log_cols;
    int rows = 1 << log_rows;
    int matrix_len = cols * rows;
    int index_in_matrix = i % matrix_len;
    int initial_shift = i - index_in_matrix;
    int new_row = index_in_matrix % cols;
    int new_col = index_in_matrix / cols;
    return initial_shift + (new_row * rows) + new_col;
}

__device__ __forceinline__ int trailing_zeros(unsigned int x)
{
    if (x == 0)
    {
        return 32;
    }
    return __ffs(x) - 1;
}
