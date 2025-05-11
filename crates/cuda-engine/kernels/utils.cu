

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

typedef struct Transpositions
{
    uint32_t n_transpositions;
    uint32_t tr_row_0;
    uint32_t tr_col_0;
    uint32_t tr_row_1;
    uint32_t tr_col_1;
    uint32_t tr_row_2;
    uint32_t tr_col_2;

    __device__ Transpositions(uint32_t n_transpositions, uint32_t tr_row_0, uint32_t tr_col_0,
                              uint32_t tr_row_1, uint32_t tr_col_1,
                              uint32_t tr_row_2, uint32_t tr_col_2)
        : n_transpositions(n_transpositions), tr_row_0(tr_row_0), tr_col_0(tr_col_0),
          tr_row_1(tr_row_1), tr_col_1(tr_col_1),
          tr_row_2(tr_row_2), tr_col_2(tr_col_2) {}

    __device__ static Transpositions empty()
    {
        return Transpositions(0, 0, 0, 0, 0, 0, 0);
    }

    __device__ int transpose(int i)
    {
        if (n_transpositions >= 1)
        {
            i = index_transpose(i, tr_row_0, tr_col_0);
        }
        if (n_transpositions >= 2)
        {
            i = index_transpose(i, tr_row_1, tr_col_1);
        }
        if (n_transpositions >= 3)
        {
            i = index_transpose(i, tr_row_2, tr_col_2);
        }
        return i;
    }
} Transpositions;
