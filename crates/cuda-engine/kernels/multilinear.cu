
#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "ff_wrapper.cu"

extern "C" __global__ void lagrange_to_monomial_basis_end(Field_A *input, Field_A *output, uint32_t n_vars, uint32_t missing_steps)
{
    // we may have input and output overlapping
    assert(missing_steps <= n_vars);
    assert(1 << (missing_steps - 1) == blockDim.x);

    extern __shared__ Field_A shared_cache[];

    int n_virtual_blocks = 1 << (n_vars - missing_steps);
    int n_repetitions = (n_virtual_blocks + gridDim.x - 1) / gridDim.x;

    for (int b = 0; b < n_repetitions; b++)
    {
        int v_block = blockIdx.x + b * gridDim.x;
        int tid = threadIdx.x;

        if (v_block >= n_virtual_blocks)
        {
            return;
        }

        shared_cache[tid] = input[v_block * blockDim.x * 2 + tid];
        shared_cache[tid + blockDim.x] = input[(v_block * 2 + 1) * blockDim.x + tid];

        for (int step = 0; step < missing_steps; step++)
        {
            __syncthreads();

            int half = 1 << (missing_steps - step - 1);

            int start = (tid / half) * 2 * half;
            int offset = tid % half;

            Field_A left = shared_cache[start + offset];
            Field_A right = shared_cache[start + offset + half];

            __syncthreads();

            Field_A diff;
            SUB_AA(right, left, diff);
            shared_cache[start + offset + half] = diff;
        }

        __syncthreads();
        output[v_block * blockDim.x * 2 + tid] = shared_cache[tid];
        output[(v_block * 2 + 1) * blockDim.x + tid] = shared_cache[tid + blockDim.x];
    }
}

template <int STEPS>
__device__ inline void
lagrange_to_monomial_inner(const Field_A *input,
                           Field_A *output,
                           int offset,
                           int index_in_chunk,
                           int jump)
{
    Field_A reg[1 << STEPS];

#pragma unroll
    for (int i = 0; i < (1 << STEPS); ++i)
    {
        reg[i] = input[offset + index_in_chunk + i * jump];
    }

#pragma unroll
    for (int step = 0; step < STEPS; ++step)
    {
        const int half = 1 << (STEPS - step - 1);

#pragma unroll
        for (int j = 0; j < (1 << (STEPS - 1)); ++j)
        {
            const int strt = (j / half) * 2 * half;
            const int ofst = j % half;
            SUB_AA(reg[strt + ofst + half],
                   reg[strt + ofst],
                   reg[strt + ofst + half]);
        }
    }

    // write the result back (if required)
#pragma unroll
    for (int i = 0; i < (1 << STEPS); ++i)
        output[offset + index_in_chunk + i * jump] = reg[i];
}

extern "C" __global__ void lagrange_to_monomial_basis_steps(Field_A *input, Field_A *output, uint32_t n_vars, uint32_t previous_steps, uint32_t steps_to_perform)
{
    // we may have input and output overlapping

    int total_n_threads = blockDim.x * gridDim.x;
    int n_ops = 1 << (n_vars - steps_to_perform);
    int n_repetitions = (n_ops + total_n_threads - 1) / total_n_threads;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int tid = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (tid >= n_ops)
        {
            return;
        }

        int index_in_chunk = tid % (1 << (n_vars - previous_steps - steps_to_perform));

        int offset = (tid >> (n_vars - previous_steps - steps_to_perform)) << (n_vars - previous_steps);

        int jump = 1 << (n_vars - steps_to_perform - previous_steps);

        switch (steps_to_perform)
        {
        case 1:
            lagrange_to_monomial_inner<1>(input, output, offset, index_in_chunk, jump);
            break;
        case 2:
            lagrange_to_monomial_inner<2>(input, output, offset, index_in_chunk, jump);
            break;
        case 3:
            lagrange_to_monomial_inner<3>(input, output, offset, index_in_chunk, jump);
            break;
        case 4:
            lagrange_to_monomial_inner<4>(input, output, offset, index_in_chunk, jump);
            break;
        case 5:
            lagrange_to_monomial_inner<5>(input, output, offset, index_in_chunk, jump);
            break;
        default:
            assert(false); // not compiled in
        }
    }
}

template <int extra_steps>
__device__ void process_eq_mle_steps_helper(Field_A *__restrict__ point, Field_A *starting_element, Field_A *__restrict__ res,
                                            int n_vars, int offset, int res_spacing, int step)
{
    Field_A buff[(1 << extra_steps)];
    buff[0] = *starting_element;
    for (int i = 0; i < extra_steps; i++)
    {
        Field_A point_value = point[n_vars - 1 - step];

        for (int j = 0; j < 1 << i; j++)
        {
            Field_A current = buff[j];

            Field_A left;
            Field_A right;
            MUL_AA(current, point_value, right);
            SUB_AA(current, right, left);

            buff[j] = left;
            buff[j + (1 << i)] = right;
        }

        step++;
    }

    for (int i = 0; i < (1 << extra_steps); i++)
    {
        res[offset + i * res_spacing] = buff[i];
    }
}

__device__ void process_eq_mle_steps(int n_steps_to_compute, Field_A *__restrict__ point, Field_A *starting_element, Field_A *__restrict__ res,
                                     int n_vars, int offset, int res_spacing, int step)
{
    switch (n_steps_to_compute)
    {
    case 0:
        process_eq_mle_steps_helper<0>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    case 1:
        process_eq_mle_steps_helper<1>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    case 2:
        process_eq_mle_steps_helper<2>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    case 3:
        process_eq_mle_steps_helper<3>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    case 4:
        process_eq_mle_steps_helper<4>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    case 5:
        process_eq_mle_steps_helper<5>(point, starting_element, res, n_vars, offset, res_spacing, step);
        break;
    default:
        // not supported (would require too much registers)
        assert(0);
    }
}

extern "C" __global__ void eq_mle_start(Field_A *__restrict__ point,
                                        uint32_t n_vars,
                                        uint32_t n_steps_within_shared_memory,
                                        uint32_t n_steps,
                                        Field_A *__restrict__ res)
{
    assert(n_steps <= n_vars);
    assert(gridDim.x == 1);

    extern __shared__ Field_A shared_cache[]; // first n_vars values = point, then the rest is used for computation
    const uint32_t tid = threadIdx.x;

    if (tid < n_vars)
    {
        shared_cache[tid] = point[tid];
    }

    if (tid == 0)
        shared_cache[n_vars] = Field_A::one();
    __syncthreads();

    int step = 0;
    for (; step < n_steps_within_shared_memory; step++)
    {
        if (tid < (1 << step))
        {
            Field_A current = shared_cache[n_vars + tid];

            Field_A left;
            Field_A right;
            Field_A point_value = shared_cache[n_vars - 1 - step];
            MUL_AA(current, point_value, right);
            SUB_AA(current, right, left);

            shared_cache[n_vars + tid] = left;
            shared_cache[n_vars + tid + (1 << step)] = right;
        }
        __syncthreads();
    }

    Field_A starting_element = shared_cache[n_vars + tid];
    int missing_steps_to_compute = n_steps - n_steps_within_shared_memory;
    process_eq_mle_steps(missing_steps_to_compute, shared_cache, &starting_element, res, n_vars, tid, 1 << n_steps_within_shared_memory, step);
}

extern "C" __global__ void eq_mle_steps(Field_A *__restrict__ point,
                                        uint32_t n_vars,
                                        uint32_t start_step,
                                        uint32_t additional_steps,
                                        Field_A *__restrict__ res)
{
    assert(start_step > 0); // otherwise, use eq_mle_start
    assert(start_step + additional_steps <= n_vars);
    assert(n_vars <= blockDim.x);

    extern __shared__ Field_A cached_point[];
    if (threadIdx.x < n_vars)
    {
        cached_point[threadIdx.x] = point[threadIdx.x];
    }
    __syncthreads();

    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_repetitions = ((1 << start_step) + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int tid = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (tid >= 1 << start_step)
        {
            return;
        }

        Field_A starting_element = res[tid];

        process_eq_mle_steps(additional_steps, cached_point, &starting_element, res, n_vars, tid, 1 << start_step, start_step);
    }
}

extern "C" __global__ void scale_in_place(Field_A *slice, const uint32_t len, Field_A scalar)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            Field_A prod;
            MUL_AA(slice[threadIndex], scalar, prod);
            slice[threadIndex] = prod;
        }
    }
}

extern "C" __global__ void add_slices(Field_A **slices, Field_A *res, const uint32_t n_slices, const uint32_t len)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            Field_A sum = {0};
            for (int i = 0; i < n_slices; i++)
            {
                ADD_AA(slices[i][threadIndex], sum, sum);
            }
            res[threadIndex] = sum;
        }
    }
}

extern "C" __global__ void add_assign_slices(LARGER_AB *left, Field_A *right, const uint32_t len)
{
    // a += b
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            LARGER_AB l = left[threadIndex];
            Field_A r = right[threadIndex];
            LARGER_AB sum;
            ADD_A_AND_MAX_AB(r, l, sum);
            left[threadIndex] = sum;
        }
    }
}

// for the AIR columns
extern "C" __global__ void multilinears_up(const Field_A **columns, const uint32_t n_columns, const int n_vars, Field_A **result)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int total = n_columns * (1 << n_vars);
    const int n_repetitions = (total + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < total)
        {
            const int column_index = threadIndex / (1 << n_vars);
            int coeff_index = threadIndex % (1 << n_vars);
            result[column_index][coeff_index] = coeff_index == (1 << n_vars) - 1
                                                    ? columns[column_index][coeff_index - 1]
                                                    : columns[column_index][coeff_index];
        }
    }
}

// for the AIR columns
extern "C" __global__ void multilinears_down(Field_A **columns, const uint32_t n_columns, const int n_vars, Field_A **result)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int total = n_columns * (1 << n_vars);
    const int n_repetitions = (total + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < total)
        {
            const int column_index = threadIndex / (1 << n_vars);
            int coeff_index = threadIndex % (1 << n_vars);
            result[column_index][coeff_index] = coeff_index == (1 << n_vars) - 1
                                                    ? columns[column_index][coeff_index]
                                                    : columns[column_index][coeff_index + 1];
        }
    }
}

extern "C" __global__ void fold_rectangular(const Field_A **inputs, LARGER_AB **res, const Field_B *scalars, const uint32_t n_slices, const uint32_t slice_log_len, const uint32_t log_n_scalars)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_folding_ops = n_slices << (slice_log_len - log_n_scalars);
    const int n_repetitions = (n_folding_ops + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - log_n_scalars));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - log_n_scalars));
        LARGER_AB sum = {0};
        for (int i = 0; i < 1 << log_n_scalars; i++)
        {
            LARGER_AB term;
            Field_B scalar = scalars[i];
            Field_A factor = inputs[slice_index][slice_offset + (i << (slice_log_len - log_n_scalars))];
            MUL_BA(scalar, factor, term);
            ADD_MAX_AB(sum, term, sum);
        }
        res[slice_index][slice_offset] = sum;
    }
}

extern "C" __global__ void dot_product(Field_A *a, Field_B *b, LARGER_AB *res, uint32_t log_len, uint32_t n_big_steps, uint32_t n_small_steps)
{
    // will alter terms, final result is in terms[0]

    assert(n_big_steps <= log_len);
    int n_ops;
    if (n_small_steps == 0)
    {
        n_ops = 1 << (log_len - n_big_steps);
    }
    else
    {
        assert(n_big_steps + n_small_steps == log_len);
        assert(blockDim.x == 1 << n_small_steps);
        assert(gridDim.x == 1);

        n_ops = 1 << n_small_steps;
    }

    int n_total_threads = blockDim.x * gridDim.x;

    int n_reps = (n_ops + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= n_ops)
        {
            return;
        }

        LARGER_AB sum;
        Field_A l = a[idx];
        Field_B r = b[idx];
        MUL_AB(l, r, sum);
        for (int i = 1; i < 1 << n_big_steps; i++)
        {
            Field_A l = a[idx + i * n_ops];
            Field_B r = b[idx + i * n_ops];
            LARGER_AB term;
            MUL_AB(l, r, term);
            ADD_MAX_AB(sum, term, sum);
        }

        if (n_small_steps == 0)
        {
            // copy back to global memory
            res[idx] = sum;
            continue;
        }
        // write back the result to shared memory

        extern __shared__ LARGER_AB my_cache[];
        my_cache[threadIdx.x] = sum;
        __syncthreads();

        // sum in small_steps steps
        for (int step = 0; step < n_small_steps; step++)
        {
            int half = 1 << (n_small_steps - step - 1);
            if (threadIdx.x < half)
            {
                LARGER_AB left = my_cache[threadIdx.x];
                LARGER_AB right = my_cache[threadIdx.x + half];
                ADD_MAX_AB(left, right, left);
                my_cache[threadIdx.x] = left;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            res[idx] = my_cache[0];
        }
    }
}

extern "C" __global__ void linear_combination_at_row_level(const Field_A *input, LARGER_AB *output, Field_B *scalars, const uint32_t len, const uint32_t n_scalars)
{
    // len must be a multiple of n_scalars
    // input has size len
    // output has size len / n_scalars
    // output[0] = input[0].scalars[0] + input[1].scalars[1] + ... + input[n_scalars - 1].scalars[n_scalars - 1]
    // output[1] = input[n_scalars].scalars[0] + input[n_scalars + 1].scalars[1] + ... + input[2.n_scalars - 1].scalars[n_scalars - 1]
    // ...
    // Current implem is suited for small values of n_scalars

    // TODO store scalars in constant memory

    const int n_total_threads = blockDim.x * gridDim.x;
    const int output_len = len / n_scalars;
    const int n_reps = (output_len + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < output_len)
        {
            LARGER_AB comb = {0};
            for (int i = 0; i < n_scalars; i++)
            {
                Field_B scalar = scalars[i];
                Field_A curr_input = input[idx * n_scalars + i];
                LARGER_AB prod;
                MUL_BA(scalar, curr_input, prod);
                ADD_MAX_AB(comb, prod, comb);
            }
            output[idx] = comb;
        }
    }
}

extern "C" __global__ void linear_combination(const Field_A **inputs, LARGER_AB *output, Field_B *scalars, const uint32_t len, const uint32_t n_scalars)
{
    // inputs has size n_scalars, and each inputs[i] has size len
    // output has size len
    // scalars has size n_scalars
    // output[0] = input[0][0].scalars[0] + input[0][1].scalars[1] + ... + input[0][n_scalars - 1].scalars[n_scalars - 1]
    // output[1] = input[1][0].scalars[0] + input[1][1].scalars[1] + ... + input[1][n_scalars - 1].scalars[n_scalars - 1]
    // ...
    // Current implem is suited for small values of n_scalars

    // TODO store scalars in constant memory

    const int n_total_threads = blockDim.x * gridDim.x;
    const int n_reps = (len + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= len)
        {
            return;
        }
        LARGER_AB comb = {0};
        for (int i = 0; i < n_scalars; i++)
        {
            Field_B scalar = scalars[i];
            LARGER_AB prod;
            MUL_BA(scalar, inputs[i][idx], prod);
            ADD_MAX_AB(comb, prod, comb);
        }
        output[idx] = comb;
    }
}

extern "C" __global__ void piecewise_sum(Field_A *input, Field_A *output, const uint32_t len, const uint32_t sum_size)
{
    // len must be a multiple of sum_size
    // input has size len
    // output has size len / sum_size
    // output[0] = input[0] + input[output_len] + ... + input[output_len * (sum_size - 1)]
    // output[1] = input[1] + input[output_len + 1] + ... + input[output_len * (sum_size - 1) + 1]
    // ...
    // Current implem is suited for small values of sum_size

    const int n_total_threads = blockDim.x * gridDim.x;
    const int output_len = len / sum_size;
    const int n_reps = (output_len + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < output_len)
        {
            Field_A sum = {0};
            for (int i = 0; i < sum_size; i++)
            {
                ADD_AA(sum, input[idx + output_len * i], sum);
            }
            output[idx] = sum;
        }
    }
}

extern "C" __global__ void repeat_slice_from_outside(const Field_A *input, Field_A *output, const uint32_t len, const uint32_t n_repetitions)
{
    // Optimized for a small number of repetitions
    const int n_total_threads = blockDim.x * gridDim.x;
    const int n_reps = (len + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= len)
        {
            return;
        }
        Field_A value = input[idx];
        for (int i = 0; i < n_repetitions; i++)
        {
            output[idx + i * len] = value;
        }
    }
}

extern "C" __global__ void repeat_slice_from_inside(const Field_A *input, Field_A *output, const uint32_t len, const uint32_t n_repetitions)
{
    // Optimized for a large number of repetitions
    const int n_total_threads = blockDim.x * gridDim.x;
    const int n_reps = (len * n_repetitions + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < len * n_repetitions)
        {
            output[idx] = input[idx / n_repetitions];
        }
    }
}

extern "C" __global__ void sum_in_place(Field_A *terms, uint32_t log_len, uint32_t n_big_steps, uint32_t n_small_steps)
{
    // will alter terms, final result is in terms[0]

    assert(n_big_steps <= log_len);
    int n_ops;
    if (n_small_steps == 0)
    {
        n_ops = 1 << (log_len - n_big_steps);
    }
    else
    {
        assert(n_big_steps + n_small_steps == log_len);
        assert(blockDim.x == 1 << n_small_steps);
        assert(gridDim.x == 1);

        n_ops = 1 << n_small_steps;
    }

    int n_total_threads = blockDim.x * gridDim.x;

    int n_reps = (n_ops + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= n_ops)
        {
            return;
        }

        Field_A sum = terms[idx];
        for (int i = 1; i < 1 << n_big_steps; i++)
        {
            ADD_AA(sum, terms[idx + i * n_ops], sum);
        }

        if (n_small_steps == 0)
        {
            // copy back to global memory
            terms[idx] = sum;
            continue;
        }
        // write back the result to shared memory

        extern __shared__ Field_A shared_cache[];
        shared_cache[threadIdx.x] = sum;
        __syncthreads();

        // sum in small_steps steps
        for (int step = 0; step < n_small_steps; step++)
        {
            int half = 1 << (n_small_steps - step - 1);
            if (threadIdx.x < half)
            {
                Field_A left = shared_cache[threadIdx.x];
                Field_A right = shared_cache[threadIdx.x + half];
                ADD_AA(left, right, left);
                shared_cache[threadIdx.x] = left;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            terms[idx] = shared_cache[0];
        }
    }
}