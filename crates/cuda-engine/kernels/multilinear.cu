
#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "ff_wrapper.cu"

// Could also be done in place
extern "C" __global__ void lagrange_to_monomial_basis(Field_A *input_evals, Field_A *buff, Field_A *output_coeffs, uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int n_total_threads = blockDim.x * gridDim.x;
    int n_iters = (1 << (n_vars - 1));
    int n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    for (int step = 0; step < n_vars; step++)
    {
        int half = 1 << (n_vars - step - 1);
        for (int rep = 0; rep < n_repetitions; rep++)
        {

            int idx = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (idx >= n_iters)
            {
                continue;
            }

            Field_A *inputs = step == 0 ? input_evals : (n_vars % 2 != step % 2 ? buff : output_coeffs);
            Field_A *outputs = n_vars % 2 == step % 2 ? buff : output_coeffs;

            int start = (idx / half) * 2 * half;
            int offset = idx % half;

            Field_A left = inputs[start + offset];
            Field_A right = inputs[start + offset + half];
            Field_A diff;
            SUB_AA(right, left, diff);

            outputs[start + offset] = left;
            outputs[start + offset + half] = diff;
        }
        grid.sync();
    }
}

extern "C" __global__ void eval_multilinear_in_lagrange_basis(Field_A *coeffs, Field_B *point, uint32_t n_vars, LARGER_AB *buff)
{
    // coeffs has size 2^n_vars
    // buff has size 2^n_vars - 1
    // result is stored at the last index of the buffer

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int total_n_threads = blockDim.x * gridDim.x;

    int n_iters = (1 << (n_vars - 1));
    int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < n_iters)
        {
            Field_A left = coeffs[threadIndex];
            Field_A right = coeffs[threadIndex + (1 << (n_vars - 1))];
            Field_A diff;
            SUB_AA(right, left, diff);
            Field_B p = point[0];
            LARGER_AB prod;
            MUL_AB(diff, p, prod);
            ADD_A_AND_MAX_AB(left, prod, buff[threadIndex]);
        }
    }

    grid.sync();

    for (int step = 1; step < n_vars; step++)
    {

        // 2^(nvars - 1) + 2^(nvars - 2) + ... + 2^(nvars - step + 1) = 2^(nvars - step + 1) (2^(step - 1) - 1) = 2^nvars - 2^(nvars - step + 1)
        LARGER_AB *input = &buff[(1 << n_vars) - (1 << (n_vars - step + 1))];
        // same formula, just shifted by one
        LARGER_AB *output = &buff[(1 << n_vars) - (1 << (n_vars - step))];

        int n_iters = (1 << (n_vars - step - 1));
        int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                LARGER_AB left = input[threadIndex];
                LARGER_AB right = input[threadIndex + (1 << (n_vars - 1 - step))];
                LARGER_AB diff;
                SUB_MAX_AB(right, left, diff);
                Field_B p = point[step];
                LARGER_AB prod;
                MUL_B_and_MAX_AB(p, diff, prod);
                ADD_MAX_AB(left, prod, output[threadIndex]);
            }
        }
        grid.sync();
    }
}

// extern "C" __global__ void eq_mle(Field_A *point, const uint32_t n_vars, Field_A *res)
// {
//     namespace cg = cooperative_groups;
//     cg::grid_group grid = cg::this_grid();

//     const int total_n_threads = blockDim.x * gridDim.x;

//     if (threadIdx.x == 0 && blockIdx.x == 0)
//     {
//         res[0] = Field_A::one();
//     }
//     grid.sync();

//     for (int step = 0; step < n_vars; step++)
//     {
//         const int n_iters = (1 << step);
//         const int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
//         for (int rep = 0; rep < n_repetitions; rep++)
//         {
//             const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
//             if (threadIndex < n_iters)
//             {
//                 MUL_AA(res[threadIndex], point[n_vars - 1 - step], res[threadIndex + (1 << step)]);
//                 SUB_AA(res[threadIndex], res[threadIndex + (1 << step)], res[threadIndex]);
//             }
//         }
//         grid.sync();
//     }
// }

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
extern "C" __global__ void dot_product(Field_A *a, Field_B *b, LARGER_AB *res, const uint32_t log_len)
{
    // a, b and res have size 2^log_len
    // the final result is stored in res[0]

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int n_total_threads = blockDim.x * gridDim.x;

    // 1) Product
    const int len = (1 << log_len);
    const int n_repetitions = (len + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < len)
        {
            MUL_BA(b[idx], a[idx], res[idx]);
        }
    }

    // 2) Sum
    for (int step = 0; step < log_len; step++)
    {
        grid.sync();
        const int half_len = 1 << (log_len - step - 1);
        const int n_reps = (half_len + n_total_threads - 1) / n_total_threads;
        for (int rep = 0; rep < n_reps; rep++)
        {
            const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
            if (thread_index < half_len)
            {
                ADD_MAX_AB(res[thread_index], res[thread_index + half_len], res[thread_index]);
            }
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

extern "C" __global__ void sum_in_place(Field_A *terms, const uint32_t log_len)
{
    // will alter terms. The final result will be in terms[0]

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_total_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < log_len; step++)
    {
        const int half_len = 1 << (log_len - step - 1);
        const int n_reps = (half_len + n_total_threads - 1) / n_total_threads;
        for (int rep = 0; rep < n_reps; rep++)
        {
            const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
            if (thread_index < half_len)
            {
                ADD_AA(terms[thread_index], terms[thread_index + half_len], terms[thread_index]);
            }
        }
        grid.sync();
    }
}