
#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "finite_field.cu"

// (+ reverse vars)
extern "C" __global__ void monomial_to_lagrange_basis(ExtField *input_coeffs, ExtField *buff, ExtField *output_evals, const uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_iters = (1 << (n_vars - 1));
    const int n_total_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    for (int step = 0; step < n_vars; step++)
    {
        // switch back and forth between buffers

        ExtField *input;
        ExtField *output;

        if (step == 0)
        {
            input = input_coeffs;
            output = n_vars % 2 == 0 ? buff : output_evals;
        }
        else
        {
            if (step % 2 != n_vars % 2)
            {
                input = buff;
                output = output_evals;
            }
            else
            {
                input = output_evals;
                output = buff;
            }
        }

        const int half_size = 1 << (n_vars - step - 1);
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                const int x = (threadIndex / half_size) * 2 * half_size;
                const int y = threadIndex % half_size;
                ExtField left = input[x + 2 * y];
                output[x + y] = left;
                ext_field_add(&left, &input[x + (2 * y) + 1], &output[x + y + half_size]);
            }
        }

        grid.sync();
    }
}

// Could also be done in place
extern "C" __global__ void lagrange_to_monomial_basis(ExtField *input_evals, ExtField *output_coeffs, const uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int n_iters = (1 << n_vars);
    int n_total_threads = blockDim.x * gridDim.x;
    int n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    // 1) copy input_evals to output_coeffs
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < n_iters)
        {
            output_coeffs[threadIndex] = input_evals[threadIndex];
        }
    }
    grid.sync();

    n_iters = (1 << (n_vars - 1));
    n_total_threads = blockDim.x * gridDim.x;
    n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    // 2) compute the monomial ceffs
    for (int step = 0; step < n_vars; step++)
    {
        const int half_size = 1 << step;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                const int x = (threadIndex / half_size) * 2 * half_size;
                const int y = threadIndex % half_size;
                ext_field_sub(&output_coeffs[x + y + half_size], &output_coeffs[x + y], &output_coeffs[x + y + half_size]);
            }
        }

        grid.sync();
    }
}

extern "C" __global__ void eval_multilinear_in_monomial_basis(ExtField *coeffs, ExtField *point, const uint32_t n_vars, ExtField *buff)
{
    // coeffs and buff have size 2^n_vars
    // result is stored at the last index of the buffer

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < n_vars; step++)
    {
        ExtField *input;
        ExtField *output;

        if (step == 0)
        {
            input = coeffs;
            output = buff;
        }
        else
        {
            // 2^(nvars - 1) + 2^(nvars - 2) + ... + 2^(nvars - step + 1) = 2^(nvars - step + 1) (2^(step - 1) - 1) = 2^nvars - 2^(nvars - step + 1)
            input = &buff[(1 << n_vars) - (1 << (n_vars - step + 1))];
            // same formula, just shifted by one
            output = &buff[(1 << n_vars) - (1 << (n_vars - step))];
        }

        const int n_iters = (1 << (n_vars - step - 1));
        const int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                ExtField prod;
                ext_field_mul(&input[threadIndex + (1 << (n_vars - 1 - step))], &point[step], &prod);
                ext_field_add(&input[threadIndex], &prod, &output[threadIndex]);
            }
        }
        grid.sync();
    }
}

extern "C" __global__ void eval_multilinear_in_lagrange_basis(ExtField *coeffs, ExtField *point, const uint32_t n_vars, ExtField *buff)
{
    // coeffs has size 2^n_vars
    // buff has size 2^n_vars - 1
    // result is stored at the last index of the buffer

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < n_vars; step++)
    {
        ExtField *input;
        ExtField *output;

        if (step == 0)
        {
            input = coeffs;
            output = buff;
        }
        else
        {
            // 2^(nvars - 1) + 2^(nvars - 2) + ... + 2^(nvars - step + 1) = 2^(nvars - step + 1) (2^(step - 1) - 1) = 2^nvars - 2^(nvars - step + 1)
            input = &buff[(1 << n_vars) - (1 << (n_vars - step + 1))];
            // same formula, just shifted by one
            output = &buff[(1 << n_vars) - (1 << (n_vars - step))];
        }

        const int n_iters = (1 << (n_vars - step - 1));
        const int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                ExtField diff;
                ext_field_sub(&input[threadIndex + (1 << (n_vars - 1 - step))], &input[threadIndex], &diff);
                ExtField prod;
                ext_field_mul(&diff, &point[step], &prod);
                ext_field_add(&prod, &input[threadIndex], &output[threadIndex]);
            }
        }
        grid.sync();
    }
}

extern "C" __global__ void eq_mle(ExtField *point, const uint32_t n_vars, ExtField *res)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        res[0] = ExtField{0};
        res[0].coeffs[0] = to_monty(1);
    }
    grid.sync();

    for (int step = 0; step < n_vars; step++)
    {
        const int n_iters = (1 << step);
        const int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                ext_field_mul(&res[threadIndex], &point[n_vars - 1 - step], &res[threadIndex + (1 << step)]);
                ext_field_sub(&res[threadIndex], &res[threadIndex + (1 << step)], &res[threadIndex]);
            }
        }
        grid.sync();
    }
}

extern "C" __global__ void scale_ext_slice_in_place(ExtField *slice, const uint32_t len, ExtField *scalar)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            ExtField prod;
            ext_field_mul(&slice[threadIndex], scalar, &prod);
            slice[threadIndex] = prod;
        }
    }
}

extern "C" __global__ void scale_prime_slice_by_ext(uint32_t *slice, const uint32_t len, ExtField *scalar, ExtField *res)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            mul_prime_and_ext_field(scalar, slice[threadIndex], &res[threadIndex]);
        }
    }
}

extern "C" __global__ void add_slices(const ExtField *a, const ExtField *b, ExtField *res, const uint32_t len)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            ext_field_add(&a[threadIndex], &b[threadIndex], &res[threadIndex]);
        }
    }
}

extern "C" __global__ void add_assign_slices(ExtField *a, ExtField *b, const uint32_t len)
{
    // a += b
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            ext_field_add(&a[threadIndex], &b[threadIndex], &a[threadIndex]);
        }
    }
}

extern "C" __global__ void whir_fold(ExtField *coeffs, const uint32_t n_vars, const uint32_t folding_factor, ExtField *folding_randomness, ExtField *buff, ExtField *res)
{
    // coeffs has length 2^n_vars

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < folding_factor; step++)
    {
        ExtField *input;
        ExtField *output;

        if (folding_factor == 1)
        {
            input = coeffs;
            output = res;
        }
        else if (step == 0)
        {
            input = coeffs;
            output = buff;
        }
        else
        {
            // 2^(nvars - 1) + 2^(nvars - 2) + ... + 2^(nvars - step + 1) = 2^(nvars - step + 1) (2^(step - 1) - 1) = 2^nvars - 2^(nvars - step + 1)
            input = &buff[(1 << n_vars) - (1 << (n_vars - step + 1))];
            if (step == folding_factor - 1)
            {
                output = res;
            }
            else
            {
                // same formula, just shifted by one
                output = &buff[(1 << n_vars) - (1 << (n_vars - step))];
            }
        }

        const int helf_step_size = 1 << (folding_factor - step - 1);

        const int n_iters = (1 << (n_vars - step - 1));
        const int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (threadIndex < n_iters)
            {
                const int x = (threadIndex / helf_step_size) * 2 * helf_step_size;
                const int y = threadIndex % helf_step_size;
                ExtField prod;
                ext_field_mul(&input[x + y + helf_step_size], &folding_randomness[step], &prod);
                ext_field_add(&input[x + y], &prod, &output[threadIndex]);
            }
        }
        grid.sync();
    }
}

// for the AIR columns
extern "C" __global__ void multilinears_up(const uint32_t **columns, const uint32_t n_columns, const int n_vars, uint32_t **result)
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
extern "C" __global__ void multilinears_down(uint32_t **columns, const uint32_t n_columns, const int n_vars, uint32_t **result)
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

__device__ int n_folding_repetitions(const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_folding_ops = n_slices * (1 << (slice_log_len - 1));
    return (n_folding_ops + total_n_threads - 1) / total_n_threads;
}

extern "C" __global__ void fold_ext_by_prime(const ExtField **inputs, ExtField **res, const uint32_t scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        ExtField diff;
        ext_field_sub(&inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], &inputs[slice_index][slice_offset], &diff);
        ExtField mul;
        mul_prime_and_ext_field(&diff, scalar, &mul);
        ext_field_add(&mul, &inputs[slice_index][slice_offset], &res[slice_index][slice_offset]);
    }
}

extern "C" __global__ void fold_ext_by_ext(const ExtField **inputs, ExtField **res, const ExtField *scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        ExtField diff;
        ext_field_sub(&inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], &inputs[slice_index][slice_offset], &diff);
        ExtField mul;
        ext_field_mul(&diff, scalar, &mul);
        ext_field_add(&mul, &inputs[slice_index][slice_offset], &res[slice_index][slice_offset]);
    }
}

extern "C" __global__ void dot_product(ExtField *a, ExtField *b, ExtField *res, const uint32_t log_len)
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
            ext_field_mul(&a[idx], &b[idx], &res[idx]);
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
                ext_field_add(&res[thread_index], &res[thread_index + half_len], &res[thread_index]);
            }
        }
    }
}

extern "C" __global__ void fold_sum(const ExtField *input, ExtField *output, const uint32_t len, const uint32_t sum_size)
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
            output[idx] = {0};
            for (int i = 0; i < sum_size; i++)
            {
                ext_field_add(&output[idx], &input[idx + output_len * i], &output[idx]);
            }
        }
    }
}

extern "C" __global__ void sum_in_place(ExtField *terms, const uint32_t log_len)
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
                ext_field_add(&terms[thread_index], &terms[thread_index + half_len], &terms[thread_index]);
            }
        }
        grid.sync();
    }
}