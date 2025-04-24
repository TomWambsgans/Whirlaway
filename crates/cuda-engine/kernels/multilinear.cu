
#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "finite_field.cu"

// (+ reverse vars)
extern "C" __global__ void monomial_to_lagrange_basis_rev(BigField *input_coeffs, BigField *buff, BigField *output_evals, const uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_iters = (1 << (n_vars - 1));
    const int n_total_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    for (int step = 0; step < n_vars; step++)
    {
        // switch back and forth between buffers

        BigField *input;
        BigField *output;

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
                BigField left = input[x + 2 * y];
                output[x + y] = left;
                BigField::add(&left, &input[x + (2 * y) + 1], &output[x + y + half_size]);
            }
        }

        grid.sync();
    }
}

// Could also be done in place
extern "C" __global__ void lagrange_to_monomial_basis(BigField *input_evals, BigField *output_coeffs, const uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_total_threads = blockDim.x * gridDim.x;
    int n_iters = (1 << n_vars);
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
    n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    // 2) compute the monomial ceffs
    for (int step = 0; step < n_vars; step++)
    {
        const int half_size = 1 << step;
        for (int rep = 0; rep < n_repetitions; rep++)
        {
            const int idx = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (idx < n_iters)
            {
                const int x = (idx / half_size) * 2 * half_size;
                const int y = idx % half_size;
                BigField::sub(&output_coeffs[x + y + half_size], &output_coeffs[x + y], &output_coeffs[x + y + half_size]);
            }
        }

        grid.sync();
    }
}

// Could also be done in place
extern "C" __global__ void lagrange_to_monomial_basis_rev(BigField *input_evals, BigField *buff, BigField *output_coeffs, const uint32_t n_vars)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_total_threads = blockDim.x * gridDim.x;
    int n_iters = (1 << (n_vars - 1));
    int n_repetitions = (n_iters + n_total_threads - 1) / n_total_threads;

    // 2) compute the monomial ceffs
    for (int step = 0; step < n_vars; step++)
    {
        const int half = 1 << (n_vars - step - 1);
        for (int rep = 0; rep < n_repetitions; rep++)
        {

            const int idx = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
            if (idx >= n_iters)
            {
                continue;
            }

            const int start = (idx / half) * 2 * half;
            const int offset = idx % half;

            BigField *inputs = step == 0 ? input_evals : (n_vars % 2 != step % 2 ? buff : output_coeffs);
            BigField *outputs = n_vars % 2 == step % 2 ? buff : output_coeffs;

            BigField even = inputs[start + offset * 2];
            BigField odd = inputs[start + offset * 2 + 1];
            BigField diff;
            BigField::sub(&odd, &even, &diff);

            outputs[start + offset] = even;
            outputs[start + offset + half] = diff;
        }
        grid.sync();
    }
}

extern "C" __global__ void eval_ext_multilinear_at_ext_point_in_monomial_basis(BigField *coeffs, BigField *point, const uint32_t n_vars, BigField *buff)
{
    // coeffs and buff have size 2^n_vars
    // result is stored at the last index of the buffer

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < n_vars; step++)
    {
        BigField *input;
        BigField *output;

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
                BigField prod;
                BigField::mul(&input[threadIndex + (1 << (n_vars - 1 - step))], &point[step], &prod);
                BigField::add(&input[threadIndex], &prod, &output[threadIndex]);
            }
        }
        grid.sync();
    }
}

extern "C" __global__ void eq_mle(BigField *point, const uint32_t n_vars, BigField *res)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        res[0] = BigField{0};
        res[0].coeffs[0] = SmallField::from_canonical(1);
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
                BigField::mul(&res[threadIndex], &point[n_vars - 1 - step], &res[threadIndex + (1 << step)]);
                BigField::sub(&res[threadIndex], &res[threadIndex + (1 << step)], &res[threadIndex]);
            }
        }
        grid.sync();
    }
}

extern "C" __global__ void scale_ext_slice_in_place(BigField *slice, const uint32_t len, BigField *scalar)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            BigField prod;
            BigField::mul(&slice[threadIndex], scalar, &prod);
            slice[threadIndex] = prod;
        }
    }
}

extern "C" __global__ void scale_prime_slice_by_ext(SmallField *slice, const uint32_t len, BigField *scalar, BigField *res)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            BigField::mul_small_field(scalar, slice[threadIndex], &res[threadIndex]);
        }
    }
}

extern "C" __global__ void add_slices(const BigField **slices, BigField *res, const uint32_t n_slices, const uint32_t len)
{
    const int total_n_threads = blockDim.x * gridDim.x;

    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            BigField sum = {0};
            for (int i = 0; i < n_slices; i++)
            {
                BigField::add(&slices[i][threadIndex], &sum, &sum);
            }
            res[threadIndex] = sum;
        }
    }
}

extern "C" __global__ void add_assign_slices(BigField *a, BigField *b, const uint32_t len)
{
    // a += b
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_repetitions = (len + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex < len)
        {
            BigField::add(&a[threadIndex], &b[threadIndex], &a[threadIndex]);
        }
    }
}

extern "C" __global__ void whir_fold(BigField *coeffs, const uint32_t n_vars, const uint32_t folding_factor, BigField *folding_randomness, BigField *buff, BigField *res)
{
    // coeffs has length 2^n_vars

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int total_n_threads = blockDim.x * gridDim.x;

    for (int step = 0; step < folding_factor; step++)
    {
        BigField *input;
        BigField *output;

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
                BigField prod;
                BigField::mul(&input[x + y + helf_step_size], &folding_randomness[step], &prod);
                BigField::add(&input[x + y], &prod, &output[threadIndex]);
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

extern "C" __global__ void fold_prime_by_prime(const SmallField **inputs, SmallField **res, const SmallField *scalars, const uint32_t n_slices, const uint32_t slice_log_len, const uint32_t log_n_scalars)
{
    // scalar contains 2^folding_factor elements
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
        SmallField sum = SmallField::from_canonical(0);
        for (int i = 0; i < 1 << log_n_scalars; i++)
        {
            SmallField term = SmallField::mul(inputs[slice_index][slice_offset + (i << (slice_log_len - log_n_scalars))], scalars[i]);
            sum = SmallField::add(sum, term);
        }
        res[slice_index][slice_offset] = sum;
    }
}

extern "C" __global__ void fold_prime_by_ext(const SmallField **inputs, BigField **res, const BigField *scalars, const uint32_t n_slices, const uint32_t slice_log_len, const uint32_t log_n_scalars)
{
    // scalar contains 2^folding_factor elements
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
        BigField sum = {0};
        for (int i = 0; i < 1 << log_n_scalars; i++)
        {
            BigField term;
            BigField::mul_small_field(&scalars[i], inputs[slice_index][slice_offset + (i << (slice_log_len - log_n_scalars))], &term);
            BigField::add(&sum, &term, &sum);
        }
        res[slice_index][slice_offset] = sum;
    }
}

extern "C" __global__ void fold_ext_by_prime(const BigField **inputs, BigField **res, const SmallField *scalars, const uint32_t n_slices, const uint32_t slice_log_len, const uint32_t log_n_scalars)
{
    // scalar contains 2^folding_factor elements
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
        BigField sum = {0};
        for (int i = 0; i < 1 << log_n_scalars; i++)
        {
            BigField term;
            BigField::mul_small_field(&inputs[slice_index][slice_offset + (i << (slice_log_len - log_n_scalars))], scalars[i], &term);
            BigField::add(&sum, &term, &sum);
        }
        res[slice_index][slice_offset] = sum;
    }
}

extern "C" __global__ void fold_ext_by_ext(const BigField **inputs, BigField **res, const BigField *scalars, const uint32_t n_slices, const uint32_t slice_log_len, const uint32_t log_n_scalars)
{
    // scalar contains 2^folding_factor elements
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
        BigField sum = {0};
        for (int i = 0; i < 1 << log_n_scalars; i++)
        {
            BigField term;
            BigField::mul(&scalars[i], &inputs[slice_index][slice_offset + (i << (slice_log_len - log_n_scalars))], &term);
            BigField::add(&sum, &term, &sum);
        }
        res[slice_index][slice_offset] = sum;
    }
}

extern "C" __global__ void dot_product_ext_ext(BigField *a, BigField *b, BigField *res, const uint32_t log_len)
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
            BigField::mul(&a[idx], &b[idx], &res[idx]);
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
                BigField::add(&res[thread_index], &res[thread_index + half_len], &res[thread_index]);
            }
        }
    }
}

extern "C" __global__ void dot_product_ext_prime(BigField *a, SmallField *b, BigField *res, const uint32_t log_len)
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
            BigField::mul_small_field(&a[idx], b[idx], &res[idx]);
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
                BigField::add(&res[thread_index], &res[thread_index + half_len], &res[thread_index]);
            }
        }
    }
}

extern "C" __global__ void piecewise_linear_comb(const SmallField *input, BigField *output, BigField *scalars, const uint32_t len, const uint32_t n_scalars)
{
    // len must be a multiple of n_scalars
    // input has size len
    // output has size len / n_scalars
    // output[0] = input[0].scalars[0] + input[output_len].scalars[1] + ... + input[output_len * (n_scalars - 1)].scalars[n_scalars - 1]
    // output[1] = input[1].scalars[0] + input[output_len + 1].scalars[1] + ... + input[output_len * (n_scalars - 1) + 1].scalars[n_scalars - 1]
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
            BigField comb = {0};
            for (int i = 0; i < n_scalars; i++)
            {
                BigField scalar = scalars[i];
                BigField prod;
                BigField::mul_small_field(&scalar, input[idx * n_scalars + i], &prod);
                BigField::add(&comb, &prod, &comb);
            }
            output[idx] = comb;
        }
    }
}

extern "C" __global__ void linear_combination_of_prime_slices_by_ext_scalars(const SmallField **inputs, BigField *output, BigField *scalars, const uint32_t len, const uint32_t n_scalars)
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
        BigField comb = {0};
        for (int i = 0; i < n_scalars; i++)
        {
            BigField scalar = scalars[i];
            BigField prod;
            BigField::mul_small_field(&scalar, inputs[i][idx], &prod);
            BigField::add(&comb, &prod, &comb);
        }
        output[idx] = comb;
    }
}

extern "C" __global__ void piecewise_sum(const BigField *input, BigField *output, const uint32_t len, const uint32_t sum_size)
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
            BigField sum = {0};
            for (int i = 0; i < sum_size; i++)
            {
                BigField::add(&sum, &input[idx + output_len * i], &sum);
            }
            output[idx] = sum;
        }
    }
}

extern "C" __global__ void repeat_slice_from_outside(const BigField *input, BigField *output, const uint32_t len, const uint32_t n_repetitions)
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
        BigField value = input[idx];
        for (int i = 0; i < n_repetitions; i++)
        {
            output[idx + i * len] = value;
        }
    }
}

extern "C" __global__ void repeat_slice_from_inside(const BigField *input, BigField *output, const uint32_t len, const uint32_t n_repetitions)
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

extern "C" __global__ void tensor_algebra_dot_product(const BigField *left, BigField *right, SmallField *buff, SmallField *result, const uint32_t log_len, const uint32_t log_n_tasks_per_thread)
{
    // left and right have size 2^log_len
    // buff has size EXT_DEGREE^2 * 2^(log_len - log_n_tasks_per_thread)
    // res has size EXT_DEGREE^2

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_total_threads = blockDim.x * gridDim.x;
    int n_reps = ((1 << (log_len - log_n_tasks_per_thread)) + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= 1 << (log_len - log_n_tasks_per_thread))
        {
            break;
        }
        TensorAlgebra sum = {0};
        for (int task = 0; task < 1 << log_n_tasks_per_thread; task++)
        {
            const int offset = idx * (1 << log_n_tasks_per_thread) + task;
            BigField l = left[offset];
            BigField r = right[offset];
            TensorAlgebra res;
            TensorAlgebra::phi_0_times_phi_1(&l, &r, &res);
            TensorAlgebra::add(&sum, &res, &sum);
        }
        int shift = 0;
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            for (int j = 0; j < EXT_DEGREE; j++)
            {
                buff[shift + idx] = sum.coeffs[i][j];
                shift += 1 << (log_len - log_n_tasks_per_thread);
            }
        }
    }

    const int w = log_len - log_n_tasks_per_thread;
    // Sum
    for (int step = 0; step < w; step++)
    {
        grid.sync();
        const int half_size = 1 << (w - step - 1);
        const int n_ops = half_size * EXT_DEGREE * EXT_DEGREE;
        n_reps = (n_ops + n_total_threads - 1) / n_total_threads;
        for (int rep = 0; rep < n_reps; rep++)
        {
            const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
            if (thread_index < n_ops)
            {
                const int offset = (thread_index / half_size) << w;
                const int m = thread_index % half_size;
                buff[offset + m] = SmallField::add(buff[offset + m], buff[offset + m + half_size]);
            }
        }
    }

    grid.sync();
    n_reps = (EXT_DEGREE * EXT_DEGREE + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        const int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < EXT_DEGREE * EXT_DEGREE)
        {
            result[idx] = buff[idx << (log_len - log_n_tasks_per_thread)];
        }
    }
}

extern "C" __global__ void sum_in_place(BigField *terms, const uint32_t log_len)
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
                BigField::add(&terms[thread_index], &terms[thread_index + half_len], &terms[thread_index]);
            }
        }
        grid.sync();
    }
}