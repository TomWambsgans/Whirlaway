
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

extern "C" __global__ void scale_slice_in_place(ExtField *slice, const uint32_t len, ExtField *scalar)
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
