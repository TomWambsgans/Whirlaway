
#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "ff_wrapper.cu"

template <typename Field_A, typename Field_B>
struct TensorAlgebra
{
    Field_A coeffs[Field_B::EXTENSION_DEGREE][Field_B::EXTENSION_DEGREE];

    __device__ static void add(TensorAlgebra<Field_A, Field_B> *a,
                               TensorAlgebra<Field_A, Field_B> *b,
                               TensorAlgebra<Field_A, Field_B> *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < Field_B::EXTENSION_DEGREE; i++)
        {
            for (int j = 0; j < Field_B::EXTENSION_DEGREE; j++)
            {
                ADD_AA(a->coeffs[i][j], b->coeffs[i][j], result->coeffs[i][j]);
            }
        }
    }

    __device__ static void phi_0_times_phi_1(Field_B *a,
                                             Field_B *b,
                                             TensorAlgebra<Field_A, Field_B> *result)
    {
        for (int i = 0; i < Field_B::EXTENSION_DEGREE; i++)
        {
            for (int j = 0; j < Field_B::EXTENSION_DEGREE; j++)
            {
                MUL_AA(a->coeffs[i], b->coeffs[j], result->coeffs[i][j]);
            }
        }
    }
};

extern "C" __global__ void tensor_algebra_dot_product(Field_B *left, Field_B *right, Field_A *buff, Field_A *result, uint32_t log_len, uint32_t log_n_tasks_per_thread)
{
    // left and right have size 2^log_len
    // buff has size EXT_DEGREE^2 * 2^(log_len - log_n_tasks_per_thread)
    // res has size EXT_DEGREE^2

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int n_total_threads = blockDim.x * gridDim.x;
    int n_reps = ((1 << (log_len - log_n_tasks_per_thread)) + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx >= 1 << (log_len - log_n_tasks_per_thread))
        {
            break;
        }
        TensorAlgebra<Field_A, Field_B> sum = {0};
        for (int task = 0; task < 1 << log_n_tasks_per_thread; task++)
        {
            int offset = idx * (1 << log_n_tasks_per_thread) + task;
            Field_B l = left[offset];
            Field_B r = right[offset];
            TensorAlgebra<Field_A, Field_B> res;
            TensorAlgebra<Field_A, Field_B>::phi_0_times_phi_1(&l, &r, &res);
            TensorAlgebra<Field_A, Field_B>::add(&sum, &res, &sum);
        }
        int shift = 0;
        for (int i = 0; i < Field_B::EXTENSION_DEGREE; i++)
        {
            for (int j = 0; j < Field_B::EXTENSION_DEGREE; j++)
            {
                buff[shift + idx] = sum.coeffs[i][j];
                shift += 1 << (log_len - log_n_tasks_per_thread);
            }
        }
    }

    int w = log_len - log_n_tasks_per_thread;
    // Sum
    for (int step = 0; step < w; step++)
    {
        grid.sync();
        int half_size = 1 << (w - step - 1);
        int n_ops = half_size * Field_B::EXTENSION_DEGREE * Field_B::EXTENSION_DEGREE;
        n_reps = (n_ops + n_total_threads - 1) / n_total_threads;
        for (int rep = 0; rep < n_reps; rep++)
        {
            int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
            if (thread_index < n_ops)
            {
                int offset = (thread_index / half_size) << w;
                int m = thread_index % half_size;
                ADD_AA(buff[offset + m], buff[offset + m + half_size], buff[offset + m]);
            }
        }
    }

    grid.sync();
    n_reps = (Field_B::EXTENSION_DEGREE * Field_B::EXTENSION_DEGREE + n_total_threads - 1) / n_total_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        int idx = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        if (idx < Field_B::EXTENSION_DEGREE * Field_B::EXTENSION_DEGREE)
        {
            result[idx] = buff[idx << (log_len - log_n_tasks_per_thread)];
        }
    }
}