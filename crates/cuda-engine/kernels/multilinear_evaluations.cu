#include "cassert"

#include "ff_wrapper.cu"

template <int STEPS>
__device__ __forceinline__ void eval_multilinear_in_lagrange_basis_template(uint32_t tid,
                                                                            uint32_t n_vars,
                                                                            Field_A *input,
                                                                            const Field_B *point,
                                                                            LARGER_AB *output,
                                                                            bool skip_me,
                                                                            bool sync_threads)
{

    LARGER_AB registers[1 << (STEPS - 1)];

    if (!skip_me)
    {
        for (int i = 0; i < 1 << (STEPS - 1); i++)
        {
            Field_A left = input[tid + i * (1 << (n_vars - STEPS))];
            Field_A right = input[tid + i * (1 << (n_vars - STEPS)) + (1 << (n_vars - 1))];
            Field_A diff;
            SUB_AA(right, left, diff);
            LARGER_AB prod;
            MUL_AB(diff, point[0], prod);

            ADD_A_AND_MAX_AB(left, prod, registers[i]);
        }

        for (int s = 1; s < STEPS; s++)
        {
            for (int i = 0; i < (1 << (STEPS - s - 1)); i++)
            {
                LARGER_AB left = registers[i];
                LARGER_AB right = registers[i + (1 << (STEPS - s - 1))];
                LARGER_AB diff;
                SUB_MAX_AB(right, left, diff);
                LARGER_AB prod;
                MUL_B_and_MAX_AB(point[s], diff, prod);
                ADD_MAX_AB(left, prod, registers[i]);
            }
        }
    }
    if (sync_threads)
    {
        __syncthreads();
    }

    if (!skip_me)
    {
        output[tid] = registers[0];
    }
}

__device__ __forceinline__ void eval_multilinear_in_lagrange_basis_helper(uint32_t tid,
                                                                          uint32_t n_vars,
                                                                          Field_A *input,
                                                                          const Field_B *point,
                                                                          LARGER_AB *output,
                                                                          int steps,
                                                                          bool skip_me,
                                                                          bool sync_threads)
{
    switch (steps)
    {
    case 1:
        eval_multilinear_in_lagrange_basis_template<1>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 2:
        eval_multilinear_in_lagrange_basis_template<2>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 3:
        eval_multilinear_in_lagrange_basis_template<3>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 4:
        eval_multilinear_in_lagrange_basis_template<4>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 5:
        eval_multilinear_in_lagrange_basis_template<5>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 6:
        eval_multilinear_in_lagrange_basis_template<6>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    default:
        assert(false);
    }
}

template <int STEPS>
__device__ __forceinline__ void eval_multilinear_in_lagrange_basis_template_2(uint32_t tid,
                                                                              uint32_t n_vars,
                                                                              LARGER_AB *input,
                                                                              const Field_B *point,
                                                                              LARGER_AB *output,
                                                                              bool skip_me,
                                                                              bool sync_threads)
{

    LARGER_AB registers[1 << (STEPS - 1)];

    if (!skip_me)
    {
        for (int i = 0; i < 1 << (STEPS - 1); i++)
        {

            LARGER_AB left = input[tid + i * (1 << (n_vars - STEPS))];
            LARGER_AB right = input[tid + i * (1 << (n_vars - STEPS)) + (1 << (n_vars - 1))];
            LARGER_AB diff;
            SUB_MAX_AB(right, left, diff);
            LARGER_AB prod;
            MUL_B_and_MAX_AB(point[0], diff, prod);
            ADD_MAX_AB(left, prod, registers[i]);
        }

        for (int s = 1; s < STEPS; s++)
        {
            for (int i = 0; i < (1 << (STEPS - s - 1)); i++)
            {
                LARGER_AB left = registers[i];
                LARGER_AB right = registers[i + (1 << (STEPS - s - 1))];
                LARGER_AB diff;
                SUB_MAX_AB(right, left, diff);
                LARGER_AB prod;
                MUL_B_and_MAX_AB(point[s], diff, prod);
                ADD_MAX_AB(left, prod, registers[i]);
            }
        }
    }

    if (sync_threads)
    {
        __syncthreads();
    }

    if (!skip_me)
    {
        output[tid] = registers[0];
    }
}

__device__ __forceinline__ void eval_multilinear_in_lagrange_basis_helper_2(uint32_t tid,
                                                                            uint32_t n_vars,
                                                                            LARGER_AB *input,
                                                                            const Field_B *point,
                                                                            LARGER_AB *output,
                                                                            int steps,
                                                                            bool skip_me,
                                                                            bool sync_threads)
{
    switch (steps)
    {
    case 1:
        eval_multilinear_in_lagrange_basis_template_2<1>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 2:
        eval_multilinear_in_lagrange_basis_template_2<2>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 3:
        eval_multilinear_in_lagrange_basis_template_2<3>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 4:
        eval_multilinear_in_lagrange_basis_template_2<4>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 5:
        eval_multilinear_in_lagrange_basis_template_2<5>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    case 6:
        eval_multilinear_in_lagrange_basis_template_2<6>(tid, n_vars, input, point, output, skip_me, sync_threads);
        break;
    default:
        assert(false);
    }
}

__device__ void load_point_into_shared_memory(const Field_B *point, Field_B *my_cached_point, uint32_t n_vars)
{
    int reps = (n_vars + blockDim.x - 1) / blockDim.x;
    for (int rep = 0; rep < reps; rep++)
    {
        int tid = threadIdx.x + rep * blockDim.x;
        if (tid < n_vars)
        {
            my_cached_point[tid] = point[tid];
        }
    }
    __syncthreads();
}

extern "C" __global__ void eval_multilinear_in_lagrange_basis_steps(Field_A *input, LARGER_AB *output, const Field_B *point, uint32_t n_vars, int steps_to_perform)
{
    // input and output may overlap

    extern __shared__ Field_B my_cached_point[]; // TODO constant memory
    load_point_into_shared_memory(point, my_cached_point, n_vars);

    int total_n_threads = blockDim.x * gridDim.x;

    int n_iters = 1 << (n_vars - steps_to_perform);
    int n_repetitions = (n_iters + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int tid = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (tid >= n_iters)
        {
            return;
        }

        eval_multilinear_in_lagrange_basis_helper(tid, n_vars, input, my_cached_point, output, steps_to_perform, false, false);
    }
}
// can only be used at the end
extern "C" __global__ void eval_multilinear_in_lagrange_basis_shared_memory(Field_A *input, LARGER_AB *output, const Field_B *point, uint32_t n_vars)
{
    // input and output may overlap

    assert(gridDim.x == 1);
    int log_n_threads = 31 - __clz(blockDim.x);

    extern __shared__ Field_B cache[]; // first n_vars: cached point, then the rest is used for computation
    Field_B *cached_point = cache;
    LARGER_AB *cached_output = (LARGER_AB *)(cache + n_vars);

    load_point_into_shared_memory(point, cached_point, n_vars);

    assert(log_n_threads + 1 <= n_vars);
    int steps_before_shared_memory = n_vars - log_n_threads - 1;

    if (steps_before_shared_memory > 0)
    {
        steps_before_shared_memory += 1;
        eval_multilinear_in_lagrange_basis_helper(threadIdx.x, n_vars, input, cached_point, cached_output, steps_before_shared_memory, false, false);
    }
    else
    {
        CONVERT_A_TO_MAX_AB(input[threadIdx.x], cached_output[threadIdx.x]);
        CONVERT_A_TO_MAX_AB(input[threadIdx.x + blockDim.x], cached_output[threadIdx.x + blockDim.x]);
    }

    __syncthreads();

    int remaining_steps = n_vars - steps_before_shared_memory;

    for (int i = 0; i < remaining_steps; i++)
    {
        bool skip_me = threadIdx.x > (1 << (remaining_steps - i - 1));
        eval_multilinear_in_lagrange_basis_helper_2(threadIdx.x, n_vars - steps_before_shared_memory - i, cached_output, &cached_point[steps_before_shared_memory + i], cached_output, 1, skip_me, true);
        __syncthreads();
    }

        if (threadIdx.x == 0)
    {
        output[0] = cached_output[0];
    }
}