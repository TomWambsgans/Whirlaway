use std::hash::{DefaultHasher, Hasher};

use arithmetic_circuit::{CircuitComputation, CircuitOp, ComputationInput, max_stack_size};
use p3_field::PrimeField32;
use std::hash::Hash;

use crate::*;

const MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING: usize = 10;

const MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT: usize = 3;

#[derive(Clone, Debug, Hash)]
pub struct SumcheckComputation<'a, F> {
    pub exprs: &'a [CircuitComputation<F>], // each one is multiplied by a 'batching scalar'. We assume the first batching scalar is always 1.
    pub n_multilinears: usize,              // including the eq_mle multiplier (if any)
    pub eq_mle_multiplier: bool,
}

impl<'a, F> SumcheckComputation<'a, F> {
    pub fn total_n_instructions(&self) -> usize {
        self.exprs.iter().map(|c| c.instructions.len()).sum()
    }

    pub fn n_cuda_compute_units(&self) -> usize {
        self.exprs
            .len()
            .div_ceil(MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT)
    }

    pub fn uuid(&self) -> u64
    where
        F: Hash,
    {
        // TODO avoid, use a custom string instead
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT.hash(&mut hasher);
        hasher.finish()
    }
}

pub fn cuda_preprocess_sumcheck_computation<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
) {
    let cuda = cuda_engine();
    let mut guard = cuda.functions.write().unwrap();
    let module = format!("sumcheck_{:x}", sumcheck_computation.uuid());
    if guard.contains_key(&module) {
        return;
    }
    let cuda_file = cuda_synthetic_dir().join(format!("{module}.cu"));
    if !cuda_file.exists() {
        let specialized_sumcheck_template =
            std::fs::read_to_string(kernels_folder().join("sumcheck_template.txt")).unwrap();
        let cuda_code =
            get_sumcheck_cuda_code(&specialized_sumcheck_template, sumcheck_computation);
        std::fs::write(&cuda_file, &cuda_code).unwrap();
    }

    // To avoid huge PTX file, and reduce compilation time, we may remove inlining
    let use_noinline =
        sumcheck_computation.total_n_instructions() > MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING;

    compile_module(
        cuda.dev.clone(),
        &cuda_synthetic_dir(),
        &module,
        use_noinline,
        &mut *guard,
    );
}

fn get_sumcheck_cuda_code<F: PrimeField32>(
    template: &str,
    composition: &SumcheckComputation<F>,
) -> String {
    let instructions = get_cuda_instructions(composition);
    template
        .replace(
            "N_BATCHING_SCALARS_PLACEHOLDER",
            &composition.exprs.len().to_string(),
        )
        .replace("COMPOSITION_PLACEHOLDER", &instructions)
}

fn get_cuda_instructions<F: PrimeField32>(sumcheck_computation: &SumcheckComputation<F>) -> String {
    let mut res = String::new();
    let n_compute_units = sumcheck_computation.n_cuda_compute_units();
    let blank = "            ";
    for compute_unit in 0..n_compute_units {
        let start = compute_unit * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT;
        let end = ((compute_unit + 1) * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT)
            .min(sumcheck_computation.exprs.len());
        let instructions = compute_unit_instructions(sumcheck_computation.exprs, start, end);
        res += &format!("\n{blank}case {}:\n{blank}{{\n", compute_unit);
        res += &instructions;
        res += &format!("{blank}    break;\n{blank}}}\n");
    }

    res
}

fn compute_unit_instructions<F: PrimeField32>(
    exprs: &[CircuitComputation<F>],
    start: usize,
    end: usize,
) -> String {
    let mut res = String::new();
    let blank = "                ";

    let mut n_registers = max_stack_size(exprs);
    if (end - start) > 1 {
        n_registers += 2;
    }
    for i in 0..n_registers {
        res += &format!("{}ExtField reg_{};\n", blank, i);
    }

    for (i, expr) in exprs[start..end].iter().enumerate() {
        res += &format!(
            "\n{blank}// computation {}/{}\n\n",
            i + start + 1,
            exprs.len()
        );
        for instr in &expr.instructions {
            let op_str = match instr.op {
                CircuitOp::Product => "mul",
                CircuitOp::Sum => "add",
            };

            match (&instr.left, &instr.right) {
                (ComputationInput::Stack(stack_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}{}_prime_and_ext_field(&reg_{}, to_monty({}), &reg_{});\n",
                        blank,
                        op_str,
                        stack_index,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}{}_prime_and_ext_field(&multilinears[{}][hypercube_point], to_monty({}), &reg_{});\n",
                        blank,
                        op_str,
                        node_index,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_left), ComputationInput::Node(node_right)) => {
                    // TODO avoid passing a global memory ref to ext_field_mul, cache it first to thread registers
                    res += &format!(
                        "{}ext_field_{}(&multilinears[{}][hypercube_point], &multilinears[{}][hypercube_point], &reg_{});\n",
                        blank, op_str, node_left, node_right, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_index), ComputationInput::Node(node_index))
                | (ComputationInput::Node(node_index), ComputationInput::Stack(stack_index)) => {
                    res += &format!(
                        "{}ext_field_{}(&reg_{}, &multilinears[{}][hypercube_point], &reg_{});\n",
                        blank, op_str, stack_index, node_index, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_left), ComputationInput::Stack(stack_right)) => {
                    res += &format!(
                        "{}ext_field_{}(&reg_{}, &reg_{}, &reg_{});\n",
                        blank, op_str, stack_left, stack_right, instr.result_location
                    )
                }
                (ComputationInput::Scalar(_), _) => {
                    unreachable!("Scalar should always be on the right")
                }
            }
        }

        if exprs.len() > 1 {
            if i == 0 && start == 0 {
                res += &format!(
                    "{}reg_{} = reg_{};\n",
                    blank,
                    n_registers - 1,
                    expr.stack_size - 1
                );
            } else {
                // multiply by batching scalar
                assert!(expr.stack_size >= 2, "TODO edge case");
                res += &format!(
                    "{}ext_field_mul(&reg_{}, &batching_scalars[{}], &reg_{});\n",
                    blank,
                    expr.stack_size - 1,
                    i + start,
                    n_registers - 2,
                );

                if i == 0 {
                    res += &format!("{}reg_{} = {{0}};\n", blank, n_registers - 1);
                }
                res += &format!(
                    "{}ext_field_add(&reg_{}, &reg_{}, &reg_{});\n",
                    blank,
                    n_registers - 2,
                    n_registers - 1,
                    n_registers - 1,
                );
            }
        }
    }

    res += &format!("\n{}sums[thread_index] = reg_{};\n", blank, n_registers - 1);

    res
}
