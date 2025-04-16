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
    let instructions_prime = get_cuda_instructions(composition, true);
    let instructions_ext = get_cuda_instructions(composition, false);
    template
        .replace(
            "N_BATCHING_SCALARS_PLACEHOLDER",
            &composition.exprs.len().to_string(),
        )
        .replace("COMPOSITION_PLACEHOLDER_PRIME", &instructions_prime)
        .replace("COMPOSITION_PLACEHOLDER_EXT", &instructions_ext)
}

fn get_cuda_instructions<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
    prime_field: bool,
) -> String {
    let mut res = String::new();
    let n_compute_units = sumcheck_computation.n_cuda_compute_units();
    let blank = "            ";
    for compute_unit in 0..n_compute_units {
        let start = compute_unit * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT;
        let end = ((compute_unit + 1) * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT)
            .min(sumcheck_computation.exprs.len());
        let instructions = if prime_field {
            compute_unit_instructions_prime(sumcheck_computation.exprs, start, end)
        } else {
            compute_unit_instructions_ext(sumcheck_computation.exprs, start, end)
        };
        res += &format!("\n{blank}case {}:\n{blank}{{\n", compute_unit);
        res += &instructions;
        res += &format!("{blank}    break;\n{blank}}}\n");
    }

    res
}

// TODO unify/simplify cuda generation

fn compute_unit_instructions_ext<F: PrimeField32>(
    exprs: &[CircuitComputation<F>],
    start: usize,
    end: usize,
) -> String {
    let mut res = String::new();
    let blank = "                ";

    let n_registers = max_stack_size(exprs);
    for i in 0..n_registers {
        res += &format!("{}ExtField reg_{};\n", blank, i);
    }
    res += &format!("{}ExtField temp_a;\n", blank);
    res += &format!("{}ExtField temp_b;\n", blank);
    res += &format!("{}ExtField computed = {{0}};\n", blank);

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
                CircuitOp::Sub => "sub",
            };

            match (&instr.left, &instr.right) {
                (ComputationInput::Stack(stack_index), ComputationInput::Scalar(scalar)) => {
                    let func_name = match instr.op {
                        CircuitOp::Product => "mul_prime_and_ext_field",
                        CircuitOp::Sum => "add_prime_and_ext_field",
                        CircuitOp::Sub => "sub_ext_field_and_prime",
                    };
                    res += &format!(
                        "{}{}(&reg_{}, to_monty({}), &reg_{});\n",
                        blank,
                        func_name,
                        stack_index,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}temp_a = multilinears[{}][hypercube_point];\n",
                        blank, node_index
                    );
                    let func_name = match instr.op {
                        CircuitOp::Product => "mul_prime_and_ext_field",
                        CircuitOp::Sum => "add_prime_and_ext_field",
                        CircuitOp::Sub => "sub_ext_field_and_prime",
                    };
                    res += &format!(
                        "{}{}(&temp_a, to_monty({}), &reg_{});\n",
                        blank,
                        func_name,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_left), ComputationInput::Node(node_right)) => {
                    res += &format!(
                        "{}temp_a = multilinears[{}][hypercube_point];\n",
                        blank, node_left
                    );
                    res += &format!(
                        "{}temp_b = multilinears[{}][hypercube_point];\n",
                        blank, node_right
                    );
                    res += &format!(
                        "{}ext_field_{}(&temp_a, &temp_b, &reg_{});\n",
                        blank, op_str, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_index), ComputationInput::Node(node_index)) => {
                    res += &format!(
                        "{}temp_a = multilinears[{}][hypercube_point];\n",
                        blank, node_index
                    );
                    res += &format!(
                        "{}ext_field_{}(&reg_{}, &temp_a, &reg_{});\n",
                        blank, op_str, stack_index, instr.result_location
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Stack(stack_index)) => {
                    res += &format!(
                        "{}temp_a = multilinears[{}][hypercube_point];\n",
                        blank, node_index
                    );
                    res += &format!(
                        "{}ext_field_{}(&temp_a, &reg_{}, &reg_{});\n",
                        blank, op_str, stack_index, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_left), ComputationInput::Stack(stack_right)) => {
                    res += &format!(
                        "{}ext_field_{}(&reg_{}, &reg_{}, &reg_{});\n",
                        blank, op_str, stack_left, stack_right, instr.result_location
                    )
                }
                (ComputationInput::Scalar(scalar), ComputationInput::Stack(stack_index)) => {
                    match instr.op {
                        CircuitOp::Product | CircuitOp::Sum => {
                            res += &format!(
                                "{}{}_prime_and_ext_field(&reg_{}, to_monty({}), &reg_{});\n",
                                blank,
                                op_str,
                                stack_index,
                                scalar.as_canonical_u32(),
                                instr.result_location
                            )
                        }
                        CircuitOp::Sub => {
                            res += &format!(
                                "{}sub_prime_and_ext_field(to_monty({}), &reg_{}, &reg_{});\n",
                                blank,
                                scalar.as_canonical_u32(),
                                stack_index,
                                instr.result_location
                            )
                        }
                    };
                }
                (ComputationInput::Scalar(scalar), ComputationInput::Node(node_index)) => {
                    res += &format!(
                        "{}temp_a = multilinears[{}][hypercube_point];\n",
                        blank, node_index
                    );
                    match instr.op {
                        CircuitOp::Product | CircuitOp::Sum => {
                            res += &format!(
                                "{}{}_prime_and_ext_field(&temp_a, to_monty({}), &reg_{});\n",
                                blank,
                                op_str,
                                scalar.as_canonical_u32(),
                                instr.result_location
                            )
                        }
                        CircuitOp::Sub => {
                            res += &format!(
                                "{}sub_prime_and_ext_field(to_monty({}), &temp_a, &reg_{});\n",
                                blank,
                                scalar.as_canonical_u32(),
                                instr.result_location
                            )
                        }
                    };
                }
                (ComputationInput::Scalar(_), ComputationInput::Scalar(_)) => {
                    unreachable!("Useless computation")
                }
            }
        }

        if i == 0 && start == 0 {
            res += &format!("{}computed = reg_{};\n", blank, expr.stack_size - 1);
        } else {
            // multiply by batching scalar
            res += &format!("{}temp_a = batching_scalars[{}];\n", blank, i + start,);
            res += &format!(
                "{}ext_field_mul(&reg_{}, &temp_a, &temp_b);\n",
                blank,
                expr.stack_size - 1,
            );
            res += &format!("{}ext_field_add(&temp_b, &computed, &computed);\n", blank,);
        }
    }

    res += &format!("\n{}sums[thread_index] = computed;\n", blank);

    res
}

fn compute_unit_instructions_prime<F: PrimeField32>(
    exprs: &[CircuitComputation<F>],
    start: usize,
    end: usize,
) -> String {
    let mut res = String::new();
    let blank = "                ";

    let n_registers = max_stack_size(exprs);

    for i in 0..n_registers {
        res += &format!("{}uint32_t reg_{};\n", blank, i);
    }
    res += &format!("{}ExtField temp = {{0}};\n", blank);
    res += &format!("{}ExtField computed = {{0}};\n", blank);

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
                CircuitOp::Sub => "sub",
            };

            match (&instr.left, &instr.right) {
                (ComputationInput::Stack(stack_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(reg_{}, to_monty({}));\n",
                        blank,
                        instr.result_location,
                        op_str,
                        stack_index,
                        scalar.as_canonical_u32(),
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(multilinears[{}][hypercube_point], to_monty({}));\n",
                        blank,
                        instr.result_location,
                        op_str,
                        node_index,
                        scalar.as_canonical_u32(),
                    )
                }
                (ComputationInput::Node(node_left), ComputationInput::Node(node_right)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(multilinears[{}][hypercube_point], multilinears[{}][hypercube_point]);\n",
                        blank, instr.result_location, op_str, node_left, node_right,
                    )
                }
                (ComputationInput::Stack(stack_index), ComputationInput::Node(node_index)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(reg_{}, multilinears[{}][hypercube_point]);\n",
                        blank, instr.result_location, op_str, stack_index, node_index,
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Stack(stack_index)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(multilinears[{}][hypercube_point], reg_{});\n",
                        blank, instr.result_location, op_str, node_index, stack_index,
                    )
                }
                (ComputationInput::Stack(stack_left), ComputationInput::Stack(stack_right)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(reg_{}, reg_{});\n",
                        blank, instr.result_location, op_str, stack_left, stack_right,
                    )
                }
                (ComputationInput::Scalar(scalar), ComputationInput::Stack(stack_index)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(to_monty({}), reg_{});\n",
                        blank,
                        instr.result_location,
                        op_str,
                        stack_index,
                        scalar.as_canonical_u32(),
                    )
                }
                (ComputationInput::Scalar(scalar), ComputationInput::Node(node_index)) => {
                    res += &format!(
                        "{}reg_{} = monty_field_{}(to_monty({}), multilinears[{}][hypercube_point]);\n",
                        blank,
                        instr.result_location,
                        op_str,
                        scalar.as_canonical_u32(),
                        node_index,
                    )
                }
                (ComputationInput::Scalar(_), ComputationInput::Scalar(_)) => {
                    unreachable!("Useless computation")
                }
            }
        }

        if i == 0 && start == 0 {
            res += &format!(
                "{}computed.coeffs[0] = reg_{};\n",
                blank,
                expr.stack_size - 1
            );
        } else {
            // multiply by batching scalar
            res += &format!(
                "{}mul_prime_and_ext_field(&batching_scalars[{}], reg_{}, &temp);\n",
                blank,
                i + start,
                expr.stack_size - 1
            );

            res += &format!("{}ext_field_add(&temp, &computed, &computed);\n", blank,);
        }
    }

    res += &format!("\n{}sums[thread_index] = computed;\n", blank);

    res
}
