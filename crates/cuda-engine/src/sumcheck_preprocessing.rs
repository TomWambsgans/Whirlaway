use std::hash::{DefaultHasher, Hasher};

use arithmetic_circuit::{CircuitComputation, CircuitOp, ComputationInput};
use p3_field::PrimeField32;
use std::hash::Hash;

use crate::*;

const MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING: usize = 10;

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

    pub fn stack_size(&self) -> usize {
        if self.exprs.len() == 1 {
            self.exprs[0].stack_size
        } else {
            2 + self.exprs.iter().map(|c| c.stack_size).max().unwrap()
        }
    }

    pub fn uuid(&self) -> u64
    where
        F: Hash,
    {
        // TODO avoid, use a custom string instead
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
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
            std::fs::read_to_string(kernel_folder().join("sumcheck_template.txt")).unwrap();
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
    let instructions = get_sumcheck_cuda_instructions(composition);
    template
        .replace(
            "N_REGISTERS_PLACEHOLDER",
            &composition.stack_size().to_string(),
        )
        .replace(
            "N_BATCHING_SCALARS_PLACEHOLDER",
            &composition.exprs.len().to_string(),
        )
        .replace("COMPOSITION_PLACEHOLDER", &instructions)
}

fn get_sumcheck_cuda_instructions<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
) -> String {
    let mut res = String::new();
    let blank = "            ";
    let total_stack_size = sumcheck_computation.stack_size();

    for (i, inner) in sumcheck_computation.exprs.iter().enumerate() {
        res += &format!(
            "\n{blank}// computation {}/{}\n\n",
            i + 1,
            sumcheck_computation.exprs.len()
        );
        for instr in &inner.instructions {
            let op_str = match instr.op {
                CircuitOp::Product => "mul",
                CircuitOp::Sum => "add",
            };

            match (&instr.left, &instr.right) {
                (ComputationInput::Stack(stack_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}{}_prime_and_ext_field(&regs[{}], to_monty({}), &regs[{}]);\n",
                        blank,
                        op_str,
                        stack_index,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_index), ComputationInput::Scalar(scalar)) => {
                    res += &format!(
                        "{}{}_prime_and_ext_field(&multilinears[{}][thread_index], to_monty({}), &regs[{}]);\n",
                        blank,
                        op_str,
                        node_index,
                        scalar.as_canonical_u32(),
                        instr.result_location
                    )
                }
                (ComputationInput::Node(node_left), ComputationInput::Node(node_right)) => {
                    res += &format!(
                        "{}ext_field_{}(&multilinears[{}][thread_index], &multilinears[{}][thread_index], &regs[{}]);\n",
                        blank, op_str, node_left, node_right, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_index), ComputationInput::Node(node_index))
                | (ComputationInput::Node(node_index), ComputationInput::Stack(stack_index)) => {
                    res += &format!(
                        "{}ext_field_{}(&regs[{}], &multilinears[{}][thread_index], &regs[{}]);\n",
                        blank, op_str, stack_index, node_index, instr.result_location
                    )
                }
                (ComputationInput::Stack(stack_left), ComputationInput::Stack(stack_right)) => {
                    res += &format!(
                        "{}ext_field_{}(&regs[{}], &regs[{}], &regs[{}]);\n",
                        blank, op_str, stack_left, stack_right, instr.result_location
                    )
                }
                (ComputationInput::Scalar(_), _) => {
                    unreachable!("Scalar should always be on the right")
                }
            }
        }

        if sumcheck_computation.exprs.len() > 1 {
            if i == 0 {
                res += &format!(
                    "{}regs[{}] = regs[{}];\n",
                    blank,
                    total_stack_size - 1,
                    inner.stack_size - 1
                );
            } else {
                // multiply by batching scalar
                assert!(inner.stack_size >= 2, "TODO edge case");
                res += &format!(
                    "{}ext_field_mul(&regs[{}], &cached_batching_scalars[{}], &regs[{}]);\n",
                    blank,
                    inner.stack_size - 1,
                    i,
                    total_stack_size - 2,
                );
                res += &format!(
                    "{}ext_field_add(&regs[{}], &regs[{}], &regs[{}]);\n",
                    blank,
                    total_stack_size - 2,
                    total_stack_size - 1,
                    total_stack_size - 1,
                );
            }
        }
    }

    if sumcheck_computation.eq_mle_multiplier {
        assert!(total_stack_size >= 2, "TODO edge case");
        res += &format!(
            "{}regs[{}] = regs[{}];\n",
            blank,
            total_stack_size - 2,
            total_stack_size - 1
        );
        res += &format!(
            "{}ext_field_mul(&regs[{}], &multilinears[{}][thread_index], &regs[{}]);\n",
            blank,
            total_stack_size - 2,
            sumcheck_computation.n_multilinears - 1,
            total_stack_size - 1
        );
    }

    res
}
