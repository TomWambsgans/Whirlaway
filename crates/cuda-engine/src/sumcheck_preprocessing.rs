use arithmetic_circuit::{
    CircuitComputation, CircuitOp, ComputationInput, all_nodes_involved, max_stack_size,
};
use p3_field::PrimeField32;
use std::{hash::Hash, ops::Range};
use utils::{SupportedField, default_hash};

use crate::*;

const MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING: usize = 10;

const MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT: usize = 2;
const MAX_COMPUTE_UNITS_PER_KERNEL: usize = 200;

#[derive(Clone, Debug, Hash)]
pub struct SumcheckComputation<'a, F> {
    pub exprs: &'a [CircuitComputation<F>], // each one is multiplied by a 'batching scalar'. We assume the first batching scalar is always 1.
    pub n_multilinears: usize,              // including the eq_mle multiplier (if any)
    pub eq_mle_multiplier: bool,
}

impl<F> SumcheckComputation<'_, F> {
    pub fn total_n_instructions(&self) -> usize {
        self.exprs.iter().map(|c| c.instructions.len()).sum()
    }

    pub fn no_inline_cuda_ops(&self) -> bool {
        // To avoid huge PTX file, and reduce compilation time, we may remove inlining
        self.total_n_instructions() > MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING
    }

    pub fn n_cuda_compute_units(&self) -> usize {
        self.exprs
            .len()
            .div_ceil(MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT)
    }

    pub fn n_cuda_kernels(&self) -> usize {
        self.n_cuda_compute_units()
            .div_ceil(MAX_COMPUTE_UNITS_PER_KERNEL)
    }

    pub fn compute_units_range_for_kernel(&self, kernel_idx: usize) -> Range<usize> {
        let n_compute_units = self.n_cuda_compute_units();
        let n_compute_units_per_kernel = n_compute_units / self.n_cuda_kernels();
        let start = kernel_idx * n_compute_units_per_kernel;
        let end = if kernel_idx == self.n_cuda_kernels() - 1 {
            n_compute_units
        } else {
            start + n_compute_units_per_kernel
        };
        start..end
    }

    pub fn uuid(&self, compute_units_range: &Range<usize>) -> String
    where
        F: Hash,
    {
        // TODO avoid, use a custom string instead
        let hash = default_hash((
            self,
            MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT,
            MAX_COMPUTE_UNITS_PER_KERNEL,
        ));
        format!(
            "{:x}_{}_{}",
            hash, compute_units_range.start, compute_units_range.end
        )
    }
}

pub fn cuda_preprocess_many_sumcheck_computations<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
    ext_degrees: &[(usize, usize, usize)],
) {
    for (extension_degree_a, extension_degree_b, extension_degree_c) in ext_degrees {
        cuda_preprocess_sumcheck_computation(
            sumcheck_computation,
            *extension_degree_a,
            *extension_degree_b,
            *extension_degree_c,
        );
    }
}

pub fn cuda_preprocess_sumcheck_computation<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
    extension_degree_a: usize,
    extension_degree_b: usize,
    extension_degree_c: usize,
) {
    // a: scalars inside each constraint
    // b: input data (the multilinears)
    // c: batching_scalars (ie output data)
    for kernel_idx in 0..sumcheck_computation.n_cuda_kernels() {
        cuda_preprocess_sumcheck_computation_chunck(
            sumcheck_computation,
            extension_degree_a,
            extension_degree_b,
            extension_degree_c,
            kernel_idx,
        );
    }
}

fn cuda_preprocess_sumcheck_computation_chunck<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
    extension_degree_a: usize,
    extension_degree_b: usize,
    extension_degree_c: usize,
    kernel_idx: usize,
) {
    let compute_units_range = sumcheck_computation.compute_units_range_for_kernel(kernel_idx);
    let cuda = cuda_engine();
    let field = SupportedField::guess::<F>();
    let module = sumcheck_computation.uuid(&compute_units_range);
    let cache_memory_reads = extension_degree_b == 1; // TODO: use a better heuristic
    let cuda_file = cuda_synthetic_dir().join(format!("{module}.cu"));
    let options = CudaFunctionInfo {
        cuda_file: cuda_file.clone(),
        function_name: "compute_over_hypercube".to_string(),
        field: Some(field),
        extension_degree_a: Some(extension_degree_a),
        extension_degree_b: Some(extension_degree_b),
        extension_degree_c: Some(extension_degree_c),
        max_ntt_log_size_at_block_level: None,
        no_inline: sumcheck_computation.no_inline_cuda_ops(),
        cache_memory_reads,
    };
    let mut functions = cuda.functions.write().unwrap();
    if functions.contains_key(&options) {
        return;
    }
    if !cuda_file.exists() {
        let template =
            std::fs::read_to_string(kernels_folder().join("sumcheck_template.txt")).unwrap();
        let cuda_code = template.replace(
            "COMPOSITION_PLACEHOLDER",
            &get_cuda_instructions(sumcheck_computation, compute_units_range),
        );
        std::fs::write(&cuda_file, &cuda_code).unwrap();
    }

    load_function(options, &mut functions);
}

fn get_cuda_instructions<F: PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
    compute_units_range: Range<usize>,
) -> String {
    let mut res = String::new();
    let blank = "        ";
    for compute_unit in compute_units_range {
        let start = compute_unit * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT;
        let end = ((compute_unit + 1) * MAX_CONSTRAINTS_PER_CUDA_COMPUTE_UNIT)
            .min(sumcheck_computation.exprs.len());
        res += &format!("\n{blank}case {}:\n{blank}{{\n", compute_unit);
        res += &compute_unit_instructions(sumcheck_computation.exprs, start, end);
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
    let blank = "            ";

    let n_registers = max_stack_size(exprs);

    res += &format!("{blank}#if CACHED\n");
    for i in all_nodes_involved(&exprs[start..end]) {
        res += &format!(
            "{}Field_B node_{} = multilinears[{}][hypercube_point];\n",
            blank, i, i
        );
    }
    res += &format!("{blank}#endif\n\n");

    for i in 0..n_registers {
        res += &format!("{}Field_B reg_{};\n", blank, i);
    }
    res += &format!("{}Field_B temp_b0;\n", blank);
    res += &format!("{}Field_B temp_b1;\n", blank);
    res += &format!("{}Field_C temp_c0;\n", blank);
    res += &format!("{}Field_C temp_c1;\n", blank);
    res += &format!("{}Field_C computed = {{0}};\n", blank);

    for (i, expr) in exprs[start..end].iter().enumerate() {
        res += &format!(
            "\n{blank}// computation {}/{}\n\n",
            i + start + 1,
            exprs.len()
        );
        for instr in &expr.instructions {
            let op_str = match instr.op {
                CircuitOp::Product => "MUL",
                CircuitOp::Sum => "ADD",
                CircuitOp::Sub => "SUB",
            };
            let (left_type, left) =
                type_and_input_str(instr.op, &instr.left, &mut res, blank, true);
            let (right_type, right) =
                type_and_input_str(instr.op, &instr.right, &mut res, blank, false);

            res += &format!(
                "{blank}{op_str}_{left_type}{right_type}({left}, {right}, reg_{});\n",
                instr.result_location
            )
        }

        if i == 0 && start == 0 {
            res += &format!(
                "{}FIELD_CONVERSION_B_C(reg_{}, computed);\n",
                blank,
                expr.stack_size - 1
            );
        } else {
            // multiply by batching scalar
            res += &format!("{}temp_c0 = batching_scalars[{}];\n", blank, i + start,);
            res += &format!(
                "{}MUL_BC(reg_{}, temp_c0, temp_c1);\n",
                blank,
                expr.stack_size - 1,
            );
            res += &format!("{}ADD_CC(temp_c1, computed, computed);\n", blank,);
        }
    }

    res += &format!("\n{}sums[target_sum_index] = computed;\n", blank);

    res
}

fn type_and_input_str<F: PrimeField32>(
    op: CircuitOp,
    input: &ComputationInput<F>,
    res: &mut String,
    blank: &str,
    left: bool,
) -> (&'static str, String) {
    match input {
        ComputationInput::Scalar(scalar) => (
            "A",
            format!("Field_A::from_canonical({})", scalar.as_canonical_u32()),
        ),
        ComputationInput::Stack(stack_index) => ("B", format!("reg_{}", stack_index)),
        ComputationInput::Node(node_index) => ("B", {
            if op == CircuitOp::Product {
                let temp_var = if left { "temp_b0" } else { "temp_b1" };
                *res += &format!("{}{temp_var} = NODE({});\n", blank, node_index);
                temp_var.to_string()
            } else {
                format!("NODE({})", node_index)
            }
        }),
    }
}
