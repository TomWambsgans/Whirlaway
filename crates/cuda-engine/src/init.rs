use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use arithmetic_circuit::{CircuitComputation, CircuitOp, ComputationInput};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use p3_field::{Field, PrimeField32, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;
use utils::powers_parallel;

pub struct CudaInfo {
    pub(crate) _dev: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    twiddles: CudaSlice<u32>, // We restrain ourseleves to the 2-addic roots of unity in the prime fields, so each one is represented by a u32
    correction_twiddles: CudaSlice<u32>, // Same remark as above
    pub whir_folding_factor: usize,
    pub two_adicity: usize,
    functions: HashMap<String, HashMap<&'static str, CudaFunction>>, // module => function_name => cuda_function
}

const MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING: usize = 10;

static CUDA_INFO: OnceLock<CudaInfo> = OnceLock::new();

pub fn cuda_info() -> &'static CudaInfo {
    CUDA_INFO.get().expect("CUDA not initialized")
}

impl CudaInfo {
    pub fn get_function(&self, module: &str, func_name: &str) -> &CudaFunction {
        self.functions
            .get(module)
            .and_then(|f| f.get(func_name))
            .unwrap_or_else(|| panic!("Function {func_name} not found in module {module}"))
    }

    pub fn twiddles<F: Field>(&self) -> CudaSlice<F> {
        // SAFETY: F should be the same field as the one used at initialization
        assert!(F::bits() <= 32);
        unsafe { std::mem::transmute(self.twiddles.clone()) }
    }

    pub fn correction_twiddles<F: Field>(&self) -> CudaSlice<F> {
        // SAFETY: F should be the same field as the one used at initialization
        assert!(F::bits() <= 32);
        unsafe { std::mem::transmute(self.correction_twiddles.clone()) }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct SumcheckComputation<F> {
    pub inner: Vec<CircuitComputation<F>>, // each one is multiplied by a 'batching scalar'. We assume the first batching scalar is always 1.
    pub n_multilinears: usize,             // including the eq_mle multiplier (if any)
    pub eq_mle_multiplier: bool,
}

impl<F> SumcheckComputation<F> {
    pub fn total_n_instructions(&self) -> usize {
        self.inner.iter().map(|c| c.instructions.len()).sum()
    }

    pub fn stack_size(&self) -> usize {
        if self.inner.len() == 1 {
            self.inner[0].stack_size
        } else {
            2 + self.inner.iter().map(|c| c.stack_size).max().unwrap()
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

#[instrument(name = "CUDA initialization", skip_all)]
pub fn init<F: TwoAdicField + PrimeField32>(
    sumcheck_computations: &[SumcheckComputation<F>],
    whir_folding_factor: usize,
) {
    let _ = CUDA_INFO.get_or_init(|| _init(sumcheck_computations, whir_folding_factor));
}

fn _init<F: TwoAdicField + PrimeField32>(
    sumcheck_computations: &[SumcheckComputation<F>],
    whir_folding_factor: usize,
) -> CudaInfo {
    let dev = CudaContext::new(0).unwrap();

    // TODO avoid this ugly trick
    let mut kernels_folder = Path::new("kernels");
    if !kernels_folder.exists() {
        kernels_folder = Path::new("crates/cuda-engine/kernels")
    };
    if !kernels_folder.exists() {
        kernels_folder = Path::new("../cuda-engine/kernels")
    }
    assert!(kernels_folder.exists());

    let build_dir = kernels_folder.join("build");
    fs::create_dir_all(&build_dir).unwrap();

    let ptx_dir = build_dir.join("ptx");
    fs::create_dir_all(&ptx_dir).unwrap();

    let cuda_synthetic_dir = build_dir.join("cuda_synthetic");
    fs::create_dir_all(&cuda_synthetic_dir).unwrap();

    let mut functions = HashMap::new();

    for (module, func_names) in [
        ("keccak", vec!["batch_keccak256"]),
        (
            "ntt",
            vec!["ntt_global", "expanded_ntt", "restructure_evaluations"],
        ),
        (
            "sumcheck_folding",
            vec![
                "fold_prime_by_prime",
                "fold_prime_by_ext",
                "fold_ext_by_prime",
                "fold_ext_by_ext",
            ],
        ),
        (
            "multilinear",
            vec![
                "monomial_to_lagrange_basis",
                "lagrange_to_monomial_basis",
                "eq_mle",
                "scale_slice_in_place",
                "add_slices",
                "add_assign_slices",
                "whir_fold",
                "eval_multilinear_in_lagrange_basis",
                "eval_multilinear_in_monomial_basis",
            ],
        ),
    ] {
        compile_module(
            dev.clone(),
            &kernels_folder.join(format!("{module}.cu")),
            &ptx_dir,
            module,
            func_names,
            false,
            &mut functions,
        );
    }

    let specialized_sumcheck_template =
        std::fs::read_to_string(Path::new(kernels_folder).join("sumcheck_template.txt")).unwrap();

    for composition in sumcheck_computations {
        let module = format!("sumcheck_{:x}", composition.uuid());
        let file = cuda_synthetic_dir.join(format!("{module}.cu"));
        if !file.exists() {
            let cuda = get_specialized_sumcheck_cuda(&specialized_sumcheck_template, composition);
            std::fs::write(&file, &cuda).unwrap();
        }

        // To avoid huge PTX file, and reduce compilation time, we may remove inlining
        let use_noinline =
            composition.total_n_instructions() > MAX_SUMCHECK_INSTRUCTIONS_TO_REMOVE_INLINING;

        compile_module(
            dev.clone(),
            &file,
            &ptx_dir,
            &module,
            vec!["sum_over_hypercube_ext"],
            use_noinline,
            &mut functions,
        );
    }

    let stream = dev.default_stream();

    let twiddles = unsafe { std::mem::transmute(store_twiddles::<F>(&stream).unwrap()) };
    let correction_twiddles = unsafe {
        std::mem::transmute(store_correction_twiddles::<F>(&stream, whir_folding_factor).unwrap())
    };

    CudaInfo {
        _dev: dev,
        stream,
        twiddles,
        correction_twiddles,
        two_adicity: F::TWO_ADICITY,
        whir_folding_factor,
        functions,
    }
}

fn compile_module(
    dev: Arc<CudaContext>,
    cuda_file: &PathBuf,
    ptx_dir: &PathBuf,
    module: &str,
    func_names: Vec<&'static str>,
    use_noinline: bool,
    functions: &mut HashMap<String, HashMap<&'static str, CudaFunction>>, // module => function_name => cuda_function
) {
    let ptx_file = ptx_dir.join(format!("{module}.ptx"));
    let (major, minor) = cuda_compute_capacity().expect("Failed to get CUDA compute capability");

    let source_modified = Path::new(&cuda_file)
        .metadata()
        .expect(&format!("Cannot find {}", cuda_file.display()))
        .modified()
        .unwrap();

    let should_compile = if ptx_file.exists() {
        let target_modified = ptx_file.metadata().unwrap().modified().unwrap();
        source_modified > target_modified
    } else {
        true
    };

    if should_compile {
        let _span = tracing::info_span!("Compiling CUDA module", module = module,).entered();
        // Create directory if it doesn't exist
        if let Some(parent) = ptx_file.parent() {
            fs::create_dir_all(parent).unwrap();
        }

        // Run nvcc to compile the CUDA code to PTX
        let mut command = Command::new("nvcc");

        let cuda_file_size = fs::metadata(cuda_file).unwrap().len() / 1024;
        if cuda_file_size > 50 {
            println!(
                "Compiling a big cuda file ({} kb), may take a while...",
                cuda_file_size
            );
        }

        command.args(&[
            &cuda_file.to_string_lossy() as &str, // Input file
            "--ptx",
            &format!("-arch=sm_{major}{minor}"), // NOT SURE OF THIS
            "-o",
            &ptx_file.to_string_lossy(), // Output file
        ]);

        if use_noinline {
            command.arg("-DUSE_NOINLINE");
        }

        let output = command
            .output()
            .expect(&format!("Failed to compile {} with nvcc", module));

        if !output.status.success() {
            panic!("NVCC error: {}", String::from_utf8_lossy(&output.stderr));
        }
    } else {
        tracing::info!("Using cached PTX for {}", module);
    }

    let ptx_content = std::fs::read_to_string(ptx_file).expect("Failed to read PTX file");
    let my_module = dev.load_module(Ptx::from_src(ptx_content)).unwrap();
    functions.insert(
        module.to_string(),
        func_names
            .iter()
            .map(|fn_name| (*fn_name, my_module.load_function(fn_name).unwrap()))
            .collect::<HashMap<_, _>>(),
    );
}

#[instrument(name = "pre-processing twiddles for CUDA NTT", skip_all)]
fn store_twiddles<F: TwoAdicField>(stream: &Arc<CudaStream>) -> Result<CudaSlice<F>, DriverError> {
    assert!(F::bits() <= 32);
    let mut all_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        all_twiddles.extend(powers_parallel(F::two_adic_generator(i), 1 << i));
    }
    let mut all_twiddles_dev = unsafe { stream.alloc::<F>(all_twiddles.len()).unwrap() };
    stream
        .memcpy_htod(&all_twiddles, &mut all_twiddles_dev)
        .unwrap();
    stream.synchronize().unwrap();
    Ok(all_twiddles_dev)
}

#[instrument(name = "pre-processing correction twiddles for CUDA NTT", skip_all)]
fn store_correction_twiddles<F: TwoAdicField>(
    stream: &Arc<CudaStream>,
    whir_folding_factor: usize,
) -> Result<CudaSlice<F>, DriverError> {
    assert!(F::bits() <= 32);
    let folding_size = 1 << whir_folding_factor;
    let size_inv = F::from_u64(folding_size).inverse();
    // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
    let mut all_correction_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY - whir_folding_factor {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        let inv_root = F::two_adic_generator(i + whir_folding_factor).inverse();
        let inv_powers = powers_parallel(inv_root, 1 << (i + whir_folding_factor));
        let correction_twiddles = (0..1 << (i + whir_folding_factor))
            .into_par_iter()
            .map(|j| size_inv * inv_powers[((j % folding_size) * (j / folding_size)) as usize])
            .collect::<Vec<_>>();
        all_correction_twiddles.extend(correction_twiddles);
    }
    let mut all_correction_twiddles_dev =
        unsafe { stream.alloc::<F>(all_correction_twiddles.len()).unwrap() };
    stream
        .memcpy_htod(&all_correction_twiddles, &mut all_correction_twiddles_dev)
        .unwrap();
    stream.synchronize().unwrap();
    Ok(all_correction_twiddles_dev)
}

fn cuda_compute_capacity() -> Result<(i32, i32), Box<dyn Error>> {
    let dev = CudaContext::new(0)?;
    let major = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

    Ok((major, minor))
}

fn get_specialized_sumcheck_cuda<F: TwoAdicField + PrimeField32>(
    template: &str,
    composition: &SumcheckComputation<F>,
) -> String {
    let composition_instructions = get_specialized_sumcheck_generic_instructions(composition);
    template
        .replace(
            "N_REGISTERS_PLACEHOLDER",
            &composition.stack_size().to_string(),
        )
        .replace(
            "N_BATCHING_SCALARS_PLACEHOLDER",
            &composition.inner.len().to_string(),
        )
        .replace("COMPOSITION_PLACEHOLDER", &composition_instructions)
}

fn get_specialized_sumcheck_generic_instructions<F: TwoAdicField + PrimeField32>(
    sumcheck_computation: &SumcheckComputation<F>,
) -> String {
    let mut res = String::new();
    let blank = "            ";
    let total_stack_size = sumcheck_computation.stack_size();

    for (i, inner) in sumcheck_computation.inner.iter().enumerate() {
        res += &format!(
            "\n{blank}// computation {}/{}\n\n",
            i + 1,
            sumcheck_computation.inner.len()
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

        if sumcheck_computation.inner.len() > 1 {
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
