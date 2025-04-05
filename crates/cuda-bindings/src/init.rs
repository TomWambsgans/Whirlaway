use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use algebra::pols::{CircuitComputation, CircuitOp, ComputationInput, TransparentComputation};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use cudarc::nvrtc::Ptx;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;

pub struct CudaInfo {
    pub dev: Arc<CudaDevice>,
    pub twiddles: CudaSlice<u32>, // For now we restrain ourseleves to the 2-addic roots of unity in the prime fields, so each one is represented by a u32
    pub two_adicity: usize,
}

const MAX_NUMBER_OF_SUMCHECK_COMPOSITION_INSTRUCTIONS_TO_REMOVE_INLINING: usize = 10;

static CUDA_INFO: OnceLock<CudaInfo> = OnceLock::new();

pub fn cuda_info() -> &'static CudaInfo {
    CUDA_INFO.get().expect("CUDA not initialized")
}

#[instrument(name = "CUDA initialization", skip_all)]
pub fn init<F: TwoAdicField + PrimeField32, EF: ExtensionField<F>>(
    sumcheck_compositions: &[&TransparentComputation<F, EF>],
) {
    let _ = CUDA_INFO.get_or_init(|| _init(sumcheck_compositions));
}

fn _init<F: TwoAdicField + PrimeField32, EF: ExtensionField<F>>(
    sumcheck_compositions: &[&TransparentComputation<F, EF>],
) -> CudaInfo {
    let dev = CudaDevice::new(0).unwrap();

    // TODO avoid this ugly trick
    let kernels_folder = if Path::new("kernels").exists() {
        Path::new("kernels")
    } else {
        Path::new("crates/cuda-bindings/kernels")
    };

    for (module, func_names) in [
        ("keccak", vec!["batch_keccak256"]),
        ("ntt", vec!["ntt"]),
        (
            "sumcheck_common",
            vec![
                "fold_prime_by_prime",
                "fold_prime_by_ext",
                "fold_ext_by_prime",
                "fold_ext_by_ext",
            ],
        ),
    ] {
        compile_module(
            dev.clone(),
            &kernels_folder.join(format!("{module}.cu")),
            module,
            func_names,
            false,
        );
    }

    let specialized_sumcheck_template =
        std::fs::read_to_string(Path::new(kernels_folder).join("specialized_sumcheck.txt"))
            .unwrap();

    let cuda_generated_dir = kernels_folder.join("auto_generated");
    fs::create_dir_all(&cuda_generated_dir).unwrap();

    for composition in sumcheck_compositions {
        let module = format!("sumcheck_{:x}", composition.uuid());
        let file = cuda_generated_dir.join(format!("{module}.cu"));
        if !file.exists() {
            let cuda = get_specialized_sumcheck_cuda(&specialized_sumcheck_template, composition);
            std::fs::write(&file, cuda).unwrap();
        }

        // To avoid huge PTX file, and reduce compilation time, we may remove inlining
        let use_noinline = composition.n_instructions()
            > MAX_NUMBER_OF_SUMCHECK_COMPOSITION_INSTRUCTIONS_TO_REMOVE_INLINING;

        compile_module(
            dev.clone(),
            &file,
            &module,
            vec!["sum_over_hypercube_ext"],
            use_noinline,
        );
    }

    let twiddles = store_twiddles::<F>(&dev).unwrap();

    CudaInfo {
        dev,
        twiddles,
        two_adicity: F::TWO_ADICITY,
    }
}

fn compile_module(
    dev: Arc<CudaDevice>,
    cuda_file: &PathBuf,
    module: &str,
    func_names: Vec<&'static str>,
    use_noinline: bool,
) {
    let ptx_file = my_temp_dir().join(format!("{module}.ptx"));
    let (major, minor) = cuda_compute_capacity().expect("Failed to get CUDA compute capability");

    let source_modified = Path::new(&cuda_file)
        .metadata()
        .unwrap()
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
    dev.load_ptx(Ptx::from_src(ptx_content), module, &func_names)
        .unwrap();
}

#[instrument(name = "pre-processing twiddles for CUDA NTT", skip_all)]
fn store_twiddles<F: TwoAdicField>(dev: &Arc<CudaDevice>) -> Result<CudaSlice<u32>, DriverError> {
    assert!(F::bits() <= 32);
    let num_threads = rayon::current_num_threads().next_power_of_two();
    let mut all_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        let root = F::two_adic_generator(i);
        let twiddles = if (1 << i) <= num_threads {
            (0..1 << i)
                .into_iter()
                .map(|j| root.exp_u64(j as u64))
                .collect::<Vec<F>>()
        } else {
            let chunk_size = (1 << i) / num_threads;
            (0..num_threads)
                .into_par_iter()
                .map(|j| {
                    let mut start = root.exp_u64(j as u64 * chunk_size as u64);
                    let mut chunck = Vec::new();
                    for _ in 0..chunk_size {
                        chunck.push(start);
                        start = start * root;
                    }
                    chunck
                })
                .flatten()
                .collect()
        };
        all_twiddles.extend(twiddles);
    }

    let all_twiddles_u32 = unsafe {
        std::slice::from_raw_parts(all_twiddles.as_ptr() as *const u32, all_twiddles.len())
    }
    .to_vec();

    let all_twiddles_dev = dev.htod_copy(all_twiddles_u32).unwrap();
    dev.synchronize().unwrap();

    Ok(all_twiddles_dev)
}

fn cuda_compute_capacity() -> Result<(i32, i32), Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    let major = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

    Ok((major, minor))
}

fn get_specialized_sumcheck_cuda<F: TwoAdicField + PrimeField32, EF: ExtensionField<F>>(
    template: &str,
    composition: &TransparentComputation<F, EF>,
) -> String {
    match composition {
        TransparentComputation::Generic(generic) => {
            let composition_instructions = get_specialized_sumcheck_generic_instructions(generic);
            template
                .replace("N_REGISTERS_PLACEHOLDER", &generic.stack_size.to_string())
                .replace("N_BATCHING_SCALARS_PLACEHOLDER", "1") // Should be 0 in theory, but this avoids "error: zero-sized variable "cached_batching_scalars" is not allowed in device code"
                .replace("COMPOSITION_PLACEHOLDER", &composition_instructions)
        }
        TransparentComputation::Custom(_custom) => {
            todo!()
        }
    }
}

fn get_specialized_sumcheck_generic_instructions<F: TwoAdicField + PrimeField32>(
    composition: &CircuitComputation<F, usize>,
) -> String {
    /*

    Example:

    regs[0] = multilinears[0][thread_index];
    mul_prime_and_ext_field(&regs[0], to_monty(11), &regs[0]);
    ext_field_mul(&regs[0], &cached_batching_scalars[0], &regs[1]);
    regs[2] = regs[1];

    regs[0] = multilinears[1][thread_index];
    mul_prime_and_ext_field(&regs[0], to_monty(22), &regs[0]);
    ext_field_mul(&regs[0], &cached_batching_scalars[1], &regs[1]);

    ext_field_add(&regs[1], &regs[2], &regs[2]);

    regs[0] = multilinears[2][thread_index];
    mul_prime_and_ext_field(&regs[0], to_monty(33), &regs[0]);
    ext_field_mul(&regs[0], &cached_batching_scalars[2], &regs[1]);

    ext_field_add(&regs[1], &regs[2], &regs[2]);

    ext_field_mul(&regs[2], &multilinears[3][thread_index], &regs[3]);

     */

    let mut res = String::new();

    for instr in &composition.instructions {
        let blank = "            ";
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

    res
}

fn my_temp_dir() -> PathBuf {
    env::temp_dir().join("whirlaway-cuda-bindings")
}
