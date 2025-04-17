use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
use cudarc::nvrtc::Ptx;
use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock, RwLock};
use tracing::instrument;
use utils::extract_cuda_global_functions;

pub(crate) struct CudaEngine {
    pub(crate) dev: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) twiddles: RwLock<BTreeMap<TypeId, CudaSlice<u32>>>, // We restrain ourseleves to the 2-addic roots of unity in the 32-bits prime field
    pub(crate) correction_twiddles: RwLock<BTreeMap<(TypeId, usize), CudaSlice<u32>>>, // (Field, whir folding factor) => correction twiddles
    pub(crate) functions: RwLock<HashMap<String, HashMap<String, CudaFunction>>>, // module => function_name => cuda_function
}

static CUDA_ENGINE: OnceLock<CudaEngine> = OnceLock::new();

pub(crate) fn cuda_engine() -> &'static CudaEngine {
    CUDA_ENGINE
        .get()
        .unwrap_or_else(|| panic!("CUDA not initialized"))
}

pub(crate) fn try_get_cuda_engine() -> Option<&'static CudaEngine> {
    CUDA_ENGINE.get()
}

#[instrument(name = "CUDA initialization", skip_all)]
pub fn cuda_init() {
    let _ = CUDA_ENGINE.get_or_init(|| CudaEngine::new());
}

impl CudaEngine {
    fn new() -> Self {
        let dev = CudaContext::new(0).unwrap();
        let mut functions = HashMap::new();
        for module in ["keccak", "ntt", "multilinear"] {
            compile_module(
                dev.clone(),
                &kernels_folder(),
                module,
                false,
                &mut functions,
            );
        }
        let stream = dev.default_stream();
        CudaEngine {
            dev,
            stream,
            twiddles: Default::default(),
            correction_twiddles: Default::default(),
            functions: RwLock::new(functions),
        }
    }
}

pub(crate) fn compile_module(
    dev: Arc<CudaContext>,
    cuda_folder: &PathBuf,
    module: &str,
    use_noinline: bool,
    functions: &mut HashMap<String, HashMap<String, CudaFunction>>, // module => function_name => cuda_function
) {
    let cuda_file = cuda_folder.join(format!("{module}.cu"));
    let ptx_file = ptx_dir().join(format!("{module}.ptx"));
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

        let cuda_file_size = fs::metadata(&cuda_file).unwrap().len() / 1024;
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
        tracing::debug!("Using cached PTX for {}", module);
    }

    let cuda_content = std::fs::read_to_string(cuda_file).expect("Failed to read CUDA file");
    let func_names = extract_cuda_global_functions(&cuda_content);
    let ptx_content = std::fs::read_to_string(ptx_file).expect("Failed to read PTX file");
    let my_module = dev.load_module(Ptx::from_src(ptx_content)).unwrap();
    assert!(!functions.contains_key(module));
    functions.insert(
        module.to_string(),
        func_names
            .into_iter()
            .map(|func_name| {
                let cuda_function = my_module.load_function(&func_name).unwrap();
                (func_name, cuda_function)
            })
            .collect::<HashMap<_, _>>(),
    );
}

fn cuda_compute_capacity() -> Result<(i32, i32), Box<dyn Error>> {
    let dev = CudaContext::new(0)?;
    let major = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

    Ok((major, minor))
}

pub(crate) fn kernels_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

pub(crate) fn ptx_dir() -> PathBuf {
    let ptx_dir = kernels_folder().join("build").join("ptx");
    fs::create_dir_all(&ptx_dir).unwrap();
    ptx_dir
}

pub(crate) fn cuda_synthetic_dir() -> PathBuf {
    let cuda_synthetic_dir = kernels_folder().join("build").join("cuda_synthetic");
    fs::create_dir_all(&cuda_synthetic_dir).unwrap();
    cuda_synthetic_dir
}
