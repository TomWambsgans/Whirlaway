use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
use cudarc::nvrtc::Ptx;
use p3_field::Field;
use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock, RwLock};
use tracing::instrument;
use utils::{SupportedField, extension_degree, log2_down};

use crate::{CudaPtr, LOG_MAX_THREADS_PER_BLOCK};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct CudaFunctionInfo {
    pub cuda_file: PathBuf,
    pub function_name: String,
    pub field: Option<SupportedField>,
    pub extension_degree_a: Option<usize>,
    pub extension_degree_b: Option<usize>,
    pub extension_degree_c: Option<usize>,
    pub max_ntt_log_size_at_block_level: Option<u32>,
    pub no_inline: bool,
    pub cache_memory_reads: bool,
}

impl CudaFunctionInfo {
    pub fn basic(cuda_file: impl Into<PathBuf>, function_name: &str) -> Self {
        Self {
            cuda_file: kernels_folder().join(cuda_file.into()),
            function_name: function_name.to_string(),
            ..Default::default()
        }
    }

    pub fn one_field<FieldA: Field>(cuda_file: impl Into<PathBuf>, function_name: &str) -> Self {
        Self {
            cuda_file: kernels_folder().join(cuda_file.into()),
            function_name: function_name.to_string(),
            field: Some(SupportedField::guess::<FieldA>()),
            extension_degree_a: Some(extension_degree::<FieldA>()),
            ..Default::default()
        }
    }

    pub fn two_fields<FieldA: Field, FieldB: Field>(
        cuda_file: impl Into<PathBuf>,
        function_name: &str,
    ) -> Self {
        assert_eq!(
            TypeId::of::<FieldA::PrimeSubfield>(),
            TypeId::of::<FieldB::PrimeSubfield>()
        );
        Self {
            cuda_file: kernels_folder().join(cuda_file.into()),
            function_name: function_name.to_string(),
            field: Some(SupportedField::guess::<FieldA>()),
            extension_degree_a: Some(extension_degree::<FieldA>()),
            extension_degree_b: Some(extension_degree::<FieldB>()),
            ..Default::default()
        }
    }

    pub fn ntt_at_block_level<F: Field>() -> Self {
        Self {
            cuda_file: kernels_folder().join("ntt.cu"),
            function_name: "ntt_at_block_level".to_string(),
            field: Some(SupportedField::guess::<F>()),
            extension_degree_a: Some(extension_degree::<F::PrimeSubfield>()), // twiddles
            extension_degree_b: Some(extension_degree::<F>()),
            max_ntt_log_size_at_block_level: Some(max_ntt_log_size_at_block_level::<F>() as u32),
            ..Default::default()
        }
    }
}

pub fn max_ntt_log_size_at_block_level<F: Field>() -> usize {
    (LOG_MAX_THREADS_PER_BLOCK as usize + 1).min(
        log2_down(
            shared_memory() / (std::mem::size_of::<F>() + std::mem::size_of::<F::PrimeSubfield>()),
        ) - 1,
    ) // TODO fix and remove -1
}

pub struct CudaEngine {
    pub(crate) dev: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    // We restrain ourseleves to the 2-addic roots of unity in the 32-bits prime field
    // Each CudaSlice<u32> contains half of the twiddles, the other half is just the opposite
    pub(crate) twiddles: RwLock<BTreeMap<TypeId, (Vec<CudaSlice<u32>>, CudaSlice<CudaPtr<u32>>)>>,
    pub(crate) functions: RwLock<HashMap<CudaFunctionInfo, CudaFunction>>,
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
    let _ = CUDA_ENGINE.get_or_init(CudaEngine::new);
}

impl CudaEngine {
    fn new() -> Self {
        let dev = CudaContext::new(0).unwrap();
        let stream = dev.default_stream();
        CudaEngine {
            dev,
            stream,
            twiddles: Default::default(),
            functions: Default::default(),
        }
    }
}

pub fn cuda_load_function(options: CudaFunctionInfo) {
    load_function(options, &mut cuda_engine().functions.write().unwrap());
}

pub(crate) fn load_function(
    options: CudaFunctionInfo,
    functions: &mut HashMap<CudaFunctionInfo, CudaFunction>,
) {
    let engine = cuda_engine();
    if functions.contains_key(&options) {
        return;
    }
    let mut ptx_file_name = options
        .cuda_file
        .to_string_lossy()
        .replace("\\", "_")
        .replace("/", "_")
        .replace(".cu", "");
    if let Some(field) = options.field {
        ptx_file_name.push_str(&format!("_{field}"));
    }
    if let Some(extension_degree_a) = options.extension_degree_a {
        ptx_file_name.push_str(&format!("_A{extension_degree_a}"));
    }
    if let Some(extension_degree_b) = options.extension_degree_b {
        ptx_file_name.push_str(&format!("_B{extension_degree_b}"));
    }
    if let Some(extension_degree_c) = options.extension_degree_c {
        ptx_file_name.push_str(&format!("_C{extension_degree_c}"));
    }
    if let Some(max_ntt_log_size_at_block_level) = options.max_ntt_log_size_at_block_level {
        ptx_file_name.push_str(&format!(
            "_max_ntt_log_size_at_block_level_{max_ntt_log_size_at_block_level}"
        ));
    }
    if options.no_inline {
        ptx_file_name.push_str("_noinline");
    }
    if options.cache_memory_reads {
        ptx_file_name.push_str("_cached");
    }
    let ptx_file = ptx_dir().join(format!("{ptx_file_name}.ptx"));
    let (major, minor) = cuda_compute_capacity();

    let source_modified = options
        .cuda_file
        .metadata()
        .unwrap_or_else(|_| panic!("Cannot find {}", options.cuda_file.display()))
        .modified()
        .unwrap();

    let should_compile = if ptx_file.exists() {
        let target_modified = ptx_file.metadata().unwrap().modified().unwrap();
        source_modified > target_modified
    } else {
        true
    };

    if should_compile {
        let _span = tracing::info_span!(
            "Compiling CUDA file",
            module = options.cuda_file.to_string_lossy().to_string(),
            function_name = options.function_name,
        )
        .entered();
        // Create directory if it doesn't exist
        if let Some(parent) = ptx_file.parent() {
            fs::create_dir_all(parent).unwrap();
        }

        // Run nvcc to compile the CUDA code to PTX
        let mut command = Command::new("nvcc");

        let cuda_file_size = fs::metadata(&options.cuda_file).unwrap().len() / 1024;
        if cuda_file_size > 50 {
            println!(
                "Compiling a big cuda file ({} kb), may take a while...",
                cuda_file_size
            );
        }

        command.args([
            &options.cuda_file.to_string_lossy() as &str, // Input file
            "--ptx",
            &format!("-arch=sm_{major}{minor}"), // NOT SURE OF THIS
            "-o",
            &ptx_file.to_string_lossy(), // Output file
        ]);

        if options.no_inline {
            command.arg("-DUSE_NOINLINE");
        }
        if options.cache_memory_reads {
            command.arg("-DCACHED");
        }
        if let Some(field) = options.field {
            command.arg(format!("-DFIELD={}", field as u32));
        }
        if let Some(extension_degree_a) = options.extension_degree_a {
            command.arg(format!("-DEXTENSION_DEGREE_A={extension_degree_a}"));
        }
        if let Some(extension_degree_b) = options.extension_degree_b {
            command.arg(format!("-DEXTENSION_DEGREE_B={extension_degree_b}"));
        }
        if let Some(extension_degree_c) = options.extension_degree_c {
            command.arg(format!("-DEXTENSION_DEGREE_C={extension_degree_c}"));
        }
        if let Some(max_ntt_log_size_at_block_level) = options.max_ntt_log_size_at_block_level {
            command.arg(format!(
                "-DMAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL={max_ntt_log_size_at_block_level}"
            ));
        }

        let output = command.output().unwrap_or_else(|_| {
            panic!(
                "Failed to compile {} with nvcc",
                options.cuda_file.to_string_lossy()
            )
        });

        if !output.status.success() {
            panic!(
                "NVCC error on {:?}\n {}",
                options,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        tracing::debug!(
            "Using cached PTX for {}",
            options.cuda_file.to_string_lossy()
        );
    }

    let ptx_content = std::fs::read_to_string(ptx_file).expect("Failed to read PTX file");
    let my_module = engine.dev.load_module(Ptx::from_src(ptx_content)).unwrap();
    let cuda_function = my_module
        .load_function(&options.function_name)
        .unwrap_or_else(|_| {
            panic!(
                "Failed to load function {} from {}",
                options.function_name,
                options.cuda_file.to_string_lossy()
            )
        });
    functions.insert(options, cuda_function);
}

fn cuda_compute_capacity() -> (u32, u32) {
    let dev = CudaContext::new(0).unwrap();
    let major = dev
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap() as u32;
    let minor = dev
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap() as u32;

    (major, minor)
}

static SHARED_MEMORY: OnceLock<usize> = OnceLock::new();
fn shared_memory() -> usize {
    *SHARED_MEMORY.get_or_init(|| {
        let dev = CudaContext::new(0).unwrap();
        dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
            .unwrap() as usize
    })
}

pub(crate) fn kernels_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

pub(crate) fn ptx_dir() -> PathBuf {
    let ptx_dir = kernels_folder().join("build").join("ptx");
    fs::create_dir_all(&ptx_dir).unwrap();
    ptx_dir
}

pub fn cuda_synthetic_dir() -> PathBuf {
    let cuda_synthetic_dir = kernels_folder().join("build").join("cuda_synthetic");
    fs::create_dir_all(&cuda_synthetic_dir).unwrap();
    cuda_synthetic_dir
}
