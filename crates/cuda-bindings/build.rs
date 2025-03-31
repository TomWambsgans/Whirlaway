use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

use cudarc::driver::CudaDevice;
use cudarc::driver::sys::CUdevice_attribute;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    handle_cuda_file("keccak");
    handle_cuda_file("ntt");
}

fn handle_cuda_file(name: &str) {
    let cuda_file = format!("kernels/{name}.cu");
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_file = Path::new(&out_dir).join(format!("{name}.ptx"));
    println!("cargo:rerun-if-changed={}", cuda_file);
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
        // Create directory if it doesn't exist
        if let Some(parent) = ptx_file.parent() {
            fs::create_dir_all(parent).unwrap();
        }

        // Run nvcc to compile the CUDA code to PTX
        let output = Command::new("nvcc")
            .args(&[
                "--ptx",
                &format!("-arch=sm_{major}{minor}"), // NOT SURE OF THIS
                "-o",
                &ptx_file.to_string_lossy(), // Output file
                &cuda_file,                  // Input file
            ])
            .output()
            .expect(&format!("Failed to compile {} with nvcc", name));

        if !output.status.success() {
            panic!("NVCC error: {}", String::from_utf8_lossy(&output.stderr));
        }

        println!(
            "Successfully compiled {} to {}",
            cuda_file,
            ptx_file.display()
        );
    }

    // Make the PTX file available to the project
    println!(
        "cargo:rustc-env=PTX_{}_PATH={}",
        name.to_uppercase(),
        ptx_file.display()
    );
}

fn cuda_compute_capacity() -> Result<(i32, i32), Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    let major = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

    // println!("Device: {}", dev.name()?);
    // println!("Compute Capability: {}.{}", major, minor);

    Ok((major, minor))
}
