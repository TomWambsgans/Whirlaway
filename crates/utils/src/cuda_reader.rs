use regex::Regex;

pub fn extract_cuda_global_functions(cuda_code: &str) -> Vec<String> {
    let function_pattern = Regex::new(r#"extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\("#)
        .expect("Failed to compile regex");

    let comment_pattern = Regex::new(r"^\s*//").expect("Failed to compile comment regex");

    let mut function_names = Vec::new();
    for line in cuda_code.lines() {
        // Skip commented lines
        if comment_pattern.is_match(line) {
            continue;
        }

        // Check for function declarations in non-commented lines
        if let Some(cap) = function_pattern.captures(line) {
            if let Some(function_name) = cap.get(1) {
                function_names.push(function_name.as_str().to_string());
            }
        }
    }

    function_names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_cuda_global_functions() {
        let cuda_code = r#"
            extern "C" __global__ void my_kernel1() {
                // kernel code
            }
            extern "C" __global__ void my_kernel2(int a) {
                // kernel code
            }
            extern "C" void not_a_kernel() {
                // not a kernel
            }
            // extern "C" __global__ void commented_kernel(int a) {
            //    // kernel code
            // }
        "#;

        let functions = extract_cuda_global_functions(cuda_code);
        assert_eq!(functions.len(), 2);
        assert_eq!(functions[0], "my_kernel1");
        assert_eq!(functions[1], "my_kernel2");
    }
}
