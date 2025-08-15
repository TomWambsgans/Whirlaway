use compiler::*;
use p3_field::BasedVectorSpace;
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng as _, rngs::StdRng};
use utils::{build_poseidon16, build_poseidon24};
use vm::*;

#[test]
fn test_fibonacci_program() {
    // a program to check the value of the 30th Fibonacci number (832040)
    let program = r#"
    fn main() {
        fibonacci(0, 1, 0, 30);
        return;
    }

    fn fibonacci(a, b, i, n) {
        if i == n {
            print(a);
            return;
        }
        fibonacci(b, a + b, i + 1, n);
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_edge_case_0() {
    let program = r#"
    fn main() {
        a = malloc(1);
        a[0] = 0;
        for i in 0..1 {
            x = 1 + a[i];
        }
        for i in 0..1 {
            y = 1 + a[i];
        }
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_unroll() {
    // a program to check the value of the 30th Fibonacci number (832040)
    let program = r#"
    fn main() {
        for i in 0..5 unroll {
            for j in i..2*i unroll {
                print(i, j);
            }
        }
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_mini_program_0() {
    let program = r#"
    fn main() {
        for i in 0..5 {
            for j in i..2*i*(2+1) {
                print(i, j);
                if i == 4 {
                    if j == 7 {
                        break;
                    }
                }
            }
        }
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_mini_program_1() {
    let program = r#"
    const N = 10;

    fn main() {
        arr = malloc(N);
        fill_array(arr);
        print_array(arr);
        return;
    }

    fn fill_array(arr) {
        for i in 0..N {
            if i == 0 {
                arr[i] = 10;
            } else {
                if i == 1 {
                    arr[i] = 20;
                } else {
                    if i == 2 {
                        arr[i] = 30;
                    } else {
                        i_plus_one = i + 1;
                        arr[i] = i_plus_one;
                    }
                }
            }
        }
        return;
    }

    fn print_array(arr) {
        for i in 0..N {
            arr_i = arr[i];
            print(arr_i);
        }
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_mini_program_2() {
    let program = r#"
    fn main() {
        for i in 0..10 {
            for j in i..10 {
                for k in j..10 {
                    sum, prod = compute_sum_and_product(i, j, k);
                    if sum == 10 {
                        print(i, j, k, prod);
                    }
                }
            }
        }
        return;
    }

    fn compute_sum_and_product(a, b, c) -> 2 {
        s1 = a + b;
        sum = s1 + c;
        p1 = a * b;
        product = p1 * c;
        return sum, product;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_mini_program_3() {
    let program = r#"
    fn main() {
        a = public_input_start / 8;
        b = a + 1;
        c = poseidon16(a, b);

        c_shifted = c * 8;
        d_shifted = (c + 1) * 8;

        for i in 0..8 {
            cc = c_shifted[i];
            print(cc);
        }
        for i in 0..8 {
            dd = d_shifted[i];
            print(dd);
        }
        return;
    }
   "#;
    let mut public_input: [F; 16] = (0..16)
        .map(|i| F::new(i))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap();
    compile_and_run(program, &public_input, &[]);

    build_poseidon16().permute_mut(&mut public_input);
    dbg!(public_input);
}

#[test]
fn test_mini_program_4() {
    let program = r#"
    fn main() {
        a = public_input_start / 8;
        c = a + 2;
        f = poseidon24(a, c);

        f_shifted = f * 8;
        for j in 0..8 {
            print(f_shifted[j]);
        }
        return;
    }
   "#;
    let mut public_input: [F; 24] = (0..24)
        .map(|i| F::new(i))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap();
    compile_and_run(program, &public_input, &[]);

    build_poseidon24().permute_mut(&mut public_input);
    dbg!(&public_input[16..]);
}

#[test]
fn test_product_extension_field() {
    let program = r#"

    const W = 3; // in the extension field, X^8 = 3

    fn main() {
        a = public_input_start;
        b = public_input_start + 8;
        sum = malloc(8);
        prod = malloc(8);
        custom_add_extension(a, b, sum);
        custom_mul_extension(a, b, prod);

        real_sum = public_input_start + 16;
        real_prod = public_input_start + 24;

        for i in 0..8 {
            assert sum[i] == real_sum[i];
            assert prod[i] == real_prod[i];
        }
        
        return;
    }

    fn custom_mul_extension(a, b, c) {
        // a, b and c are pointers
        // c = a * b

        c[0] = (a[0] * b[0]) + W * ((a[1] * b[7]) + (a[2] * b[6]) + (a[3] * b[5]) + (a[4] * b[4]) + (a[5] * b[3]) + (a[6] * b[2]) + (a[7] * b[1]));
        c[1] = (a[1] * b[0]) + (a[0] * b[1]) + W * ((a[2] * b[7]) + (a[3] * b[6]) + (a[4] * b[5]) + (a[5] * b[4]) + (a[6] * b[3]) + (a[7] * b[2]));
        c[2] = (a[2] * b[0]) + (a[1] * b[1]) + (a[0] * b[2]) + W * ((a[3] * b[7]) + (a[4] * b[6]) + (a[5] * b[5]) + (a[6] * b[4]) + (a[7] * b[3]));
        c[3] = (a[3] * b[0]) + (a[2] * b[1]) + (a[1] * b[2]) + (a[0] * b[3]) + W * ((a[4] * b[7]) + (a[5] * b[6]) + (a[6] * b[5]) + (a[7] * b[4]));
        c[4] = (a[4] * b[0]) + (a[3] * b[1]) + (a[2] * b[2]) + (a[1] * b[3]) + (a[0] * b[4]) + W * ((a[5] * b[7]) + (a[6] * b[6]) + (a[7] * b[5]));
        c[5] = (a[5] * b[0]) + (a[4] * b[1]) + (a[3] * b[2]) + (a[2] * b[3]) + (a[1] * b[4]) + (a[0] * b[5]) + W * ((a[6] * b[7]) + (a[7] * b[6]));
        c[6] = (a[6] * b[0]) + (a[5] * b[1]) + (a[4] * b[2]) + (a[3] * b[3]) + (a[2] * b[4]) + (a[1] * b[5]) + (a[0] * b[6]) + W * (a[7] * b[7]);
        c[7] = (a[7] * b[0]) + (a[6] * b[1]) + (a[5] * b[2]) + (a[4] * b[3]) + (a[3] * b[4]) + (a[2] * b[5]) + (a[1] * b[6]) + (a[0] * b[7]);

        return;
    }

    fn custom_add_extension(a, b, c) {
        // a, b and c are pointers
        // c = a + b
        for i in 0..8 unroll {
            c[i] = a[i] + b[i];
        }
        return;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let a: EF = rng.random();
    let b: EF = rng.random();

    let mut public_input = Vec::<F>::new();
    public_input.extend(a.as_basis_coefficients_slice());
    public_input.extend(b.as_basis_coefficients_slice());
    public_input.extend((a + b).as_basis_coefficients_slice());
    public_input.extend((a * b).as_basis_coefficients_slice());

    compile_and_run(program, &public_input, &[]);
}
