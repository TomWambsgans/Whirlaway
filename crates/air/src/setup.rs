use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{
    CudaFunctionInfo, SumcheckComputation, cuda_init, cuda_load_function,
    cuda_preprocess_many_sumcheck_computations, cuda_preprocess_twiddles,
};
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use utils::extension_degree;

use crate::table::AirTable;

impl<F: PrimeField32 + TwoAdicField> AirTable<F> {
    pub fn cuda_setup<EF: ExtensionField<F>, WhirF: ExtensionField<F>>(&self) {
        cuda_init();
        let constraint_sumcheck_computations = SumcheckComputation::<F> {
            exprs: &self.constraints,
            n_multilinears: self.n_columns * 2 + 1,
            eq_mle_multiplier: true,
        };
        let prod_sumcheck = SumcheckComputation::<F> {
            exprs: &[
                (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1)).fix_computation(false)
            ],
            n_multilinears: 2,
            eq_mle_multiplier: false,
        };
        let inner_air_sumcheck = SumcheckComputation::<F> {
            exprs: &[(ArithmeticCircuit::Node(4)
                * ((ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                    + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3))))
            .fix_computation(false)],
            n_multilinears: 5,
            eq_mle_multiplier: false,
        };

        assert!(extension_degree::<F>() == 1);
        let deg_a = extension_degree::<F>();
        let deg_b = extension_degree::<EF>();
        let deg_c = extension_degree::<WhirF>();
        cuda_preprocess_many_sumcheck_computations(
            &constraint_sumcheck_computations,
            &[(deg_a, deg_a, deg_b), (deg_a, deg_b, deg_b)],
        );
        cuda_preprocess_many_sumcheck_computations(
            &prod_sumcheck,
            &[(deg_a, deg_b, deg_b), (deg_a, deg_c, deg_c)],
        );
        cuda_preprocess_many_sumcheck_computations(&inner_air_sumcheck, &[(deg_a, deg_b, deg_b)]);
        cuda_preprocess_twiddles::<F>();

        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "monomial_to_lagrange_basis_rev",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "multilinears_up",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "multilinears_down",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "multilinear.cu",
            "eval_multilinear_in_lagrange_basis",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear.cu",
            "eval_multilinear_in_lagrange_basis",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "eq_mle",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, F>(
            "multilinear.cu",
            "fold_rectangular",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "dot_product",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
            "multilinear.cu",
            "dot_product",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear.cu",
            "dot_product",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "multilinear.cu",
            "dot_product",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "scale_in_place",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "add_slices",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "add_assign_slices",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "piecewise_sum",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "linear_combination",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "repeat_slice_from_outside",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "repeat_slice_from_inside",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "sum_in_place",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "sum_in_place",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "tensor_algebra.cu",
            "tensor_algebra_dot_product",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "ntt/transpose.cu",
            "transpose",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "ntt/bit_reverse.cu",
            "reverse_bit_order_for_ntt",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "ntt/ntt.cu",
            "ntt_step",
        ));
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<WhirF>());
    }
}
