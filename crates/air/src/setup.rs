use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{
    CudaFunctionInfo, SumcheckComputation, cuda_init, cuda_load_function,
    cuda_preprocess_many_sumcheck_computations, cuda_preprocess_twiddles,
};
use p3_field::BasedVectorSpace;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use utils::log2_up;

use crate::{AirSettings, table::AirTable};

impl<F: PrimeField32 + TwoAdicField> AirTable<F> {
    pub fn cuda_setup<EF: ExtensionField<F>, WhirF: ExtensionField<F>>(
        &self,
        settings: &AirSettings,
    ) {
        cuda_init();

        let n_vars = log2_up(self.n_columns) + self.log_length;
        cuda_preprocess_twiddles::<F>(
            n_vars + settings.whir_log_inv_rate - settings.whir_folding_factor.maximum(),
        );

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

        assert!(F::DIMENSION == 1);
        let deg_a = F::DIMENSION;
        let deg_b = EF::DIMENSION;
        let deg_c = WhirF::DIMENSION;
        cuda_preprocess_many_sumcheck_computations(
            &constraint_sumcheck_computations,
            &[(deg_a, deg_a, deg_b), (deg_a, deg_b, deg_b)],
        );
        cuda_preprocess_many_sumcheck_computations(
            &prod_sumcheck,
            &[(deg_a, deg_b, deg_b), (deg_a, deg_c, deg_c)],
        );
        cuda_preprocess_many_sumcheck_computations(&inner_air_sumcheck, &[(deg_a, deg_b, deg_b)]);

        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "batch_keccak256"));
        cuda_load_function(CudaFunctionInfo::basic("keccak.cu", "pow_grinding"));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis_end",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis_end",
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
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<F, WhirF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_shared_memory",
        ));

        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_shared_memory",
        ));

        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, F>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_steps",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, F>(
            "multilinear_evaluations.cu",
            "eval_multilinear_in_lagrange_basis_shared_memory",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "univariate_skip.cu",
            "matrix_up_folded_with_univariate_skips",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "univariate_skip.cu",
            "matrix_down_folded_with_univariate_skips",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<WhirF>(
            "multilinear.cu",
            "eq_mle_steps",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "eq_mle_steps",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_steps",
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
        cuda_load_function(CudaFunctionInfo::two_fields::<EF, WhirF>(
            "multilinear.cu",
            "add_assign_slices",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
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
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, WhirF>(
            "multilinear.cu",
            "linear_combination",
        ));
        cuda_load_function(CudaFunctionInfo::two_fields::<WhirF, EF>(
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
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
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
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<WhirF>());
    }
}
