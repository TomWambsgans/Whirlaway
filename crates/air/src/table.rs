use std::marker::PhantomData;

use p3_air::Air;
use p3_field::{ExtensionField, Field, TwoAdicField};

use p3_uni_stark::{SymbolicAirBuilder, get_symbolic_constraints};
use utils::{FSChallenger, FSProver, PF, log2_up, univariate_selectors};
use whir_p3::{
    fiat_shamir::errors::ProofError,
    poly::{dense::WhirDensePolynomial, evals::EvaluationsList},
    whir::config::{WhirConfig, WhirConfigBuilder},
};

use crate::{AirSettings, WHIR_POW_BITS};

pub struct AirTable<EF: Field, A> {
    pub log_length: usize,
    pub n_columns: usize,
    pub air: A,
    pub preprocessed_columns: Vec<EvaluationsList<PF<EF>>>, // TODO 'sparse' preprocessed columns (with non zero values at cylic shifts)
    pub n_constraints: usize,
    pub constraint_degree: usize,
    pub(crate) univariate_selectors: Vec<WhirDensePolynomial<PF<EF>>>,

    _phantom: std::marker::PhantomData<EF>,
}

impl<EF, A> AirTable<EF, A>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>> + TwoAdicField,
    PF<EF>: TwoAdicField,
{
    pub fn new(
        air: A,
        log_length: usize,
        univariate_skips: usize,
        preprocessed_columns: Vec<EvaluationsList<PF<EF>>>,
        constraint_degree: usize,
    ) -> Self
    where
        A: Air<SymbolicAirBuilder<PF<EF>>>,
    {
        let symbolic_constraints = get_symbolic_constraints(&air, 0, 0);
        let n_constraints = symbolic_constraints.len();

        Self {
            log_length,
            n_columns: air.width(),
            air,
            preprocessed_columns,
            n_constraints,
            constraint_degree,
            univariate_selectors: univariate_selectors(univariate_skips),
            _phantom: std::marker::PhantomData,
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn n_witness_columns(&self) -> usize {
        self.n_columns - self.preprocessed_columns.len()
    }

    /// rounded up
    pub fn log_n_witness_columns(&self) -> usize {
        log2_up(self.n_witness_columns())
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn n_preprocessed_columns(&self) -> usize {
        self.preprocessed_columns.len()
    }

    pub fn build_whir_config<H, C, const DIGEST_ELEMS: usize>(
        &self,
        settings: &AirSettings,
        merkle_hash: H,
        merkle_compress: C,
    ) -> WhirConfig<PF<EF>, EF, H, C, DIGEST_ELEMS> {
        let num_variables = self.log_length + self.log_n_witness_columns();

        let whir_config_builder = WhirConfigBuilder {
            max_num_variables_to_send_coeffs: 6,
            security_level: settings.security_bits,
            pow_bits: WHIR_POW_BITS,
            folding_factor: settings.whir_folding_factor,
            merkle_hash,
            merkle_compress,
            soundness_type: settings.whir_soudness_type,
            starting_log_inv_rate: settings.whir_log_inv_rate,
            rs_domain_initial_reduction_factor: settings.whir_initial_domain_reduction_factor,
            base_field: PhantomData,
            extension_field: PhantomData,
        };

        WhirConfig::new(whir_config_builder, num_variables)
    }

    pub(crate) fn constraints_batching_pow(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        prover_state.pow_grinding(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_constraints))),
        );

        Ok(())
    }

    pub(crate) fn zerocheck_pow(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        prover_state.pow_grinding(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(self.log_length)),
        );

        Ok(())
    }

    pub(crate) fn secondary_sumchecks_batching_pow(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        prover_state.pow_grinding(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(self.log_n_witness_columns())),
        );

        Ok(())
    }
}
