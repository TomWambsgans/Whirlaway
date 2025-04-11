use std::any::TypeId;

use algebra::pols::{Multilinear, MultilinearsSlice, MultilinearsVec, UnivariatePolynomial};
use arithmetic_circuit::CircuitComputation;
use cuda_engine::{SumcheckComputation, cuda_sync};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use utils::eq_extension;

const MIN_VARS_FOR_GPU: usize = 9; // When there are a small number of variables, it's not worth using GPU

pub fn prove<
    'a,
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Into<MultilinearsSlice<'a, NF>>,
>(
    multilinears: ML,
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
) -> (Vec<EF>, Vec<Multilinear<EF>>) {
    let multilinears: MultilinearsSlice<'_, _> = multilinears.into();
    let on_device = multilinears.is_device();
    let mut n_vars = multilinears.n_vars();

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars);
    let max_degree_per_vars = exprs
        .iter()
        .map(|expr| expr.composition_degree)
        .max_by_key(|d| *d)
        .unwrap();
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars);
    }

    let sumcheck_computation = SumcheckComputation {
        exprs: &exprs,
        n_multilinears: multilinears.len() + eq_factor.is_some() as usize,
        eq_mle_multiplier: eq_factor.is_some(),
    };

    let mut sum = sum.unwrap_or_else(|| {
        multilinears.sum_over_hypercube_of_computation(&sumcheck_computation, batching_scalars)
    });
    let mut folded_multilinears;

    folded_multilinears = sc_round(
        &multilinears,
        &mut n_vars,
        &sumcheck_computation,
        batching_scalars,
        eq_factor,
        is_zerofier,
        fs_prover,
        max_degree_per_vars,
        &mut sum,
        pow_bits,
        &mut challenges,
        0,
    );

    let mut need_to_transfer_back_to_device = false;
    for i in 1..n_rounds {
        if on_device && n_vars < MIN_VARS_FOR_GPU {
            // transfer GPU -> CPU
            folded_multilinears = folded_multilinears.transfer_to_host();
            need_to_transfer_back_to_device = true;
        }

        folded_multilinears = sc_round(
            &folded_multilinears.as_ref(),
            &mut n_vars,
            &sumcheck_computation,
            batching_scalars,
            eq_factor,
            is_zerofier,
            fs_prover,
            max_degree_per_vars,
            &mut sum,
            pow_bits,
            &mut challenges,
            i,
        );

        if need_to_transfer_back_to_device {
            folded_multilinears = folded_multilinears.transfer_to_device();
        }
    }
    (challenges, folded_multilinears.decompose())
}

fn sc_round<'a, F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>(
    multilinears: &MultilinearsSlice<'a, NF>,
    n_vars: &mut usize,
    sumcheck_computation: &SumcheckComputation<F>,
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    round: usize,
) -> MultilinearsVec<EF> {
    let _span = if *n_vars >= 6 {
        Some(tracing::span!(tracing::Level::INFO, "Sumcheck round").entered())
    } else {
        None
    };
    let mut p_evals = Vec::<(EF, EF)>::new();
    let eq_mle = eq_factor
        .map(|eq_factor| Multilinear::eq_mle(&eq_factor[1 + round..], multilinears.is_device()));

    let start = if is_zerofier && round == 0 {
        p_evals.push((EF::ZERO, EF::ZERO));
        p_evals.push((EF::ONE, EF::ZERO));
        2
    } else {
        0
    };
    for z in start..=degree as u32 {
        cuda_sync(); // I don't really understand why it is neccesary but it is
        let sum_z = if z == 1 {
            if let Some(eq_factor) = eq_factor {
                let f = eq_extension(&eq_factor[..round], &challenges);
                (*sum - p_evals[0].1 * f * (EF::ONE - eq_factor[round])) / (f * eq_factor[round])
            } else {
                *sum - p_evals[0].1
            }
        } else {
            let folded = multilinears.fix_variable_in_small_field(F::from_u32(z as u32));

            // TODO very bad
            let mut folded = if TypeId::of::<NF>() == TypeId::of::<EF>() {
                unsafe { std::mem::transmute::<_, MultilinearsVec<EF>>(folded) }
            } else {
                panic!()
            };

            if let Some(eq_mle) = &eq_mle {
                folded.push(eq_mle.clone()); // TODO avoid clone
            }

            folded
                .as_ref()
                .sum_over_hypercube_of_computation(sumcheck_computation, batching_scalars)
        };
        p_evals.push((EF::from_u32(z), sum_z));
    }
    cuda_sync();

    let mut p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        let f = eq_extension(&eq_factor[..round], &challenges);
        p *= UnivariatePolynomial::new(vec![
            f * (EF::ONE - eq_factor[round]),
            f * ((eq_factor[round] * EF::TWO) - EF::ONE),
        ]);
    }

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    *n_vars -= 1;

    // Do PoW if needed
    if pow_bits > 0 {
        fs_prover.challenge_pow(pow_bits);
    }

    multilinears.fix_variable_in_big_field(challenge)
}
