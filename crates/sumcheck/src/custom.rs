use algebra::pols::{
    HypercubePoint, MultilinearPolynomial, PartialHypercubePoint, UnivariatePolynomial,
};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::instrument;

/*
TODO: The 'densification' of the sparse multilinear polynomial could be further postponed
*/

/// A 'sparse' sumcheck, with variable shifts, specifically used to reduce several polynomial evaluations
/// (where points all finish with zeros and ones for their last j variables) to one single evaluation, opened
/// later by the (ring switch) PCS.
///
/// mixed_eql_mles is a 'sparse' multilinear polynomial (in k + kappa variables). To evaluate it on a point x = (1, 0, 1, ...) of the hypercube:
/// 1) 'row' = the last k variables = x[kappa..kappa + k] (the corresponing integer beeing interpreted in big indian)
/// 2) if mixed_eql_mles[row].1 == x[kappa - j..kapaa] return mixed_eql_mles[row].0(x[0..kappa - j]), else return 0
///
#[instrument(name = "sumcheck: prove_custom", skip_all)]
pub fn prove_custom<F: Field, EF: ExtensionField<F>>(
    witness_pol: MultilinearPolynomial<F>, // kappa variables
    eq_factor: MultilinearPolynomial<EF>,  // k variables
    mixed_eql_mles: Vec<(MultilinearPolynomial<EF>, HypercubePoint)>, // (kappa - j variables, j variables), 2^(k-1) < length <= 2^k
    fs_prover: &mut FsProver,
    mut sum: EF,
) -> Vec<EF> {
    let kappa = witness_pol.n_vars;
    let k = eq_factor.n_vars;
    assert_eq!(
        k,
        mixed_eql_mles.len().next_power_of_two().trailing_zeros() as usize
    );
    let j = mixed_eql_mles[0].1.n_vars;
    let mut challenges = Vec::new();

    let (mut folded_witness_pol, mut folded_eq_factor, mut folded_mixed_eql_mles) =
        first_half_round(
            witness_pol,
            eq_factor,
            mixed_eql_mles,
            fs_prover,
            &mut sum,
            &mut challenges,
        );
    for _round in 1..kappa - j {
        (folded_witness_pol, folded_eq_factor, folded_mixed_eql_mles) = first_half_round(
            folded_witness_pol,
            folded_eq_factor,
            folded_mixed_eql_mles,
            fs_prover,
            &mut sum,
            &mut challenges,
        );
    }

    let mut densified_eql_mles = densify_sparse_mle(folded_mixed_eql_mles);

    for _round in kappa - j..kappa + k {
        (folded_witness_pol, folded_eq_factor, densified_eql_mles) = second_half_round(
            folded_witness_pol,
            folded_eq_factor,
            densified_eql_mles,
            fs_prover,
            &mut sum,
            &mut challenges,
        );
    }

    challenges
}

// first kappa-j variables
fn first_half_round<F: Field, EF: ExtensionField<F>>(
    witness_pol: MultilinearPolynomial<F>, // kappa variables
    eq_factor: MultilinearPolynomial<EF>,  // k variables
    mixed_eql_mles: Vec<(MultilinearPolynomial<EF>, HypercubePoint)>, // (kappa - j variables, j variables), 2^(k-1) < length <= 2^k
    fs_prover: &mut FsProver,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
) -> (
    MultilinearPolynomial<EF>,
    MultilinearPolynomial<EF>,
    Vec<(MultilinearPolynomial<EF>, HypercubePoint)>,
) {
    let k = eq_factor.n_vars;
    assert_eq!(
        k,
        mixed_eql_mles.len().next_power_of_two().trailing_zeros() as usize
    );
    let kappa = witness_pol.n_vars;
    let j = mixed_eql_mles[0].1.n_vars;
    assert!(kappa - j >= 1);
    assert!(
        mixed_eql_mles
            .iter()
            .all(|(p, s)| p.n_vars == kappa - j && s.n_vars == j)
    );

    let mut p_evals = Vec::<(EF, EF)>::new();
    for z in 0..=2 as u32 {
        let sum_z = if z == 1 {
            *sum - p_evals[0].1
        } else {
            let sparse_hypercube = mixed_eql_mles
                .iter()
                .enumerate()
                .map(|(r, (_, m))| (0..1 << (kappa - j - 1)).map(move |l| (l, m.val, r)))
                .flatten()
                .collect::<Vec<(usize, usize, usize)>>();
            sparse_hypercube
                .into_par_iter()
                .map(|(l, m, r)| {
                    eq_factor.eval_hypercube(&HypercubePoint { n_vars: k, val: r })
                        * witness_pol.eval_partial_hypercube(&PartialHypercubePoint {
                            left: z,
                            right: HypercubePoint {
                                n_vars: kappa - 1,
                                val: (l << j) + m,
                            },
                        })
                        * mixed_eql_mles[r]
                            .0
                            .eval_partial_hypercube(&PartialHypercubePoint {
                                left: z,
                                right: HypercubePoint {
                                    n_vars: kappa - j - 1,
                                    val: l,
                                },
                            })
                })
                .sum::<EF>()
        };
        p_evals.push((EF::from_u32(z), sum_z));
    }
    let p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];

    let folded_witness_pol = witness_pol.fix_variable(challenge);
    let folded_eq_factor = eq_factor;
    let folded_mixed_eql_mles = mixed_eql_mles
        .into_iter()
        .map(|(p, s)| (p.fix_variable(challenge), s))
        .collect::<Vec<_>>();
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    (folded_witness_pol, folded_eq_factor, folded_mixed_eql_mles)
}

// last j+k variables
fn second_half_round<EF: ExtensionField<EF>>(
    witness_pol: MultilinearPolynomial<EF>,
    eq_factor: MultilinearPolynomial<EF>,
    mixed_eql_mles: MultilinearPolynomial<EF>,
    fs_prover: &mut FsProver,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
) -> (
    MultilinearPolynomial<EF>,
    MultilinearPolynomial<EF>,
    MultilinearPolynomial<EF>,
) {
    let k = eq_factor.n_vars;
    let kappa = witness_pol.n_vars;

    let mut p_evals = Vec::<(EF, EF)>::new();
    for z in 0..=2 as u32 {
        let sum_z = if z == 1 {
            *sum - p_evals[0].1
        } else {
            (0..(1 << (kappa + k - 1)))
                .into_par_iter()
                .map(|x| {
                    mixed_eql_mles.eval_partial_hypercube(&PartialHypercubePoint {
                        left: z,
                        right: HypercubePoint {
                            n_vars: kappa + k - 1,
                            val: x,
                        },
                    }) * if kappa != 0 {
                        witness_pol.eval_partial_hypercube(&PartialHypercubePoint {
                            left: z,
                            right: HypercubePoint {
                                n_vars: kappa - 1,
                                val: x >> k,
                            },
                        }) * eq_factor.eval_hypercube(&HypercubePoint {
                            n_vars: k,
                            val: x & ((1 << k) - 1),
                        })
                    } else {
                        witness_pol.eval::<EF>(&[])
                            * eq_factor.eval_partial_hypercube(&PartialHypercubePoint {
                                left: z,
                                right: HypercubePoint {
                                    n_vars: k - 1,
                                    val: x & ((1 << (k - 1)) - 1),
                                },
                            })
                    }
                })
                .sum::<EF>()
        };
        p_evals.push((EF::from_u32(z), sum_z));
    }

    let p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];

    let (folded_witness_pol, folded_eq_factor) = if kappa == 0 {
        (witness_pol, eq_factor.fix_variable(challenge))
    } else {
        (witness_pol.fix_variable(challenge), eq_factor)
    };
    let folded_mixed_eql_mles = mixed_eql_mles.fix_variable(challenge);
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    (folded_witness_pol, folded_eq_factor, folded_mixed_eql_mles)
}

fn densify_sparse_mle<F: Field>(
    sparse: Vec<(MultilinearPolynomial<F>, HypercubePoint)>,
) -> MultilinearPolynomial<F> {
    let k = sparse.len().next_power_of_two().trailing_zeros() as usize;
    let j = sparse[0].1.n_vars;
    let kappa = sparse[0].0.n_vars + j;
    let n_vars = k + kappa;
    let mut evals = vec![F::ZERO; 1 << n_vars];
    for (row_index, (row_pol, final_bits)) in sparse.iter().enumerate() {
        for (i, v) in row_pol.evals.iter().enumerate() {
            let mut index = i << (j + k); // first vars
            index += final_bits.val << k;
            index += row_index;
            evals[index] = *v;
        }
    }
    MultilinearPolynomial::new(evals)
}
