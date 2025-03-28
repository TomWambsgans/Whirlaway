use algebra::{
    field_utils::{dot_product, eq_extension},
    pols::{
        ArithmeticCircuit, ComposedPolynomial, GenericTransparentMultivariatePolynomial,
        HypercubePoint, MultilinearPolynomial, TransparentMultivariatePolynomial,
        UnivariatePolynomial,
    },
    utils::expand_randomness,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

use crate::SumcheckError;

// TODO
// re-add `eq_factor`
// re-add `sum`
// some evaluations can be avoided in the context of zero check
pub fn prove_with_univariate_skip<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    pol: ComposedPolynomial<F, NF, EF>,
    fs_prover: &mut FsProver,
    n: usize, // the first round will fold 2^n (instead of 2 in the basic sumcheck)
) -> Vec<EF> {
    assert!(pol.n_vars >= n);
    let max_degree_per_vars = pol.max_degree_per_vars();
    assert!(
        max_degree_per_vars
            .iter()
            .all(|x| *x == max_degree_per_vars[0])
    );

    let degree = (1 << n) * max_degree_per_vars[0];

    let selectors = compute_selectors::<F>(n);

    let mut evals = Vec::<(EF, EF)>::new();

    for z in 0usize..=degree {
        let selector_evals = selectors
            .iter()
            .map(|s| s.eval(&F::from_usize(z)))
            .collect::<Vec<_>>();

        let hypercube_partial_sum = (0..1 << (pol.n_vars - n))
            .into_par_iter()
            .map(|x| {
                let mut node_evals = Vec::new();
                for node in &pol.nodes {
                    node_evals.push(
                        (0..1 << n)
                            .map(|y| {
                                node.eval_hypercube(&HypercubePoint {
                                    n_vars: pol.n_vars,
                                    val: (y << n) | x,
                                })
                            })
                            .zip(&selector_evals)
                            .map(|(a, b)| a * *b)
                            .sum::<NF>(),
                    );
                }
                pol.structure.eval(&node_evals)
            })
            .sum::<EF>();
        evals.push((EF::from_usize(z), hypercube_partial_sum));
    }

    let p = UnivariatePolynomial::lagrange_interpolation(&evals).unwrap();

    debug_assert_eq!(
        (0..1 << n).map(|i| p.eval(&EF::from_usize(i))).sum::<EF>(),
        pol.sum_over_hypercube()
    );

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars::<EF>(1)[0];

    let selector_evals = selectors
        .iter()
        .map(|s| s.eval(&challenge))
        .collect::<Vec<_>>();

    // folding
    let mut folded_nodes = Vec::new();
    for node in &pol.nodes {
        let mut folded = Vec::with_capacity(1 << (pol.n_vars - n));
        for x in 0..1 << (pol.n_vars - n) {
            folded.push(
                (0..1 << n)
                    .map(|y| {
                        node.eval_hypercube(&HypercubePoint {
                            n_vars: pol.n_vars,
                            val: (y << n) | x,
                        })
                    })
                    .zip(&selector_evals)
                    .map(|(a, b)| *b * a)
                    .sum::<EF>(),
            );
        }
        folded_nodes.push(MultilinearPolynomial::new(folded));
    }

    let initial_nodes = pol.nodes;
    let folded_pol = ComposedPolynomial {
        n_vars: pol.n_vars - n,
        nodes: folded_nodes,
        structure: pol.structure,
        max_degree_per_vars: max_degree_per_vars[n..].to_vec(),
    };

    debug_assert_eq!(folded_pol.sum_over_hypercube(), p.eval(&challenge));

    let (next_challenges, final_node_evals) = super::prove(
        folded_pol,
        None,
        fs_prover,
        Some(p.eval(&challenge)),
        None,
        0,
    );
    let final_node_evals = final_node_evals
        .nodes
        .iter()
        .map(|n| n.eval::<EF>(&[]))
        .collect::<Vec<_>>();
    fs_prover.add_scalars(&final_node_evals);

    // sumcheck for each node, because the final claim is on a "rectangular" polynomial

    let batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];
    let expanded_randomness = expand_randomness(batching_scalar, initial_nodes.len());

    let final_batched_eval = dot_product(&final_node_evals, &expanded_randomness);

    let mut batched_evals = MultilinearPolynomial::<EF>::zero(pol.n_vars);
    for (pol, sc) in initial_nodes.iter().zip(&expanded_randomness) {
        batched_evals += pol.scale(*sc);
    }

    let next_chllenges_mle = MultilinearPolynomial::eq_mle(&next_challenges);
    let mut eq_factor = vec![];
    for sel in &selector_evals {
        eq_factor.extend(next_chllenges_mle.scale(*sel).evals);
    }
    let eq_factor = MultilinearPolynomial::new(eq_factor);

    let final_pol = ComposedPolynomial::<EF, EF>::new(
        pol.n_vars,
        vec![eq_factor, batched_evals],
        GenericTransparentMultivariatePolynomial::new(
            ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1),
            2,
        ),
    );

    debug_assert_eq!(final_pol.sum_over_hypercube(), final_batched_eval);

    let (final_challenges, _) = super::prove(
        final_pol,
        None,
        fs_prover,
        Some(final_batched_eval),
        None,
        0,
    );

    let final_evals = initial_nodes
        .iter()
        .map(|n| n.eval(&final_challenges))
        .collect::<Vec<_>>();
    fs_prover.add_scalars(&final_evals);

    final_challenges
}

pub fn verify_with_univariate_skip<F: Field, EF: ExtensionField<F>>(
    fs_verifier: &mut FsVerifier,
    composition_degree: usize,
    n_vars: usize,
    composition: &TransparentMultivariatePolynomial<F, EF>,
    n: usize,
) -> Result<(EF, Vec<EF>, Vec<EF>), SumcheckError> {
    let mut challenges = Vec::new();
    let mut sum = EF::ZERO;
    let mut target = EF::ZERO;

    for i in 0..1 + n_vars - n {
        let d = if i == 0 {
            (1 << n) * composition_degree
        } else {
            composition_degree
        };
        let coefs = fs_verifier.next_scalars::<EF>(d + 1)?;
        let pol = UnivariatePolynomial::new(coefs);
        dbg!(&pol);
        if i == 0 {
            sum = (0..1 << n).map(|i| pol.eval(&EF::from_usize(i))).sum();
            target = sum;
        }
        dbg!(target);

        if i != 0 && target != pol.eval(&EF::ZERO) + pol.eval(&EF::ONE) {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = fs_verifier.challenge_scalars(1)[0];

        target = pol.eval(&challenge);
        challenges.push(challenge);
    }

    let selector_evals = compute_selectors::<F>(n)
        .iter()
        .map(|s| s.eval(&challenges[0]))
        .collect::<Vec<_>>();

    let inner_evals = fs_verifier.next_scalars::<EF>(composition.n_vars())?;

    if composition.fix_computation().eval::<EF>(&inner_evals) != target {
        return Err(SumcheckError::InvalidRound);
    }

    // now it remains to verify that inner_evals are correct

    let batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];
    let expanded_randomness = expand_randomness(batching_scalar, composition.n_vars());

    let final_batched_eval = dot_product(&inner_evals, &expanded_randomness);

    let (final_value, final_point) = super::verify::<EF>(fs_verifier, &vec![2; n_vars], 0)?;

    if final_value != final_batched_eval {
        return Err(SumcheckError::InvalidRound);
    }

    let final_inner_evals = fs_verifier.next_scalars::<EF>(composition.n_vars())?;

    if final_point.value
        != dot_product(&expanded_randomness, &final_inner_evals)
            * eq_extension(&final_point.point[n..], &challenges[1..])
            * MultilinearPolynomial::new(selector_evals).eval(&final_point.point[..n])
    {
        return Err(SumcheckError::InvalidRound);
    }

    Ok((sum, final_point.point, final_inner_evals))
}

fn compute_selectors<F: Field>(n: usize) -> Vec<UnivariatePolynomial<F>> {
    (0..1 << n)
        .map(|i| {
            let values = (0..1 << n)
                .map(|j| (F::from_u64(j), if i == j { F::ONE } else { F::ZERO }))
                .collect::<Vec<_>>();
            UnivariatePolynomial::lagrange_interpolation(&values).unwrap()
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type F = KoalaBear;
    type EF = BinomialExtensionField<KoalaBear, 4>;

    #[test]
    fn test_univariate_skip() {
        let n_vars = 6;
        let n_nodes = 3;
        let n = 3;
        let composition = ArithmeticCircuit::Node(0)
            * (ArithmeticCircuit::Node(1)
                + ArithmeticCircuit::Node(0) * ArithmeticCircuit::Scalar(F::from_usize(785)))
            * ArithmeticCircuit::Node(2);
        let composition = TransparentMultivariatePolynomial::Generic(
            GenericTransparentMultivariatePolynomial::new(composition, 3),
        );

        let rng = &mut StdRng::seed_from_u64(0);
        let nodes = (0..n_nodes)
            .map(|_| MultilinearPolynomial::<F>::random(rng, n_vars))
            .collect::<Vec<_>>();

        let composed = ComposedPolynomial::<F, F, EF>::new(n_vars, nodes, composition.clone());
        let mut fs_prover = FsProver::new();
        let challenges =
            prove_with_univariate_skip::<F, F, EF>(composed.clone(), &mut fs_prover, n);

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let (sum, final_point, final_inner_evals) =
            verify_with_univariate_skip::<F, EF>(&mut fs_verifier, 3, n_vars, &composition, n)
                .unwrap();

        assert_eq!(sum, composed.sum_over_hypercube());

        assert_eq!(&challenges, &final_point);

        for i in 0..n_nodes {
            assert_eq!(composed.nodes[i].eval(&final_point), final_inner_evals[i]);
        }
    }
}
