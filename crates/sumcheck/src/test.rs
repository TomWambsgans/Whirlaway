use algebra::pols::{ComposedPolynomial, MultilinearPolynomial};
use fiat_shamir::{FsProver, FsVerifier};
use rand::{SeedableRng, rngs::StdRng};

use super::*;

type F = p3_koala_bear::KoalaBear;

#[test]
fn test_sumcheck() {
    let n_vars = 10;
    let rng = &mut StdRng::seed_from_u64(0);
    let pol = ComposedPolynomial::new_product(
        n_vars,
        (0..5)
            .map(|_| MultilinearPolynomial::<F>::random(rng, n_vars))
            .collect::<Vec<_>>(),
    );
    let mut fs_prover = FsProver::new();
    let sum = pol.sum_over_hypercube();
    prove(pol.clone(), None, false, &mut fs_prover, None, None, 0);

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let (claimed_sum, postponed_verification) =
        verify::<F>(&mut fs_verifier, &pol.max_degree_per_vars(), 0).unwrap();
    assert_eq!(sum, claimed_sum);
    assert_eq!(
        pol.eval(&postponed_verification.point),
        postponed_verification.value
    );
}
