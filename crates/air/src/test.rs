use fiat_shamir::{FsProver, FsVerifier};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use pcs::{BatchSettings, RingSwitch, WhirPCS, WhirParameters};

use algebra::pols::MultilinearPolynomial;

use crate::AirBuilder;

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 4>;

#[test]
fn test_air_fibonacci() {
    let log_length = 8;
    let security_bits = 45;
    let log_inv_rate = 2;

    let mut builder = AirBuilder::<F, 2>::new();
    builder.set_fixed_value(0, 0, F::ZERO);
    builder.set_fixed_value(1, 0, F::ONE);
    builder.set_fixed_value(1, (1 << log_length) - 1, nth_fibonacci(1 << log_length));
    let ([c0_up, c1_up], [c0_down, c1_down]) = builder.vars();
    builder.assert_eq(c1_down, c0_up + c1_up.clone());
    builder.assert_eq(c0_down, c1_up);

    let table = builder.build();

    let mut col_1 = vec![F::ZERO];
    let mut col_2 = vec![F::ONE];
    for i in 1..(1 << log_length) {
        col_1.push(col_2[i - 1]);
        col_2.push(col_1[i - 1] + col_2[i - 1]);
    }

    let witnesses = vec![
        MultilinearPolynomial::new(col_1),
        MultilinearPolynomial::new(col_2),
    ];

    table.check_validity(&witnesses);

    let batch = BatchSettings::<F, EF, RingSwitch<F, EF, WhirPCS<EF>>>::new(
        2,
        log_length,
        &WhirParameters::standard(security_bits, log_inv_rate, false),
    );

    let mut batch_prover = batch.clone();

    let mut fs_prover = FsProver::new();
    let batch_witness = batch_prover.commit(&mut fs_prover, witnesses);
    table.prove(
        &mut fs_prover,
        &mut batch_prover,
        &batch_witness.polys,
        false,
    );
    batch_prover.prove(batch_witness, &mut fs_prover);

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let mut batch_verifier = batch.clone();
    let commitment = batch_verifier.parse_commitment(&mut fs_verifier).unwrap();
    table
        .verify(&mut fs_verifier, &mut batch_verifier, log_length)
        .unwrap();
    batch_verifier
        .verify(&mut fs_verifier, &commitment)
        .unwrap();
}

#[test]
fn test_air_complex() {
    let log_length = 8;
    let security_bits = 45;
    let log_inv_rate = 2;

    let mut builder = AirBuilder::<F, 3>::new();
    builder.set_fixed_value(0, 0, F::ZERO);
    builder.set_fixed_value(1, 0, F::ONE);
    builder.set_fixed_value(1, (1 << log_length) - 1, nth_fibonacci(1 << log_length));
    let ([c0_up, c1_up, c2_up], [c0_down, c1_down, c2_down]) = builder.vars();
    builder.assert_eq(c1_down.clone(), c0_up.clone() + c1_up.clone());
    builder.assert_eq(c0_down.clone(), c1_up.clone());
    // c2 + c1 * c0 + c1 * (c1 + c0 * c1 * 10) = 0
    builder.assert_zero(
        c2_down.clone()
            + c2_up.square()
            + (c1_up.clone() * c0_up.clone())
            + (c1_up.clone() * (c1_up.clone() + (c0_up * c1_up * F::from_usize(10)))),
    );

    let table = builder.build();

    let mut col_1 = vec![F::ZERO];
    let mut col_2 = vec![F::ONE];
    let mut col_3 = vec![F::NEG_ONE];
    for i in 1..(1 << log_length) {
        col_1.push(col_2[i - 1]);
        col_2.push(col_1[i - 1] + col_2[i - 1]);
        let c1 = col_1[i - 1];
        let c2 = col_2[i - 1];
        let c3 = col_3[i - 1];
        col_3.push(-(c3.square() + c2 * c1 + c2 * (c2 + c1 * c2 * F::from_usize(10))));
    }

    let witnesses = vec![
        MultilinearPolynomial::new(col_1),
        MultilinearPolynomial::new(col_2),
        MultilinearPolynomial::new(col_3),
    ];

    table.check_validity(&witnesses);

    let batch = BatchSettings::<F, EF, RingSwitch<F, EF, WhirPCS<EF>>>::new(
        table.n_columns,
        log_length,
        &WhirParameters::standard(security_bits, log_inv_rate, false),
    );

    let mut batch_prover = batch.clone();

    let mut fs_prover = FsProver::new();
    let batch_witness = batch_prover.commit(&mut fs_prover, witnesses);
    table.prove(
        &mut fs_prover,
        &mut batch_prover,
        &batch_witness.polys,
        false,
    );
    batch_prover.prove(batch_witness, &mut fs_prover);

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let mut batch_verifier = batch.clone();
    let commitment = batch_verifier.parse_commitment(&mut fs_verifier).unwrap();
    table
        .verify(&mut fs_verifier, &mut batch_verifier, log_length)
        .unwrap();
    batch_verifier
        .verify(&mut fs_verifier, &commitment)
        .unwrap();
}

fn nth_fibonacci(n: usize) -> F {
    let mut x = F::ZERO;
    let mut y = F::ONE;
    for _ in 0..n {
        let tmp = x;
        x = y;
        y += tmp;
    }
    x
}
