use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{
    SumcheckComputation, cuda_init, cuda_preprocess_all_twiddles,
    cuda_preprocess_sumcheck_computation,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use pcs::{PCS, RingSwitch, WhirPCS, WhirParameters};

use algebra::pols::MultilinearHost;

use crate::AirBuilder;

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 8>;

#[test]
fn test_air_fibonacci() {
    let log_length = 10;
    let security_bits = 45;
    let log_inv_rate = 2;
    let cuda = false;

    let mut builder = AirBuilder::<F, 4>::new(log_length);
    let mut first_row_selector = vec![F::ZERO; 1 << log_length];
    first_row_selector[0] = F::ONE;
    builder.add_preprocess_column(first_row_selector);
    let mut last_row_selector = vec![F::ZERO; 1 << log_length];
    last_row_selector[(1 << log_length) - 1] = F::ONE;
    builder.add_preprocess_column(last_row_selector);
    let (
        [first_row_selector_up, _last_row_selector_up, c0_up, c1_up],
        [
            _first_row_selector_down,
            last_row_selector_down,
            c0_down,
            c1_down,
        ],
    ) = builder.vars();
    builder.assert_eq(c1_down.clone(), c0_up.clone() + c1_up.clone());
    builder.assert_eq(c0_down, c1_up.clone());

    builder.assert_eq_if(F::ZERO.into(), c0_up, first_row_selector_up.clone());
    builder.assert_eq_if(F::ONE.into(), c1_up, first_row_selector_up);

    builder.assert_eq_if(
        nth_fibonacci(1 << log_length).into(),
        c1_down,
        last_row_selector_down,
    );

    let table = builder.build();

    if cuda {
        let constraint_sumcheck_computations = SumcheckComputation::<F> {
            exprs: &table.constraints,
            n_multilinears: table.n_columns * 2 + 1,
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
            exprs: &[((ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3)))
            .fix_computation(false)],
            n_multilinears: 4,
            eq_mle_multiplier: false,
        };

        cuda_init();
        cuda_preprocess_sumcheck_computation(&constraint_sumcheck_computations);
        cuda_preprocess_sumcheck_computation(&prod_sumcheck);
        cuda_preprocess_sumcheck_computation(&inner_air_sumcheck);
        cuda_preprocess_all_twiddles::<F>(4);
    }

    let mut col_1 = vec![F::ZERO];
    let mut col_2 = vec![F::ONE];
    for i in 1..(1 << log_length) {
        col_1.push(col_2[i - 1]);
        col_2.push(col_1[i - 1] + col_2[i - 1]);
    }

    let witnesses = vec![MultilinearHost::new(col_1), MultilinearHost::new(col_2)];

    table.check_validity(&witnesses);

    let pcs = RingSwitch::<F, EF, WhirPCS<F, EF>>::new(
        log_length + 1,
        &WhirParameters::standard(security_bits, log_inv_rate, cuda),
    );

    let mut fs_prover = FsProver::new();
    table.prove(&mut fs_prover, &pcs, witnesses, cuda);

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table.verify(&mut fs_verifier, &pcs, log_length).unwrap();
}

#[test]
fn test_air_complex() {
    let log_length = 11;
    let security_bits = 45;
    let log_inv_rate = 2;
    let cuda = false;

    let mut builder = AirBuilder::<F, 8>::new(log_length);
    let mut first_row_selector = vec![F::ZERO; 1 << log_length];
    first_row_selector[0] = F::ONE;
    builder.add_preprocess_column(first_row_selector);
    let mut last_row_selector = vec![F::ZERO; 1 << log_length];
    last_row_selector[(1 << log_length) - 1] = F::ONE;
    builder.add_preprocess_column(last_row_selector);
    let counter = (0..(1 << log_length))
        .map(|i| F::new(i))
        .collect::<Vec<_>>();
    builder.add_preprocess_column(counter);

    let (
        [
            first_row_selector_up,
            _last_row_selector_up,
            counter_up,
            c0_up,
            c1_up,
            c2_up,
            c3_up,
            _c4_up,
        ],
        [
            _first_row_selector_down,
            last_row_selector_down,
            counter_down,
            c0_down,
            c1_down,
            c2_down,
            _c3_down,
            c4_down,
        ],
    ) = builder.vars();
    builder.assert_eq(c1_down.clone(), c0_up.clone() + c1_up.clone());
    builder.assert_eq(c0_down, c1_up.clone());

    builder.assert_eq_if(F::ZERO.into(), c0_up.clone(), first_row_selector_up.clone());
    builder.assert_eq_if(F::ONE.into(), c1_up.clone(), first_row_selector_up);

    builder.assert_eq_if(
        nth_fibonacci(1 << log_length).into(),
        c1_down,
        last_row_selector_down,
    );
    builder.assert_zero(
        c2_down.clone()
            + c2_up.square()
            + (c1_up.clone() * c0_up.clone())
            + (c1_up.clone()
                * (c1_up.clone() + (c0_up.clone() * c1_up.clone() * F::from_usize(10))))
            + counter_down.clone().square(),
    );
    builder.assert_eq(
        c3_up,
        counter_up.clone().cube() * F::new(78) + c0_up.clone(),
    );
    builder.assert_eq(
        c4_down,
        counter_up.clone().cube() * F::new(1111) + c1_up.clone(),
    );

    let table = builder.build();

    if cuda {
        let constraint_sumcheck_computations = SumcheckComputation::<F> {
            exprs: &table.constraints,
            n_multilinears: table.n_columns * 2 + 1,
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
            exprs: &[((ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3)))
            .fix_computation(false)],
            n_multilinears: 4,
            eq_mle_multiplier: false,
        };

        cuda_init();
        cuda_preprocess_sumcheck_computation(&constraint_sumcheck_computations);
        cuda_preprocess_sumcheck_computation(&prod_sumcheck);
        cuda_preprocess_sumcheck_computation(&inner_air_sumcheck);
        cuda_preprocess_all_twiddles::<F>(4);
    }
    let mut col_0 = vec![F::ZERO];
    let mut col_1 = vec![F::ONE];
    let mut col_2 = vec![F::NEG_ONE];
    let mut col_3 = vec![F::ZERO];
    let mut col_4 = vec![F::ZERO];
    for i in 1..(1 << log_length) {
        col_0.push(col_1[i - 1]);
        col_1.push(col_0[i - 1] + col_1[i - 1]);
        let c1 = col_0[i - 1];
        let c2 = col_1[i - 1];
        let c3 = col_2[i - 1];
        col_2.push(
            -(c3.square()
                + c2 * c1
                + c2 * (c2 + c1 * c2 * F::from_usize(10))
                + F::from_usize(i).square()),
        );
        col_3.push(col_0[i] + F::from_usize(78) * F::from_usize(i).cube());
        col_4.push(col_1[i - 1] + F::from_usize(1111) * F::from_usize(i - 1).cube());
    }

    let witnesses = vec![
        MultilinearHost::new(col_0),
        MultilinearHost::new(col_1),
        MultilinearHost::new(col_2),
        MultilinearHost::new(col_3),
        MultilinearHost::new(col_4),
    ];

    table.check_validity(&witnesses);

    let pcs = RingSwitch::<F, EF, WhirPCS<F, EF>>::new(
        log_length + 3,
        &WhirParameters::standard(security_bits, log_inv_rate, cuda),
    );

    let mut fs_prover = FsProver::new();
    table.prove(&mut fs_prover, &pcs, witnesses, cuda);

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table.verify(&mut fs_verifier, &pcs, log_length).unwrap();
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
