use ::air::AirSettings;
use air::table::AirTable;
use p3_challenger::DuplexChallenger;
use p3_field::PrimeField64;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator, parameters::FoldingFactor,
    whir::parameters::WhirConfig,
};

// Koalabear
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

// Koalabear
type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

// BabyBear
// type F = BabyBear;
// type EF = BinomialExtensionField<F, 4>;
// type LinearLayers = GenericPoseidon2LinearLayersBabyBear;
// const SBOX_DEGREE: u64 = 7;
// const SBOX_REGISTERS: usize = 1;
// const HALF_FULL_ROUNDS: usize = 4;
// const PARTIAL_ROUNDS: usize = 13;

const WIDTH: usize = 16;

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub settings: AirSettings,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: f64, // in bytes
}

impl fmt::Display for Poseidon2Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Security level: {} bits ({:?}), starting rate: 1/{}, folding factor: {}",
            self.settings.security_bits,
            self.settings.whir_soudness_type,
            1 << self.settings.whir_log_inv_rate,
            match self.settings.whir_folding_factor {
                FoldingFactor::Constant(factor) => format!("{factor}"),
                FoldingFactor::ConstantFromSecondRound(first, then) =>
                    format!("1st: {first} then {then}"),
            }
        )?;
        let n_rows = 1 << self.log_n_rows;
        writeln!(
            f,
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

pub fn prove_poseidon2(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    if display_logs {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

    let n_rows = 1 << log_n_rows;

    let mut rng = StdRng::seed_from_u64(0);
    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

    let poseidon_air = Poseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

    let inputs: Vec<[F; WIDTH]> = (0..n_rows)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

    let witness_matrix = generate_trace_rows::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, &constants, 0)
    .transpose();

    let mut witness = witness_matrix
        .rows()
        .map(|col| whir_p3::poly::evals::EvaluationsList::new(col.collect()))
        .collect::<Vec<_>>();

    let preprocessed_columns = witness.drain(..n_preprocessed_columns).collect::<Vec<_>>();

    let table = AirTable::<F, EF, _>::new(
        poseidon_air,
        log_n_rows,
        settings.univariate_skips,
        preprocessed_columns,
        3,
    );

    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);
    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let t = Instant::now();

    let whir_params: WhirConfig<_, _, _, _, MyChallenger> =
        table.build_whir_params(&settings, merkle_hash.clone(), merkle_compress.clone());
    let mut domainsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, 8>(&whir_params);
    domainsep.add_whir_proof::<_, _, _, 8>(&whir_params);

    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    table.prove(
        &settings,
        merkle_hash.clone(),
        merkle_compress.clone(),
        &mut prover_state,
        witness,
    );
    // let proof_size = prover_state.narg_string().len();

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

    table
        .verify(
            &settings,
            merkle_hash,
            merkle_compress,
            &mut verifier_state,
            log_n_rows,
        )
        .unwrap();
    let verifier_time = time.elapsed();

    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;

    Poseidon2Benchmark {
        log_n_rows,
        settings,
        prover_time,
        verifier_time,
        proof_size,
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::BasedVectorSpace;
    use p3_field::PackedField;
    use p3_field::PackedValue;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::dense::RowMajorMatrixView;
    use rayon::prelude::*;
    use utils::ConstraintFolder;

    type PackingF = <F as Field>::Packing;
    type PackingExtension = <EF as ExtensionField<F>>::ExtensionPacking;

    const N_COLS: usize = 3;
    const HEIGHT: usize = 1 << 24;
    const N_CONSTRAINTS: usize = 150;

    struct MyAir;

    impl<F: Field> BaseAir<F> for MyAir {
        fn width(&self) -> usize {
            N_COLS
        }
    }

    impl<AB: AirBuilder> Air<AB> for MyAir {
        #[inline]
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let up = &main.row_slice(0).expect("The matrix is empty?")[..];
            let down = &main.row_slice(1).expect("The matrix is empty?")[..];
            let up = up
                .iter()
                .map(|x| x.clone().into())
                .collect::<Vec<AB::Expr>>();
            let down = down
                .iter()
                .map(|x| x.clone().into())
                .collect::<Vec<AB::Expr>>();

            for _ in 0..N_CONSTRAINTS / 3 {
                builder.assert_eq(
                    up[0].clone() * up[1].clone(),
                    down[1].clone() * down[2].clone(),
                );
                builder.assert_eq(
                    up[1].clone() * up[1].clone(),
                    down[2].clone() * down[2].clone(),
                );
                builder.assert_eq(
                    up[0].clone() * up[0].clone(),
                    down[2].clone() * down[2].clone(),
                );
            }
        }
    }

    #[test]
    fn test_all() {
        test();
        test_packed();
    }

    #[test]
    fn test() {
        let mut rng = StdRng::seed_from_u64(0);

        let my_air = MyAir;

        let trace = RowMajorMatrix::new(
            (0..N_COLS * HEIGHT)
                .map(|_| rng.random())
                .collect::<Vec<F>>(),
            N_COLS,
        );

        let alpha: EF = rng.random();
        let alpha_powers = alpha.powers().take(N_CONSTRAINTS).collect();

        let time = Instant::now();

        let res = (0..HEIGHT)
            .into_par_iter()
            .map(|i| {
                let mut point = trace.row(i).unwrap().into_iter().collect::<Vec<F>>();
                point.extend(
                    trace
                        .row((i + 1) % HEIGHT)
                        .unwrap()
                        .into_iter()
                        .collect::<Vec<F>>(),
                );
                let mut folder = ConstraintFolder {
                    main: RowMajorMatrixView::new(&point, point.len() / 2),
                    alpha_powers: &alpha_powers,
                    accumulator: EF::ZERO,
                    constraint_index: 0,
                    _phantom: std::marker::PhantomData,
                };
                my_air.eval(&mut folder);
                folder.accumulator
            })
            .collect::<Vec<_>>();

        dbg!(res.into_par_iter().sum::<EF>());
        dbg!(time.elapsed());
    }

    #[derive(Debug)]
    pub struct ConstraintFolderPacked<'a> {
        pub main: RowMajorMatrixView<'a, PackingF>,
        pub alpha_powers: &'a [EF],
        pub decomposed_alpha_powers: &'a [Vec<F>],
        pub accumulator: PackingExtension,
        pub constraint_index: usize,
    }

    impl<'a> AirBuilder for ConstraintFolderPacked<'a> {
        type F = F;
        type Expr = PackingF;
        type Var = PackingF;
        type M = RowMajorMatrixView<'a, PackingF>;

        #[inline]
        fn main(&self) -> Self::M {
            self.main
        }

        #[inline]
        fn is_first_row(&self) -> Self::Expr {
            unreachable!()
        }

        #[inline]
        fn is_last_row(&self) -> Self::Expr {
            unreachable!()
        }

        /// Returns an expression indicating rows where transition constraints should be checked.
        ///
        /// # Panics
        /// This function panics if `size` is not `2`.
        #[inline]
        fn is_transition_window(&self, _: usize) -> Self::Expr {
            unreachable!()
        }

        #[inline]
        fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
            let alpha_power = self.alpha_powers[self.constraint_index];
            self.accumulator += Into::<PackingExtension>::into(alpha_power) * x.into();
            self.constraint_index += 1;
        }

        #[inline]
        fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
            let expr_array = array.map(Into::into);
            self.accumulator += PackingExtension::from_basis_coefficients_fn(|i| {
                let alpha_powers = &self.decomposed_alpha_powers[i]
                    [self.constraint_index..(self.constraint_index + N)];
                PackingF::packed_linear_combination::<N>(alpha_powers, &expr_array)
            });
            self.constraint_index += N;
        }
    }

    #[test]
    fn test_packed() {
        let mut rng = StdRng::seed_from_u64(0);

        let my_air = MyAir;

        let trace = RowMajorMatrix::new(
            (0..N_COLS * HEIGHT)
                .map(|_| rng.random())
                .collect::<Vec<F>>(),
            N_COLS,
        );

        let alpha: EF = rng.random();
        let alpha_powers = alpha.powers().take(N_CONSTRAINTS).collect();

        let decomposed_alpha_powers: Vec<_> = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|i| {
                alpha_powers
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();

        let time = Instant::now();

        let res = (0..HEIGHT)
            .into_par_iter()
            .step_by(PackingF::WIDTH)
            .flat_map_iter(|i_start| {
                let main =
                    RowMajorMatrix::new(trace.vertically_packed_row_pair(i_start, 1), N_COLS);

                let mut folder = ConstraintFolderPacked {
                    main: main.as_view(),
                    alpha_powers: &alpha_powers,
                    decomposed_alpha_powers: &decomposed_alpha_powers,
                    accumulator: PackingExtension::ZERO,
                    constraint_index: 0,
                };
                my_air.eval(&mut folder);

                (0..core::cmp::min(HEIGHT, PackingF::WIDTH)).map(move |idx_in_packing| {
                    EF::from_basis_coefficients_fn(|coeff_idx| {
                        BasedVectorSpace::<PackingF>::as_basis_coefficients_slice(
                            &folder.accumulator,
                        )[coeff_idx]
                            .as_slice()[idx_in_packing]
                    })
                })
            })
            .collect::<Vec<EF>>();

        dbg!(res.into_par_iter().sum::<EF>());
        dbg!(time.elapsed());
    }
}

#[cfg(test)]
mod tests_ef {

    use super::*;
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::BasedVectorSpace;
    use p3_field::PackedValue;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::dense::RowMajorMatrixView;
    use rayon::prelude::*;
    use utils::ConstraintFolder;

    type PackingF = <F as Field>::Packing;
    type PackingEF = <EF as Field>::Packing;
    type PackingExtension = <EF as ExtensionField<F>>::ExtensionPacking;

    const N_COLS: usize = 3;
    const HEIGHT: usize = 1 << 20;
    const N_CONSTRAINTS: usize = 150;

    struct MyAir;

    impl<F: Field> BaseAir<F> for MyAir {
        fn width(&self) -> usize {
            N_COLS
        }
    }

    impl<AB: AirBuilder> Air<AB> for MyAir {
        #[inline]
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let up = &main.row_slice(0).expect("The matrix is empty?")[..];
            let down = &main.row_slice(1).expect("The matrix is empty?")[..];
            let up = up
                .iter()
                .map(|x| x.clone().into())
                .collect::<Vec<AB::Expr>>();
            let down = down
                .iter()
                .map(|x| x.clone().into())
                .collect::<Vec<AB::Expr>>();

            for _ in 0..N_CONSTRAINTS / 3 {
                builder.assert_eq(
                    up[0].clone() * up[1].clone(),
                    down[1].clone() * down[2].clone(),
                );
                builder.assert_eq(
                    up[1].clone() * up[1].clone(),
                    down[2].clone() * down[2].clone(),
                );
                builder.assert_eq(
                    up[0].clone() * up[0].clone(),
                    down[2].clone() * down[2].clone(),
                );
            }
        }
    }

    #[test]
    fn test() {
        let mut rng = StdRng::seed_from_u64(0);

        let my_air = MyAir;

        let trace = RowMajorMatrix::new(
            (0..N_COLS * HEIGHT)
                .map(|_| rng.random())
                .collect::<Vec<EF>>(),
            N_COLS,
        );

        let alpha: EF = rng.random();
        let alpha_powers = alpha.powers().take(N_CONSTRAINTS).collect();

        let time = Instant::now();

        let res = (0..HEIGHT)
            .into_par_iter()
            .map(|i| {
                let mut point = trace.row(i).unwrap().into_iter().collect::<Vec<EF>>();
                point.extend(
                    trace
                        .row((i + 1) % HEIGHT)
                        .unwrap()
                        .into_iter()
                        .collect::<Vec<EF>>(),
                );
                let mut folder = ConstraintFolder {
                    main: RowMajorMatrixView::new(&point, point.len() / 2),
                    alpha_powers: &alpha_powers,
                    accumulator: EF::ZERO,
                    constraint_index: 0,
                    _phantom: std::marker::PhantomData::<F>,
                };
                my_air.eval(&mut folder);
                folder.accumulator
            })
            .collect::<Vec<_>>();

        dbg!(res.into_par_iter().sum::<EF>());
        dbg!(time.elapsed());
    }

    #[derive(Debug)]
    pub struct ConstraintFolderPacked<'a> {
        pub main: RowMajorMatrixView<'a, PackingEF>,
        pub alpha_powers: &'a [EF],
        pub accumulator: PackingExtension,
        pub constraint_index: usize,
    }

    impl<'a> AirBuilder for ConstraintFolderPacked<'a> {
        type F = F;
        type Expr = PackingEF;
        type Var = PackingEF;
        type M = RowMajorMatrixView<'a, PackingEF>;

        #[inline]
        fn main(&self) -> Self::M {
            self.main
        }

        #[inline]
        fn is_first_row(&self) -> Self::Expr {
            unreachable!()
        }

        #[inline]
        fn is_last_row(&self) -> Self::Expr {
            unreachable!()
        }

        /// Returns an expression indicating rows where transition constraints should be checked.
        ///
        /// # Panics
        /// This function panics if `size` is not `2`.
        #[inline]
        fn is_transition_window(&self, _: usize) -> Self::Expr {
            unreachable!()
        }

        #[inline]
        fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
            let alpha_power = self.alpha_powers[self.constraint_index];
            self.accumulator += Into::<PackingExtension>::into(alpha_power) * x.into();
            self.constraint_index += 1;
        }

        #[inline]
        fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
            for (i, a) in array.into_iter().enumerate() {
                let alpha_power = self.alpha_powers[self.constraint_index + i];
                self.accumulator += alpha_power * a.into();
            }
            self.constraint_index += N;
        }
    }

    #[test]
    fn test_all() {
        test();
        test_packed();
    }

    #[test]
    fn test_packed() {
        let mut rng = StdRng::seed_from_u64(0);

        let my_air = MyAir;

        let trace = RowMajorMatrix::new(
            (0..N_COLS * HEIGHT)
                .map(|_| rng.random())
                .collect::<Vec<EF>>(),
            N_COLS,
        );

        let alpha: EF = rng.random();
        let alpha_powers = alpha.powers().take(N_CONSTRAINTS).collect();

        let time = Instant::now();

        let res = (0..HEIGHT)
            .into_par_iter()
            .step_by(PackingEF::WIDTH)
            .flat_map_iter(|i_start| {
                let main =
                    RowMajorMatrix::new(trace.vertically_packed_row_pair(i_start, 1), N_COLS);

                let mut folder = ConstraintFolderPacked {
                    main: main.as_view(),
                    alpha_powers: &alpha_powers,
                    accumulator: PackingExtension::ZERO,
                    constraint_index: 0,
                };
                my_air.eval(&mut folder);

                (0..core::cmp::min(HEIGHT, PackingEF::WIDTH)).map(move |idx_in_packing| {
                    EF::from_basis_coefficients_fn(|coeff_idx| {
                        BasedVectorSpace::<PackingF>::as_basis_coefficients_slice(
                            &folder.accumulator,
                        )[coeff_idx]
                            .as_slice()[idx_in_packing]
                    })
                })
            })
            .collect::<Vec<EF>>();

        dbg!(res.into_par_iter().sum::<EF>());
        dbg!(time.elapsed());
    }
}
