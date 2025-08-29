use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::{RngCore, rng};
use std::time::Duration;
use utils::{fold_multilinear_in_large_field, fold_multilinear_in_small_field};
use whir_p3::poly::evals::EvaluationsList;

use p3_field::BasedVectorSpace;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

const SAMPLE_SIZE: usize = 50;
const MEASUREMENT_TIME: u64 = 15;

fn create_random_base_field_element_polys(log_2_rows: usize) -> EvaluationsList<F> {
    let mut rng = rng();

    let evals: Vec<F> = (0..(1 << log_2_rows))
        .map(|_| {
            let val: u32 = rng.next_u32();
            F::new(val)
        })
        .collect();

    EvaluationsList::new(evals)
}

fn create_random_extension_field_element_polys(log_2_rows: usize) -> EvaluationsList<EF> {
    let mut rng = rng();

    let evals: Vec<EF> = (0..(1 << log_2_rows))
        .map(|_| EF::from_basis_coefficients_fn(|_| F::new(rng.next_u32())))
        .collect();

    EvaluationsList::new(evals)
}

fn create_random_base_field_element_scalars(log_n: usize) -> Vec<F> {
    let mut rng = rng();

    (0..(1 << log_n))
        .map(|_| {
            let val: u32 = rng.next_u32();
            F::new(val)
        })
        .collect()
}

fn create_random_extension_field_element_scalars(log_n: usize) -> Vec<EF> {
    let mut rng = rng();

    (0..(1 << log_n))
        .map(|_| EF::from_basis_coefficients_fn(|_| F::new(rng.next_u32())))
        .collect()
}

// Represents the case: Round 0 (polynomials in base field), skips == 1, scalars in base field.
fn bench_fold_multilinear_in_small_field_with_base_field_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_small_field with base field polys (Round 0) and skips = 1",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_base_field_element_polys(log_2_rows);
        let scalars = create_random_base_field_element_scalars(1);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_small_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

// Represents the case: Rounds > 0 (polynomials in extension field), skips == 1, scalars in base field.
fn bench_fold_multilinear_in_small_field_with_extension_field_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_small_field with extension field polys (Rounds > 0) and skips = 1",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_extension_field_element_polys(log_2_rows);
        let scalars = create_random_base_field_element_scalars(1);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_small_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

// Represents the case: Round 0 (polynomials in base field), skips == 4, scalars in base field.
fn bench_fold_multilinear_in_small_field_with_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_small_field with base field polys (Round 0) and skips = 4",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];
    let log_2_scalars = 4; // Skip = 4

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_base_field_element_polys(log_2_rows);
        let scalars = create_random_base_field_element_scalars(log_2_scalars);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_small_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

// Represents the case: Round 0 (polynomials in base field), skips == 1, scalars in extension field.
fn bench_fold_multilinear_in_large_field_with_base_field_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_large_field with base field polys (Round 0) and skips = 1",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_base_field_element_polys(log_2_rows);
        let scalars = create_random_extension_field_element_scalars(1);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_large_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

// Represents the case: Rounds > 0 (polynomials in extension field), skips == 1, scalars in extension field.
fn bench_fold_multilinear_in_large_field_with_extension_field_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_large_field with extension field polys (Rounds > 0) and skips = 1",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_extension_field_element_polys(log_2_rows);
        let scalars = create_random_extension_field_element_scalars(1);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_large_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

// Represents the case: Round 0 (polynomials in base field), skips == 4, scalars in extension field.
fn bench_fold_multilinear_in_large_field_with_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group(
        "fold_multilinear_in_large_field with base field polys (Round 0) and skips = 4",
    );
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME));

    let log2_rows_list = [22];
    let log_2_scalars = 4; // Skip = 4

    for &log_2_rows in &log2_rows_list {
        let polys = create_random_base_field_element_polys(log_2_rows);
        let scalars = create_random_extension_field_element_scalars(log_2_scalars);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Rows: 2ˆ{}", log_2_rows)),
            &polys,
            |b, polys| {
                b.iter(|| fold_multilinear_in_large_field(black_box(polys), black_box(&scalars)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fold_multilinear_in_small_field_with_base_field_polys,
    bench_fold_multilinear_in_small_field_with_extension_field_polys,
    bench_fold_multilinear_in_small_field_with_skip,
    bench_fold_multilinear_in_large_field_with_base_field_polys,
    bench_fold_multilinear_in_large_field_with_extension_field_polys,
    bench_fold_multilinear_in_large_field_with_skip
);

criterion_main!(benches);
