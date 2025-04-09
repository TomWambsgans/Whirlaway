use std::time::Instant;

use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

#[test]
#[ignore]
fn bench_fields() {
    bench_field::<KoalaBear>("KoalaBear");
    bench_field::<BinomialExtensionField<KoalaBear, 4>>("KoalaBear^4");
    bench_field::<BinomialExtensionField<KoalaBear, 8>>("KoalaBear^8");
}

fn bench_field<F: Field>(name: &str) {
    println!("{}:", name);

    let n_adds = 1000_000_000;
    let time = Instant::now();
    let mut v = F::ONE;
    for i in 1..=n_adds {
        v = v + F::from_u64(i)
    }
    assert!(v != F::ZERO);
    let duration = time.elapsed();
    let avg_addition = duration.as_nanos() / n_adds as u128;
    println!(
        "- {} additions in {:?} ({} ns per addition)",
        n_adds,
        duration,
        avg_addition
    );

    let n_muls = 10_000_000;
    let time = Instant::now();
    let mut v = F::ONE;
    for i in 1..=n_muls {
        v = v * F::from_u64(i)
    }
    assert!(v != F::ZERO);
    let duration = time.elapsed();
    let avg_multiplication = duration.as_nanos() / n_muls as u128;
    println!(
        "- {} multiplications in {:?} ({} ns per multiplication)",
        n_muls,
        duration,
        avg_multiplication
    );
    println!();
}
