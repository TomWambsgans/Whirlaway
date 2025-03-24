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
    println!(
        "- {} additions in {:?} ({:?} per addition)",
        n_adds,
        time.elapsed(),
        time.elapsed() / n_adds as u32
    );

    let n_muls = 10_000_000;
    let time = Instant::now();
    let mut v = F::ONE;
    for i in 1..=n_muls {
        v = v * F::from_u64(i)
    }
    assert!(v != F::ZERO);
    println!(
        "- {} multiplications in {:?} ({:?} per multiplication)",
        n_muls,
        time.elapsed(),
        time.elapsed() / n_muls as u32
    );
    println!();
}
