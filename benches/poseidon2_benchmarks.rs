use air::AirSettings;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

use whirlaway::examples::poseidon2::prove_poseidon2;

fn poseidon2_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2_protocol");

    // Configurar para solo 10 muestras para que sea más rápido
    group.sample_size(30);

    group.bench_function("complete_protocol", |b| {
        b.iter(|| {
            let (log_n_rows, log_inv_rate) = (18, 1);
            let benchmark = prove_poseidon2(
                log_n_rows,
                AirSettings::new(
                    128, // SECURITY_BITS
                    SecurityAssumption::CapacityBound,
                    FoldingFactor::ConstantFromSecondRound(7, 4),
                    log_inv_rate,
                    4,
                    5,
                ),
                0,
                false, // Sin logs para evitar conflictos
            );
            black_box(benchmark);
        });
    });

    group.finish();
}

criterion_group!(benches, poseidon2_benchmark);
criterion_main!(benches);
