use air::AirSettings;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::{Subscriber, span};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};
use whirlaway::examples::poseidon2::prove_poseidon2;

/// Parameters:
const LOG_N_ROWS: usize = 18;
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 128;

// adjust here if needed.
const UNIVARIATE_SKIPS: usize = 1;

/// Structure to accumulate times from `sumcheck_round` spans.
#[derive(Default)]
struct SumcheckAccum {
    // Times per span-id (to calculate duration)
    enters: HashMap<tracing::span::Id, Instant>,
    // Phase associated with the span when created (1=Zerocheck, 2=Inner)
    span_phase: HashMap<tracing::span::Id, u8>,
    // Current detected phase (0: none; 1: zerocheck; 2: inner)
    current_phase: u8,
    // Accumulated durations
    zerocheck_total: Duration,
    inner_total: Duration,
}

/// Tracing layer that measures spans called "sumcheck_round"
struct SumcheckTimingLayer {
    state: Arc<Mutex<SumcheckAccum>>,
}

impl SumcheckTimingLayer {
    fn new() -> (Self, Arc<Mutex<SumcheckAccum>>) {
        let state = Arc::new(Mutex::new(SumcheckAccum::default()));
        (
            Self {
                state: state.clone(),
            },
            state,
        )
    }
}

impl<S> Layer<S> for SumcheckTimingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        _attrs: &tracing::span::Attributes<'_>,
        id: &span::Id,
        ctx: Context<'_, S>,
    ) {
        if let Some(meta) = ctx.metadata(id) {
            let mut guard = self.state.lock().unwrap();

            // Detect "zerocheck" span to switch to phase 1
            if meta.name() == "zerocheck" {
                guard.current_phase = 1;
                return;
            }

            // Detect "inner sumchecks" span to switch to phase 2
            if meta.name() == "inner sumchecks" {
                guard.current_phase = 2;
                return;
            }

            // For "sumcheck_round" spans, assign the current phase
            if meta.name() == "sumcheck_round" {
                let current_phase = guard.current_phase;
                guard.span_phase.insert(id.clone(), current_phase);
            }
        }
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(meta) = ctx.metadata(id) {
            if meta.name() == "sumcheck_round" {
                let mut guard = self.state.lock().unwrap();
                guard.enters.insert(id.clone(), Instant::now());
            }
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(meta) = ctx.metadata(id) {
            if meta.name() == "sumcheck_round" {
                let mut guard = self.state.lock().unwrap();
                if let Some(start) = guard.enters.remove(id) {
                    let dur = start.elapsed();
                    // Attribute the duration to the phase saved when creating the span
                    match guard.span_phase.get(id).copied().unwrap_or(0) {
                        1 => {
                            guard.zerocheck_total += dur;
                        }
                        2 => {
                            guard.inner_total += dur;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

pub fn sumcheck_timings_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_timings");
    group.sample_size(10);

    // AIR configuration
    let settings = AirSettings::new(
        SECURITY_BITS,
        SecurityAssumption::CapacityBound,
        FoldingFactor::ConstantFromSecondRound(7, 4),
        LOG_INV_RATE,
        UNIVARIATE_SKIPS,
        5,
    );

    // Separate benchmark for Zerocheck
    group.bench_function(BenchmarkId::new("zerocheck_only", LOG_N_ROWS), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (layer, state) = SumcheckTimingLayer::new();
                tracing::subscriber::with_default(
                    tracing_subscriber::registry().with(layer),
                    || {
                        let _ = prove_poseidon2(LOG_N_ROWS, settings.clone(), 0, false);
                    },
                );
                total += state.lock().unwrap().zerocheck_total;
            }
            black_box(total)
        });
    });

    // Separate benchmark for Inner Sumcheck
    group.bench_function(BenchmarkId::new("inner_only", LOG_N_ROWS), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (layer, state) = SumcheckTimingLayer::new();
                tracing::subscriber::with_default(
                    tracing_subscriber::registry().with(layer),
                    || {
                        let _ = prove_poseidon2(LOG_N_ROWS, settings.clone(), 0, false);
                    },
                );
                total += state.lock().unwrap().inner_total;
            }
            black_box(total)
        });
    });

    group.finish();
}

criterion_group!(benches, sumcheck_timings_bench);
criterion_main!(benches);
