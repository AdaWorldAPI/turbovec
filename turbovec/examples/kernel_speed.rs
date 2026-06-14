//! Kernel speed + recall harness for the AVX2 `ndarray::simd` migration.
//!
//! Times three scoring paths of the SAME index on this host and reports
//! ns/query + recall@k against an exact f32 brute-force:
//!
//! - **native**  — default runtime dispatch (AVX-512BW on an AVX-512 host).
//!   This kernel is UNCHANGED by the migration, so it doubles as the
//!   "upstream" production baseline.
//! - **avx2**    — the migrated `search_multi_query_avx2`, forced via
//!   `FORCE_AVX2_PATH`. Every wide op now flows through `ndarray::simd`
//!   (`U8x32::shuffle_bytes`, native `U16x16`, `F32x8::mul_add`).
//! - **scalar**  — the `score_query_into_heap` reference, forced.
//!
//! Run:
//! ```text
//!   cargo run --release --example kernel_speed --features bench-internals
//! ```
//! To compare the AVX2 kernel against UPSTREAM's raw-intrinsic AVX2 kernel,
//! run the same example from a worktree checked out before the migration
//! commit (with the `FORCE_AVX2_PATH` hook cherry-picked in).

use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use turbovec::search::{FORCE_AVX2_PATH, FORCE_SCALAR_FALLBACK};
use turbovec::TurboQuantIndex;

/// Deterministic unit vectors via a SplitMix-ish LCG (no `rand` churn).
fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = vec![0.0f32; n * dim];
    for row in out.chunks_mut(dim) {
        let mut norm = 0.0f64;
        for x in row.iter_mut() {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            *x = v as f32;
            norm += v * v;
        }
        let inv = 1.0 / (norm.sqrt() + 1e-9);
        for x in row.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
    out
}

/// Exact top-k by f32 inner product — recall ground truth.
fn exact_topk(db: &[f32], queries: &[f32], n: usize, nq: usize, dim: usize, k: usize) -> Vec<BTreeSet<i64>> {
    (0..nq)
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, i64)> = (0..n)
                .map(|v| {
                    let row = &db[v * dim..(v + 1) * dim];
                    let dot: f32 = row.iter().zip(q).map(|(a, b)| a * b).sum();
                    (dot, v as i64)
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.iter().take(k).map(|p| p.1).collect()
        })
        .collect()
}

fn recall(approx_idx: &[i64], nq: usize, k: usize, exact: &[BTreeSet<i64>]) -> f64 {
    let mut hit = 0usize;
    for qi in 0..nq {
        for &i in &approx_idx[qi * k..(qi + 1) * k] {
            if exact[qi].contains(&i) {
                hit += 1;
            }
        }
    }
    hit as f64 / (nq * k) as f64
}

fn time_search(index: &TurboQuantIndex, queries: &[f32], k: usize, nq: usize, repeats: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..repeats {
        black_box(index.search(queries, k));
    }
    t.elapsed().as_nanos() as f64 / (repeats * nq) as f64
}

fn main() {
    let dim = 512usize;
    let n = 20_000usize;
    let nq = 256usize;
    let k = 10usize;
    let bits = 4usize;
    let repeats = 20usize;

    println!("turbovec kernel speed — n={n} dim={dim} nq={nq} k={k} bits={bits}");

    let db = unit_vectors(n, dim, 11);
    let queries = unit_vectors(nq, dim, 22);
    let mut index = TurboQuantIndex::new(dim, bits).unwrap();
    index.add(&db);
    index.prepare();
    let exact = exact_topk(&db, &queries, n, nq, dim, k);

    // ── native (default dispatch — AVX-512BW here; == upstream production) ──
    FORCE_AVX2_PATH.store(false, Ordering::SeqCst);
    FORCE_SCALAR_FALLBACK.store(false, Ordering::SeqCst);
    let native = index.search(&queries, k);
    let native_ns = time_search(&index, &queries, k, nq, repeats);

    // ── migrated AVX2 kernel (ndarray::simd), forced ──
    FORCE_AVX2_PATH.store(true, Ordering::SeqCst);
    let avx2 = index.search(&queries, k);
    let avx2_ns = time_search(&index, &queries, k, nq, repeats);
    FORCE_AVX2_PATH.store(false, Ordering::SeqCst);

    // ── scalar reference, forced (fewer reps; it's slow) ──
    FORCE_SCALAR_FALLBACK.store(true, Ordering::SeqCst);
    let scalar = index.search(&queries, k);
    let scalar_ns = time_search(&index, &queries, k, nq, repeats.min(3));
    FORCE_SCALAR_FALLBACK.store(false, Ordering::SeqCst);

    let r_native = recall(&native.indices, nq, k, &exact);
    let r_avx2 = recall(&avx2.indices, nq, k, &exact);
    let r_scalar = recall(&scalar.indices, nq, k, &exact);

    println!("\n                       ns/query   recall@{k}");
    println!("  native (AVX-512BW)   {native_ns:9.1}   {r_native:.4}");
    println!("  avx2 (ndarray::simd) {avx2_ns:9.1}   {r_avx2:.4}");
    println!("  scalar reference     {scalar_ns:9.1}   {r_scalar:.4}");
    println!(
        "\n  migrated avx2 vs native(avx512): {:.2}x  ({})",
        avx2_ns / native_ns.max(1e-9),
        if avx2_ns <= native_ns { "avx2 ≤ avx512" } else { "avx512 faster (expected: wider)" }
    );
    println!(
        "  migrated avx2 vs scalar        : {:.1}x faster",
        scalar_ns / avx2_ns.max(1e-9)
    );
}
