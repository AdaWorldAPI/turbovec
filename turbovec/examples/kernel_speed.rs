//! Native LUT-ADC kernel vs the `ndarray::simd` polyfill GEMM vs scalar —
//! speed (ns/query) and recall@k against an exact f32 brute-force.
//!
//! This is the "test speed differences" harness for the turbovec ⇄ ndarray
//! integration. The polyfill path scores through
//! [`ndarray::simd::matmul_i8_to_i32`], so the SIMD backend (AMX `TDPBUSD`
//! tile → AVX-512 VPDPBUSD → AVX-VNNI → scalar) is chosen *inside ndarray*.
//!
//! Run (this AVX-512 host → VNNI path inside ndarray):
//! ```text
//!   cargo run --release --example kernel_speed --features ndarray-simd,bench-internals
//! ```
//! On a Sapphire-Rapids / AMX host, add `RUSTFLAGS="-C target-cpu=native"`
//! (or `-C target-cpu=sapphirerapids`) and `matmul_i8_to_i32` lights up the
//! AMX tile path automatically — turbovec needs no change.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::time::Instant;

use turbovec::search::FORCE_SCALAR_FALLBACK;
use turbovec::TurboQuantIndex;

/// Deterministic unit vectors via a SplitMix-ish LCG (no rand dep churn).
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
fn exact_topk(
    db: &[f32],
    queries: &[f32],
    n: usize,
    nq: usize,
    dim: usize,
    k: usize,
) -> Vec<BTreeSet<i64>> {
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
            scored.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
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

fn main() {
    let dim = 512usize; // multiple of 64 → AMX tile-aligned on Sapphire Rapids
    let n = 20_000usize;
    let nq = 256usize;
    let k = 10usize;
    let bits = 4usize;
    let repeats = 20usize;

    println!("turbovec kernel speed — n={n} dim={dim} nq={nq} k={k} bits={bits}");
    println!(
        "ndarray AMX tile tier available on this host: {}",
        ndarray::simd::amx_available()
    );

    let db = unit_vectors(n, dim, 11);
    let queries = unit_vectors(nq, dim, 22);

    let mut index = TurboQuantIndex::new(dim, bits).unwrap();
    index.add(&db);
    index.prepare();

    let exact = exact_topk(&db, &queries, n, nq, dim, k);

    // ── Native LUT-ADC (runtime AVX-512BW on this host) ──
    FORCE_SCALAR_FALLBACK.store(false, Ordering::Relaxed);
    let native = index.search(&queries, k);
    let t = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(index.search(&queries, k));
    }
    let native_ns = t.elapsed().as_nanos() as f64 / (repeats * nq) as f64;

    // ── Scalar fallback (turbovec's own reference kernel) ──
    FORCE_SCALAR_FALLBACK.store(true, Ordering::Relaxed);
    let reps_scalar = repeats.min(5);
    let t = Instant::now();
    for _ in 0..reps_scalar {
        std::hint::black_box(index.search(&queries, k));
    }
    let scalar_ns = t.elapsed().as_nanos() as f64 / (reps_scalar * nq) as f64;
    FORCE_SCALAR_FALLBACK.store(false, Ordering::Relaxed);

    // ── Polyfill GEMM (ndarray::simd: AMX→VNNI→scalar inside ndarray) ──
    let db_i8 = index.reconstruct_db_i8_transposed();
    let poly = index.search_polyfill_with_db(&queries, k, &db_i8);
    let t = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(index.search_polyfill_with_db(&queries, k, &db_i8));
    }
    let poly_ns = t.elapsed().as_nanos() as f64 / (repeats * nq) as f64;

    let r_native = recall(&native.indices, nq, k, &exact);
    let r_poly = recall(&poly.indices, nq, k, &exact);

    println!("\n                 ns/query   recall@{k}");
    println!("  native LUT-ADC  {native_ns:9.1}   {r_native:.4}");
    println!("  polyfill GEMM   {poly_ns:9.1}   {r_poly:.4}");
    println!("  scalar ref      {scalar_ns:9.1}   (top-k == native LUT-ADC)");

    println!(
        "\n  native vs scalar     : {:.1}x faster",
        scalar_ns / native_ns.max(1e-9)
    );
    println!(
        "  polyfill vs native   : {:.2}x ({})",
        poly_ns / native_ns.max(1e-9),
        if poly_ns < native_ns {
            "polyfill faster"
        } else {
            "native faster"
        }
    );
    let mem_native = n * dim * bits / 8;
    println!(
        "  DB memory            : native {} KB ({bits}-bit packed), polyfill {} KB (i8 recon)",
        mem_native / 1024,
        db_i8.len() / 1024
    );
}
