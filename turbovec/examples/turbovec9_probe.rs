//! TurboVec9 research probe: 9-bit Lloyd-Max codes using turbovec's existing
//! codebook and rotation, plus a scalar 4-bit control and shipped 4-bit index.
//!
//! This does NOT change the production index, persistence or SIMD layout.
//! The 9-bit code is a 0..511 Lloyd-Max centroid ID, NOT ndarray Codec9's
//! DR512 presence/polarity encoding.
//!
//! From the standalone AdaWorldAPI/turbovec checkout:
//!   cargo test --release -p turbovec --features turbovec9-lab --example turbovec9_probe
//!   cargo run --release -p turbovec --features turbovec9-lab --example turbovec9_probe
//!   TV9_WORKLOAD=graph cargo run --release -p turbovec --features turbovec9-lab --example turbovec9_probe
//! Environment overrides: TV9_N, TV9_NQ, TV9_DIM, TV9_K, TV9_REPEATS,
//! TV9_WORKLOAD, TV9_NOISE (only for the graph workload).

use std::hint::black_box;
use std::time::Instant;
use turbovec::{codebook, rotation::Rotation, TurboQuantIndex};

const N_BITS: usize = 9;

fn row_bytes(dim: usize, bits: usize) -> usize {
    assert!(dim > 0 && dim % 8 == 0);
    bits * (dim / 8)
}

/// Reuse turbovec's bit-plane storage law, with u16 codewords for 9 bits.
fn write_code(packed: &mut [u8], dim: usize, bits: usize, row: usize, d: usize, code: u16) {
    assert!(code < (1u16 << bits));
    let base = row * row_bytes(dim, bits);
    let byte = d / 8;
    let mask = 1u8 << (7 - (d % 8));
    for p in 0..bits {
        if code & (1u16 << p) != 0 {
            packed[base + p * (dim / 8) + byte] |= mask;
        }
    }
}

fn read_code(packed: &[u8], dim: usize, bits: usize, row: usize, d: usize) -> usize {
    let base = row * row_bytes(dim, bits);
    let byte = d / 8;
    let mask = 1u8 << (7 - (d % 8));
    let mut code = 0usize;
    for p in 0..bits {
        if packed[base + p * (dim / 8) + byte] & mask != 0 {
            code |= 1usize << p;
        }
    }
    code
}

/// Uses the exact upstream TurboVec block-Hadamard rotation.
/// Deliberately scalar: measure algorithmic quality before any new kernel.
fn rotate(row: &[f32], rotation: &Rotation) -> Vec<f32> {
    let mut out = row.to_vec();
    rotation.apply(&mut out);
    out
}

struct ScalarTurbo {
    dim: usize,
    bits: usize,
    n: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    centroids: Vec<f32>,
}

impl ScalarTurbo {
    fn encode(vectors: &[f32], dim: usize, bits: usize, rotation: &Rotation) -> Result<Self, String> {
        if !(bits == 4 || bits == N_BITS) || dim == 0 || dim % 8 != 0
            || vectors.len() % dim != 0
            || vectors.iter().any(|x| !x.is_finite())
        {
            return Err("expected finite, whole vectors; dim multiple of 8; bits=4 or 9".into());
        }
        // Call the actual turbovec Lloyd-Max generator rather than making a
        // competing quantizer. The 512-level numerical stability is a falsifier.
        let (boundaries, centroids) = if bits == N_BITS {
            codebook::lab_codebook9(dim)
        } else {
            turbovec::expected_codebook(bits, dim)
        };
        if boundaries.len() != (1 << bits) - 1
            || centroids.len() != 1 << bits
            || boundaries.iter().any(|x| !x.is_finite())
            || centroids.iter().any(|x| !x.is_finite())
            || !boundaries.windows(2).all(|w| w[0] <= w[1])
            || !centroids.windows(2).all(|w| w[0] <= w[1])
        {
            return Err(format!("Lloyd-Max codebook at {bits} bits is not finite/ordered"));
        }
        let n = vectors.len() / dim;
        let mut packed = vec![0u8; n * row_bytes(dim, bits)];
        let mut scales = Vec::with_capacity(n);
        for (vi, v) in vectors.chunks_exact(dim).enumerate() {
            let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32;
            let unit: Vec<f32> = v.iter()
                .map(|&x| if norm > 1e-10 { x / norm } else { 0.0 })
                .collect();
            let rot = rotate(&unit, rotation);
            let mut inner = 0.0f64;
            for (d, &x) in rot.iter().enumerate() {
                // Exactly the upstream comparator (x > boundary), with a
                // binary search instead of the 511-comparison u8 scan.
                let code = boundaries.partition_point(|&boundary| x > boundary);
                write_code(&mut packed, dim, bits, vi, d, code as u16);
                inner += (x as f64) * (centroids[code] as f64);
            }
            // RaBitQ-style norm / self-dot correction; calibration disabled in this lab.
            scales.push(if norm <= 1e-10 {
                0.0
            } else {
                norm / (inner.max(1e-10) as f32)
            });
        }
        Ok(Self { dim, bits, n, packed, scales, centroids })
    }

    fn topk(&self, query: &[f32], rotation: &Rotation, k: usize) -> Vec<usize> {
        let q = rotate(query, rotation);
        let mut scored: Vec<(f32, usize)> = (0..self.n)
            .map(|vi| {
                let dot = (0..self.dim)
                    .map(|d| {
                        (q[d] as f64)
                            * (self.centroids[read_code(&self.packed, self.dim, self.bits, vi, d)] as f64)
                    })
                    .sum::<f64>();
                ((dot as f32) * self.scales[vi], vi)
            })
            .collect();
        scored.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        });
        scored.into_iter().take(k).map(|x| x.1).collect()
    }
}

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = vec![0.0f32; n * dim];
    for row in out.chunks_mut(dim) {
        let mut norm = 0.0f64;
        for x in row.iter_mut() {
            state = state.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            *x = v as f32;
            norm += v * v;
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for x in row.iter_mut() { *x = (*x as f64 * inv) as f32; }
    }
    out
}

fn exact_topk(db: &[f32], q: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let mut scored: Vec<(f64, usize)> = db.chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| {
            (v.iter().zip(q).map(|(&a, &b)| (a as f64) * (b as f64)).sum(), i)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|x| x.1).collect()
}

fn recall(found: &[usize], truth: &[usize]) -> f64 {
    let hits = found.iter().filter(|i| truth.contains(i)).count();
    hits as f64 / truth.len().max(1) as f64
}

/// Graph-regularization **input** probe: each group of four database
/// vectors is an explicit ground-truth neighborhood, independent of the
/// retrieval encoder. A query is a separate noisy observation of one group.
/// This measures neighbor selection only, NOT backprop/training accuracy.
fn graph_vectors(n: usize, nq: usize, dim: usize, noise: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(n >= 4 && n % 4 == 0 && nq > 0 && dim % 8 == 0);
    assert!(noise >= 0.0 && noise.is_finite());
    let groups = n / 4;
    let centers = unit_vectors(groups, dim, 0xD0C);
    let perturb_db = unit_vectors(n, dim, 0xBAD);
    let perturb_q = unit_vectors(nq, dim, 0xF00);
    let mut db = vec![0.0; n * dim];
    let mut queries = vec![0.0; nq * dim];
    fn mix(dst: &mut [f32], center: &[f32], jitter: &[f32], noise: f32) {
        let mut norm = 0f64;
        for d in 0..dst.len() {
            dst[d] = center[d] + noise * jitter[d];
            norm += (dst[d] as f64) * (dst[d] as f64);
        }
        let inv = norm.sqrt().recip() as f32;
        for x in dst { *x *= inv; }
    }
    for row in 0..n {
        let group = row / 4;
        mix(
            &mut db[row * dim..(row + 1) * dim],
            &centers[group * dim..(group + 1) * dim],
            &perturb_db[row * dim..(row + 1) * dim],
            noise,
        );
    }
    for q in 0..nq {
        let group = q % groups;
        mix(
            &mut queries[q * dim..(q + 1) * dim],
            &centers[group * dim..(group + 1) * dim],
            &perturb_q[q * dim..(q + 1) * dim],
            noise,
        );
    }
    (db, queries)
}

/// Independent explicit graph: group q%groups has exactly four edges.
/// "hit" means any graph edge appears in top-k, "coverage" counts all four.
fn graph_edges_at_k(found: &[usize], query: usize, groups: usize) -> (bool, f64) {
    let group = query % groups;
    let relevant = (group * 4)..(group * 4 + 4);
    let hits = found.iter().filter(|&&i| relevant.contains(&i)).count();
    (hits > 0, hits as f64 / 4.0)
}

fn run_graph_probe() {
    let n = env_usize("TV9_N", 512);
    let nq = env_usize("TV9_NQ", 64);
    let dim = env_usize("TV9_DIM", 64);
    let k = env_usize("TV9_K", 10).min(n);
    let noise: f32 = std::env::var("TV9_NOISE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.7);
    assert!(n >= 4 && n % 4 == 0 && dim > 0 && dim % 8 == 0 && nq > 0 && k > 0);
    let (db, queries) = graph_vectors(n, nq, dim, noise);
    let groups = n / 4;
    let rotation = Rotation::new(dim);
    let i4 = ScalarTurbo::encode(&db, dim, 4, &rotation).expect("4-bit");
    let i9 = ScalarTurbo::encode(&db, dim, 9, &rotation).expect("9-bit");
    let mut prod4 = TurboQuantIndex::new(dim, 4).expect("shipped 4-bit");
    prod4.add(&db);
    prod4.prepare();
    let truth: Vec<Vec<usize>> = queries.chunks_exact(dim)
        .map(|q| exact_topk(&db, q, dim, k)).collect();

    println!(
        "TurboVec9 GRAPH-LAB | n={n} dim={dim} queries={nq} k={k} noise={noise} \\
         graph=4-positives/group (fixed ground truth, independent of retrieval)"
    );
    println!("METRICS: topk_recall_f32 / graph_any_positive@k / graph_edge_coverage@k");
    for (name, bits, search) in [
        ("f32-exact", 32usize, 0usize),
        ("scalar-i4", 4, 1),
        ("scalar-i9", 9, 2),
        ("shipped-i4", 4, 3),
    ] {
        let mut recall_sum = 0.0f64;
        let mut hit_sum = 0usize;
        let mut edge_sum = 0.0f64;
        let t = Instant::now();
        // This loop is the same for all encoders; production search
        // includes its own per-query dispatch and index lookup.
        for (qi, query) in queries.chunks_exact(dim).enumerate() {
            let top = match search {
                0 => truth[qi].clone(),
                1 => i4.topk(query, &rotation, k),
                2 => i9.topk(query, &rotation, k),
                _ => prod4.search(query, k).indices_for_query(0)
                    .iter().map(|&i| i as usize).collect(),
            };
            recall_sum += recall(&top, &truth[qi]);
            let (hit, coverage) = graph_edges_at_k(&top, qi, groups);
            hit_sum += hit as usize;
            edge_sum += coverage;
        }
        println!(
            "{name:10} payload={:4} B/vector recall_f32@{k}={:.5} graph_any@{k}={:.5} \\
             graph_coverage@{k}={:.5} query={:.1} us",
            if search == 0 { dim * 4 } else { row_bytes(dim, bits) },
            recall_sum / nq as f64,
            hit_sum as f64 / nq as f64,
            edge_sum / nq as f64,
            t.elapsed().as_secs_f64() * 1e6 / nq as f64,
        );
    }
    println!("Interpretation: neighbor-selection proxy only; no NSL graph loss/gradient/training run.");
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    if std::env::var("TV9_WORKLOAD").ok().as_deref() == Some("graph") {
        run_graph_probe();
        return;
    }
    let n = env_usize("TV9_N", 512);
    let nq = env_usize("TV9_NQ", 32);
    let dim = env_usize("TV9_DIM", 64);
    let k = env_usize("TV9_K", 10).min(n);
    let repeats = env_usize("TV9_REPEATS", 3);
    assert!(dim > 0 && dim % 8 == 0 && n > 0 && nq > 0 && k > 0 && repeats > 0);
    println!("TurboVec9 LAB | n={n} dim={dim} queries={nq} k={k} repeats={repeats}");
    println!("No TQ+ calibration in scalar controls. Pure Rust scalar scan, not SIMD.");

    let db = unit_vectors(n, dim, 11);
    let queries = unit_vectors(nq, dim, 22);
    let rotation = Rotation::new(dim);

    let t = Instant::now();
    let i4 = ScalarTurbo::encode(&db, dim, 4, &rotation).expect("i4 codebook");
    let i4_encode = t.elapsed();
    let t = Instant::now();
    let i9 = ScalarTurbo::encode(&db, dim, N_BITS, &rotation).expect("i9 codebook");
    let i9_encode = t.elapsed();

    let mut prod4 = TurboQuantIndex::new(dim, 4).expect("production i4");
    prod4.add(&db);
    prod4.prepare();

    let truth: Vec<Vec<usize>> = queries.chunks_exact(dim)
        .map(|q| exact_topk(&db, q, dim, k))
        .collect();
    let prod = prod4.search(&queries, k);
    let mut prod_recall = 0.0;
    for (qi, gt) in truth.iter().enumerate() {
        let found: Vec<usize> = prod.indices_for_query(qi)
            .iter().map(|&i| i as usize).collect();
        prod_recall += recall(&found, gt);
    }
    prod_recall /= nq as f64;

    for (name, index, encode_elapsed) in [
        ("scalar-i4", &i4, i4_encode),
        ("scalar-i9", &i9, i9_encode),
    ] {
        let mut score = 0.0f64;
        let t = Instant::now();
        for _ in 0..repeats {
            for (qi, q) in queries.chunks_exact(dim).enumerate() {
                let results = black_box(index.topk(q, &rotation, k));
                score += recall(&results, &truth[qi]);
            }
        }
        let ns_per_query = t.elapsed().as_nanos() as f64 / (repeats * nq) as f64;
        println!(
            "{name:10} payload={:5} B/vector recall@{k}={:.5} scan={:.1} ns/query encode={:.3} ms",
            row_bytes(dim, index.bits),
            score / (repeats * nq) as f64,
            ns_per_query,
            encode_elapsed.as_secs_f64() * 1000.0
        );
    }
    let t = Instant::now();
    for _ in 0..repeats {
        black_box(prod4.search(&queries, k));
    }
    println!(
        "shipped-i4 payload={:5} B/vector recall@{k}={:.5} scan={:.1} ns/query (SIMD path, including top-k)",
        row_bytes(dim, 4),
        prod_recall,
        t.elapsed().as_nanos() as f64 / (repeats * nq) as f64
    );
    println!("Interpretation: scalar i4↔i9 isolates the bit-width effect; production i4 is a separate performance anchor.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_512_codewords_roundtrip_in_nine_planes() {
        let dim = 512;
        let mut packed = vec![0u8; row_bytes(dim, N_BITS)];
        for code in 0..512 {
            write_code(&mut packed, dim, N_BITS, 0, code, code as u16);
        }
        for code in 0..512 {
            assert_eq!(read_code(&packed, dim, N_BITS, 0, code), code);
        }
        assert_eq!(packed.len(), 576); // 512*9/8 bytes.
    }

    #[test]
    fn scalar_codebook_and_search_are_finite_at_nine_bits() {
        let dim = 32;
        let rotation = Rotation::new(dim);
        let db = unit_vectors(24, dim, 123);
        let index = ScalarTurbo::encode(&db, dim, N_BITS, &rotation).unwrap();
        assert_eq!(index.centroids.len(), 512);
        assert_eq!(index.packed.len(), 24 * 36);
        assert!(index.scales.iter().all(|s| s.is_finite()));
        let top = index.topk(&db[..dim], &rotation, 5);
        assert_eq!(top.len(), 5);
        assert!(top.iter().all(|&id| id < 24));
    }

    #[test]
    fn four_explicit_graph_edges_and_query_groups_are_stable() {
        let (db, q) = graph_vectors(24, 7, 32, 1.7);
        assert_eq!(db.len(), 24 * 32);
        assert_eq!(q.len(), 7 * 32);
        assert_eq!(graph_edges_at_k(&[0, 3, 8], 0, 6), (true, 0.5));
        assert_eq!(graph_edges_at_k(&[0, 3, 8], 1, 6), (false, 0.0));
        assert_eq!(graph_edges_at_k(&[0, 3, 8], 6, 6), (true, 0.5));
    }

    #[test]
    fn graph_probe_retrieval_has_valid_metrics_and_no_duplicate_ids() {
        let (db, queries) = graph_vectors(32, 5, 32, 1.7);
        let r = Rotation::new(32);
        let ix = ScalarTurbo::encode(&db, 32, 9, &r).unwrap();
        for (i, query) in queries.chunks_exact(32).enumerate() {
            let top = ix.topk(query, &r, 10);
            assert_eq!(top.len(), 10);
            let unique: std::collections::BTreeSet<usize> = top.iter().copied().collect();
            assert_eq!(unique.len(), top.len());
            let (hit, coverage) = graph_edges_at_k(&top, i, 8);
            assert_eq!(hit, coverage > 0.0);
            assert!((0.0..=1.0).contains(&coverage));
        }
    }

    #[test]
    fn nullable_values_must_refuse_before_encoding() {
        let dim = 8;
        let rotation = Rotation::new(dim);
        let mut db = vec![0.0f32; dim];
        db[3] = f32::NAN;
        assert!(ScalarTurbo::encode(&db, dim, N_BITS, &rotation).is_err());
    }
}
