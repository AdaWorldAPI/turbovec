//! TurboVec9 research probe: 9-bit Lloyd-Max codes using turbovec's existing
//! codebook and rotation, plus a scalar 4-bit control and shipped 4-bit index.
//!
//! This does NOT change the production index, persistence or SIMD layout.
//! The 9-bit code is a 0..511 Lloyd-Max centroid ID, NOT ndarray Codec9's
//! DR512 presence/polarity encoding.
//!
//! From the AdaWorldAPI/turbovec checkout with sibling ../ndarray:
//!   cargo test -p turbovec --release --example turbovec9_probe
//!   cargo run -p turbovec --release --example turbovec9_probe
//! Environment overrides: TV9_N, TV9_NQ, TV9_DIM, TV9_K, TV9_REPEATS.

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Instant;
use turbovec::{codebook, rotation, TurboQuantIndex};

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

/// Same row-vector @ rotation^T orientation as turbovec encode/search.
/// Deliberately scalar: measure algorithmic quality before any new kernel.
fn rotate(row: &[f32], rotation: &[f32]) -> Vec<f32> {
    let dim = row.len();
    assert_eq!(rotation.len(), dim * dim);
    (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| (row[j] as f64) * (rotation[i * dim + j] as f64))
                .sum::<f64>() as f32
        })
        .collect()
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
    fn encode(vectors: &[f32], dim: usize, bits: usize, rotation: &[f32]) -> Result<Self, String> {
        if !(bits == 4 || bits == N_BITS) || dim == 0 || dim % 8 != 0
            || vectors.len() % dim != 0
            || vectors.iter().any(|x| !x.is_finite())
        {
            return Err("expected finite, whole vectors; dim multiple of 8; bits=4 or 9".into());
        }
        // Call the actual turbovec Lloyd-Max generator rather than making a
        // competing quantizer. The 512-level numerical stability is a falsifier.
        let (boundaries, centroids) = codebook::codebook(bits, dim);
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
            // Same RaBitQ-style norm / self-dot correction as encode.rs.
            scales.push(if norm <= 1e-10 {
                0.0
            } else {
                norm / (inner.max(1e-10) as f32)
            });
        }
        Ok(Self { dim, bits, n, packed, scales, centroids })
    }

    fn topk(&self, query: &[f32], rotation: &[f32], k: usize) -> Vec<usize> {
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
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
    let rotation = rotation::make_rotation_matrix(dim);

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
        let rotation = rotation::make_rotation_matrix(dim);
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
    fn nullable_values_must_refuse_before_encoding() {
        let dim = 8;
        let rotation = rotation::make_rotation_matrix(dim);
        let mut db = vec![0.0f32; dim];
        db[3] = f32::NAN;
        assert!(ScalarTurbo::encode(&db, dim, N_BITS, &rotation).is_err());
    }
}
