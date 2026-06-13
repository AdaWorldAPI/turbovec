//! Polyfill scoring path — TurboQuant ADC expressed as a batched int8 GEMM
//! routed entirely through `ndarray::simd`.
//!
//! Gated behind the `ndarray-simd` feature. Compiled ALONGSIDE the native
//! kernel so `benches/kernel_speed.rs` can run them head-to-head.
//!
//! # The contract: the polyfill does the work, ndarray ships AMX
//!
//! turbovec writes **zero** raw SIMD intrinsics here. Every wide op goes
//! through the `ndarray::simd::*` surface, and ndarray owns the backend
//! dispatch — including AMX. The scoring GEMM calls
//! [`ndarray::simd::matmul_i8_to_i32`], whose runtime ladder is
//!
//! ```text
//!   AMX TDPBUSD tile (byte-asm, 16 384 MAC/instr, Sapphire Rapids+)
//!     → AVX-512 VPDPBUSD zmm (64 MAC/instr)
//!     → AVX-VNNI ymm (32 MAC/instr)
//!     → scalar i32 reference
//! ```
//!
//! all bit-identical. So a turbovec built with `-C target-cpu=sapphirerapids`
//! (or run on any AMX host) ships AMX *for free*; on this AVX-512 host the
//! same call runs VPDPBUSD-zmm; on a Pi it runs scalar. That is the whole
//! point of routing through the polyfill instead of hand-writing a kernel.
//!
//! # Why a GEMM (and not turbovec's nibble LUT)
//!
//! turbovec's native kernel scores via a per-query nibble lookup table
//! (`pshufb` gather + accumulate). A gather is **not** a matmul, so AMX —
//! a tile *matrix-multiply* unit — cannot accelerate it. The polyfill takes
//! the other road that TurboQuant's paper explicitly trades away: it
//! reconstructs each database vector's calibrated coordinates to i8 (the
//! centroid value per dim) and scores a whole query batch as one matmul
//!
//! ```text
//!   S[nq × n]  =  Q_i8[nq × dim]  ·  X̂ᵀ_i8[dim × n]
//! ```
//!
//! which is exactly the shape AMX tiles eat. The trade-off the benchmark
//! quantifies: i8 reconstruction is 8 bits/dim (finer than the 2/4-bit
//! packed codes, so recall is typically *higher*, but it costs 2–4× the
//! memory) and the matmul does the full `dim`-length dot per (query, vector)
//! pair rather than the LUT's O(1) table hit. LUT-ADC wins on low-bit
//! memory + raw op count; the GEMM wins when an AMX tile engine is present
//! and the query batch is wide enough to amortise it.
//!
//! Ranking note: within one query, the per-query quant scale and the i8
//! centroid scale are constants and do not change the argsort. Only the
//! per-vector renorm `scales[v]` (and, under TQ+, the per-query bias
//! correction) affect the ranking, so the f32 epilogue applies exactly
//! those — see [`TurboQuantIndex::search_polyfill_with_db`].

use crate::codebook;
use crate::rotation;
use crate::search::calibrate_queries;
use crate::{SearchResults, TurboQuantIndex};
use ndarray::simd::matmul_i8_to_i32;
use ndarray::{ArrayView2, ArrayViewMut2};
use rayon::prelude::*;

/// i8 scale for reconstructed centroid values. Lloyd-Max centroids live in
/// `[-1, 1]`; ×127 maps them onto the full signed-i8 range with one code to
/// spare (we clamp to ±127 so the AMX i8→u8 sign-shift never overflows).
const XHAT_I8_SCALE: f32 = 127.0;

impl TurboQuantIndex {
    /// Reconstruct the database into a **transposed** i8 matrix `X̂ᵀ` of
    /// shape `(dim × n_vectors)`, row-major:
    /// `out[d * n_vectors + v] = i8(round(centroid[code(v, d)] * 127))`.
    ///
    /// This is the polyfill analogue of `pack::repack` (the native blocked
    /// layout): a one-time, cacheable transform of `packed_codes` into the
    /// right-hand-side layout the scoring GEMM consumes. Cost is `O(n·dim)`;
    /// build it once and reuse it across many `search_polyfill_with_db`
    /// calls (the benchmark does exactly this, mirroring how the native path
    /// reuses its `OnceLock<BlockedCache>`).
    ///
    /// Returns an empty `Vec` for a lazy/empty index (no committed dim or
    /// zero vectors).
    pub fn reconstruct_db_i8_transposed(&self) -> Vec<i8> {
        let Some(dim) = self.dim_opt() else {
            return Vec::new();
        };
        let n = self.len();
        if n == 0 {
            return Vec::new();
        }
        let bits = self.bit_width();

        // Lloyd-Max centroids are a deterministic function of (bit_width, dim).
        // Reuse the same codebook the encoder/searcher use so the
        // reconstruction lands in the identical calibrated coordinate system.
        let (_boundaries, centroids) = codebook::codebook(bits, dim);

        let packed = self.packed_codes();
        let bytes_per_plane = dim / 8;
        let bytes_per_row = bits * bytes_per_plane;

        // Transposed layout: each ROW of the output is one coordinate `d`
        // across all `n` vectors. Build per-d in parallel so the GEMM's
        // right-hand side is contiguous over the vector axis.
        let mut out = vec![0i8; dim * n];
        out.par_chunks_mut(n).enumerate().for_each(|(d, row)| {
            let byte_in_plane = d / 8;
            let bit_in_byte = 7 - (d % 8);
            let mask = 1u8 << bit_in_byte;
            for (v, slot) in row.iter_mut().enumerate() {
                let row_base = v * bytes_per_row;
                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed[row_base + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }
                let q = (centroids[code as usize] * XHAT_I8_SCALE).round();
                *slot = q.clamp(-127.0, 127.0) as i8;
            }
        });
        out
    }

    /// Top-`k` search via the int8-GEMM polyfill, using a pre-built
    /// transposed reconstruction `db_i8_t` (from
    /// [`Self::reconstruct_db_i8_transposed`], shape `dim × n_vectors`).
    ///
    /// Mirrors [`TurboQuantIndex::search`]: rotate the queries into the
    /// codebook domain, apply the TQ+ inverse calibration, then score. The
    /// only difference is the scoring kernel — here it is a single
    /// `matmul_i8_to_i32` over the whole `(nq × dim) · (dim × n)` batch, so
    /// the AMX/VNNI/scalar choice is made inside ndarray.
    ///
    /// # Panics
    /// Panics if `queries.len()` is not a multiple of `dim`, or if
    /// `db_i8_t.len() != dim * n_vectors`.
    pub fn search_polyfill_with_db(
        &self,
        queries: &[f32],
        k: usize,
        db_i8_t: &[i8],
    ) -> SearchResults {
        let Some(dim) = self.dim_opt() else {
            return SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq: 0,
                k: 0,
            };
        };
        let n = self.len();
        let nq = queries.len() / dim;
        assert_eq!(
            queries.len(),
            nq * dim,
            "queries length must be a multiple of dim"
        );
        assert_eq!(db_i8_t.len(), dim * n, "db_i8_t must be dim * n_vectors");
        let effective_k = k.min(n);
        if effective_k == 0 || nq == 0 {
            return SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq,
                k: 0,
            };
        }

        // 1. Rotate queries into the codebook domain (q_rot = queries @ Rᵀ).
        //    Same batched GEMM the native path uses, via faer (no BLAS).
        //    Reuse the index's lazily-built rotation cache directly — a child
        //    module may touch the private `OnceLock` field (same crate).
        let rotation_mat = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(dim));
        let mut q_rot = vec![0.0f32; nq * dim];
        {
            let q_ref = faer::mat::from_row_major_slice::<f32, _, _>(queries, nq, dim);
            let r_ref = faer::mat::from_row_major_slice::<f32, _, _>(&rotation_mat[..], dim, dim);
            let out_mut = faer::mat::from_row_major_slice_mut::<f32, _, _>(&mut q_rot, nq, dim);
            faer::linalg::matmul::matmul(
                out_mut,
                q_ref,
                r_ref.transpose(),
                None,
                1.0_f32,
                faer::Parallelism::Rayon(0),
            );
        }

        // 2. TQ+ inverse calibration (identity for v2 / uncalibrated indexes).
        let (q_calib, bias_corrs) =
            calibrate_queries(&q_rot, self.tqplus_shift(), self.tqplus_scale(), nq, dim);

        // 3. Quantize each query row to i8 with a per-query symmetric scale.
        //    `inv_k[qi]` recovers the f32 inner product from the i32 GEMM
        //    output: ⟨q_calib, x̂⟩ ≈ scores_i32 / (sq · 127).
        let mut q_i8 = vec![0i8; nq * dim];
        let mut inv_k = vec![0.0f32; nq];
        q_i8.par_chunks_mut(dim)
            .zip(inv_k.par_iter_mut())
            .enumerate()
            .for_each(|(qi, (row, inv))| {
                let src = &q_calib[qi * dim..(qi + 1) * dim];
                let max_abs = src.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let sq = if max_abs > 1e-12 {
                    127.0 / max_abs
                } else {
                    0.0
                };
                for (d, slot) in row.iter_mut().enumerate() {
                    *slot = (src[d] * sq).round().clamp(-127.0, 127.0) as i8;
                }
                // K = sq · XHAT_I8_SCALE; guard the degenerate all-zero query.
                *inv = if sq > 0.0 {
                    1.0 / (sq * XHAT_I8_SCALE)
                } else {
                    0.0
                };
            });

        // 4. The scoring GEMM — THE polyfill call. ndarray picks AMX tile /
        //    VPDPBUSD / AVX-VNNI / scalar at runtime; turbovec stays clean.
        //    S[nq × n] = Q_i8[nq × dim] · X̂ᵀ_i8[dim × n].
        let mut scores = ndarray::Array2::<i32>::zeros((nq, n));
        {
            let lhs: ArrayView2<i8> =
                ArrayView2::from_shape((nq, dim), &q_i8[..]).expect("q_i8 shape");
            let rhs: ArrayView2<i8> =
                ArrayView2::from_shape((dim, n), db_i8_t).expect("db_i8_t shape");
            let out: ArrayViewMut2<i32> = scores.view_mut();
            matmul_i8_to_i32(lhs, rhs, out).expect("matmul_i8_to_i32 shape check");
        }

        // 5. f32 epilogue + per-query top-k. rank[v] = (⟨q,x̂⟩ + bias)·scale[v].
        //    The i8 scales are per-query constants and drop out of the
        //    argsort; only the per-vector renorm `scales[v]` and the TQ+
        //    bias correction change the ranking.
        let vec_scales = self.scales();
        let scores_arr = &scores;
        let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let row = scores_arr.row(qi);
                let inv = inv_k[qi];
                let bias = bias_corrs[qi];
                top_k_from_row(
                    row.as_slice().expect("contiguous score row"),
                    vec_scales,
                    inv,
                    bias,
                    effective_k,
                )
            })
            .collect();

        let mut all_scores = Vec::with_capacity(nq * effective_k);
        let mut all_indices = Vec::with_capacity(nq * effective_k);
        for (s, idx) in &results {
            all_scores.extend_from_slice(s);
            all_indices.extend_from_slice(idx);
        }
        SearchResults {
            scores: all_scores,
            indices: all_indices,
            nq,
            k: effective_k,
        }
    }

    /// Convenience wrapper: reconstruct the DB then score. For one-shot use
    /// and correctness tests; benchmarks should build the reconstruction
    /// once and call [`Self::search_polyfill_with_db`] in the timed loop.
    pub fn search_polyfill(&self, queries: &[f32], k: usize) -> SearchResults {
        let db = self.reconstruct_db_i8_transposed();
        self.search_polyfill_with_db(queries, k, &db)
    }
}

/// Per-query bounded top-k over one i32 score row. `rank = (s·inv_k + bias)·scale[v]`.
/// Returns `(scores, indices)` sorted descending by rank, length = `k`.
fn top_k_from_row(
    row: &[i32],
    vec_scales: &[f32],
    inv_k: f32,
    bias: f32,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let mut heap_s = vec![f32::NEG_INFINITY; k];
    let mut heap_i = vec![0u32; k];
    let mut sz = 0usize;
    let mut hmin = f32::NEG_INFINITY;
    let mut hmi = 0usize;
    for (v, &s) in row.iter().enumerate() {
        let rank = (s as f32 * inv_k + bias) * vec_scales[v];
        if sz < k {
            heap_s[sz] = rank;
            heap_i[sz] = v as u32;
            sz += 1;
            if sz == k {
                (hmi, hmin) = argmin_f32(&heap_s[..k]);
            }
        } else if rank > hmin {
            heap_s[hmi] = rank;
            heap_i[hmi] = v as u32;
            (hmi, hmin) = argmin_f32(&heap_s[..k]);
        }
    }
    let mut pairs: Vec<(f32, u32)> = heap_s[..sz]
        .iter()
        .zip(heap_i[..sz].iter())
        .map(|(&s, &i)| (s, i))
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let scores = pairs.iter().map(|p| p.0).collect();
    let indices = pairs.iter().map(|p| p.1 as i64).collect();
    (scores, indices)
}

/// `(index, value)` of the minimum of `xs` (NaN-tolerant via `<`; ties → first).
/// Iterator form so the heap-min rescan stays clippy-clean (no range-index loop).
#[inline]
fn argmin_f32(xs: &[f32]) -> (usize, f32) {
    let mut mi = 0usize;
    let mut mn = xs[0];
    for (i, &x) in xs.iter().enumerate().skip(1) {
        if x < mn {
            mn = x;
            mi = i;
        }
    }
    (mi, mn)
}

#[cfg(test)]
mod tests {
    use crate::TurboQuantIndex;
    use std::collections::BTreeSet;

    /// Deterministic unit vectors (same LCG as the kernel_speed example).
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

    fn exact_topk(db: &[f32], q: &[f32], n: usize, dim: usize, k: usize) -> BTreeSet<i64> {
        let mut scored: Vec<(f32, i64)> = (0..n)
            .map(|v| {
                let dot: f32 = db[v * dim..(v + 1) * dim]
                    .iter()
                    .zip(q)
                    .map(|(a, b)| a * b)
                    .sum();
                (dot, v as i64)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.iter().take(k).map(|p| p.1).collect()
    }

    #[test]
    fn polyfill_recall_tracks_brute_force() {
        let (dim, n, nq, k) = (64usize, 600usize, 8usize, 10usize);
        let db = unit_vectors(n, dim, 7);
        let queries = unit_vectors(nq, dim, 99);
        let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
        idx.add(&db);

        let res = idx.search_polyfill(&queries, k);
        assert_eq!(res.nq, nq);
        assert_eq!(res.k, k);

        // The i8-GEMM polyfill is an approximation; assert it recovers a
        // solid majority of the exact top-k (the 4-bit native kernel lands
        // ~0.78 on harder/bigger sets — on this small clean set the i8
        // reconstruction recall is high).
        let mut hits = 0usize;
        for qi in 0..nq {
            let exact = exact_topk(&db, &queries[qi * dim..(qi + 1) * dim], n, dim, k);
            for &i in &res.indices[qi * k..(qi + 1) * k] {
                if exact.contains(&i) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(
            recall >= 0.6,
            "polyfill recall@{k} = {recall:.3} below floor 0.6"
        );
    }

    #[test]
    fn reconstruct_shape_and_empty() {
        let (dim, n) = (32usize, 100usize);
        let db = unit_vectors(n, dim, 1);
        let mut idx = TurboQuantIndex::new(dim, 2).unwrap();
        idx.add(&db);
        assert_eq!(idx.reconstruct_db_i8_transposed().len(), dim * n);

        let empty = TurboQuantIndex::new(dim, 2).unwrap();
        assert!(empty.reconstruct_db_i8_transposed().is_empty());
    }
}
