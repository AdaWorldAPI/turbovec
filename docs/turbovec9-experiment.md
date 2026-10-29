# TurboVec9: first runnable 9-bit TurboQuant probe (2026-10-29)

**Status: EXPERIMENT ONLY. No production API or SIMD ABI changes.**

This experiment answers one question: on the *same* rotated vectors and Lloyd-Max
quantizer, how much does replacing 16 centroids (i4) with 512 centroids (i9)
change retrieval quality, scalar scan cost, and storage?

## Ground rules

- Use this fork's `turbovec::codebook::codebook(bits, dim)` and
  `rotation::make_rotation_matrix(dim)`. No second codebook algorithm.
- Codes are **unsigned centroid indices** 0..511, stored in nine bit planes,
  using `u16` during packing. Every code is a numeric centroid, *not*
  `ndarray::hpc::codec9`'s DR512 P×M / never-touched encoding.
- Norm/inner-product correction follows `encode.rs`; query and database
  rotate in the same orientation. No TQ+ calibration for scalar controls.
- Compare **scalar i4 ↔ scalar i9** for the bit-width effect, and report
  **shipped TurboQuantIndex i4** separately as the production speed anchor.
  Mixing scalar-i9 latency with SIMD-i4 and calling it a bit-width penalty
  would be misleading.
- No staged i4+i5 residual codec, no new gather/LUT kernel, no index format
  change, and no ndarray changes. Those require evidence from this run.
- The first run is intentionally small: this fork's existing Lloyd-Max
  generator was built for <=4 bits. Finite, ordered 512-codeword output is
  an explicit falsifier, not an assumption.

## Run

This fork currently depends on the AdaWorldAPI ndarray sibling checkout at
`../../ndarray` relative to `turbovec/Cargo.toml`.

```sh
cargo test --release -p turbovec --example turbovec9_probe
cargo run --release -p turbovec --example turbovec9_probe

# Try larger cases only once the first gate passes:
TV9_N=2048 TV9_NQ=64 TV9_DIM=128 TV9_K=10 TV9_REPEATS=3 \
  cargo run --release -p turbovec --example turbovec9_probe
```

The probe prints payload bytes per vector, Recall@k against exact f64
inner-product ordering on the same f32 inputs, encode time and warm
search ns/query. It reports the shipped i4 index as a separate anchor.

## Admission gates

1. 512/512 codeword bit-plane roundtrip, 9*dim/8 bytes, correct zero and
   high-bit handling. NaN input refuses before quantization.
2. The existing Lloyd-Max generator must return a finite, ordered 512-level
   codebook. If not, diagnose numerics before adding SIMD.
3. On identical deterministic queries compare scalar i4, scalar i9 and
   production i4 against the same brute-force oracle. Record host CPU and
   compiler when running on hardware.
4. Only if i9 materially improves recall, consider an ndarray-backed
   scoring kernel or TurboVec blocked layout; do not duplicate SIMD.

## Not yet proven

No CPU benchmarks are asserted in this PR. CI availability and the actual
512-level codebook behavior must be checked. This is **not** an NSL-training
result and does not imply TFLite Palette256 compatibility.

References: RyanCodrai/turbovec (TurboQuant Rust implementation);
arxiv.org/abs/2504.19874; AdaWorldAPI/ndarray #383 (different 9-bit
epistemic semantics); AdaWorldAPI/lance-graph #1846 (NSL graph-loss oracle).
