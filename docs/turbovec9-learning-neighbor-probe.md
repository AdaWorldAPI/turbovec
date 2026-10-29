# TurboVec9 × structured learning: graph-neighbor selection probe

**2026-10-29 · lab only · no production codec or index changes**

Why: NSL graph regularization is only as useful as the neighbor features/edges
it receives. TurboVec9's first scalar experiment showed high Recall@10 on
random unit vectors, but **did not measure labeled neighbor retention**.
This experiment tests that missing, narrower premise, not training quality.

## Oracle and graph

- **Truth graph is independent of quantization:** a seeded center is mixed
  into four database observations (same group label). Each query is a separate
  noisy observation of one center. The four group members are its *explicit
  graph edges*, even when exact cosine retrieval would choose another point.
- **f32 search oracle:** deterministic exact f64 dot-product ranking on the
  original f32 vectors. This is a retrieval reference, **not a graph-label
  oracle**, which is why both metrics are reported separately.
- **Controls:** scalar i4 and scalar i9 both use the **same upstream Lloyd-Max
  quantizer and deterministic two-round block-Hadamard rotation**. The shipped
  TurboQuantIndex i4 is listed separately because it has TQ+ calibration and
  SIMD/search policies that do not match the scalar controls.
- **Outputs:** (1) recall@k vs f32 top-k; (2) proportion of queries with
  at least one genuine group neighbor in top-k; (3) fraction of the four
  explicit edges found per query. These are *graph-construction/neighbor
  sampling proxies*; no gradients or neural training are executed.
- **Crucial limit:** improved top-k ranking is not guaranteed to improve
  NSL classification, graph loss, or robustness. Graph edges need an
  independently justified semantic source.

## Run

```sh
TV9_WORKLOAD=graph cargo run --release -p turbovec --features turbovec9-lab --example turbovec9_probe

TV9_WORKLOAD=graph TV9_N=2048 TV9_NQ=64 TV9_DIM=128 \
TV9_K=10 TV9_NOISE=1.7 cargo run --release -p turbovec \
  --features turbovec9-lab --example turbovec9_probe
```

Change `TV9_NOISE` to stress neighborhood separation; use identical seeds
and dimensions across controls. Payload counts only bit-packed vector codes,
not centroid tables, scale metadata or search caches.

## Cross-repo reuse (no copying)

1. **Numeric 9-bit**: `ndarray::hpc::codec9` owns DR512,
   `P=0,M=0` = never touched and touched zero is `+0`.
   It is **not** a 512-centroid vector quantizer. Neither codeword maps nor
   NaN policy may be conflated with TurboVec9.
2. **Neighbor indexing**: `turbovec` owns trained-data-oblivious
   Lloyd-Max centroid selection, packed code layout, search and ID/filter.
   It does *not* store NSL edge truth.
3. **Graph/loss oracle**: `lance-graph-nsl` owns the frozen Google
   graph-loss fixtures. It should consume selected neighbor IDs or an
   explicitly described feature view; it should not reimplement Lloyd-Max.
4. **Other existing substrates**: `ndarray::hpc::edge_codec` owns
   CoarseResidue/PQ32x4, `ndarray::hpc::cam_pq` owns PQ-ADC cascade.
   `bgz-tensor::turboquant_kv` owns the independent gain-shape i4
   cache probe. They are related *alternatives*, not byte-compatible codecs.

## Falsifiers and next admission

- If i9 graph-edge retention is not better at a useful cost, no promotion.
- Compare **fixed-budget** recall, bytes/query, and runtime on real embeddings
  before making a recommendation; do not infer relative performance from
  scalar i9 vs SIMD i4.
- Next: NSL loss/gradient/epoch parity on *fixed* graph labels, then *separate*
  learned graph construction using i4 vs i9 retrieved neighbor lists; report
  graph-edge precision and class accuracy under held-out labels.
- QJL residual debiasing is separate from the i9 test. The Google TurboQuant
  paper's two-stage method does not become proven merely by 9-bit Lloyd-Max.
- Do not silently coerce DR512 never-touched lanes to valid zeros in a
  learning input pipeline.

References:
- https://arxiv.org/abs/2504.19874
- https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- https://github.com/tensorflow/neural-structured-learning
- https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/codec9.rs
- https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/edge_codec.rs
- https://github.com/AdaWorldAPI/lance-graph/tree/main/crates/lance-graph-nsl
