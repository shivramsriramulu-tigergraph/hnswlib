# Design: Low-Precision Vector Storage in TigerGraph (CORE-5808)

**Author:** Shivram Sriramulu  
**Reviewers:** Jian Gong, Songting Chen, Adil (GSQL)  
**Status:** Draft  
**Jira:** [CORE-5808](https://graphsql.atlassian.net/browse/CORE-5808)  
**Related:** [CORE-5791](https://graphsql.atlassian.net/browse/CORE-5791) (High QPS)

---

## 1. Motivation

TigerGraph's vector search stores embeddings as FP32 (4 bytes/dimension). For a graph with 100M vertices and 1024-dim embeddings, that's **400 GB** of embedding data. Most modern ML models and inference hardware natively work at lower precision, so storing at full FP32 precision is wasteful.

Lower-precision formats reduce memory footprint, improve DRAM bandwidth utilization, and increase L2/L3 cache hit rates — all of which directly translate to higher QPS on vector search.

| Format | Bytes/dim | vs FP32 | Hardware support |
|--------|-----------|---------|-----------------|
| FP32   | 4         | 1×      | Universal        |
| BF16   | 2         | 2×      | NVIDIA A100/H100, Intel AMX, Google TPU |
| FP16   | 2         | 2×      | NVIDIA GPU, Apple M-series, x86 F16C (Ivy Bridge+) |
| FP8 E4M3 | 1      | 4×      | NVIDIA H100 (forward pass) |
| FP8 E5M2 | 1      | 4×      | NVIDIA H100 (gradient) |
| INT4   | 0.5       | 8×      | NVIDIA H100, Intel AMX |

For a 100M × 1024 embedding table: FP32=400GB → FP16/BF16=200GB → FP8=100GB → INT4=50GB.

---

## 2. Format Reference

### 2.1 FP16 (IEEE 754 half-precision)

```
[S][EEEEE][MMMMMMMMMM]
 1    5         10      = 16 bits
```
- Exponent bias: 15. Normal range: ~6×10⁻⁵ to 65504.
- Subnormals: exp=0, implicit leading 0 (supports gradual underflow).
- Special: exp=31, mant=0 → ±Inf; exp=31, mant≠0 → NaN.
- **x86 acceleration**: `_mm256_cvtph_ps` (F16C, Ivy Bridge+ 2012). Converts 8 FP16→FP32 per instruction.
- **Status**: ✅ Implemented (scalar + F16C SIMD, L2 and IP spaces).

### 2.2 BF16 (Brain Float 16)

```
[S][EEEEEEEE][MMMMMMM]
 1      8         7     = 16 bits
```
- Same exponent as FP32 (bias 127). Range: same as FP32 (~1.175×10⁻³⁸ to 3.4×10³⁸).
- Less mantissa precision than FP16 (7 vs 10 bits), but no overflow risk when casting from FP32.
- **Conversion from FP32**: truncate the low 16 bits (with optional rounding). Single instruction on supported hardware.
- **Hardware**: NVIDIA A100/H100 Tensor Cores, Intel AMX (Sapphire Rapids+), Google TPU.
- **Why not FP16 instead?** FP16 overflows at 65504 — large logit values in LLMs commonly exceed this. BF16 is the standard choice for training/inference on modern accelerators.
- **Status**: ❌ Not implemented.

### 2.3 FP8 E4M3 (NVIDIA H100 forward pass)

```
[S][EEEE][MMM]
 1    4     3   = 8 bits
```
- Exponent bias: 7. Normal range: ~1.95×10⁻³ to 448.
- Special: all-1 exponent + all-1 mantissa = NaN (no ±Inf, unlike IEEE 754).
- Used for inference activations and weights on H100.
- **Status**: ❌ Placeholder only (raw uint8 cast, not real FP8 conversion).

### 2.4 FP8 E5M2 (NVIDIA H100 gradients)

```
[S][EEEEE][MM]
 1    5     2   = 8 bits
```
- Exponent bias: 15. Normal range: ~1.53×10⁻⁵ to 57344.
- Supports ±Inf and NaN (IEEE-compatible special values).
- Used for gradient values (wider range needed) on H100.
- **Status**: ❌ Placeholder only.

### 2.5 INT4 / FP4

- **INT4**: 4-bit signed integer [-8, 7]. Always paired with a per-vector or per-block scale factor (dequant).
- **FP4**: 4-bit float. Multiple variants (E2M1, NF4). NF4 (NormalFloat4) used in QLoRA.
- Storage: 2 elements packed per byte. Requires explicit pack/unpack.
- No standard x86 SIMD for FP4/INT4 dot products below AVX-512VNNI (which handles INT8, not INT4 natively).
- **Status**: ❌ Not implemented. Needs design discussion.

---

## 3. Current Architecture

```
GSQL query (list<float>, FP32)
        │
        ▼
embeddingaction.cpp
  EmbeddingDataType switch
        │
        ├─ _Float       → query as-is → TopKVectorSearch<float>
        ├─ _HalfPrecision → fp32_to_fp16(query) → TopKVectorSearch (fp16 buf)
        └─ _Binary / other → exception (not implemented)
        │
        ▼
EmbeddingInstance<uint32_t, float>::TopKVectorSearch
        │
        ▼
EmbeddingIndex → hnswlib::Index (space selected by data_type string)
        │
        ├─ "float32" → L2Space / InnerProductSpace
        ├─ "float16" → L2SpaceFp16 / InnerProductSpaceFp16
        └─ "float8"  → L2SpaceFp8 / InnerProductSpaceFp8 (placeholder)
        │
        ▼
Distance function (const void* → cast to typed pointer → compute)
```

**Storage layout** (on disk):
```
tigergraph/data/gstore/.emb/<graph>/<vertex_type>/<attr_name>/
  ├── main.bin          # raw embedding vectors (element_size * dim * N bytes)
  ├── config.yaml       # dimension, metric, dtype, etc.
  └── index/            # HNSW graph structure
```

---

## 4. Implementation Plan

### Phase 1 — FP16 (complete, minor gaps)

FP16 is the highest-value, lowest-risk format. 2× memory reduction with near-zero accuracy loss for most embedding models (BERT, OpenAI ada-002, etc. were trained at or fine-tuned to FP16).

**Remaining work:**
- [ ] **Serialization**: `hnswlib_interface.h` needs to save `data_type` string in the HNSW index file header (`saveIndex`/`loadIndex`), so a reloaded index knows which space to instantiate.
- [ ] **F16C SIMD test**: Verify `L2SqrFp16_F16C` path on a Haswell+ x86 server (current testing was on Sandy Bridge which has no F16C — scalar fallback only).
- [ ] **Insert path**: `cdc_delta_transformer.cpp` needs FP32→FP16 conversion when writing embedding vectors to `main.bin` for a `HalfPrecision` attribute.

### Phase 2 — BF16 (new implementation)

BF16 is the lowest-effort format to add after FP16 — conversion is just a 16-bit truncation of FP32 bits, no complex bit manipulation needed. Many embedding models from Hugging Face and Google are natively BF16.

**hnswlib changes:**
- `hnswlib.h`: add `typedef uint16_t bfloat16_t`
- `space_l2.h`: add `bf16_to_fp32()`, `L2SqrBf16`, `L2SpaceBf16`
- `space_ip.h`: add `InnerProductBf16`, `InnerProductSpaceBf16`
- `hnswlib_interface.h`: wire `"bfloat16"` → `L2SpaceBf16` / `InnerProductSpaceBf16`

**Conversion** (simpler than FP16 — no subnormal normalization loop needed):
```cpp
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float r; memcpy(&r, &f, sizeof(float)); return r;
}
static inline uint16_t fp32_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
    // round to nearest even
    bits += 0x7fff + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}
```

**Engine changes:**
- Add `EmbeddingDataType_BF16` to the `EmbeddingDataType` enum in `topology4/metafiles.hpp`
- `embeddingaction.cpp`: add BF16 case (same pattern as FP16)
- `cdc_delta_transformer.cpp`: add BF16 insert path

**GSQL schema changes** (Adil's team):
- `ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb(dimension=1024, metric="L2", dtype="BFLOAT16")`

### Phase 3 — FP8 proper (replace placeholder)

FP8 is primarily used for H100 inference. For TigerGraph's use case (approximate nearest neighbor search, not matrix multiply), FP8 is useful when embeddings are already produced at FP8 precision or when maximum compression is needed at the cost of accuracy.

**Key decision needed: single `float8_t` or two separate types?**

Option A — Single `float8_t` with a variant flag per index:
- Simpler API: `dtype="float8_e4m3"` or `dtype="float8_e5m2"`
- Current `float8_t = uint8_t` works for both
- Pro: minimal schema change. Con: runtime check needed per distance call.

Option B — Two separate types `float8_e4m3_t` and `float8_e5m2_t`:
- Each gets its own Space class
- Pro: no runtime branch in hot path. Con: doubles the number of space classes.

**Recommendation**: Option A — the `data_type` string already distinguishes them at index construction time. One distance function per variant, selected at construction, no runtime overhead.

**Conversion functions** (both need proper implementation to replace the uint8 cast):
```cpp
// E4M3: bias=7, max=448, NaN=0xFF
static inline float fp8e4m3_to_fp32(uint8_t h) {
    uint8_t sign = h >> 7;
    uint8_t exp  = (h >> 3) & 0xf;
    uint8_t mant = h & 0x7;
    if (exp == 0xf && mant == 0x7) { /* NaN */ ... }
    if (exp == 0) { /* subnormal */ ... }
    // normal: fp32_exp = exp - 7 + 127 = exp + 120
    uint32_t f = ((uint32_t)sign << 31) | ((uint32_t)(exp + 120) << 23) | ((uint32_t)mant << 20);
    float r; memcpy(&r, &f, sizeof(float)); return r;
}

// E5M2: bias=15, max=57344, supports Inf/NaN
static inline float fp8e5m2_to_fp32(uint8_t h) {
    uint8_t sign = h >> 7;
    uint8_t exp  = (h >> 2) & 0x1f;
    uint8_t mant = h & 0x3;
    if (exp == 0x1f) { /* Inf or NaN */ ... }
    if (exp == 0)    { /* subnormal */ ... }
    // normal: fp32_exp = exp - 15 + 127 = exp + 112
    uint32_t f = ((uint32_t)sign << 31) | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 21);
    float r; memcpy(&r, &f, sizeof(float)); return r;
}
```

### Phase 4 — INT4 (future, requires design review)

INT4 is fundamentally different from the float formats — it requires a **quantization scale** (a float per vector or per block of elements). Distance computation is:
```
dist(q, v) = scale_q * scale_v * L2(q_int4, v_int4)
```
This needs a significant storage format change (scale stored alongside vectors) and a new distance function signature. **This phase requires a separate design review before implementation.**

Packed storage (2 elements per byte):
```cpp
uint8_t packed = (a & 0xF) | ((b & 0xF) << 4);
int8_t a = (packed & 0xF);           // low nibble
if (a > 7) a -= 16;                  // sign extend
int8_t b = (packed >> 4) & 0xF;      // high nibble
if (b > 7) b -= 16;
```

---

## 5. GSQL API Changes

Current syntax (FP32 only):
```sql
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb1(dimension=1024, metric="L2")
```

Proposed extension (requires GSQL team — Adil):
```sql
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb1(dimension=1024, metric="L2", dtype="FLOAT16")
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb1(dimension=1024, metric="COSINE", dtype="BFLOAT16")
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb1(dimension=512, metric="IP", dtype="FP8_E4M3")
```

**Query side stays FP32** — the user always passes `list<float>` to `VectorSearch`. The engine handles conversion from FP32 query → stored dtype before calling `searchKnn`. This is the right design because:
1. GSQL users shouldn't need to know the storage format
2. FP32 query is lossless; quantization only applies to the index

The conversion functions live in `embeddingaction.cpp`:
- FP32 → FP16: already implemented (Phase 1)
- FP32 → BF16: `fp32_to_bf16()` (Phase 2)
- FP32 → FP8 E4M3/E5M2: needs implementation (Phase 3)

---

## 6. Serialization

**Problem**: `hnswlib::Index::saveIndex()` currently saves the HNSW graph but not the `data_type`. When the server restarts and loads the index, it defaults to `float32`, causing wrong-width reads for FP16/BF16/FP8 indices.

**Fix**: Prepend a header to the HNSW index file:
```
[4 bytes: magic number 0x544746] [1 byte: version] [N bytes: data_type string (null-terminated)]
[existing HNSW binary content]
```

Or simpler: store `data_type` in the existing `config.yaml` that already accompanies each index. `config.yaml` is read at load time anyway.

```yaml
# config.yaml
dimension: 1024
metric: L2
dtype: float16   # new field
```

The engine reads `config.yaml` → passes `dtype` string to `EmbeddingIndex` constructor → selects correct space. This is the lower-risk change (no binary format change).

---

## 7. Accuracy Impact

| Format | Typical recall@10 drop vs FP32 | Notes |
|--------|---------------------------------|-------|
| FP16   | < 0.1%  | Negligible for most models |
| BF16   | < 0.1%  | Same range as FP32, less mantissa precision |
| FP8 E4M3 | 0.5–2% | Depends heavily on embedding distribution |
| FP8 E5M2 | 1–3%  | Less mantissa precision than E4M3 |
| INT4   | 2–8%   | Highly dependent on quantization quality |

Recommendation: offer FP16 and BF16 as production options; FP8 as opt-in for memory-constrained deployments with known-compatible models.

---

## 8. Open Questions

1. **FP8 variant API**: `dtype="float8_e4m3"` vs `dtype="float8_e5m2"` — or a single `dtype="float8"` with the variant auto-detected from the model? Should this be a GSQL attribute property or a TigerGraph system-level setting?

2. **Cosine normalization for FP16/BF16**: Cosine similarity = InnerProduct on normalized vectors. The current engine normalizes at query time (FP32) before search. Should we normalize before or after quantizing to FP16? (Normalize first → safer, but loses 1 bit of precision on the leading 1.)

3. **Mixed precision index**: Can a vertex have two embedding attributes at different dtypes (e.g., one FP32 and one FP16)? Based on the current architecture (separate Space per attribute), yes — but need to confirm with Songting.

4. **INT4 scale storage**: Where is the quantization scale stored? Options: (a) one scale per vector in `main.bin`, (b) per-block of 32/64 elements (like llama.cpp), (c) global scale. This decision affects accuracy, storage overhead, and distance function complexity.

5. **Index migration**: If a user upgrades TigerGraph and wants to convert an existing FP32 index to FP16, what's the migration path? Full rebuild, or can we transcode `main.bin` in-place?

---

## 9. Appendix: Bit Layout Quick Reference

```
FP32:  [S 1][E 8][M 23] = 32 bits  bias=127
BF16:  [S 1][E 8][M  7] = 16 bits  bias=127  (truncated FP32)
FP16:  [S 1][E 5][M 10] = 16 bits  bias=15
FP8 E4M3: [S 1][E 4][M 3] = 8 bits  bias=7   NaN=0xFF
FP8 E5M2: [S 1][E 5][M 2] = 8 bits  bias=15  Inf/NaN supported
INT4:  [S 1][I 3]          = 4 bits  range [-8, 7]
```
