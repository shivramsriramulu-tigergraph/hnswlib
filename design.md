# Design: Low-Precision Vector Storage (CORE-5808)

**Author:** Shivram Sriramulu  
**Reviewers:** Jian Gong, Songting Chen, Adil (GSQL)  
**Status:** Draft  
**Jira:** [CORE-5808](https://graphsql.atlassian.net/browse/CORE-5808)

---

## 1. Goal

Support FP16, BF16, FP8 (E4M3/E5M2), and INT4 as vector storage types in addition to the current FP32. Lower precision = smaller index = better cache utilization = higher QPS.

| Format | Bytes/dim | Memory vs FP32 | Hardware |
|--------|-----------|----------------|----------|
| FP32   | 4         | 1×             | Universal |
| FP16   | 2         | 2×             | x86 F16C (Ivy Bridge+), NVIDIA GPU, Apple M |
| BF16   | 2         | 2×             | NVIDIA A100/H100, Intel AMX, Google TPU |
| FP8 E4M3 | 1      | 4×             | NVIDIA H100 (forward pass) |
| FP8 E5M2 | 1      | 4×             | NVIDIA H100 (gradients) |
| INT4   | 0.5       | 8×             | NVIDIA H100, Intel AMX |

---

## 2. Format Bit Layouts

```
FP32:     [S:1][E:8][M:23]  bias=127
BF16:     [S:1][E:8][M:7]   bias=127  (top 16 bits of FP32)
FP16:     [S:1][E:5][M:10]  bias=15   range: ±65504
FP8 E4M3: [S:1][E:4][M:3]   bias=7    max=448, NaN=0xFF
FP8 E5M2: [S:1][E:5][M:2]   bias=15   max=57344, supports Inf/NaN
INT4:     [S:1][I:3]         range: [-8, 7], requires scale factor
```

---

## 3. Architecture

```
GSQL query (list<float>, always FP32)
        │
        ▼
embeddingaction.cpp  —  converts FP32 query → stored dtype
        │
        ▼
EmbeddingIndex → hnswlib::Index (space selected by data_type string)
        ├─ "float32"  → L2Space / InnerProductSpace
        ├─ "float16"  → L2SpaceFp16 / InnerProductSpaceFp16   ✅
        ├─ "bfloat16" → L2SpaceBf16 / InnerProductSpaceBf16   ❌
        ├─ "float8_e4m3" → L2SpaceFp8E4M3 / ...               ❌
        └─ "float8_e5m2" → L2SpaceFp8E5M2 / ...               ❌
        │
        ▼
Distance fn: const void* → cast to typed ptr → compute in FP32
```

The query side always stays FP32 — users pass `list<float>` to `VectorSearch`. Conversion to the stored dtype happens inside the engine before calling `searchKnn`.

---

## 4. Implementation Phases

### Phase 1 — FP16 (in progress)

PRs: [third_party#273](https://github.com/tigergraph/third_party/pull/273), [engine#4116](https://github.com/tigergraph/engine/pull/4116)

- ✅ `L2SpaceFp16`, `InnerProductSpaceFp16` (scalar + F16C SIMD)
- ✅ FP32→FP16 query conversion in `embeddingaction.cpp`
- ❌ Serialization: save `dtype` in `config.yaml` so index survives restart
- ❌ Insert path: FP32→FP16 conversion in `cdc_delta_transformer.cpp`
- ❌ F16C SIMD test on Haswell+ x86 (Sandy Bridge on mtv13 uses scalar fallback)

### Phase 2 — BF16

BF16 conversion is simpler than FP16 — just truncate/round the low 16 bits of a FP32.

```cpp
static inline float bf16_to_fp32(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float r; memcpy(&r, &f, 4); return r;
}
static inline uint16_t fp32_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    bits += 0x7fff + ((bits >> 16) & 1);  // round to nearest even
    return (uint16_t)(bits >> 16);
}
```

Changes needed:
- `hnswlib.h`: `typedef uint16_t bfloat16_t`
- `space_l2.h`: `L2SqrBf16`, `L2SpaceBf16`
- `space_ip.h`: `InnerProductBf16`, `InnerProductSpaceBf16`
- `hnswlib_interface.h`: wire `"bfloat16"` string
- Engine: `EmbeddingDataType_BF16` enum value, BF16 case in `embeddingaction.cpp`
- GSQL: `dtype="BFLOAT16"` attribute option (coordinate with Adil)

### Phase 3 — FP8 (replace placeholder)

Current `L2SqrFp8` / `InnerProductFp8` just cast `uint8_t` to `float` — incorrect. Need proper bit conversion for E4M3 and E5M2:

```cpp
// E4M3: bias=7, NaN=0xFF (no Inf)
static inline float fp8e4m3_to_fp32(uint8_t h) {
    uint32_t sign = (h >> 7) << 31;
    uint32_t exp  = (h >> 3) & 0xf;
    uint32_t mant = h & 0x7;
    if (exp == 0xf && mant == 0x7) { /* NaN */ uint32_t f = 0x7fc00000; float r; memcpy(&r,&f,4); return r; }
    if (exp == 0) { /* subnormal */ ... }
    uint32_t f = sign | ((exp + 120u) << 23) | (mant << 20);
    float r; memcpy(&r, &f, 4); return r;
}

// E5M2: bias=15, supports Inf/NaN
static inline float fp8e5m2_to_fp32(uint8_t h) {
    uint32_t sign = (h >> 7) << 31;
    uint32_t exp  = (h >> 2) & 0x1f;
    uint32_t mant = h & 0x3;
    if (exp == 0x1f) { /* Inf/NaN */ ... }
    if (exp == 0)    { /* subnormal */ ... }
    uint32_t f = sign | ((exp + 112u) << 23) | (mant << 21);
    float r; memcpy(&r, &f, 4); return r;
}
```

Two separate Space classes (`L2SpaceFp8E4M3`, `L2SpaceFp8E5M2`) selected by `dtype` string — no runtime branch in the hot path.

### Phase 4 — INT4 (future)

INT4 requires a per-vector or per-block scale factor for dequantization, which changes the storage format and distance function signature. Needs a separate design review before implementation.

---

## 5. GSQL Schema

```sql
-- current (FP32 only)
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb(dimension=1024, metric="L2")

-- proposed
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb(dimension=1024, metric="L2",    dtype="FLOAT16")
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb(dimension=1024, metric="COSINE", dtype="BFLOAT16")
ALTER VERTEX v1 ADD VECTOR ATTRIBUTE emb(dimension=512,  metric="IP",     dtype="FP8_E4M3")
```

`dtype` defaults to `"FLOAT32"` if omitted (backward compatible).

---

## 6. Serialization

`dtype` needs to survive a server restart. Simplest approach: add `dtype` field to the existing `config.yaml` that already lives alongside each index:

```yaml
dimension: 1024
metric: L2
dtype: float16
```

Engine reads `config.yaml` at load time and passes `dtype` to `EmbeddingIndex` constructor to select the correct space. No binary format change required.

---

## 7. Accuracy

| Format | Typical recall@10 drop vs FP32 |
|--------|---------------------------------|
| FP16   | < 0.1% |
| BF16   | < 0.1% |
| FP8 E4M3 | 0.5–2% |
| FP8 E5M2 | 1–3% |
| INT4   | 2–8% |
