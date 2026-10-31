/*
 * kernels.cpp - SIMD GEMM microkernels.
 *
 * Every ISA-specific routine is runtime-dispatched.  The translation unit is
 * compiled with conservative baseline flags; GCC/Clang target attributes
 * enable AVX2/AVX512 code only inside the marked functions.
 */
#include "../internal.h"

#if VNPU_X86 && (defined(__GNUC__) || defined(__clang__))
#  define VNPU_TARGET_AVX2   __attribute__((target("avx2,fma")))
#else
#  define VNPU_TARGET_AVX2
#endif

namespace vnpu {
namespace {

inline const float *a_ptr(const GemmF32Params &p, uint32_t m, uint32_t k)
{
    if (p.trans_a) return p.A + static_cast<size_t>(k) * p.lda + m;
    return p.A + static_cast<size_t>(m) * p.lda + k;
}

inline const float *b_ptr(const GemmF32Params &p, uint32_t k, uint32_t n)
{
    if (p.trans_b) return p.B + static_cast<size_t>(n) * p.ldb + k;
    return p.B + static_cast<size_t>(k) * p.ldb + n;
}

inline float load_a(const GemmF32Params &p, uint32_t m, uint32_t k)
{
    if (p.trans_a) return p.A[static_cast<size_t>(k) * p.lda + m];
    return p.A[static_cast<size_t>(m) * p.lda + k];
}

inline float load_b(const GemmF32Params &p, uint32_t k, uint32_t n)
{
    if (p.trans_b) return p.B[static_cast<size_t>(n) * p.ldb + k];
    return p.B[static_cast<size_t>(k) * p.ldb + n];
}

inline void store_c(const GemmF32Params &p, uint32_t m, uint32_t n, float value)
{
    float *dst = p.C + static_cast<size_t>(m) * p.ldc + n;
    float out = p.alpha * value;
    if (p.bias) out += p.bias[n];
    if (p.beta != 0.0f) out += p.beta * *dst;
    *dst = apply_epilogue(out, p.epilogue);
}

void zero_c_tile(const GemmF32Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        float *row = p.C + static_cast<size_t>(m) * p.ldc + tile.n0;
        for (uint32_t n = 0; n < tile.tn; ++n) {
            float out = p.bias ? p.bias[tile.n0 + n] : 0.0f;
            if (p.beta != 0.0f) out += p.beta * row[n];
            row[n] = apply_epilogue(out, p.epilogue);
        }
    }
}

/* Portable scalar path - also the correctness oracle. */
void gemm_f32_scalar_tile(const GemmF32Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < p.K; ++k)
                acc += load_a(p, m, k) * load_b(p, k, n);
            store_c(p, m, n, acc);
        }
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* B is row-major KxN: accumulate 4 rows x 8 columns in registers. */
/* One output row against row-major B, K outermost.
 *
 * This is the only loop order that walks B linearly, which is what the
 * hardware prefetcher and the L1/L2 streamers need.  Column-major ordering
 * (N outer, K inner) re-walks the whole B panel for every column and measured
 * 4-19x slower than a plain compiler loop on decode shapes.  Rows left over
 * from the MR-row block use this path, and M<=2 tiles use it entirely. */
VNPU_TARGET_AVX2
static void gemm_f32_row_stream(const GemmF32Params &p, uint32_t m,
                                uint32_t n0, uint32_t n1)
{
    constexpr uint32_t NB = 64;   /* 8 vectors = 8 of the 16 YMM registers */
    const float *arow = a_ptr(p, m, 0);
    uint32_t nb = n0;
    for (; nb + NB <= n1; nb += NB) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (uint32_t k = 0; k < p.K; ++k) {
            const __m256 va = _mm256_set1_ps(arow[k]);
            const float *brow = p.B + static_cast<size_t>(k) * p.ldb + nb;
            a0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 0), a0);
            a1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 8), a1);
            a2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 16), a2);
            a3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 24), a3);
            a4 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 32), a4);
            a5 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 40), a5);
            a6 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 48), a6);
            a7 = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + 56), a7);
        }
        float tmp[NB];
        _mm256_storeu_ps(tmp + 0, a0);
        _mm256_storeu_ps(tmp + 8, a1);
        _mm256_storeu_ps(tmp + 16, a2);
        _mm256_storeu_ps(tmp + 24, a3);
        _mm256_storeu_ps(tmp + 32, a4);
        _mm256_storeu_ps(tmp + 40, a5);
        _mm256_storeu_ps(tmp + 48, a6);
        _mm256_storeu_ps(tmp + 56, a7);
        for (uint32_t j = 0; j < NB; ++j) store_c(p, m, nb + j, tmp[j]);
    }
    if (nb < n1) {
        const uint32_t w = n1 - nb;
        float acc[NB];
        for (uint32_t j = 0; j < w; ++j) acc[j] = 0.0f;
        for (uint32_t k = 0; k < p.K; ++k) {
            const float a = arow[k];
            const float *brow = p.B + static_cast<size_t>(k) * p.ldb + nb;
            for (uint32_t j = 0; j < w; ++j) acc[j] += a * brow[j];
        }
        for (uint32_t j = 0; j < w; ++j) store_c(p, m, nb + j, acc[j]);
    }
}

/* Panel packing.
 *
 * Row-major B means every K step of a tile reads a fresh cache line of a
 * K x N matrix, and each 3-row block re-reads the same K x 32 strip: for 512^3
 * that is ~22 MB of strip traffic per N tile, almost all of it served from
 * L2/L3.  Packing the tile's columns into [kc][tn] makes the panel 32 KiB at
 * kc=128, nr=64 -- exactly L1 -- and turns the strip reads into a sequential
 * walk of a resident buffer.  This is the BLIS block structure without BLIS's
 * generality: one fixed panel shape for the one microkernel we ship. */
static void pack_b_panel(float *dst, const float *B, uint32_t ldb,
                         uint32_t k0, uint32_t kc, uint32_t n0, uint32_t nr)
{
    const size_t row_bytes = static_cast<size_t>(nr) * sizeof(float);
    for (uint32_t k = 0; k < kc; ++k) {
        const float *src = B + static_cast<size_t>(k0 + k) * ldb + n0;
        std::memcpy(dst + static_cast<size_t>(k) * nr, src, row_bytes);
    }
}

/* 3 rows x 32 columns of C per block: 12 accumulators + 3 broadcasts + 1 live
 * B vector = exactly the 16 architectural YMM registers.  Eight accumulators
 * (the previous 4x8 block) cannot hide a 5-cycle FMA latency across two ports;
 * 12 independent chains can. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_nn_tile(const GemmF32Params &p, const TileJob &tile)
{
    constexpr uint32_t MR = 3;
    constexpr uint32_t NR = 32;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;

    if (p.trans_a || tile.tm < MR) {
        for (uint32_t m = tile.m0; m < mend; ++m)
            gemm_f32_row_stream(p, m, tile.n0, nend);
        return;
    }
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        const float *a0 = a_ptr(p, m + 0, 0);
        const float *a1 = a_ptr(p, m + 1, 0);
        const float *a2 = a_ptr(p, m + 2, 0);
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
            __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10, c13 = c10;
            __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20, c23 = c20;
            for (uint32_t k = 0; k < p.K; ++k) {
                const float *brow = p.B + static_cast<size_t>(k) * p.ldb + n;
                __m256 b0 = _mm256_loadu_ps(brow + 0);
                const __m256 va0 = _mm256_set1_ps(a0[k]);
                const __m256 va1 = _mm256_set1_ps(a1[k]);
                const __m256 va2 = _mm256_set1_ps(a2[k]);
                c00 = _mm256_fmadd_ps(va0, b0, c00);
                c10 = _mm256_fmadd_ps(va1, b0, c10);
                c20 = _mm256_fmadd_ps(va2, b0, c20);
                __m256 b1 = _mm256_loadu_ps(brow + 8);
                c01 = _mm256_fmadd_ps(va0, b1, c01);
                c11 = _mm256_fmadd_ps(va1, b1, c11);
                c21 = _mm256_fmadd_ps(va2, b1, c21);
                __m256 b2 = _mm256_loadu_ps(brow + 16);
                c02 = _mm256_fmadd_ps(va0, b2, c02);
                c12 = _mm256_fmadd_ps(va1, b2, c12);
                c22 = _mm256_fmadd_ps(va2, b2, c22);
                __m256 b3 = _mm256_loadu_ps(brow + 24);
                c03 = _mm256_fmadd_ps(va0, b3, c03);
                c13 = _mm256_fmadd_ps(va1, b3, c13);
                c23 = _mm256_fmadd_ps(va2, b3, c23);
            }
            float tmp[MR][NR];
            _mm256_storeu_ps(tmp[0] + 0, c00);
            _mm256_storeu_ps(tmp[0] + 8, c01);
            _mm256_storeu_ps(tmp[0] + 16, c02);
            _mm256_storeu_ps(tmp[0] + 24, c03);
            _mm256_storeu_ps(tmp[1] + 0, c10);
            _mm256_storeu_ps(tmp[1] + 8, c11);
            _mm256_storeu_ps(tmp[1] + 16, c12);
            _mm256_storeu_ps(tmp[1] + 24, c13);
            _mm256_storeu_ps(tmp[2] + 0, c20);
            _mm256_storeu_ps(tmp[2] + 8, c21);
            _mm256_storeu_ps(tmp[2] + 16, c22);
            _mm256_storeu_ps(tmp[2] + 24, c23);
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            for (uint32_t i = 0; i < MR; ++i)
                gemm_f32_row_stream(p, m + i, n, nend);
        }
    }
    for (; m < mend; ++m)
        gemm_f32_row_stream(p, m, tile.n0, nend);
}

/* Same microkernel, but reading a packed [kc][tn] panel that stays in L1.
 * Requires !trans_a, !trans_b, tn % 32 == 0 and a scratch buffer of at least
 * 128 * tn floats. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_nn_packed_tile(const GemmF32Params &p,
                                         const TileJob &tile,
                                         uint8_t *scratch)
{
    constexpr uint32_t KC = 128;   /* 128 x 64 floats = 32 KiB = L1 */
    constexpr uint32_t MR = 3;
    constexpr uint32_t NV = 32;
    const uint32_t tn = tile.tn;
    const uint32_t nend = tile.n0 + tn;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t full_end = tile.m0 + (tile.tm / MR) * MR;

    /* Rows that do not fill a 3-row block are streamed once over the whole K,
     * so the K-blocked loop below only has to handle full blocks. */
    for (uint32_t m = full_end; m < mend; ++m)
        gemm_f32_row_stream(p, m, tile.n0, nend);
    if (full_end <= tile.m0) return;

    float *panel = reinterpret_cast<float *>(scratch);
    for (uint32_t k0 = 0; k0 < p.K; k0 += KC) {
        const uint32_t kc = std::min<uint32_t>(KC, p.K - k0);
        const bool first = (k0 == 0);
        pack_b_panel(panel, p.B, p.ldb, k0, kc, tile.n0, tn);
        for (uint32_t m = tile.m0; m < full_end; m += MR) {
            const float *a0 = a_ptr(p, m + 0, 0);
            const float *a1 = a_ptr(p, m + 1, 0);
            const float *a2 = a_ptr(p, m + 2, 0);
            for (uint32_t nb = 0; nb < tn; nb += NV) {
                __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00,
                       c03 = c00;
                __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10,
                       c13 = c10;
                __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20,
                       c23 = c20;
                /* One B vector live at a time: 12 accumulators + 3 broadcasts
                 * + 1 B vector is exactly 16 registers.  Hoisting all four
                 * loads first spills and measured ~1.5x slower. */
                for (uint32_t k = 0; k < kc; ++k) {
                    const float *brow = panel + static_cast<size_t>(k) * tn + nb;
                    const __m256 va0 = _mm256_set1_ps(a0[k0 + k]);
                    const __m256 va1 = _mm256_set1_ps(a1[k0 + k]);
                    const __m256 va2 = _mm256_set1_ps(a2[k0 + k]);
                    __m256 bv = _mm256_loadu_ps(brow + 0);
                    c00 = _mm256_fmadd_ps(va0, bv, c00);
                    c10 = _mm256_fmadd_ps(va1, bv, c10);
                    c20 = _mm256_fmadd_ps(va2, bv, c20);
                    bv = _mm256_loadu_ps(brow + 8);
                    c01 = _mm256_fmadd_ps(va0, bv, c01);
                    c11 = _mm256_fmadd_ps(va1, bv, c11);
                    c21 = _mm256_fmadd_ps(va2, bv, c21);
                    bv = _mm256_loadu_ps(brow + 16);
                    c02 = _mm256_fmadd_ps(va0, bv, c02);
                    c12 = _mm256_fmadd_ps(va1, bv, c12);
                    c22 = _mm256_fmadd_ps(va2, bv, c22);
                    bv = _mm256_loadu_ps(brow + 24);
                    c03 = _mm256_fmadd_ps(va0, bv, c03);
                    c13 = _mm256_fmadd_ps(va1, bv, c13);
                    c23 = _mm256_fmadd_ps(va2, bv, c23);
                }
                float tmp[MR][NV];
                _mm256_storeu_ps(tmp[0] + 0, c00);
                _mm256_storeu_ps(tmp[0] + 8, c01);
                _mm256_storeu_ps(tmp[0] + 16, c02);
                _mm256_storeu_ps(tmp[0] + 24, c03);
                _mm256_storeu_ps(tmp[1] + 0, c10);
                _mm256_storeu_ps(tmp[1] + 8, c11);
                _mm256_storeu_ps(tmp[1] + 16, c12);
                _mm256_storeu_ps(tmp[1] + 24, c13);
                _mm256_storeu_ps(tmp[2] + 0, c20);
                _mm256_storeu_ps(tmp[2] + 8, c21);
                _mm256_storeu_ps(tmp[2] + 16, c22);
                _mm256_storeu_ps(tmp[2] + 24, c23);
                for (uint32_t i = 0; i < MR; ++i) {
                    for (uint32_t j = 0; j < NV; ++j) {
                        const uint32_t n = tile.n0 + nb + j;
                        if (first) {
                            store_c(p, m + i, n, tmp[i][j]);
                        } else {
                            p.C[static_cast<size_t>(m + i) * p.ldc + n] +=
                                p.alpha * tmp[i][j];
                        }
                    }
                }
            }
        }
    }
}

/* B is NxK: vectorized dot product over K for each output element. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_nt_tile(const GemmF32Params &p, const TileJob &tile)
{
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m = tile.m0; m < mend; ++m) {
        const float *arow = a_ptr(p, m, 0);
        for (uint32_t n = tile.n0; n < nend; ++n) {
            const float *brow = p.B + static_cast<size_t>(n) * p.ldb;
            __m256 acc = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 8 <= p.K; k += 8) {
                __m256 av = _mm256_loadu_ps(arow + k);
                __m256 bv = _mm256_loadu_ps(brow + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            float dot = hsum256_ps(acc);
            for (; k < p.K; ++k) dot += arow[k] * brow[k];
            store_c(p, m, n, dot);
        }
    }
}

#endif /* VNPU_X86 */

#if VNPU_NEON
static void gemm_f32_neon_nn_tile(const GemmF32Params &p,
                                  const TileJob &tile)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NR = 4;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            const float *a0 = a_ptr(p, m + 0, 0);
            const float *a1 = a_ptr(p, m + 1, 0);
            const float *a2 = a_ptr(p, m + 2, 0);
            const float *a3 = a_ptr(p, m + 3, 0);
            for (uint32_t k = 0; k < p.K; ++k) {
                float32x4_t bv = vld1q_f32(p.B +
                    static_cast<size_t>(k) * p.ldb + n);
                acc0 = vfmaq_n_f32(acc0, bv, a0[k]);
                acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                acc2 = vfmaq_n_f32(acc2, bv, a2[k]);
                acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
            }
            float tmp[MR][NR];
            vst1q_f32(tmp[0], acc0);
            vst1q_f32(tmp[1], acc1);
            vst1q_f32(tmp[2], acc2);
            vst1q_f32(tmp[3], acc3);
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_f32_scalar_tile(p, tail);
        }
    }
    if (m < mend) {
        TileJob tail{m, tile.n0, mend - m, tile.tn, tile.tile_m,
                     tile.tile_n, tile.tile_k, 0, {0, 0, 0}};
        gemm_f32_scalar_tile(p, tail);
    }
}
#endif

/* Cell SPE-style path: explicit DMA into an aligned local store followed by
 * register-blocked accumulation.  It deliberately uses a smaller K window so
 * the same code can run on a remote worker with a tiny working set. */
static void gemm_f32_cell_tile(const GemmF32Params &p, const TileJob &tile,
                               uint8_t *scratch, uint64_t scratch_bytes)
{
    uint32_t MR = 4;
    uint32_t NR = (p.isa == ISA_AVX512) ? 16 : 8;
    uint32_t BK = tile.tile_k ? tile.tile_k : 64;
    BK = std::min<uint32_t>(BK, 256);
    const uint64_t a_bytes = static_cast<uint64_t>(MR) * BK * sizeof(float);
    const uint64_t b_bytes = static_cast<uint64_t>(BK) * NR * sizeof(float);
    if (!scratch || a_bytes + b_bytes + 128 > scratch_bytes) {
        gemm_f32_scalar_tile(p, tile);
        return;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(scratch);
    base = align_up(base, VNPU_CACHELINE_BYTES);
    float *A_s = reinterpret_cast<float *>(base);
    float *B_s = reinterpret_cast<float *>(base + a_bytes);
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;

    for (uint32_t m0 = tile.m0; m0 < mend; m0 += MR) {
        uint32_t rm = std::min<uint32_t>(MR, mend - m0);
        for (uint32_t n0 = tile.n0; n0 < nend; n0 += NR) {
            uint32_t rn = std::min<uint32_t>(NR, nend - n0);
            float acc[4][16] = {};
            for (uint32_t k0 = 0; k0 < p.K; k0 += BK) {
                uint32_t rk = std::min<uint32_t>(BK, p.K - k0);
                /* DMA A panel. */
                for (uint32_t i = 0; i < rm; ++i)
                    for (uint32_t k = 0; k < rk; ++k)
                        A_s[i * BK + k] = load_a(p, m0 + i, k0 + k);
                /* DMA B panel. */
                for (uint32_t k = 0; k < rk; ++k)
                    for (uint32_t j = 0; j < rn; ++j)
                        B_s[k * NR + j] = load_b(p, k0 + k, n0 + j);
                /* SPU local compute. */
                for (uint32_t i = 0; i < rm; ++i) {
                    const float *arow = A_s + i * BK;
                    for (uint32_t k = 0; k < rk; ++k) {
                        float av = arow[k];
                        const float *brow = B_s + k * NR;
                        for (uint32_t j = 0; j < rn; ++j)
                            acc[i][j] += av * brow[j];
                    }
                }
            }
            for (uint32_t i = 0; i < rm; ++i)
                for (uint32_t j = 0; j < rn; ++j)
                    store_c(p, m0 + i, n0 + j, acc[i][j]);
        }
    }
}

/* Kernel scratch, owned by the thread that uses it.
 *
 * This used to be indexed by the worker id a parallel_for hands to a task.
 * That id is a *task* slot, not a thread: two streams on one device run two
 * parallel_for calls at once, so two tasks carrying the same slot index can
 * land on different threads and overwrite each other's panel half way through
 * a tile.  Measured with two concurrent streams on a 512^3 F32 GEMM before the
 * fix: 10 of 12 runs returned wrong results (max error 20-60).  Per-thread
 * storage makes the hazard structurally impossible instead of ordering around
 * it. */
static uint8_t *thread_scratch(uint64_t &bytes)
{
    static thread_local std::vector<uint8_t> buf;
    if (buf.empty()) buf.resize(VNPU_LOCAL_STORE_BYTES + VNPU_CACHELINE_BYTES);
    bytes = buf.size();
    return buf.data();
}

static void gemm_f32_dispatch_tile(const GemmF32Params &p,
                                   const TileJob &tile,
                                   uint32_t worker_id)
{
    (void)worker_id;  /* scratch is per-thread now */
    if (tile.cull) {
        zero_c_tile(p, tile);
        return;
    }
    if (p.unit == VNPU_UNIT_CELL) {
        uint64_t bytes = 0;
        uint8_t *scratch = thread_scratch(bytes);
        gemm_f32_cell_tile(p, tile, scratch, bytes);
        return;
    }
    if (p.isa == ISA_AVX2 && !p.trans_a) {
        if (!p.trans_b) {
            if (p.pack_b && (tile.tn % 32) == 0 && tile.tm >= 3) {
                uint64_t bytes = 0;
                uint8_t *scratch = thread_scratch(bytes);
                const uint64_t need =
                    static_cast<uint64_t>(128) * tile.tn * sizeof(float);
                if (scratch && bytes >= need) {
                    gemm_f32_avx2_nn_packed_tile(p, tile, scratch);
                    return;
                }
            }
            gemm_f32_avx2_nn_tile(p, tile);
        } else {
            gemm_f32_avx2_nt_tile(p, tile);
        }
        return;
    }
#if VNPU_NEON
    if (p.isa == ISA_NEON && !p.trans_a && !p.trans_b) {
        gemm_f32_neon_nn_tile(p, tile);
        return;
    }
#endif
    gemm_f32_scalar_tile(p, tile);
}

/* ------------------------------------------------------------------------- */
/* INT8 kernels                                                              */
/* ------------------------------------------------------------------------- */

inline void store_i8_c(const GemmI8Params &p, uint32_t m, uint32_t n,
                       int32_t dot)
{
    float scale_a = p.scale_a ? p.scale_a[m] : 1.0f;
    float scale_b = p.B.scales ? p.B.scales[n] : 1.0f;
    float out = static_cast<float>(dot) * scale_a * scale_b;
    if (p.bias) out += p.bias[n];
    p.C[static_cast<size_t>(m) * p.ldc + n] =
        apply_epilogue(out, p.epilogue);
}

static void gemm_i8_scalar_tile(const GemmI8Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        const int8_t *arow = p.A + static_cast<size_t>(m) * p.lda;
        for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < p.K; ++k)
                acc += static_cast<int32_t>(arow[k]) *
                       static_cast<int32_t>(p.B.original[
                           static_cast<size_t>(k) * p.B.N + n]);
            store_i8_c(p, m, n, acc);
        }
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static inline int32_t hsum256_epi32(__m256i v)
{
    int32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), v);
    int32_t sum = 0;
    for (int32_t x : tmp) sum += x;
    return sum;
}

/* Without VNNI there is no int8 dot-product instruction, so the only exact
 * and vectorisable axis on row-major B is the column axis: broadcast one A
 * value and multiply eight contiguous B bytes.  The previous version built B's
 * operand byte-by-byte out of a strided column, which is why it measured 0.08x
 * a plain compiler loop.  Two rows share each B load; eight int32 accumulators
 * plus the operand vectors still fit the 16 YMM registers. */
/* Rank-R staged row stream (R = 2 or 3; R=4 is implemented and measured
 * worse, see the dispatch).
 *
 * The 32-column register block below cannot hold several accumulator rows at a
 * wide width without spilling, so it walks B as 64-byte runs at stride N -- the
 * same prefetch-hostile read that made the single row 3.1x slower than its
 * streaming form.  The packed panel fixes the locality but couples rows in
 * pairs, so an odd row re-reads the whole panel: at M=3 that is 128 bytes of
 * panel read per k step for three outputs against the same 128 for four at M=4,
 * which is the entire 0.65 vs 0.51 ms/row gap.  Staging the R rows in L1 and
 * sweeping B once makes every B byte worth R FMAs.  Returns false when the
 * per-thread scratch cannot hold the rows, so the caller keeps the old path. */
template <uint32_t R>
VNPU_TARGET_AVX2
static bool gemm_i8_avx2_rank_r(const GemmI8Params &p, const TileJob &tile)
{
    static_assert(R >= 2 && R <= 4, "rank stream covers 2 to 4 rows");
    const uint32_t width = tile.tn;
    const uint32_t stride = (width + 7u) & ~7u;
    uint64_t scratch_bytes = 0;
    uint8_t *scratch = thread_scratch(scratch_bytes);
    const uint64_t need = static_cast<uint64_t>(stride) * R * sizeof(int32_t) +
                          VNPU_CACHELINE_BYTES;
    if (!scratch || scratch_bytes < need) return false;
    int32_t *acc[R];
    const int8_t *ar[R];
    int32_t *base = reinterpret_cast<int32_t *>(align_up(
        reinterpret_cast<uintptr_t>(scratch), VNPU_CACHELINE_BYTES));
    for (uint32_t i = 0; i < R; ++i) {
        acc[i] = base + static_cast<size_t>(i) * stride;
        ar[i] = p.A + static_cast<size_t>(tile.m0 + i) * p.lda;
    }
    std::memset(base, 0, static_cast<size_t>(stride) * R * sizeof(int32_t));
    const uint32_t full = width & ~7u;
    for (uint32_t k = 0; k < p.K; ++k) {
        const int8_t *brow = p.B.original +
            static_cast<size_t>(k) * p.B.N + tile.n0;
        __m256i va[R];
        for (uint32_t i = 0; i < R; ++i)
            va[i] = _mm256_set1_epi32(ar[i][k]);
        for (uint32_t j = 0; j < full; j += 8) {
            const __m256i b32 = _mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(brow + j)));
            for (uint32_t i = 0; i < R; ++i) {
                const __m256i c = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(acc[i] + j));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(acc[i] + j),
                                    _mm256_add_epi32(
                                        c, _mm256_mullo_epi32(va[i], b32)));
            }
        }
        if (full < width) {
            for (uint32_t j = full; j < width; ++j) {
                const int32_t b = static_cast<int32_t>(brow[j]);
                for (uint32_t i = 0; i < R; ++i)
                    acc[i][j] += ar[i][k] * b;
            }
        }
    }
    for (uint32_t j = 0; j < width; ++j)
        for (uint32_t i = 0; i < R; ++i)
            store_i8_c(p, tile.m0 + i, tile.n0 + j, acc[i][j]);
    return true;
}

VNPU_TARGET_AVX2
static void gemm_i8_avx2_tile(const GemmI8Params &p, const TileJob &tile)
{
    if (!p.B.original) {
        gemm_i8_scalar_tile(p, tile);
        return;
    }
    constexpr uint32_t MR = 2;
    constexpr uint32_t NR = 32;   /* 4 vectors of 8 int32 lanes */
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        const int8_t *ar0 = p.A + static_cast<size_t>(m) * p.lda;
        const int8_t *ar1 = ar0 + p.lda;
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            __m256i c0[4], c1[4];
            for (uint32_t v = 0; v < 4; ++v) {
                c0[v] = _mm256_setzero_si256();
                c1[v] = _mm256_setzero_si256();
            }
            for (uint32_t k = 0; k < p.K; ++k) {
                const int8_t *brow = p.B.original +
                    static_cast<size_t>(k) * p.B.N + n;
                const __m256i va0 = _mm256_set1_epi32(ar0[k]);
                const __m256i va1 = _mm256_set1_epi32(ar1[k]);
                for (uint32_t v = 0; v < 4; ++v) {
                    uint64_t raw = 0;
                    std::memcpy(&raw, brow + 8 * v, sizeof(raw));
                    __m256i b32 = _mm256_cvtepi8_epi32(
                        _mm_cvtsi64_si128(static_cast<long long>(raw)));
                    c0[v] = _mm256_add_epi32(c0[v],
                                             _mm256_mullo_epi32(va0, b32));
                    c1[v] = _mm256_add_epi32(c1[v],
                                             _mm256_mullo_epi32(va1, b32));
                }
            }
            int32_t tmp0[NR], tmp1[NR];
            for (uint32_t v = 0; v < 4; ++v) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp0 + 8 * v),
                                   c0[v]);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp1 + 8 * v),
                                   c1[v]);
            }
            for (uint32_t j = 0; j < NR; ++j) {
                store_i8_c(p, m + 0, n + j, tmp0[j]);
                store_i8_c(p, m + 1, n + j, tmp1[j]);
            }
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_i8_scalar_tile(p, tail);
        }
    }
    /* Leftover rows, including the whole tile when M<=8 gives tm==1: this is
     * the decode shape, and sending it to the scalar kernel measured 0.02x a
     * plain compiler loop.
     *
     * For a single row there is no reuse across rows, so keeping C in registers
     * only forces the column block down to 32 bytes per B row: the walk then
     * touches every row of B at a stride of N bytes for 32 useful bytes, which
     * measured 3.69 ms (4.5 GB/s of a 19.8 GB/s ceiling) at 4096x4096 int8.
     * A rank-1 update streams B one row at a time and keeps the C row (4 bytes
     * per column) resident instead, which measured 1.19 ms / 14.1 GB/s: 3.1x
     * faster with the same arithmetic in the same k order, so every output is
     * unchanged bit for bit. */
    for (; m < mend; ++m) {
        const int8_t *arow = p.A + static_cast<size_t>(m) * p.lda;
        const uint32_t width = nend - tile.n0;
        uint64_t scratch_bytes = 0;
        uint8_t *scratch = thread_scratch(scratch_bytes);
        const uint64_t need = static_cast<uint64_t>(width) * sizeof(int32_t) +
                              VNPU_CACHELINE_BYTES;
        if (!scratch || scratch_bytes < need) {
            /* No room to stage the C row: fall back to the blocked walk. */
            uint32_t n = tile.n0;
            for (; n + NR <= nend; n += NR) {
                __m256i c0[4];
                for (uint32_t v = 0; v < 4; ++v) c0[v] = _mm256_setzero_si256();
                for (uint32_t k = 0; k < p.K; ++k) {
                    const int8_t *brow = p.B.original +
                        static_cast<size_t>(k) * p.B.N + n;
                    const __m256i va = _mm256_set1_epi32(arow[k]);
                    for (uint32_t v = 0; v < 4; ++v) {
                        uint64_t raw = 0;
                        std::memcpy(&raw, brow + 8 * v, sizeof(raw));
                        __m256i b32 = _mm256_cvtepi8_epi32(
                            _mm_cvtsi64_si128(static_cast<long long>(raw)));
                        c0[v] = _mm256_add_epi32(c0[v],
                                                 _mm256_mullo_epi32(va, b32));
                    }
                }
                int32_t tmp[NR];
                for (uint32_t v = 0; v < 4; ++v)
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(tmp + 8 * v), c0[v]);
                for (uint32_t j = 0; j < NR; ++j)
                    store_i8_c(p, m, n + j, tmp[j]);
            }
            if (n < nend) {
                TileJob tail{m, n, 1, nend - n, tile.tile_m, tile.tile_n,
                             tile.tile_k, 0, {0, 0, 0}};
                gemm_i8_scalar_tile(p, tail);
            }
            continue;
        }
        int32_t *acc = reinterpret_cast<int32_t *>(align_up(
            reinterpret_cast<uintptr_t>(scratch), VNPU_CACHELINE_BYTES));
        std::memset(acc, 0, static_cast<size_t>(width) * sizeof(int32_t));
        uint32_t full = width & ~7u;
        for (uint32_t k = 0; k < p.K; ++k) {
            const int8_t *brow = p.B.original +
                static_cast<size_t>(k) * p.B.N + tile.n0;
            const __m256i va = _mm256_set1_epi32(arow[k]);
            for (uint32_t j = 0; j < full; j += 8) {
                const __m256i b32 = _mm256_cvtepi8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(brow + j)));
                const __m256i c = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(acc + j));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(acc + j),
                                    _mm256_add_epi32(c,
                                        _mm256_mullo_epi32(va, b32)));
            }
            if (full < width) {
                const int32_t a_k = arow[k];
                for (uint32_t j = full; j < width; ++j)
                    acc[j] += a_k * static_cast<int32_t>(brow[j]);
            }
        }
        for (uint32_t j = 0; j < width; ++j)
            store_i8_c(p, m, tile.n0 + j, acc[j]);
    }
}

/* Panel-packed variant of the same microkernel.
 *
 * The row-major path re-reads a K x 32 strip of B for every 2-row block, so at
 * 1024^3 that is ~500 MB of strip traffic.  An int8 panel is four times smaller
 * than the f32 one, so the whole K fits in the per-worker local store at the
 * usual tile width (64 columns x 1024 = 64 KiB), and the int32 accumulators
 * therefore stay in registers across all of K.  Nothing about the arithmetic
 * changes -- same multipliers, same order -- so packed and unpacked results are
 * bit-identical, which is what the conformance check asserts. */
VNPU_TARGET_AVX2
static void gemm_i8_avx2_packed_tile(const GemmI8Params &p,
                                     const TileJob &tile, int8_t *panel)
{
    constexpr uint32_t MR = 2;
    constexpr uint32_t NR = 32;
    const uint32_t tn = tile.tn;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tn;

    for (uint32_t k = 0; k < p.K; ++k)
        std::memcpy(panel + static_cast<size_t>(k) * tn,
                    p.B.original + static_cast<size_t>(k) * p.B.N + tile.n0, tn);

    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        const int8_t *ar0 = p.A + static_cast<size_t>(m) * p.lda;
        const int8_t *ar1 = ar0 + p.lda;
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            const uint32_t off = n - tile.n0;
            __m256i c0[4], c1[4];
            for (uint32_t v = 0; v < 4; ++v) {
                c0[v] = _mm256_setzero_si256();
                c1[v] = _mm256_setzero_si256();
            }
            for (uint32_t k = 0; k < p.K; ++k) {
                const int8_t *brow = panel + static_cast<size_t>(k) * tn + off;
                const __m256i va0 = _mm256_set1_epi32(ar0[k]);
                const __m256i va1 = _mm256_set1_epi32(ar1[k]);
                for (uint32_t v = 0; v < 4; ++v) {
                    uint64_t raw = 0;
                    std::memcpy(&raw, brow + 8 * v, sizeof(raw));
                    __m256i b32 = _mm256_cvtepi8_epi32(
                        _mm_cvtsi64_si128(static_cast<long long>(raw)));
                    c0[v] = _mm256_add_epi32(c0[v],
                                             _mm256_mullo_epi32(va0, b32));
                    c1[v] = _mm256_add_epi32(c1[v],
                                             _mm256_mullo_epi32(va1, b32));
                }
            }
            int32_t tmp0[NR], tmp1[NR];
            for (uint32_t v = 0; v < 4; ++v) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp0 + 8 * v),
                                    c0[v]);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp1 + 8 * v),
                                    c1[v]);
            }
            for (uint32_t j = 0; j < NR; ++j) {
                store_i8_c(p, m + 0, n + j, tmp0[j]);
                store_i8_c(p, m + 1, n + j, tmp1[j]);
            }
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_i8_scalar_tile(p, tail);
        }
    }
    for (; m < mend; ++m) {
        const int8_t *arow = p.A + static_cast<size_t>(m) * p.lda;
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            const uint32_t off = n - tile.n0;
            __m256i c0[4];
            for (uint32_t v = 0; v < 4; ++v) c0[v] = _mm256_setzero_si256();
            for (uint32_t k = 0; k < p.K; ++k) {
                const int8_t *brow = panel + static_cast<size_t>(k) * tn + off;
                const __m256i va = _mm256_set1_epi32(arow[k]);
                for (uint32_t v = 0; v < 4; ++v) {
                    uint64_t raw = 0;
                    std::memcpy(&raw, brow + 8 * v, sizeof(raw));
                    __m256i b32 = _mm256_cvtepi8_epi32(
                        _mm_cvtsi64_si128(static_cast<long long>(raw)));
                    c0[v] = _mm256_add_epi32(c0[v],
                                             _mm256_mullo_epi32(va, b32));
                }
            }
            int32_t tmp[NR];
            for (uint32_t v = 0; v < 4; ++v)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp + 8 * v),
                                    c0[v]);
            for (uint32_t j = 0; j < NR; ++j) store_i8_c(p, m, n + j, tmp[j]);
        }
        if (n < nend) {
            TileJob tail{m, n, 1, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_i8_scalar_tile(p, tail);
        }
    }
}

#endif /* VNPU_X86 */

static void gemm_i8_cell_tile(const GemmI8Params &p, const TileJob &tile,
                              uint8_t *scratch, uint64_t scratch_bytes)
{
    uint32_t MR = 4;
    uint32_t NR = 16;
    uint32_t BK = std::min<uint32_t>(tile.tile_k ? tile.tile_k : 64, 256);
    const uint64_t a_bytes = static_cast<uint64_t>(MR) * BK;
    const uint64_t b_bytes = static_cast<uint64_t>(BK) * NR;
    if (!scratch || a_bytes + b_bytes + 128 > scratch_bytes) {
        gemm_i8_scalar_tile(p, tile);
        return;
    }
    uintptr_t base = align_up(reinterpret_cast<uintptr_t>(scratch),
                              VNPU_CACHELINE_BYTES);
    int8_t *A_s = reinterpret_cast<int8_t *>(base);
    int8_t *B_s = reinterpret_cast<int8_t *>(base + a_bytes);
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m0 = tile.m0; m0 < mend; m0 += MR) {
        uint32_t rm = std::min<uint32_t>(MR, mend - m0);
        for (uint32_t n0 = tile.n0; n0 < nend; n0 += NR) {
            uint32_t rn = std::min<uint32_t>(NR, nend - n0);
            int32_t acc[4][16] = {};
            for (uint32_t k0 = 0; k0 < p.K; k0 += BK) {
                uint32_t rk = std::min<uint32_t>(BK, p.K - k0);
                for (uint32_t i = 0; i < rm; ++i)
                    std::memcpy(A_s + i * BK,
                                p.A + static_cast<size_t>(m0 + i) *
                                          p.lda + k0,
                                rk);
                for (uint32_t k = 0; k < rk; ++k)
                    for (uint32_t j = 0; j < rn; ++j)
                        B_s[k * NR + j] = p.B.original[
                            static_cast<size_t>(k0 + k) * p.B.N + n0 + j];
                for (uint32_t i = 0; i < rm; ++i)
                    for (uint32_t k = 0; k < rk; ++k) {
                        int32_t av = A_s[i * BK + k];
                        for (uint32_t j = 0; j < rn; ++j)
                            acc[i][j] += av *
                                static_cast<int32_t>(B_s[k * NR + j]);
                    }
            }
            for (uint32_t i = 0; i < rm; ++i)
                for (uint32_t j = 0; j < rn; ++j)
                    store_i8_c(p, m0 + i, n0 + j, acc[i][j]);
        }
    }
}

static void gemm_i8_dispatch_tile(const GemmI8Params &p,
                                  const TileJob &tile,
                                  uint32_t worker_id)
{
    (void)worker_id;  /* scratch is per-thread now */
    if (tile.cull) {
        for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m)
            for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n)
                store_i8_c(p, m, n, 0);
        return;
    }
    if (p.unit == VNPU_UNIT_CELL) {
        uint64_t bytes = 0;
        uint8_t *scratch = thread_scratch(bytes);
        gemm_i8_cell_tile(p, tile, scratch, bytes);
        return;
    }
#if VNPU_X86
    if (p.isa == ISA_AVX2) {
        /* Rank-R stream ahead of the packed panel: the panel couples rows in
         * pairs, so an odd row re-reads the whole panel.  It only pays over a
         * long run -- at 512 columns or fewer the staged rows cost more than
         * the walk they replace, which is why 2x4096x1024 measured 0.40 ms
         * streamed against 0.28 ms blocked. */
        if (p.B.original && tile.tn >= 1024u) {
            switch (tile.tm) {
            case 2: if (gemm_i8_avx2_rank_r<2>(p, tile)) return; break;
            case 3: if (gemm_i8_avx2_rank_r<3>(p, tile)) return; break;
            /* M=4 staged four rows and measured 1.98 -> 2.32 ms at
             * 4x4096x4096: four accumulator rows thrash L1 badly enough that
             * the panel's register-held accumulators win, so it keeps the
             * panel and the narrow tile. */
            default: break;
            }
        }
        /* A single-row tile reads each panel entry exactly once, so packing
         * it is a pure copy: that is the decode shape, which stays row-major. */
        if (p.pack_b && tile.tm >= 3 && p.B.original) {
            uint64_t bytes = 0;
            uint8_t *scratch = thread_scratch(bytes);
            /* Whole-K panel plus cache-line slack for alignment. */
            const uint64_t need = static_cast<uint64_t>(p.K) * tile.tn +
                                  VNPU_CACHELINE_BYTES;
            if (scratch && bytes >= need) {
                int8_t *panel = reinterpret_cast<int8_t *>(align_up(
                    reinterpret_cast<uintptr_t>(scratch),
                    VNPU_CACHELINE_BYTES));
                gemm_i8_avx2_packed_tile(p, tile, panel);
                return;
            }
        }
        gemm_i8_avx2_tile(p, tile);
        return;
    }
#endif
    gemm_i8_scalar_tile(p, tile);
}

} /* anonymous namespace */

void kernel_gemm_f32_range(const GemmF32Params &p, size_t begin, size_t end,
                           uint32_t worker_id)
{
    if (!p.tiles) return;
    for (size_t i = begin; i < end; ++i)
        gemm_f32_dispatch_tile(p, (*p.tiles)[i], worker_id);
}

void kernel_gemm_i8_range(const GemmI8Params &p, size_t begin, size_t end,
                          uint32_t worker_id)
{
    if (!p.tiles) return;
    for (size_t i = begin; i < end; ++i)
        gemm_i8_dispatch_tile(p, (*p.tiles)[i], worker_id);
}

} /* namespace vnpu */
