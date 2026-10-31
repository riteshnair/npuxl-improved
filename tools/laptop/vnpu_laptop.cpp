/*
 * Virtual NPU INT8 GEMM, adapted and tuned for THIS laptop, with Sherlock-it
 * style tracing on every component.
 *
 * Target: Intel Core Ultra 5 135H (Meteor Lake) - AVX2 + FMA + AVX-VNNI (VEX-256).
 * No AVX-512. MSVC, so no __attribute__((target)): the ISA path is chosen by
 * CPUID at runtime instead of by per-function target attributes.
 *
 * COMPETITION = RAW CPU. The baseline is a plain, compiler-vectorisable C++
 * triple loop, plus a scalar oracle for correctness.
 *
 * Tracing: VNPU_TRACE=1 for the text report, =2 for chrome trace too.
 * Traps are placed at phase boundaries only (pack stages, avec build, k loop,
 * epilogue, block shaders) - never inside a K loop.
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include "sherlock.h"

#ifdef _MSC_VER
#  define NOINLINE __declspec(noinline)
#else
#  define NOINLINE __attribute__((noinline))
#endif

static inline double now_s(void) { return (double)sherlock_impl::now_ns() / 1e9; }

/* ------------------------------------------------------------------ CPUID -- */
static bool cpu_avx2(void)    { int r[4]; __cpuidex(r, 7, 0); return (r[1] & (1 << 5)) != 0; }
static bool cpu_avxvnni(void) { int r[4]; __cpuidex(r, 7, 1); return (r[0] & (1 << 4)) != 0; }

static double clock_ghz(void) {
    __m256 x = _mm256_set1_ps(1.0000001f);
    const __m256 a = _mm256_set1_ps(1.0000001f), b = _mm256_set1_ps(1e-7f);
    long N = 200000000L;
    double t0 = now_s();
    for (long i = 0; i < N; i++) x = _mm256_fmadd_ps(x, a, b);
    double t1 = now_s();
    volatile float sink = _mm256_cvtss_f32(x); (void)sink;
    return (double)N * 4.0 / (t1 - t0) / 1e9;   /* FMA latency = 4 cycles */
}

/* Streaming read bandwidth at a chosen working-set size, GB/s. Used to build a
 * cache-roof ladder so the trace can say WHICH level is the limiter. */
static double bench_read_bw(size_t nfloats) {
    static std::vector<float> a, b;
    a.assign(nfloats, 1.0f);
    b.assign(nfloats, 2.0f);
    __m256 s = _mm256_setzero_ps();
    int reps = (int)((32u << 20) / (nfloats * 4u));
    if (reps < 2) reps = 2;
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i + 8 <= nfloats; i += 8)
            s = _mm256_add_ps(s, _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    double t1 = now_s();
    volatile float sink = _mm256_cvtss_f32(s); (void)sink;
    return (double)reps * nfloats * 4 * 2 / (t1 - t0) / 1e9;
}

static double stream_gbs(void) { return bench_read_bw(16u << 20); }

/* ------------------------------------------------------------ packed B ---- */
struct PackedI8 {
    int N = 0, K = 0, Kpad = 0;
    std::vector<int8_t>  rows;     /* [N][K]                */
    std::vector<uint8_t> blk;      /* [N/8][Kpad/4][32]     */
    std::vector<uint8_t> pan8;     /* [N/64][Kpad/4][8][32] */
    std::vector<uint8_t> pan2;     /* [N/16][Kpad/4][2][32] : M-blocked kernel */
    std::vector<int32_t> colsums;  /* [N]                   */
    std::vector<float>   scales;   /* [N]                   */
};

static void pack_i8(PackedI8& B, const std::vector<int8_t>& w, int N, int K) {
    B.N = N; B.K = K; B.Kpad = (K + 3) & ~3;
    B.colsums.assign(N, 0);
    B.scales.assign(N, 1.0f);
    const int kg = B.Kpad / 4;

    {   SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "pack.rows", 0, (uint64_t)N * K);
        B.rows = w; }

    B.blk.assign((size_t)(N / 8) * kg * 32, 0);
    {   SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "pack.blk", 0, B.blk.size());
        for (int cb = 0; cb < N / 8; cb++)
            for (int g = 0; g < kg; g++) {
                uint8_t* d = B.blk.data() + ((size_t)cb * kg + g) * 32;
                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 4; j++) {
                        const int k = g * 4 + j;
                        d[i * 4 + j] = (k < K) ? (uint8_t)w[(size_t)(cb * 8 + i) * K + k] : 0;
                    }
            } }

    if (N % 64 == 0) {
        B.pan8.assign((size_t)(N / 64) * kg * 256, 0);
        SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "pack.pan8", 0, B.pan8.size());
        for (int p = 0; p < N / 64; p++)
            for (int g = 0; g < kg; g++) {
                uint8_t* d = B.pan8.data() + ((size_t)p * kg + g) * 256;
                for (int b = 0; b < 8; b++)
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 4; j++) {
                            const int k = g * 4 + j;
                            d[b * 32 + i * 4 + j] =
                                (k < K) ? (uint8_t)w[(size_t)(p * 64 + b * 8 + i) * K + k] : 0;
                        }
            }
    }

    if (N % 16 == 0) {
        B.pan2.assign((size_t)(N / 16) * kg * 64, 0);
        SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "pack.pan2", 0, B.pan2.size());
        for (int p = 0; p < N / 16; p++)
            for (int g = 0; g < kg; g++) {
                uint8_t* d = B.pan2.data() + ((size_t)p * kg + g) * 64;
                for (int b = 0; b < 2; b++)
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 4; j++) {
                            const int k = g * 4 + j;
                            d[b * 32 + i * 4 + j] =
                                (k < K) ? (uint8_t)w[(size_t)(p * 16 + b * 8 + i) * K + k] : 0;
                        }
            }
    }

    {   SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "pack.colsums", (uint64_t)N * K, 0);
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += w[(size_t)n * K + k];
            B.colsums[n] = s;
        } }
}

static inline __m256i make_avec(const int8_t* row, int K, int g) {
    uint32_t w = 0;
    uint8_t* wb = (uint8_t*)&w;
    for (int j = 0; j < 4; j++) {
        const int k = g * 4 + j;
        wb[j] = (k < K) ? (uint8_t)((int16_t)row[k] + 128) : 0;
    }
    return _mm256_set1_epi32((int)w);
}

/* 3DNow!/AltiVec-style packed broadcast: build 8 broadcast A-vectors from 32
 * input bytes with packed ops only - no scalar byte assembly.
 *
 * 3DNow! part: signed int8 -> unsigned for dpbusd is just XOR 0x80 (two's
 * complement), converting all 32 bytes in one instruction.
 *
 * AltiVec part: vec_perm is full-width, but AVX2 pshufb is limited to 128-bit
 * lanes. The AltiVec method is therefore emulated as permute2x128 (bring the
 * wanted lane into both lanes) + pshufb (replicate the 4-byte group). One
 * lane-duplicate serves 4 groups, so it is 2 permutes + 8 shuffles per 8
 * groups - about 1.3 ops per group against ~10 for the scalar form.
 *
 * B is zero-padded past K, so padding bytes (which become 0x80 unsigned)
 * contribute 128*0 = 0 and the colsum correction stays exact.
 */
static inline void make_avec_row(const int8_t* row, int K, int kg, __m256i* out) {
    const __m256i x80 = _mm256_set1_epi8(-128);
    int g = 0;
    for (; g + 8 <= kg; g += 8) {
        const __m256i raw = _mm256_loadu_si256((const __m256i*)(row + (size_t)g * 4));
        const __m256i uns = _mm256_xor_si256(raw, x80);           /* signed -> unsigned */
        const __m256i lo  = _mm256_permute2x128_si256(uns, uns, 0x00); /* lane0 -> both */
        const __m256i hi  = _mm256_permute2x128_si256(uns, uns, 0x11); /* lane1 -> both */
        for (int j = 0; j < 4; j++) {
            const __m256i ctl = _mm256_setr_epi8(
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3),
                (char)(4*j+0), (char)(4*j+1), (char)(4*j+2), (char)(4*j+3));
            out[g + j]     = _mm256_shuffle_epi8(lo, ctl);
            out[g + j + 4] = _mm256_shuffle_epi8(hi, ctl);
        }
    }
    for (; g < kg; g++) out[g] = make_avec(row, K, g);   /* tail */
}

/* ------------------------------------------------------- raw CPU baseline -- */
NOINLINE
static void gemm_i8_rawcpu(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "rawcpu.kloop",
                   (uint64_t)M * B.N * K, (uint64_t)M * K + (uint64_t)B.N * K);
    for (int m = 0; m < M; m++) {
        const int8_t* ar = A + (size_t)m * K;
        for (int n = 0; n < B.N; n++) {
            const int8_t* br = B.rows.data() + (size_t)n * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)ar[k] * (int32_t)br[k];
            C[(size_t)m * B.N + n] = (float)s * B.scales[n];
        }
    }
}

/* -------------------------------------------------------- scalar oracle --- */
static void gemm_i8_scalar(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < B.N; n++) {
            const int8_t* ar = A + (size_t)m * K;
            const int8_t* br = B.rows.data() + (size_t)n * K;
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int64_t)ar[k] * (int64_t)br[k];
            C[(size_t)m * B.N + n] = (float)s * B.scales[n];
        }
}

/* ------------------------------------------------------- AVX2 fallback ---- */
static inline __m256i mul_sum_i8_pairs(__m256i x, __m256i y) {
    __m256i ax = _mm256_sign_epi8(x, x);
    __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(ax, sy));
}
static inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
NOINLINE
static void gemm_i8_avx2(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "avx2.kloop",
                   (uint64_t)M * B.N * K, (uint64_t)M * K + (uint64_t)B.N * K);
    for (int m = 0; m < M; m++) {
        const int8_t* ar = A + (size_t)m * K;
        for (int n = 0; n < B.N; n++) {
            const int8_t* br = B.rows.data() + (size_t)n * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= K; k += 32)
                acc = _mm256_add_epi32(acc, mul_sum_i8_pairs(
                    _mm256_loadu_si256((const __m256i*)(ar + k)),
                    _mm256_loadu_si256((const __m256i*)(br + k))));
            int32_t d = hsum_epi32(acc);
            for (; k < K; k++) d += (int32_t)ar[k] * (int32_t)br[k];
            C[(size_t)m * B.N + n] = (float)d * B.scales[n];
        }
    }
}

/* ---------------------------------------------------- VEX-256 VNNI, blk --- */
NOINLINE
static void gemm_i8_vnni_blk(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    static const int PREP = sherlock_impl::slot_for("avec.build", SHERLOCK_TRAP_INSTRUCTION);
    const int kg = B.Kpad / 4, NB = 8, UN = 4;
    const int NC = B.N / NB;
    std::vector<__m256i> avec(kg);

    for (int m = 0; m < M; m++) {
        const int8_t* ar = A + (size_t)m * K;
        {   SHERLOCK_SCOPE(SHERLOCK_TRAP_INSTRUCTION, "avec.build", 0, 0);
            for (int g = 0; g < kg; g++) avec[g] = make_avec(ar, K, g); }
        float* cr = C + (size_t)m * B.N;
        for (int cb = 0; cb < NC; cb += UN) {
            const int nb = (cb + UN <= NC) ? UN : (NC - cb);
            __m256i acc[UN];
            const uint8_t* base[UN];
            for (int b = 0; b < nb; b++) {
                acc[b]  = _mm256_setzero_si256();
                base[b] = B.blk.data() + (size_t)(cb + b) * kg * 32;
            }
            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "vnni_blk.kloop",
                               (uint64_t)nb * NB * K, (uint64_t)nb * kg * 32);
                for (int g = 0; g < kg; g++) {
                    const __m256i a = avec[g];
                    for (int b = 0; b < nb; b++)
                        acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a,
                                 _mm256_loadu_si256((const __m256i*)(base[b] + (size_t)g * 32)));
                } }
            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_MEMORY, "vnni_blk.epilogue", 0, (uint64_t)nb * 32);
                for (int b = 0; b < nb; b++) {
                    const int n0 = (cb + b) * NB;
                    __m256i sums = _mm256_loadu_si256((const __m256i*)(B.colsums.data() + n0));
                    __m256i fx = _mm256_sub_epi32(acc[b], _mm256_slli_epi32(sums, 7));
                    _mm256_storeu_ps(cr + n0,
                        _mm256_mul_ps(_mm256_cvtepi32_ps(fx), _mm256_loadu_ps(B.scales.data() + n0)));
                } }
        }
    }
    (void)PREP;
}

/* ---------------------------------- memory reuse hints (AltiVec dst / SSE) */
/* AltiVec's vec_dst family and SSE's non-temporal hints are the same idea:
 * tell the hardware how the data will be reused, instead of letting it guess.
 *   decode  : weights are touched once   -> NTA, keep them out of L2/L3
 *   prefill : weights are re-read        -> T0, keep them close
 *   C writes: write-once                 -> streaming store, no read-for-ownership
 */
enum { PF_NONE = 0, PF_T0 = 1, PF_NTA = 2 };
enum { ST_CACHED = 0, ST_STREAM = 1 };

template <int PF>
static inline void dst_touch(const void* p) {
    if constexpr (PF == PF_T0)       _mm_prefetch((const char*)p, _MM_HINT_T0);
    else if constexpr (PF == PF_NTA) _mm_prefetch((const char*)p, _MM_HINT_NTA);
}

/* --------------------------------------------------- VEX-256 VNNI, pan8 --- */
template <int STORE, int PF>
NOINLINE
static void gemm_i8_vnni_pan8_s(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    const int kg = B.Kpad / 4, NB = 8, PB = 8, PCOLS = PB * NB;
    const int NP = B.N / PCOLS;
    const size_t pan_bytes = (size_t)kg * PB * 32;
    std::vector<__m256i> avec(kg);

    for (int m = 0; m < M; m++) {
        const int8_t* ar = A + (size_t)m * K;
        {   SHERLOCK_SCOPE(SHERLOCK_TRAP_INSTRUCTION, "pan8.avec.build", 0, 0);
            make_avec_row(ar, K, kg, avec.data()); }
        float* cr = C + (size_t)m * B.N;

        for (int p = 0; p < NP; p++) {
            const uint8_t* pb0 = B.pan8.data() + (size_t)p * pan_bytes;
            __m256i acc[PB];
            for (int b = 0; b < PB; b++) acc[b] = _mm256_setzero_si256();

            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "pan8.kloop",
                               (uint64_t)PCOLS * K, pan_bytes);
                int g = 0;
                for (; g + 2 <= kg; g += 2) {
                    const uint8_t* q0 = pb0 + (size_t)g * PB * 32;
                    const uint8_t* q1 = q0 + PB * 32;
                    dst_touch<PF>(q1 + PB * 32);
                    const __m256i a0 = avec[g], a1 = avec[g + 1];
                    for (int b = 0; b < PB; b++) {
                        acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                                 _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
                        acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a1,
                                 _mm256_loadu_si256((const __m256i*)(q1 + (size_t)b * 32)));
                    }
                }
                for (; g < kg; g++) {
                    const uint8_t* q0 = pb0 + (size_t)g * PB * 32;
                    const __m256i a0 = avec[g];
                    for (int b = 0; b < PB; b++)
                        acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                                 _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
                } }
            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_MEMORY, "pan8.epilogue", 0, (uint64_t)PCOLS * 4);
                for (int b = 0; b < PB; b++) {
                    const int n0 = p * PCOLS + b * NB;
                    __m256i sums = _mm256_loadu_si256((const __m256i*)(B.colsums.data() + n0));
                    __m256i fx = _mm256_sub_epi32(acc[b], _mm256_slli_epi32(sums, 7));
                    const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(fx),
                                                   _mm256_loadu_ps(B.scales.data() + n0));
                    float* dst = cr + n0;
                    if constexpr (STORE == ST_STREAM) {
                        if (((uintptr_t)dst & 31u) == 0) _mm256_stream_ps(dst, v);
                        else                             _mm256_storeu_ps(dst, v);
                    } else {
                        _mm256_storeu_ps(dst, v);
                    }
                } }
        }
    }
    if constexpr (STORE == ST_STREAM) _mm_sfence();
}

NOINLINE
static void gemm_i8_vnni_pan8(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    gemm_i8_vnni_pan8_s<ST_CACHED, PF_T0>(A, M, K, B, C);
}

/* ------------------------------------------- partitioning (mainframe/DB2) -- */
/*
 * DB2's lesson: do not let one worker scan the whole table - partition it so
 * each worker's slice is contiguous and cache-friendly. And point
 * z/Architecture / UltraSPARC T hide memory latency with many runnable threads.
 *
 * CRITICAL: the pool is created ONCE and the workers spin on an epoch counter.
 * Creating threads inside the timed loop measures thread-creation cost
 * (~0.5 ms each on Windows), not the kernel. That is the real lesson of the
 * review's "batch submissions, reduce synchronisation" point: for a decode
 * GEMV the work is tens of microseconds, so dispatch must be amortised over
 * many GEMMs (a whole layer) or it dominates.
 */
template <int STORE, int PF>
static void pan8_range_m(const int8_t* A, int M0, int M1, int K, const PackedI8& B,
                         float* C, int p0, int p1) {
    const int kg = B.Kpad / 4, NB = 8, PB = 8, PCOLS = PB * NB;
    const size_t pan_bytes = (size_t)kg * PB * 32;
    std::vector<__m256i> avec(kg);

    for (int m = M0; m < M1; m++) {
        const int8_t* ar = A + (size_t)m * K;
        make_avec_row(ar, K, kg, avec.data());
        float* cr = C + (size_t)m * B.N;
        for (int p = p0; p < p1; p++) {
            const uint8_t* pb0 = B.pan8.data() + (size_t)p * pan_bytes;
            __m256i acc[PB];
            for (int b = 0; b < PB; b++) acc[b] = _mm256_setzero_si256();
            int g = 0;
            for (; g + 2 <= kg; g += 2) {
                const uint8_t* q0 = pb0 + (size_t)g * PB * 32;
                const uint8_t* q1 = q0 + PB * 32;
                dst_touch<PF>(q1 + PB * 32);
                const __m256i a0 = avec[g], a1 = avec[g + 1];
                for (int b = 0; b < PB; b++) {
                    acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                             _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
                    acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a1,
                             _mm256_loadu_si256((const __m256i*)(q1 + (size_t)b * 32)));
                }
            }
            for (; g < kg; g++) {
                const uint8_t* q0 = pb0 + (size_t)g * PB * 32;
                const __m256i a0 = avec[g];
                for (int b = 0; b < PB; b++)
                    acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                             _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
            }
            for (int b = 0; b < PB; b++) {
                const int n0 = p * PCOLS + b * NB;
                __m256i sums = _mm256_loadu_si256((const __m256i*)(B.colsums.data() + n0));
                __m256i fx = _mm256_sub_epi32(acc[b], _mm256_slli_epi32(sums, 7));
                const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(fx),
                                               _mm256_loadu_ps(B.scales.data() + n0));
                float* dst = cr + n0;
                if constexpr (STORE == ST_STREAM) {
                    if (((uintptr_t)dst & 31u) == 0) _mm256_stream_ps(dst, v);
                    else                             _mm256_storeu_ps(dst, v);
                } else {
                    _mm256_storeu_ps(dst, v);
                }
            }
        }
    }
    if constexpr (STORE == ST_STREAM) _mm_sfence();
}

template <int STORE, int PF>
static void pan8_range(const int8_t* A, int M, int K, const PackedI8& B, float* C,
                       int p0, int p1) {
    pan8_range_m<STORE, PF>(A, 0, M, K, B, C, p0, p1);
}

/* --------------------------------------- Cell-style staging tile (SPE model) */
/*
 * Cell's SPE model is a small software-managed local store plus a double-buffered
 * DMA. The x86 analogue used here is a 64-byte-aligned buffer sized to fit L1
 * (48 KB on this P-core) and a sequential copy of one K-tile issued one tile
 * ahead, so filling tile t+1 overlaps the arithmetic on tile t.
 *
 * A "tile" is a run of K-groups belonging to one 64-column panel. The point of
 * the staging buffer is that the compute loop then reads with unit stride out of
 * a known-resident buffer instead of walking the whole panel, which is what Cell
 * bought by making the local store explicit.
 *
 * Two modes, because they cost different things:
 *
 *   STAGE_COPY     - explicit memcpy into the staging buffer: the literal DMA.
 *                    Pays an extra read+write of every weight byte.
 *   STAGE_PREFETCH - no copy, only T0/NTA touches issued one whole tile ahead.
 *                    Same lookahead, no extra bandwidth. The plain pan8 kernel
 *                    only looks one cache line ahead, so the depth is the change.
 *
 * The staging buffer is a persistent thread-local "local store", allocated once
 * and grown as needed, so per-call timing does not include malloc.
 */
static constexpr int PAN8_PB = 8;   /* 8 blocks of 8 output columns per panel */

/* Accumulate K-groups [g0,g1) of one panel. Group g0 sits at offset 0 of base,
 * so base is either the panel itself or a staged copy of one tile of it. */
static inline void pan8_acc_tile(__m256i* acc, const uint8_t* base,
                                 const __m256i* avec, int g0, int g1) {
    constexpr int PB = PAN8_PB;
    int t = 0, g = g0;
    for (; g + 2 <= g1; g += 2, t += 2) {
        const __m256i a0 = avec[g], a1 = avec[g + 1];
        const uint8_t* q0 = base + (size_t)t * PB * 32;
        const uint8_t* q1 = q0 + PB * 32;
        for (int b = 0; b < PB; b++) {
            acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                     _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
            acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a1,
                     _mm256_loadu_si256((const __m256i*)(q1 + (size_t)b * 32)));
        }
    }
    for (; g < g1; g++, t++) {
        const __m256i a0 = avec[g];
        const uint8_t* q0 = base + (size_t)t * PB * 32;
        for (int b = 0; b < PB; b++)
            acc[b] = _mm256_dpbusd_avx_epi32(acc[b], a0,
                     _mm256_loadu_si256((const __m256i*)(q0 + (size_t)b * 32)));
    }
}

/* Touch a whole tile into cache, one line at a time. */
template <int PF>
static inline void pan8_pf_bytes(const uint8_t* base, int bytes) {
    if constexpr (PF != PF_NONE)
        for (int o = 0; o < bytes; o += 64) dst_touch<PF>(base + o);
}

/* The literal DMA: move one tile into the local store. */
static inline void pan8_stage(uint8_t* dst, const uint8_t* panel, int g0, int tile_g) {
    memcpy(dst, panel + (size_t)g0 * PAN8_PB * 32, (size_t)tile_g * PAN8_PB * 32);
}

/* Cached epilogue for panel p: undo the +128 bias, apply the row scale. */
static inline void pan8_epilogue(const PackedI8& B, const __m256i* acc, float* cr, int p) {
    constexpr int PB = PAN8_PB, NB = 8, PCOLS = PB * NB;
    for (int b = 0; b < PB; b++) {
        const int n0 = p * PCOLS + b * NB;
        const __m256i sums = _mm256_loadu_si256((const __m256i*)(B.colsums.data() + n0));
        const __m256i fx   = _mm256_sub_epi32(acc[b], _mm256_slli_epi32(sums, 7));
        const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(fx),
                                       _mm256_loadu_ps(B.scales.data() + n0));
        _mm256_storeu_ps(cr + n0, v);
    }
}

enum { STAGE_COPY = 0, STAGE_PREFETCH = 1 };

template <int MODE, int PF>
NOINLINE
static void gemm_i8_vnni_stage(const int8_t* A, int M, int K, const PackedI8& B,
                               float* C, int tile_g) {
    constexpr int PB = PAN8_PB, PCOLS = PAN8_PB * 8;
    const int kg = B.Kpad / 4;
    const int NP = B.N / PCOLS;
    const size_t pan_bytes  = (size_t)kg * PB * 32;
    const size_t tile_bytes = (size_t)tile_g * PB * 32;
    const int ntiles = (kg + tile_g - 1) / tile_g;

    /* persistent thread-local local store */
    static thread_local uint8_t* stage = nullptr;
    static thread_local size_t stage_cap = 0;
    if (stage_cap < tile_bytes * 2) {
        if (stage) _aligned_free(stage);
        stage = (uint8_t*)_aligned_malloc(tile_bytes * 2, 64);
        stage_cap = tile_bytes * 2;
    }

    std::vector<__m256i> avec(kg);

    for (int m = 0; m < M; m++) {
        const int8_t* ar = A + (size_t)m * K;
        {   SHERLOCK_SCOPE(SHERLOCK_TRAP_INSTRUCTION, "stage.avec.build", 0, 0);
            make_avec_row(ar, K, kg, avec.data()); }
        float* cr = C + (size_t)m * B.N;

        for (int p = 0; p < NP; p++) {
            const uint8_t* pb0 = B.pan8.data() + (size_t)p * pan_bytes;
            __m256i acc[PB];
            for (int b = 0; b < PB; b++) acc[b] = _mm256_setzero_si256();

            if constexpr (MODE == STAGE_COPY) {
                /* prologue fills buffer 0, then keep exactly one tile in flight */
                int g0 = 0;
                int g1 = (g0 + tile_g < kg) ? g0 + tile_g : kg;
                {   SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "stage.dma", 0, tile_bytes);
                    pan8_stage(stage, pb0, g0, g1 - g0); }
                for (int t = 0; t < ntiles; t++) {
                    const int n0 = g1;
                    const int n1 = (n0 + tile_g < kg) ? n0 + tile_g : kg;
                    if (t + 1 < ntiles) {
                        SHERLOCK_SCOPE(SHERLOCK_TRAP_TRANSFER, "stage.dma", 0, tile_bytes);
                        pan8_stage(stage + (size_t)((t + 1) & 1) * tile_bytes, pb0, n0, n1 - n0);
                    }
                    {   SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "stage.kloop",
                                       (uint64_t)PCOLS * K, tile_bytes);
                        pan8_acc_tile(acc, stage + (size_t)(t & 1) * tile_bytes,
                                      avec.data(), g0, g1); }
                    g0 = n0; g1 = n1;
                }
            } else {
                int g0 = 0;
                for (int t = 0; t < ntiles; t++, g0 += tile_g) {
                    const int g1 = (g0 + tile_g < kg) ? g0 + tile_g : kg;
                    const int n1 = (g1 + tile_g < kg) ? g1 + tile_g : kg;
                    if (t + 1 < ntiles) pan8_pf_bytes<PF>(pb0 + (size_t)g1 * PB * 32, (n1 - g1) * PB * 32);
                    {   SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "stage.kloop",
                                       (uint64_t)PCOLS * K, (uint64_t)(g1 - g0) * PB * 32);
                        pan8_acc_tile(acc, pb0 + (size_t)g0 * PB * 32, avec.data(), g0, g1); }
                }
            }
            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_MEMORY, "stage.epilogue", 0, (uint64_t)PCOLS * 4);
                pan8_epilogue(B, acc, cr, p); }
        }
    }
}

static void k_stage_pf64  (const int8_t* A, int M, int K, const PackedI8& B, float* C) { gemm_i8_vnni_stage<STAGE_PREFETCH, PF_T0 >(A, M, K, B, C,  64); }
static void k_stage_pf128 (const int8_t* A, int M, int K, const PackedI8& B, float* C) { gemm_i8_vnni_stage<STAGE_PREFETCH, PF_T0 >(A, M, K, B, C, 128); }
static void k_stage_pfnta64(const int8_t* A, int M, int K, const PackedI8& B, float* C) { gemm_i8_vnni_stage<STAGE_PREFETCH, PF_NTA>(A, M, K, B, C,  64); }
static void k_stage_cp64  (const int8_t* A, int M, int K, const PackedI8& B, float* C) { gemm_i8_vnni_stage<STAGE_COPY,     PF_NONE>(A, M, K, B, C, 64); }
static void k_stage_cp32  (const int8_t* A, int M, int K, const PackedI8& B, float* C) { gemm_i8_vnni_stage<STAGE_COPY,     PF_NONE>(A, M, K, B, C, 32); }

/*
 * NPU cluster: one worker per P-core, plus an E-core governor.
 *
 * This is the Cell split (PPE governs, SPEs compute) and the Kryo/hybrid split
 * (place by core class). No P-core is spent on scheduling, the governor is not
 * a worker, and workers claim units from a shared cursor so the load is
 * distributed dynamically rather than statically partitioned.
 *
 * The governor's useful job while workers run: look ahead over the weight
 * buffer and warm the shared L3, so the P-cores stream from cache instead of
 * DRAM. That is the Cell DMA engine role, done with prefetch instead of DMA.
 */
struct npu_cluster {
    const int8_t* A = nullptr; int M = 0, K = 0;
    const PackedI8* B = nullptr; float* C = nullptr;
    int nblk = 1, npart = 1, nunits = 1, npan = 0;
    std::vector<DWORD_PTR> pcores;
    DWORD_PTR govcore = 0;
    std::vector<std::thread> th;
    std::atomic<int> cursor{0};
    std::atomic<long> epoch{0}, busy{0}, done{0};
    std::atomic<bool> quit{false};
    std::atomic<long> gov_touched{0};

    void do_unit(int u) {
        if (u >= nunits) return;
        const int blk = u / npart, part = u % npart;
        const int m0 = (int)((long long)M * blk / nblk);
        const int m1 = (int)((long long)M * (blk + 1) / nblk);
        const int p0 = (int)((long long)npan * part / npart);
        const int p1 = (int)((long long)npan * (part + 1) / npart);
        pan8_range_m<ST_CACHED, PF_T0>(A, m0, m1, K, *B, C, p0, p1);
    }

    void worker(int i) {
        if (i < (int)pcores.size()) SetThreadAffinityMask(GetCurrentThread(), pcores[i]);
        long seen = 0;
        for (;;) {
            if (quit.load(std::memory_order_relaxed)) return;
            const long e = epoch.load(std::memory_order_acquire);
            if (e == seen) { _mm_pause(); continue; }
            seen = e;
            for (;;) {
                const int u = cursor.fetch_add(1, std::memory_order_relaxed);
                if (u >= nunits) break;
                do_unit(u);
            }
            busy.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void governor() {
        if (govcore) SetThreadAffinityMask(GetCurrentThread(), govcore);
        long seen = 0;
        for (;;) {
            if (quit.load(std::memory_order_relaxed)) return;
            const long e = epoch.load(std::memory_order_acquire);
            if (e == seen) { _mm_pause(); continue; }
            seen = e;
            /* Pure control plane: block until the crew drains, then release the
             * caller. A look-ahead prefetch walk of the weight buffer used to
             * live here; it was removed because it competed for exactly the
             * L3/DRAM bandwidth the workers need, and it pushed run-to-run
             * variance from ~6% to ~150% while never winning on the mean. */
            while (busy.load(std::memory_order_acquire) > 0) _mm_pause();
            done.store(e, std::memory_order_release);
        }
    }

    void start(int nb, int npart_, DWORD_PTR e_core) {
        nblk = nb; npart = npart_;
        nunits = nblk * npart;
        npan = B->N / 64;
        govcore = e_core;
        for (size_t i = 0; i < pcores.size(); i++) th.emplace_back([this, i]{ worker((int)i); });
        th.emplace_back([this]{ governor(); });
        for (size_t i = 0; i < th.size(); i++) { /* nothing: threads already running */ }
        run();   /* warm up */
    }
    void run() {
        const long e = epoch.load(std::memory_order_relaxed) + 1;
        cursor.store(0, std::memory_order_relaxed);
        busy.store((long)pcores.size(), std::memory_order_relaxed);
        epoch.store(e, std::memory_order_release);
        while (done.load(std::memory_order_acquire) != e) _mm_pause();
    }
    void stop() {
        quit.store(true, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_release);
        for (size_t i = 0; i < th.size(); i++) th[i].join();
        th.clear();
    }
};
/*
 * Heterogeneous work stealing across all three core classes.
 *
 * Thread Director already moves work between P- and E-cores, but it does so
 * with no knowledge of our tiles, and it silently parks threads on E-cores -
 * which is why the *unpinned* pool stalled at 4 workers and then regressed.
 * So we pin one worker per physical core and let the scheduler see all of them.
 *
 * No capability model is needed. Partitioning (rows via nblk, N panels via
 * npart) produces far more tiles than there are workers, and every worker pulls
 * from ONE atomic cursor: a core that runs at half rate simply claims half as
 * many tiles. Balance falls out of the granularity, not out of us guessing
 * which core is fast. Row-splitting is preferred where M allows it because the
 * activation expansion is per-row, so it is not duplicated across tiles.
 */
struct hetero_pool {
    const int8_t* A = nullptr; int M = 0, K = 0;
    const PackedI8* B = nullptr; float* C = nullptr;
    int nblk = 1, npart = 1, nunits = 1, npan = 0;
    std::vector<DWORD_PTR> cores;      /* one physical core per worker */
    std::vector<std::thread> th;
    std::atomic<int> cursor{0};
    std::atomic<long> epoch{0}, busy{0};
    std::atomic<bool> quit{false};
    DWORD_PTR prior = 0; bool caller_pinned = false;

    void do_unit(int u) {
        if (u >= nunits) return;
        const int blk = u / npart, part = u % npart;
        const int m0 = (int)((long long)M * blk / nblk);
        const int m1 = (int)((long long)M * (blk + 1) / nblk);
        const int p0 = (int)((long long)npan * part / npart);
        const int p1 = (int)((long long)npan * (part + 1) / npart);
        pan8_range_m<ST_CACHED, PF_T0>(A, m0, m1, K, *B, C, p0, p1);
    }

    void worker(int i) {
        if (cores[i]) SetThreadAffinityMask(GetCurrentThread(), cores[i]);
        long seen = 0;
        for (;;) {
            if (quit.load(std::memory_order_relaxed)) return;
            const long e = epoch.load(std::memory_order_acquire);
            if (e == seen) { _mm_pause(); continue; }
            seen = e;
            for (;;) {
                const int u = cursor.fetch_add(1, std::memory_order_relaxed);
                if (u >= nunits) break;
                do_unit(u);
            }
            busy.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    /* caller_core keeps the joining thread off the P-cores while it spin-waits */
    void start(int nb, int npart_, DWORD_PTR caller_core) {
        nblk = nb; npart = npart_;
        nunits = nblk * npart;
        npan = B->N / 64;
        for (size_t i = 0; i < cores.size(); i++) th.emplace_back([this, i]{ worker((int)i); });
        if (caller_core) {
            prior = SetThreadAffinityMask(GetCurrentThread(), caller_core);
            caller_pinned = true;
        }
        run();   /* warm up */
    }
    void run() {
        const long e = epoch.load(std::memory_order_relaxed) + 1;
        cursor.store(0, std::memory_order_relaxed);
        busy.store((long)cores.size(), std::memory_order_relaxed);
        epoch.store(e, std::memory_order_release);
        while (busy.load(std::memory_order_acquire) != 0) _mm_pause();
    }
    void stop() {
        quit.store(true, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_release);
        for (size_t i = 0; i < th.size(); i++) th[i].join();
        th.clear();
        if (caller_pinned) { SetThreadAffinityMask(GetCurrentThread(), prior); caller_pinned = false; }
    }
};

struct pan8_pool {
    const int8_t* A = nullptr; int M = 0, K = 0;
    const PackedI8* B = nullptr; float* C = nullptr;
    int npan = 0, nth = 0, gang = 0;
    bool pin = false;
    std::vector<DWORD_PTR> masks;
    std::vector<std::thread> th;
    std::atomic<long> epoch{0}, remaining{0};
    std::atomic<bool> quit{false};

    void build_masks(DWORD_PTR pcore_mask) {
        masks.clear();
        for (int b = 0; b < 64; b++)
            if (pcore_mask & ((DWORD_PTR)1 << b)) masks.push_back((DWORD_PTR)1 << b);
    }

    void worker(int t) {
        if (pin && !masks.empty())
            SetThreadAffinityMask(GetCurrentThread(), masks[(size_t)t % masks.size()]);
        long seen = 0;
        for (;;) {
            if (quit.load(std::memory_order_relaxed)) return;
            const long e = epoch.load(std::memory_order_acquire);
            if (e == seen) { _mm_pause(); continue; }
            seen = e;
            if (t < gang) {
                const int p0 = (int)((long long)npan * t / gang);
                const int p1 = (int)((long long)npan * (t + 1) / gang);
                pan8_range<ST_CACHED, PF_T0>(A, M, K, *B, C, p0, p1);
            }
            remaining.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void start(int n, int g) {
        nth = n; gang = g; npan = B->N / 64;
        for (int t = 0; t < nth; t++) th.emplace_back([this, t]{ worker(t); });
        run();                       /* warm the workers up */
    }
    void run() {
        remaining.store(nth, std::memory_order_relaxed);
        epoch.fetch_add(1, std::memory_order_release);
        while (remaining.load(std::memory_order_acquire) != 0) _mm_pause();
    }
    void stop() {
        quit.store(true, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_release);
        for (size_t i = 0; i < th.size(); i++) th[i].join();
        th.clear(); quit.store(false); epoch.store(0);
    }
};

/* Kryo / Meteor Lake heterogeneity: this CPU is 4 P-cores + 8 E-cores + 2 LP-E.
 * A memory/compute bound GEMM should not be scheduled onto an E-core. Enumerate
 * the core classes and build a mask of the highest-efficiency (P) cores. */
struct core_classes {
    int np = 0, ne = 0, nlpe = 0;
    BYTE pclass = 0;
    DWORD_PTR pmask = 0, emask = 0;
    std::vector<DWORD_PTR> pfirst;    /* one primary thread per P-core */
    std::vector<DWORD_PTR> efirst;    /* one primary thread per E-core */
    std::vector<DWORD_PTR> lpefirst;  /* one primary thread per LP-E core */
};

static core_classes core_scan(void) {
    core_classes cc;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (!len) return cc;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len)) return cc;
    DWORD off = 0;
    BYTE maxclass = 0;
    while (off < len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.EfficiencyClass > maxclass)
            maxclass = e->Processor.EfficiencyClass;
        off += e->Size;
    }
    cc.pclass = maxclass;
    off = 0;
    while (off < len) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off);
        if (e->Relationship == RelationProcessorCore) {
            /* each core lists its logical processors; take one so we place one
             * worker per physical core, never two on the same hyperthread pair */
            const DWORD_PTR m = e->Processor.GroupMask[0].Mask;
            DWORD_PTR first = 0;
            for (int b = 0; b < 64; b++) if (m & ((DWORD_PTR)1 << b)) { first = (DWORD_PTR)1 << b; break; }
            const BYTE c = e->Processor.EfficiencyClass;
            /* pmask/emask keep EVERY logical processor of the class (the legacy
             * pan8_pool hands one per worker, so it needs the hyperthread
             * siblings); pfirst/efirst keep exactly one per physical core, for
             * the pools that must never place two workers on one hyperthread. */
            if (c == maxclass)          { cc.np++;   cc.pmask |= m; cc.pfirst.push_back(first); }
            else if (c + 1 == maxclass) { cc.ne++;   cc.emask |= m; cc.efirst.push_back(first); }
            else                        { cc.nlpe++; cc.lpefirst.push_back(first); }
        }
        off += e->Size;
    }
#endif
    return cc;
}

/* z/Architecture DAT / huge pages: fewer TLB entries for the weight stream.
 * Windows needs SeLockMemoryPrivilege for MEM_LARGE_PAGES, so probe and report
 * rather than assume. */
static bool large_page_probe(void) {
#ifdef _WIN32
    SIZE_T sz = GetLargePageMinimum();
    if (!sz) return false;
    void* p = VirtualAlloc(NULL, sz, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (!p) return false;
    VirtualFree(p, 0, MEM_RELEASE);
    return true;
#else
    return false;
#endif
}

/* ------------------------------------------- VEX-256 VNNI, M-blocked ------ */
/*
 * Classic GEMM register blocking over the M dimension, which is THE lever when
 * M > 1. The streaming side is B; A (the broadcast vectors) is tiny and stays
 * resident in L1. So the traffic that matters is B, and it is loaded
 *     M            times   without blocking  (one full sweep per output row)
 *     M/MR         times   with an MR block  (MR rows share one sweep)
 * i.e. weight traffic drops by MR. That is the whole trick, and it is why
 * prefill (large M) can go much faster than decode (M = 1).
 *
 * Registers: MR*PB accumulators + MR a[] + 1 bvec.
 *   MR=4, PB=2 -> 8 + 4 + 1 = 13  (bvec bytes/MAC = 1/MR = 0.25)
 *   MR=2, PB=2 -> 4 + 2 + 1 =  7  (0.5 bvec bytes/MAC)
 */
template <int MR>
NOINLINE
static void gemm_i8_vnni_rowblk(const int8_t* A, int M, int K, const PackedI8& B, float* C) {
    const int NB = 8, PB = 2, PCOLS = PB * NB;   /* 16 columns per panel */
    const int kg = B.Kpad / 4;
    const int NP = B.N / PCOLS;
    const size_t pan_bytes = (size_t)kg * PB * 32;

    for (int m0 = 0; m0 < M; m0 += MR) {
        const int mr = (m0 + MR <= M) ? MR : (M - m0);
        std::vector<__m256i> avec((size_t)mr * kg);
        {   SHERLOCK_SCOPE(SHERLOCK_TRAP_INSTRUCTION, "rowblk.avec", 0, 0);
            /* The packed 3DNow!/AltiVec broadcast builder, one row at a time.
             * Using the scalar make_avec() here - as this kernel first did -
             * costs ~10 ops per group against ~1.25, and it is rebuilt for
             * every M-block, so it dominated the whole kernel and hid the
             * benefit of reusing each B operand across MR rows. */
            for (int i = 0; i < mr; i++) {
                const int8_t* ar = A + (size_t)(m0 + i) * K;
                make_avec_row(ar, K, kg, avec.data() + (size_t)i * kg);
            } }

        for (int p = 0; p < NP; p++) {
            const uint8_t* pb0 = B.pan2.data() + (size_t)p * pan_bytes;
            __m256i acc[MR][PB];
            for (int i = 0; i < mr; i++)
                for (int b = 0; b < PB; b++) acc[i][b] = _mm256_setzero_si256();

            {   /* only pan_bytes of B is read here, however many rows reuse it */
                SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "rowblk.kloop",
                               (uint64_t)mr * PCOLS * K, pan_bytes);
                for (int g = 0; g < kg; g++) {
                    __m256i a[MR];
                    for (int i = 0; i < mr; i++) a[i] = avec[(size_t)i * kg + g];
                    const uint8_t* q = pb0 + (size_t)g * PB * 32;
                    _mm_prefetch((const char*)(q + 2 * PB * 32), _MM_HINT_T0);
                    for (int b = 0; b < PB; b++) {
                        const __m256i bv = _mm256_loadu_si256((const __m256i*)(q + (size_t)b * 32));
                        for (int i = 0; i < mr; i++)
                            acc[i][b] = _mm256_dpbusd_avx_epi32(acc[i][b], a[i], bv);
                    }
                } }

            {   SHERLOCK_SCOPE(SHERLOCK_TRAP_MEMORY, "rowblk.epilogue", 0, (uint64_t)mr * PCOLS * 4);
                for (int i = 0; i < mr; i++) {
                    float* cr = C + (size_t)(m0 + i) * B.N;
                    for (int b = 0; b < PB; b++) {
                        const int n0 = p * PCOLS + b * NB;
                        __m256i sums = _mm256_loadu_si256((const __m256i*)(B.colsums.data() + n0));
                        __m256i fx = _mm256_sub_epi32(acc[i][b], _mm256_slli_epi32(sums, 7));
                        _mm256_storeu_ps(cr + n0,
                            _mm256_mul_ps(_mm256_cvtepi32_ps(fx), _mm256_loadu_ps(B.scales.data() + n0)));
                    }
                } }
        }
    }
}

/* ==================================================================
 * LiGGUF's "shaders": fused dequant-free dots over the real GGUF block
 * formats, reproduced from arch_avx2.h. Storage density is the point:
 *   int8 plain : 1.0000 B/weight
 *   Q8_0 block : 34/32 = 1.0625 B/weight
 *   Q5_0 block : 22/32 = 0.6875 B/weight
 *   Q4_0 block : 18/32 = 0.5625 B/weight
 * Decode (M=1) is memory bound, so fewer bytes/weight is a direct win.
 * ================================================================== */

static inline float f16_to_f32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    float exp_scale; uint32_t b1 = 0x7800000u; memcpy(&exp_scale, &b1, 4);
    float norm; uint32_t b2 = (two_w >> 4) + (0xE0u << 23); memcpy(&norm, &b2, 4);
    norm *= exp_scale;
    float denorm; uint32_t b3 = (two_w >> 17) | (126u << 23); memcpy(&denorm, &b3, 4);
    denorm -= 0.5f;
    uint32_t rb;
    if (two_w < (1u << 27)) memcpy(&rb, &denorm, 4); else memcpy(&rb, &norm, 4);
    rb |= sign;
    float r; memcpy(&r, &rb, 4);
    return r;
}

struct blk_q8_0 { uint16_t d; int8_t   qs[32]; };
struct blk_q4_0 { uint16_t d; uint8_t  qs[16]; };
struct blk_q5_0 { uint16_t d; uint8_t  qh[4]; uint8_t qs[16]; };

/* The engine converts fp16 scales through a 64K LUT (lil_math.cpp fp1632_init)
 * rather than the scalar bit twiddle - the scalar form is called twice per
 * 32-value block and dominated the block shaders. Match the engine. */
static float g_f16lut[65536];
static void f16_lut_init(void) {
    for (int i = 0; i < 65536; i++) g_f16lut[i] = f16_to_f32((uint16_t)i);
}
#define F16(h) (g_f16lut[(h)])

static inline float hsum_ps(__m256 v) {
    /* packed reduction: no memory round trip, no scalar adds */
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

NOINLINE
static float dot_q8_0_q8_0(const blk_q8_0* x, const blk_q8_0* y, int n) {
    SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "shader.Q8_0xQ8_0", (uint64_t)n, (uint64_t)n * 34 / 32 + n);
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        __m256 d = _mm256_set1_ps(F16(x[i].d) * F16(y[i].d));
        __m256i qx = _mm256_loadu_si256((const __m256i*)x[i].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i*)y[i].qs);
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(mul_sum_i8_pairs(qx, qy)), acc);
    }
    return hsum_ps(acc);
}

NOINLINE
static float dot_q8_0_q4_0(const blk_q8_0* y, const blk_q4_0* x, int n) {
    SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "shader.Q8_0xQ4_0", (uint64_t)n, (uint64_t)n * 18 / 32 + n);
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const float d = F16(x[i].d) * F16(y[i].d);
        __m256i q8 = _mm256_loadu_si256((const __m256i*)y[i].qs);
        __m128i qs128 = _mm_loadu_si128((const __m128i*)x[i].qs);
        __m128i lo = _mm_and_si128(qs128, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(qs128, 4), _mm_set1_epi8(0x0F));
        __m256i qs = _mm256_set_m128i(hi, lo);
        __m256i q4 = _mm256_sub_epi8(qs, _mm256_set1_epi8(8));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d),
                              _mm256_cvtepi32_ps(mul_sum_i8_pairs(q4, q8)), acc);
    }
    return hsum_ps(acc);
}

NOINLINE
static float dot_q8_0_q5_0(const blk_q8_0* y, const blk_q5_0* x, int n) {
    SHERLOCK_SCOPE(SHERLOCK_TRAP_COMPUTE, "shader.Q8_0xQ5_0", (uint64_t)n, (uint64_t)n * 22 / 32 + n);
    const int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    /* expand the 32 qh bits to 32 0/1 bytes, all in SIMD:
     * lane L = qh32 >> 4L (low nibble = bits 4L..4L+3), replicated into the
     * lane's 4 bytes, then compared against the per-byte bit mask. */
    const __m256i counts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i bit    = _mm256_setr_epi8(
        1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8,
        1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i nib = _mm256_set1_epi32(0x0F);
    const __m256i splat = _mm256_set1_epi32(0x01010101);

    for (int i = 0; i < nb; i++) {
        const float d = F16(x[i].d) * F16(y[i].d);
        __m256i q8 = _mm256_loadu_si256((const __m256i*)y[i].qs);

        uint32_t qh32 = (uint32_t)x[i].qh[0] | ((uint32_t)x[i].qh[1] << 8) |
                        ((uint32_t)x[i].qh[2] << 16) | ((uint32_t)x[i].qh[3] << 24);
        __m256i q = _mm256_set1_epi32((int)qh32);
        __m256i n4 = _mm256_and_si256(_mm256_srlv_epi32(q, counts), nib);
        __m256i rep = _mm256_mullo_epi32(n4, splat);
        __m256i bits = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(rep, bit), bit), one);
        __m256i qh = _mm256_slli_epi16(bits, 4);

        __m128i qs128 = _mm_loadu_si128((const __m128i*)x[i].qs);
        __m128i lo = _mm_and_si128(qs128, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(qs128, 4), _mm_set1_epi8(0x0F));
        __m256i qs = _mm256_set_m128i(hi, lo);
        __m256i q5 = _mm256_sub_epi8(_mm256_or_si256(qs, qh), _mm256_set1_epi8(16));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d),
                              _mm256_cvtepi32_ps(mul_sum_i8_pairs(q5, q8)), acc);
    }
    return hsum_ps(acc);
}

/* ---------------------------------------------------------------- bench --- */
static int g_N = 2048, g_K = 2048;

template <typename F>
static double timeit(int reps, F fn) {
    double t0 = now_s();
    for (int r = 0; r < reps; r++) fn();
    return (now_s() - t0) / reps;
}

/* ------------------------------------------------------------------------
 * What is the actual VNNI issue ceiling on this core?
 *
 * The headline "compute peak" is ghz * 32 MAC, i.e. one VPDPBUSD per cycle.
 * Whether that is reachable decides whether further kernel work is worth
 * anything, so measure it directly: independent accumulator chains, operands
 * already in registers, nothing else in the loop.
 * ------------------------------------------------------------------------ */
static volatile int g_sink;

NOINLINE
static double vnni_peak_reg(long iters, int nchains) {
    const __m256i a = _mm256_set1_epi8(3);
    const __m256i b = _mm256_set1_epi8(5);
    __m256i acc[16];
    for (int i = 0; i < nchains; i++) acc[i] = _mm256_set1_epi32(i);
    const double t0 = now_s();
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < nchains; i++)
            acc[i] = _mm256_dpbusd_avx_epi32(acc[i], a, b);
    const double dt = now_s() - t0;
    int s = 0;
    for (int i = 0; i < nchains; i++) s += _mm_extract_epi32(_mm256_castsi256_si128(acc[i]), 0);
    g_sink = s;
    return dt;
}

NOINLINE
static double vnni_peak_l1(long iters, int nchains, const uint8_t* buf, size_t mask) {
    const __m256i a = _mm256_set1_epi8(3);
    __m256i acc[16];
    for (int i = 0; i < nchains; i++) acc[i] = _mm256_set1_epi32(i);
    size_t off = 0;
    const double t0 = now_s();
    for (long it = 0; it < iters; it++) {
        const uint8_t* q = buf + (off & mask);
        for (int i = 0; i < nchains; i++)
            acc[i] = _mm256_dpbusd_avx_epi32(acc[i], a,
                     _mm256_loadu_si256((const __m256i*)(q + (size_t)i * 32)));
        off += (size_t)nchains * 32;
    }
    const double dt = now_s() - t0;
    int s = 0;
    for (int i = 0; i < nchains; i++) s += _mm_extract_epi32(_mm256_castsi256_si128(acc[i]), 0);
    g_sink = s;
    return dt;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const double wall0 = now_s();
    const int N = g_N, K = g_K;

    double ghz, bw;
    ghz = clock_ghz(); bw = stream_gbs();   /* startup probe, not NPU work */
    const double bw_l1 = bench_read_bw(4096);
    const double bw_l2 = bench_read_bw(256u << 10);
    const double bw_l3 = bench_read_bw(2u << 20);
    sherlock_set_peaks(ghz * 32e9, bw * 1e9);
    sherlock_set_bw_ladder(bw_l1, bw_l2, bw_l3, bw);
    sherlock_enable_tracing(1);
    f16_lut_init();

    printf("Virtual NPU (adapted) - competition = RAW CPU\n");
    printf("AVX2=%d  AVX-VNNI(VEX-256)=%d\n", (int)cpu_avx2(), (int)cpu_avxvnni());
    printf("clock=%.2f GHz  compute peak~%.1f GMAC/s\n", ghz, ghz * 32.0);
    printf("bw ladder: L1 %.0f  L2 %.0f  L3 %.0f  DRAM %.0f GB/s\n\n",
           bw_l1, bw_l2, bw_l3, bw);

    std::vector<int8_t> w((size_t)N * K), a((size_t)K * 256);
    for (size_t i = 0; i < w.size(); i++) w[i] = (int8_t)((i * 131 + 7) % 255 - 127);
    for (size_t i = 0; i < a.size(); i++) a[i] = (int8_t)((i * 97 + 13) % 255 - 127);

    PackedI8 B; pack_i8(B, w, N, K);
    std::vector<float> c_ref((size_t)N * 256), c_t((size_t)N * 256);

    /* ---- correctness vs scalar oracle ---- */
    const int Mc = 3;
    gemm_i8_scalar(a.data(), Mc, K, B, c_ref.data());
    auto md = [&](void (*fn)(const int8_t*, int, int, const PackedI8&, float*)) {
        fn(a.data(), Mc, K, B, c_t.data());
        double m = 0;
        for (int i = 0; i < Mc * N; i++) {
            const double d = fabs((double)c_t[i] - (double)c_ref[i]);
            if (d > m) m = d;
        }
        return m;
    };
    printf("correctness vs scalar oracle (max abs diff):\n");
    printf("  rawcpu      : %.6g\n", md(gemm_i8_rawcpu));
    printf("  avx2        : %.6g\n", md(gemm_i8_avx2));
    if (cpu_avxvnni()) {
        printf("  vnni_blk    : %.6g\n", md(gemm_i8_vnni_blk));
        printf("  vnni_pan8   : %.6g\n", md(gemm_i8_vnni_pan8));
        printf("  rowblk MR=2 : %.6g\n", md(gemm_i8_vnni_rowblk<2>));
        printf("  rowblk MR=4 : %.6g\n", md(gemm_i8_vnni_rowblk<4>));
    }
    printf("\n");

    /* ---- VNNI issue ceiling: is 32 MAC/cycle actually reachable? ---------- */
    {
        const long IT = 2000000;
        const int chain[4] = { 1, 2, 4, 8 };
        printf("VNNI issue ceiling (register-only and L1-fed, independent chains)\n");
        printf("theoretical peak = ghz * 32 = %.1f GMAC/s\n", ghz * 32.0);
        printf("%-10s %-8s %12s %12s %10s\n", "source", "chains", "GMAC/s", "MAC/cycle", "vs peak");
        printf("--------------------------------------------------------------\n");
        for (int c = 0; c < 4; c++) {
            const int nc = chain[c];
            const double t = vnni_peak_reg(IT, nc);
            const double mac = (double)IT * nc * 32;
            const double g = mac / t / 1e9;
            printf("%-10s %-8d %12.1f %12.2f %9.0f%%\n", "registers", nc, g, g / ghz, 100.0 * g / (ghz * 32.0));
        }
        /* L1-resident operand: 4 KB walks in a loop, so it stays in L1 */
        std::vector<uint8_t> l1(4096);
        for (size_t i = 0; i < l1.size(); i++) l1[i] = (uint8_t)(i * 7 + 1);
        for (int c = 0; c < 4; c++) {
            const int nc = chain[c];
            const double t = vnni_peak_l1(IT / 2, nc, l1.data(), l1.size() - (size_t)nc * 32);
            const double mac = (double)(IT / 2) * nc * 32;
            const double g = mac / t / 1e9;
            printf("%-10s %-8d %12.1f %12.2f %9.0f%%\n", "L1", nc, g, g / ghz, 100.0 * g / (ghz * 32.0));
        }
        printf("\n");
    }

    /* ---- throughput ---- */
    const double macs = (double)N * K;
    const int Ms[] = { 1, 8, 32, 128 };
    printf("%-6s %-14s %10s %12s %12s\n", "M", "kernel", "ms", "GMAC/s", "vs rawcpu");
    printf("--------------------------------------------------------\n");

    for (int si = 0; si < 4; si++) {
        const int M = Ms[si];
        std::vector<float> C((size_t)M * N);
        const int reps = (M <= 8) ? 20 : (M <= 32 ? 8 : 4);
        const double bytes = (double)M * K + (double)N * K + (double)M * N * 4;

        double t_raw = timeit(reps, [&]{ gemm_i8_rawcpu(a.data(), M, K, B, C.data()); });
        printf("%-6d %-14s %10.3f %12.1f %11.2fx\n", M, "rawcpu", t_raw * 1e3, macs * M / t_raw / 1e9, 1.0);

        struct V { const char* nm; void (*fn)(const int8_t*, int, int, const PackedI8&, float*); };
        V vs[] = {
            { "avx2",      gemm_i8_avx2 },
            { "vnni_blk",  gemm_i8_vnni_blk },
            { "vnni_pan8", gemm_i8_vnni_pan8 },
        };
        for (int v = 0; v < 3; v++) {
            if (v && !cpu_avxvnni()) continue;
            const double t = timeit(reps, [&]{ vs[v].fn(a.data(), M, K, B, C.data()); });
            printf("%-6d %-14s %10.3f %12.1f %11.2fx\n", M, vs[v].nm, t * 1e3, macs * M / t / 1e9, t_raw / t);
            (void)bytes;
        }
        printf("\n");
    }

    /* ---- M-blocking: the prefill lever. Weight traffic per pass. ---- */
    {
        printf("M-blocking (prefill shaped): B is re-read once per row unless blocked\n");
        printf("%-6s %-14s %10s %12s %10s %9s %8s\n", "M", "kernel", "ms", "GMAC/s", "GB/s", "B/MAC", "vs MR=1");
        printf("----------------------------------------------------------------------------\n");
        const int Ms2[] = { 8, 32, 128 };
        for (int si = 0; si < 3; si++) {
            const int M = Ms2[si];
            std::vector<float> C((size_t)M * N);
            const int reps = (M <= 32) ? 8 : 4;
            const double macc = (double)M * N * K;

            double t_ref = timeit(reps, [&]{ gemm_i8_vnni_pan8(a.data(), M, K, B, C.data()); });
            double bytes = (double)M * N * K;
            printf("%-6d %-14s %10.3f %12.1f %10.1f %9.3f %7.2fx\n",
                   M, "pan8 (MR=1)", t_ref * 1e3, macc / t_ref / 1e9,
                   bytes / t_ref / 1e9, bytes / macc, 1.0);

            struct RB { const char* nm; int mr; void (*fn)(const int8_t*, int, int, const PackedI8&, float*); };
            RB rbs[] = {
                { "rowblk MR=1", 1, gemm_i8_vnni_rowblk<1> },
                { "rowblk MR=2", 2, gemm_i8_vnni_rowblk<2> },
                { "rowblk MR=4", 4, gemm_i8_vnni_rowblk<4> },
            };
            for (int v = 0; v < 3; v++) {
                const double t = timeit(reps, [&]{ rbs[v].fn(a.data(), M, K, B, C.data()); });
                const double nb = (double)((M + rbs[v].mr - 1) / rbs[v].mr) * N * K;
                printf("%-6d %-14s %10.3f %12.1f %10.1f %9.3f %7.2fx\n",
                       M, rbs[v].nm, t * 1e3, macc / t / 1e9, nb / t / 1e9,
                       nb / macc, t_ref / t);
            }
            printf("\n");
        }
    }

    /* ---- methodology: 3DNow! packed + SSE streaming + AltiVec dst ---- */
    {
        printf("methodology sweep - packed / streaming / reuse hints\n");
        printf("%-6s %-22s %10s %12s %9s\n", "M", "variant", "ms", "GMAC/s", "vs base");
        printf("--------------------------------------------------------------\n");
        typedef void (*kfn)(const int8_t*, int, int, const PackedI8&, float*);
        struct V { const char* nm; kfn fn; };
        V vs[] = {
            { "cached  + T0",   gemm_i8_vnni_pan8_s<ST_CACHED, PF_T0>   },
            { "cached  + NTA",  gemm_i8_vnni_pan8_s<ST_CACHED, PF_NTA>  },
            { "cached  + none", gemm_i8_vnni_pan8_s<ST_CACHED, PF_NONE> },
            { "stream  + T0",   gemm_i8_vnni_pan8_s<ST_STREAM, PF_T0>   },
            { "stream  + NTA",  gemm_i8_vnni_pan8_s<ST_STREAM, PF_NTA>  },
            { "stream  + none", gemm_i8_vnni_pan8_s<ST_STREAM, PF_NONE> },
        };
        const int Ms3[] = { 1, 128 };
        for (int si = 0; si < 2; si++) {
            const int M = Ms3[si];
            float* C = (float*)_aligned_malloc((size_t)M * N * 4, 32);
            memset(C, 0, (size_t)M * N * 4);
            const int reps = (M <= 1) ? 20 : 4;
            sherlock_mute(true);
            const double base = timeit(reps, [&]{ vs[0].fn(a.data(), M, K, B, C); });
            for (int v = 0; v < 6; v++) {
                const double t = timeit(reps, [&]{ vs[v].fn(a.data(), M, K, B, C); });
                printf("%-6d %-22s %10.3f %12.1f %8.2fx\n",
                       M, vs[v].nm, t * 1e3, (double)M * N * K / t / 1e9, base / t);
            }
            sherlock_mute(false);
            _aligned_free(C);
            printf("\n");
        }
    }

    /* ---- Cell-style staging tile: double-buffered local store ------------- */
    {
        printf("staging tile (Cell SPE model): L1-resident buffer, one tile in flight\n");
        typedef void (*kfn)(const int8_t*, int, int, const PackedI8&, float*);
        struct SV { const char* nm; kfn fn; };

        /* correctness at tile sizes that do not divide the K-group count */
        {
            const int Mc3 = 3;
            float* Cc = (float*)_aligned_malloc((size_t)Mc3 * N * 4, 32);
            sherlock_mute(true);
            auto maxdiff = [&](int tile_g) {
                gemm_i8_vnni_stage<STAGE_PREFETCH, PF_T0>(a.data(), Mc3, K, B, Cc, tile_g);
                double m = 0;
                for (int i = 0; i < Mc3 * N; i++) m = (std::max)(m, fabs((double)Cc[i] - (double)c_ref[i]));
                return m;
            };
            const double d97 = maxdiff(97);
            gemm_i8_vnni_stage<STAGE_COPY, PF_NONE>(a.data(), Mc3, K, B, Cc, 61);
            double dcp = 0;
            for (int i = 0; i < Mc3 * N; i++) dcp = (std::max)(dcp, fabs((double)Cc[i] - (double)c_ref[i]));
            sherlock_mute(false);
            _aligned_free(Cc);
            printf("  correctness vs oracle: prefetch tile97 diff=%.6g  copy tile61 diff=%.6g\n\n", d97, dcp);
        }

        SV sv[] = {
            { "stage pf tile64",   k_stage_pf64   },
            { "stage pf tile128",  k_stage_pf128  },
            { "stage pfNTA tile64",k_stage_pfnta64},
            { "stage copy tile64", k_stage_cp64   },
            { "stage copy tile32", k_stage_cp32   },
        };
        const int NSV = (int)(sizeof(sv) / sizeof(sv[0]));

        /* (a) the geometry used everywhere else: B fits well inside L3 */
        {
            printf("  B = %d x %d (%.1f MB), fits L3\n", N, K, (double)N * K / 1e6);
            printf("  %-6s %-20s %10s %12s %10s %9s\n", "M", "variant", "ms", "GMAC/s", "GB/s", "vs pan8");
            printf("  -----------------------------------------------------------------\n");
            const int Ms6[] = { 1, 128 };
            for (int si = 0; si < 2; si++) {
                const int M = Ms6[si];
                float* C = (float*)_aligned_malloc((size_t)M * N * 4, 32);
                const double macc = (double)M * N * K;
                const int reps = (M <= 1) ? 20 : 4;
                sherlock_mute(true);
                const double base = timeit(reps, [&]{ gemm_i8_vnni_pan8(a.data(), M, K, B, C); });
                sherlock_mute(false);
                printf("  %-6d %-20s %10.3f %12.1f %10.1f %8.2fx\n", M, "pan8 (baseline)",
                       base * 1e3, macc / base / 1e9, (double)N * K / base / 1e9, 1.0);
                for (int v = 0; v < NSV; v++) {
                    sherlock_mute(true);
                    const double t = timeit(reps, [&]{ sv[v].fn(a.data(), M, K, B, C); });
                    sherlock_mute(false);
                    printf("  %-6d %-20s %10.3f %12.1f %10.1f %8.2fx\n", M, sv[v].nm,
                           t * 1e3, macc / t / 1e9, (double)N * K / t / 1e9, base / t);
                }
                printf("\n");
                _aligned_free(C);
            }
        }

        /* (b) the regime the technique is actually for: B far larger than L3, so
         *     a decode pass cannot lean on cache and the read stream is the limit */
        {
            const int N2 = 8192, K2 = 4096;      /* 32 MB of int8 weights */
            printf("  B = %d x %d (%.1f MB), exceeds L3 (%s)\n",
                   N2, K2, (double)N2 * K2 / 1e6, "18 MB");
            std::vector<int8_t> w2((size_t)N2 * K2), a2((size_t)K2 * 256);
            for (size_t i = 0; i < w2.size(); i++) w2[i] = (int8_t)((i * 131 + 7) % 255 - 127);
            for (size_t i = 0; i < a2.size(); i++) a2[i] = (int8_t)((i * 97 + 13) % 255 - 127);
            PackedI8 B2; pack_i8(B2, w2, N2, K2);

            printf("  %-6s %-20s %10s %12s %10s %9s\n", "M", "variant", "ms", "GMAC/s", "GB/s", "vs pan8");
            printf("  -----------------------------------------------------------------\n");
            const int Ms7[] = { 1, 32 };
            for (int si = 0; si < 2; si++) {
                const int M = Ms7[si];
                float* C = (float*)_aligned_malloc((size_t)M * N2 * 4, 32);
                const double macc = (double)M * N2 * K2;
                const int reps = (M <= 1) ? 20 : 4;
                sherlock_mute(true);
                const double base = timeit(reps, [&]{ gemm_i8_vnni_pan8(a2.data(), M, K2, B2, C); });
                sherlock_mute(false);
                printf("  %-6d %-20s %10.3f %12.1f %10.1f %8.2fx\n", M, "pan8 (baseline)",
                       base * 1e3, macc / base / 1e9, (double)N2 * K2 / base / 1e9, 1.0);
                for (int v = 0; v < NSV; v++) {
                    sherlock_mute(true);
                    const double t = timeit(reps, [&]{ sv[v].fn(a2.data(), M, K2, B2, C); });
                    sherlock_mute(false);
                    printf("  %-6d %-20s %10.3f %12.1f %10.1f %8.2fx\n", M, sv[v].nm,
                           t * 1e3, macc / t / 1e9, (double)N2 * K2 / t / 1e9, base / t);
                }
                printf("\n");
                _aligned_free(C);
            }
        }
    }

    /* ---- partitioning across threads (mainframe CMT / DB2 partitioning) ---- */
    {
        const core_classes cc = core_scan();
        const unsigned hw = std::thread::hardware_concurrency();
        printf("partitioned across threads (persistent pool, N split into panel ranges)\n");
        printf("hardware threads: %u   P-cores: %d  E-cores: %d  LP-E: %d   large pages: %s\n",
               hw, cc.np, cc.ne, cc.nlpe,
               large_page_probe() ? "YES" : "no (needs SeLockMemoryPrivilege)");
        printf("workers pinned to P-cores: %s\n", cc.pmask ? "yes" : "no");
        printf("%-6s %-9s %10s %12s %9s %9s\n",
               "M", "workers", "ms", "GMAC/s", "speedup", "scaling");
        printf("------------------------------------------------------------------\n");
        const int Ms4[] = { 1, 128 };
        const int gang[] = { 1, 2, 4, 8, 14 };
        for (int si = 0; si < 2; si++) {
            const int M = Ms4[si];
            float* C = (float*)_aligned_malloc((size_t)M * N * 4, 32);
            const double macc = (double)M * N * K;
            const int reps = (M <= 1) ? 20 : 4;
            sherlock_mute(true);
            double t1 = 0;
            for (int k = 0; k < 5; k++) {
                pan8_pool pool;
                pool.A = a.data(); pool.M = M; pool.K = K; pool.B = &B; pool.C = C;
                pool.pin = (cc.pmask != 0);
                pool.build_masks(cc.pmask);
                pool.start(gang[k], gang[k]);
                const double t = timeit(reps, [&]{ pool.run(); });
                pool.stop();
                if (k == 0) t1 = t;
                printf("%-6d %-9d %10.3f %12.1f %9.2fx %8.2fx\n",
                       M, gang[k], t * 1e3, macc / t / 1e9, t1 / t, (t1 / t) / gang[k]);
            }
            sherlock_mute(false);
            _aligned_free(C);
            printf("\n");
        }
    }

    /* ---- NPU cluster: 1 worker per P-core, E-core governor ---- */
    {
        const core_classes cc2 = core_scan();
        const DWORD_PTR gov = cc2.efirst.empty() ? 0 : cc2.efirst[0];
        printf("NPU cluster: 1 worker per P-core, governor on an E-core\n");
        printf("P-core workers available: %d   E-core governor: %s\n",
               (int)cc2.pfirst.size(), gov ? "yes" : "no");
        printf("%-6s %-18s %9s %10s %12s %9s\n",
               "M", "unit split", "workers", "ms", "GMAC/s", "vs 1thr");
        printf("----------------------------------------------------------------------\n");
        const int Ms5[] = { 1, 128 };
        const int cfg[4][2] = { {1, 1}, {1, 4}, {4, 1}, {2, 2} };
        for (int si = 0; si < 2; si++) {
            const int M = Ms5[si];
            float* C = (float*)_aligned_malloc((size_t)M * N * 4, 32);
            const double macc = (double)M * N * K;
            const int reps = (M <= 1) ? 20 : 4;
            sherlock_mute(true);
            double t1 = 0;
            for (int c = 0; c < 4; c++) {
                npu_cluster cl;
                cl.A = a.data(); cl.M = M; cl.K = K; cl.B = &B; cl.C = C;
                cl.pcores = cc2.pfirst;
                cl.start(cfg[c][0], cfg[c][1], gov);
                const double t = timeit(reps, [&]{ cl.run(); });
                cl.stop();
                if (c == 0) t1 = t;
                char nm[32];
                snprintf(nm, sizeof(nm), "mblk=%d x part=%d", cfg[c][0], cfg[c][1]);
                printf("%-6d %-18s %9d %10.3f %12.1f %8.2fx\n",
                       M, nm, (int)cc2.pfirst.size(), t * 1e3, macc / t / 1e9, t1 / t);
            }
            sherlock_mute(false);
            _aligned_free(C);
            printf("\n");
        }
    }

    /* ---- heterogeneous work stealing: P-cores + E-cores + LP-E ---- */
    {
        const core_classes cc3 = core_scan();

        std::vector<DWORD_PTR> pv(cc3.pfirst.begin(), cc3.pfirst.end());
        std::vector<DWORD_PTR> ev(cc3.efirst.begin(), cc3.efirst.end());
        std::vector<DWORD_PTR> lv(cc3.lpefirst.begin(), cc3.lpefirst.end());
        /* Reserve one low-power core for the joining thread, so the thread that
         * spin-waits on the barrier can never steal a compute core. */
        DWORD_PTR reserved = 0;
        if (!lv.empty())      { reserved = lv.back(); lv.pop_back(); }
        else if (!ev.empty()) { reserved = ev.back(); ev.pop_back(); }
        else if (!pv.empty()) { reserved = pv.back(); pv.pop_back(); }

        std::vector<DWORD_PTR> pe = pv;
        pe.insert(pe.end(), ev.begin(), ev.end());
        std::vector<DWORD_PTR> pel = pe;
        pel.insert(pel.end(), lv.begin(), lv.end());

        printf("heterogeneous work stealing: fine tiles, one atomic cursor\n");
        printf("workers -- P: %d   E: %d   LP-E: %d   (joiner parked on 1 low-power core)\n",
               (int)pv.size(), (int)ev.size(), (int)lv.size());
        printf("%-6s %-16s %-11s %5s %9s %11s %10s %10s\n",
               "M", "unit split", "cores", "thr", "ms", "GMAC/s", "per-core", "vs P-only");
        printf("-----------------------------------------------------------------------------------------\n");

        const std::vector<DWORD_PTR> eonly = ev;
        const int granularity[3] = { 8, 16, 32 };   /* more tiles than workers */
        for (int Mi = 0; Mi < 2; Mi++) {
            const int M = Mi == 0 ? 1 : 128;
            float* C = (float*)_aligned_malloc((size_t)M * N * 4, 32);
            const double macc = (double)M * N * K;
            const int reps = (M <= 1) ? 50 : 4;

            for (int gi = 0; gi < 3; gi++) {
                /* Panel-split for both shapes. This matters more than it looks:
                 * splitting by N gives each worker a slice of B that stays in
                 * its PRIVATE L2; splitting by rows makes every worker stream
                 * the whole 4 MB weight matrix out of shared L3, which measured
                 * ~3x slower at M=128 (108 vs 300 GMAC/s at equal work). */
                const int nb  = 1;
                const int np_ = granularity[gi];
                const char* names[4] = { "P only", "E only", "P + E", "P + E + LP" };
                const std::vector<DWORD_PTR>* lst[4] = { &pv, &eonly, &pe, &pel };
                double base = 0.0;
                for (int ci = 0; ci < 4; ci++) {
                    hetero_pool hp;
                    hp.A = a.data(); hp.M = M; hp.K = K; hp.B = &B; hp.C = C;
                    hp.cores = *lst[ci];
                    sherlock_mute(true);
                    hp.start(nb, np_, reserved);
                    const double t = timeit(reps, [&]{ hp.run(); });
                    hp.stop();
                    sherlock_mute(false);
                    if (ci == 0) base = t;
                    const double gmac = macc / t / 1e9;
                    char nm[40];
                    snprintf(nm, sizeof(nm), "mblk=%d x part=%d", nb, np_);
                    printf("%-6d %-16s %-11s %5d %9.3f %11.1f %10.2f %9.2fx\n",
                           M, nm, names[ci], (int)lst[ci]->size(), t * 1e3, gmac,
                           gmac / (double)lst[ci]->size(), base / t);
                }
            }
            _aligned_free(C);
            printf("\n");
        }
    }

    /* ---- LiGGUF block-format shaders vs int8, decode shaped (M=1) ---- */
    {
        printf("LiGGUF block shaders vs int8, decode shaped (M=1, K=%d):\n", K);
        const int nb = K / 32;
        std::vector<blk_q8_0> av(nb);
        for (int i = 0; i < nb; i++) {
            av[i].d = (uint16_t)0x3C00;
            for (int j = 0; j < 32; j++) av[i].qs[j] = a[(size_t)i * 32 + j];
        }
        std::vector<blk_q8_0> w8((size_t)N * nb);
        std::vector<blk_q4_0> w4((size_t)N * nb);
        std::vector<blk_q5_0> w5((size_t)N * nb);
        for (int n = 0; n < N; n++)
            for (int i = 0; i < nb; i++) {
                w8[(size_t)n * nb + i].d = w4[(size_t)n * nb + i].d = w5[(size_t)n * nb + i].d = (uint16_t)0x3C00;
                uint32_t qh = 0;
                for (int j = 0; j < 16; j++) {
                    const int lo = (int)w[(size_t)n * K + i * 32 + j] & 0x0F;
                    const int hi = (int)w[(size_t)n * K + i * 32 + 16 + j] & 0x0F;
                    w4[(size_t)n * nb + i].qs[j] = (uint8_t)(lo | (hi << 4));
                    w5[(size_t)n * nb + i].qs[j] = (uint8_t)(lo | (hi << 4));
                }
                for (int j = 0; j < 32; j++) {
                    if (((int)w[(size_t)n * K + i * 32 + j] >> 4) & 1) qh |= (1u << j);
                    w8[(size_t)n * nb + i].qs[j] = w[(size_t)n * K + i * 32 + j];
                }
                w5[(size_t)n * nb + i].qh[0] = (uint8_t)(qh);
                w5[(size_t)n * nb + i].qh[1] = (uint8_t)(qh >> 8);
                w5[(size_t)n * nb + i].qh[2] = (uint8_t)(qh >> 16);
                w5[(size_t)n * nb + i].qh[3] = (uint8_t)(qh >> 24);
            }

        struct Fmt { const char* nm; double bpw; };
        Fmt fmts[] = { {"int8 (raw)", 1.0}, {"Q8_0", 34.0 / 32}, {"Q5_0", 22.0 / 32}, {"Q4_0", 18.0 / 32} };
        printf("%-12s %9s %10s %12s %10s %10s\n", "weights", "B/weight", "ms", "GMAC/s", "GB/s", "vs int8");
        printf("--------------------------------------------------------------------\n");
        const double macs1 = (double)N * K;
        const int reps = 20;
        double t_int8 = 0;

        for (int f = 0; f < 4; f++) {
            const double wbytes = macs1 * fmts[f].bpw;
            std::vector<float> C(N);
            double t = 0;
            if (f == 0) {
                /* fair baseline: the SIMD int8 path, not the scalar loop */
                t = timeit(reps, [&]{ gemm_i8_avx2(a.data(), 1, K, B, C.data()); });
                t_int8 = t;
            } else if (f == 1) {
                t = timeit(reps, [&]{ for (int n = 0; n < N; n++) C[n] = dot_q8_0_q8_0(av.data(), &w8[(size_t)n * nb], K); });
            } else if (f == 2) {
                t = timeit(reps, [&]{ for (int n = 0; n < N; n++) C[n] = dot_q8_0_q5_0(av.data(), &w5[(size_t)n * nb], K); });
            } else {
                t = timeit(reps, [&]{ for (int n = 0; n < N; n++) C[n] = dot_q8_0_q4_0(av.data(), &w4[(size_t)n * nb], K); });
            }
            const double gbs = (wbytes + K) / t / 1e9;
            printf("%-12s %9.4f %10.3f %12.1f %10.1f %9.2fx  %s\n",
                   fmts[f].nm, fmts[f].bpw, t * 1e3, macs1 / t / 1e9, gbs, t_int8 / t,
                   gbs / bw > 0.80 ? "MEMORY-bound" : "compute-bound");
        }
        printf("\n");
    }

    sherlock_impl::tr().total_time = now_s() - wall0;
    sherlock_dump_trace("vnpu-laptop");
    return 0;
}
