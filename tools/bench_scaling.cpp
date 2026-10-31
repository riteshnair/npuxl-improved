/*
 * bench_scaling.cpp - what limits the virtual NPU, and what closes the gap.
 *
 * Three measurements:
 *   1. tile geometry sweep  -> how much tiling is worth right now
 *   2. M curve              -> how much batching is worth (decode -> prefill)
 *   3. instruction ceiling  -> FMA / int-MAC peak of this core, so kernel
 *                              efficiency can be stated as a fraction of peak
 *
 * Build: g++ -O2 -std=c++17 -Iinclude tools/bench_scaling.cpp -Llib -lvnpu \
 *            -Wl,-rpath,'$ORIGIN/../lib' -o bin/bench_scaling
 */
#include "vnpu.h"

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch()).count();
}

template <typename F>
static double best_of(int iters, F fn)
{
    fn();
    double best = 1e300;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        fn();
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* ------------------------------------------------------------------------- */
/* instruction ceilings                                                      */
/* ------------------------------------------------------------------------- */

__attribute__((target("avx2,fma"), noinline))
static double peak_fma(long iters)
{
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = a0, a2 = a0, a3 = a0;
    __m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0;
    __m256 b = _mm256_set1_ps(0.999f), c = _mm256_set1_ps(0.001f);
    double t0 = now_ms();
    for (long i = 0; i < iters; ++i) {
        a0 = _mm256_fmadd_ps(a0, b, c); a1 = _mm256_fmadd_ps(a1, b, c);
        a2 = _mm256_fmadd_ps(a2, b, c); a3 = _mm256_fmadd_ps(a3, b, c);
        a4 = _mm256_fmadd_ps(a4, b, c); a5 = _mm256_fmadd_ps(a5, b, c);
        a6 = _mm256_fmadd_ps(a6, b, c); a7 = _mm256_fmadd_ps(a7, b, c);
        a8 = _mm256_fmadd_ps(a8, b, c); a9 = _mm256_fmadd_ps(a9, b, c);
    }
    double ms = now_ms() - t0;
    float sink[8];
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(a4, a5),
                                       _mm256_add_ps(_mm256_add_ps(a6, a7),
                                                     _mm256_add_ps(a8, a9))));
    _mm256_storeu_ps(sink, s);
    volatile float keep = sink[0];
    (void)keep;
    /* 10 lanes x 8 floats x 2 flop x iters */
    return (double)iters * 10.0 * 8.0 * 2.0 / (ms * 1e6);
}

__attribute__((target("avx2"), noinline))
static double peak_madd(long iters)
{
    __m256i a0 = _mm256_set1_epi16(1), a1 = a0, a2 = a0, a3 = a0;
    __m256i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    __m256i b = _mm256_set1_epi16(1);
    double t0 = now_ms();
    for (long i = 0; i < iters; ++i) {
        a0 = _mm256_madd_epi16(a0, b); a1 = _mm256_madd_epi16(a1, b);
        a2 = _mm256_madd_epi16(a2, b); a3 = _mm256_madd_epi16(a3, b);
        a4 = _mm256_madd_epi16(a4, b); a5 = _mm256_madd_epi16(a5, b);
        a6 = _mm256_madd_epi16(a6, b); a7 = _mm256_madd_epi16(a7, b);
    }
    double ms = now_ms() - t0;
    int32_t sink[8];
    __m256i s = _mm256_add_epi32(_mm256_add_epi32(a0, a1),
                                 _mm256_add_epi32(a2, a3));
    s = _mm256_add_epi32(s, _mm256_add_epi32(_mm256_add_epi32(a4, a5),
                                             _mm256_add_epi32(a6, a7)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(sink), s);
    volatile int32_t keep = sink[0];
    (void)keep;
    /* 8 lanes x 8 int32 x 2 MACs per lane per instr x iters */
    return (double)iters * 8.0 * 8.0 * 2.0 / (ms * 1e6);
}

/* ------------------------------------------------------------------------- */
/* helpers                                                                   */
/* ------------------------------------------------------------------------- */

static void fill_f32(std::vector<float> &v, uint32_t seed)
{
    uint32_t s = seed * 2654435761u + 1u;
    for (auto &x : v) {
        s = s * 1664525u + 1013904223u;
        x = ((float)(int32_t)(s >> 8) / 8388608.0f) - 0.25f;
    }
}

struct Ctx {
    vnpu_device_t *dev = nullptr;
    vnpu_stream_t *stream = nullptr;
};

static bool make_ctx(uint32_t threads, Ctx &c)
{
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.num_threads = threads;
    return vnpu_device_create(&cfg, &c.dev) == VNPU_OK &&
           vnpu_stream_create(c.dev, 0, &c.stream) == VNPU_OK;
}

int main(int argc, char **argv)
{
    int iters = 5;
    uint32_t threads = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = (uint32_t)std::atoi(argv[++i]);
    }

    std::printf("# ceilings on this core (register-only loops)\n");
    double pk_fma = peak_fma(2000000);
    double pk_mad = peak_madd(2000000);
    std::printf("#   f32 FMA peak      : %.1f GFLOP/s\n", pk_fma);
    std::printf("#   int16 MAC peak    : %.1f GMAC/s\n", pk_mad);

    Ctx c;
    if (!make_ctx(threads, c)) {
        std::fprintf(stderr, "device create failed\n");
        return 1;
    }
    vnpu_device_info_t info{};
    vnpu_device_get_info(c.dev, &info);

    /* ---- 1. tile geometry sweep --------------------------------------- */
    struct Case { uint32_t M, K, N; const char *dtype; };
    const Case cases[] = {
        {512, 512, 512, "f32"},
        {512, 512, 512, "i8"},
        {32, 4096, 4096, "i8"},
    };
    const uint32_t tms[] = {16, 32, 64, 128};
    const uint32_t tks[] = {32, 64, 128, 256};
    const uint32_t tns[] = {32, 64, 128};

    std::printf("\n# 1. tile sweep (threads=%u)\n", threads);
    std::printf("%-6s %-12s %-18s %10s %10s\n", "dtype", "shape",
                "tile m/k/n", "ms", "GFLOP/s");
    for (const Case &cs : cases) {
        std::vector<float> A((size_t)cs.M * cs.K), B((size_t)cs.K * cs.N),
                           C((size_t)cs.M * cs.N);
        fill_f32(A, 3);
        fill_f32(B, 5);
        std::vector<int8_t> A8(A.size()), B8(B.size());
        std::vector<float> sa(cs.M, 1.0f), sb(cs.N, 1.0f);
        for (size_t i = 0; i < A.size(); ++i)
            A8[i] = (int8_t)std::max(-127, std::min(127,
                        (int)std::lround(A[i] * 97.0f)));
        for (size_t i = 0; i < B.size(); ++i)
            B8[i] = (int8_t)std::max(-127, std::min(127,
                        (int)std::lround(B[i] * 97.0f)));
        vnpu_packed_weights_t *W = nullptr;
        if (!std::strcmp(cs.dtype, "i8"))
            vnpu_pack_weights_i8(c.dev, B8.data(), sb.data(), cs.K, cs.N,
                                 0, 0, &W);
        const double flops = 2.0 * cs.M * cs.K * cs.N;
        double best = 1e300;
        char best_cfg[32] = "";
        for (uint32_t tm : tms)
            for (uint32_t tk : tks)
                for (uint32_t tn : tns) {
                    vnpu_device_set_tiles(c.dev, tm, tk, tn);
                    double t;
                    if (!std::strcmp(cs.dtype, "f32")) {
                        vnpu_matmul_f32_args_t a{};
                        a.A = A.data(); a.B = B.data(); a.C = C.data();
                        a.M = cs.M; a.K = cs.K; a.N = cs.N;
                        a.alpha = 1.0f; a.unit = VNPU_UNIT_VOODOO;
                        t = best_of(iters, [&] {
                            vnpu_matmul_f32(c.stream, &a);
                        });
                    } else {
                        vnpu_matmul_i8_args_t a{};
                        a.A = A8.data(); a.B = W; a.C = C.data();
                        a.scale_a = sa.data();
                        a.M = cs.M; a.K = cs.K; a.N = cs.N;
                        a.unit = VNPU_UNIT_VOODOO;
                        t = best_of(iters, [&] {
                            vnpu_matmul_i8(c.stream, &a);
                        });
                    }
                    if (t < best) {
                        best = t;
                        std::snprintf(best_cfg, sizeof(best_cfg), "%u/%u/%u",
                                      tm, tk, tn);
                    }
                }
        std::printf("%-6s %ux%ux%-6u %-18s %10.4f %10.2f\n", cs.dtype,
                    cs.M, cs.K, cs.N, best_cfg, best, flops / (best * 1e6));
        if (W) vnpu_packed_weights_destroy(W);
    }
    vnpu_device_set_tiles(c.dev, 64, 128, 64);

    /* ---- 2. M curve (batching) ---------------------------------------- */
    std::printf("\n# 2. batching curve, K=N=2048, i8 and f32 (threads=%u)\n",
                threads);
    std::printf("%6s %12s %12s\n", "M", "i8 GFLOP/s", "f32 GFLOP/s");
    const uint32_t K = 2048, N = 2048;
    const uint32_t ms[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (uint32_t M : ms) {
        std::vector<float> A((size_t)M * K), B((size_t)K * N),
                           C((size_t)M * N);
        fill_f32(A, 7);
        fill_f32(B, 9);
        std::vector<int8_t> A8(A.size()), B8(B.size());
        std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
        for (size_t i = 0; i < A.size(); ++i)
            A8[i] = (int8_t)std::max(-127, std::min(127,
                        (int)std::lround(A[i] * 97.0f)));
        for (size_t i = 0; i < B.size(); ++i)
            B8[i] = (int8_t)std::max(-127, std::min(127,
                        (int)std::lround(B[i] * 97.0f)));
        vnpu_packed_weights_t *W = nullptr;
        vnpu_pack_weights_i8(c.dev, B8.data(), sb.data(), K, N, 0, 0, &W);
        const double flops = 2.0 * M * K * N;

        vnpu_matmul_i8_args_t ai{};
        ai.A = A8.data(); ai.B = W; ai.C = C.data();
        ai.scale_a = sa.data(); ai.M = M; ai.K = K; ai.N = N;
        ai.unit = VNPU_UNIT_VOODOO;
        double ti = best_of(iters, [&] { vnpu_matmul_i8(c.stream, &ai); });

        vnpu_matmul_f32_args_t af{};
        af.A = A.data(); af.B = B.data(); af.C = C.data();
        af.M = M; af.K = K; af.N = N; af.alpha = 1.0f;
        af.unit = VNPU_UNIT_VOODOO;
        double tf = best_of(iters, [&] { vnpu_matmul_f32(c.stream, &af); });

        std::printf("%6u %12.2f %12.2f\n", M, flops / (ti * 1e6),
                    flops / (tf * 1e6));
        vnpu_packed_weights_destroy(W);
    }

    /* ---- 3. per-command overhead -------------------------------------- */
    std::printf("\n# 3. fixed per-command cost (stream submit + sync)\n");
    {
        std::vector<float> A(16 * 16), B(16 * 16), C(16 * 16, 0.0f);
        fill_f32(A, 1);
        fill_f32(B, 2);
        vnpu_matmul_f32_args_t a{};
        a.A = A.data(); a.B = B.data(); a.C = C.data();
        a.M = 16; a.K = 16; a.N = 16; a.alpha = 1.0f;
        a.unit = VNPU_UNIT_VOODOO;
        double t = best_of(20, [&] { vnpu_matmul_f32(c.stream, &a); });
        std::printf("#   16x16x16 GEMM (8 K MACs per output): %.4f ms "
                    "= %.1f us/command\n", t, t * 1000.0);
    }

    std::printf("\n# backend=%s simd_bits=%u threads=%u\n", info.backend,
                info.simd_bits, info.num_threads);
    vnpu_stream_destroy(c.stream);
    vnpu_device_destroy(c.dev);
    return 0;
}
