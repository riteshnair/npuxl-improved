/* What is the FMA ceiling on this machine?
 *
 * `bench` reports GFLOP/s, but a GFLOP/s figure with no denominator says
 * nothing about how much of the CPU we are actually using.  This probe
 * measures the denominator three ways:
 *
 *   reg   - 12 independent FMA chains, no loads at all.  Pure port throughput.
 *   mix   - the exact instruction mix of gemm_f32_avx2_nn_packed_tile:
 *           3 broadcasts + 4 B loads + 12 FMAs per k, reading a 32 KiB panel.
 *   copy  - memcpy-bandwidth bound, for the packing overhead estimate.
 *
 * Run with 1 and 2 threads: the second number is the ceiling for a "dual NPU"
 * that is really two cores.
 */
#include <immintrin.h>

#include <algorithm>
#include <chrono>

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch())
        .count();
}

__attribute__((target("avx2,fma"), noinline))
static float reg_only(long iters) {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0,
           a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
    const __m256 b = _mm256_set1_ps(0.999999f);
    const __m256 c = _mm256_set1_ps(1e-7f);
    for (long i = 0; i < iters; ++i) {
        a0 = _mm256_fmadd_ps(a0, b, c);
        a1 = _mm256_fmadd_ps(a1, b, c);
        a2 = _mm256_fmadd_ps(a2, b, c);
        a3 = _mm256_fmadd_ps(a3, b, c);
        a4 = _mm256_fmadd_ps(a4, b, c);
        a5 = _mm256_fmadd_ps(a5, b, c);
        a6 = _mm256_fmadd_ps(a6, b, c);
        a7 = _mm256_fmadd_ps(a7, b, c);
        a8 = _mm256_fmadd_ps(a8, b, c);
        a9 = _mm256_fmadd_ps(a9, b, c);
        a10 = _mm256_fmadd_ps(a10, b, c);
        a11 = _mm256_fmadd_ps(a11, b, c);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(a4, a5),
                                       _mm256_add_ps(a6, a7)));
    s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(a8, a9),
                                       _mm256_add_ps(a10, a11)));
    float out[8];
    _mm256_storeu_ps(out, s);
    float t = 0.0f;
    for (float v : out) t += v;
    return t;
}

/* Same 3x32 register tile as the real kernel, same 16-register budget. */
__attribute__((target("avx2,fma"), noinline))
static float mix_shape(long iters, const float *panel, const float *a0,
                       const float *a1, const float *a2, uint32_t tn) {
    float sink = 0.0f;
    for (long it = 0; it < iters; ++it) {
        __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
        __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10, c13 = c10;
        __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20, c23 = c20;
        for (uint32_t k = 0; k < 128; ++k) {
            const float *brow = panel + static_cast<size_t>(k) * tn;
            const __m256 va0 = _mm256_set1_ps(a0[k]);
            const __m256 va1 = _mm256_set1_ps(a1[k]);
            const __m256 va2 = _mm256_set1_ps(a2[k]);
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
        __m256 s = _mm256_add_ps(_mm256_add_ps(c00, c01),
                                 _mm256_add_ps(c02, c03));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c10, c11),
                                           _mm256_add_ps(c12, c13)));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c20, c21),
                                           _mm256_add_ps(c22, c23)));
        float out[8];
        _mm256_storeu_ps(out, s);
        for (float v : out) sink += v;
    }
    return sink;
}

/* Same as mix_shape but with the three A broadcasts hoisted out of the k loop:
 * isolates the cost of the broadcast uops from the cost of the B loads. */
__attribute__((target("avx2,fma"), noinline))
static float mix_nobcast(long iters, const float *panel, const float *a0,
                          uint32_t tn) {
    const __m256 va0 = _mm256_set1_ps(a0[0]);
    const __m256 va1 = _mm256_set1_ps(a0[1]);
    const __m256 va2 = _mm256_set1_ps(a0[2]);
    float sink = 0.0f;
    for (long it = 0; it < iters; ++it) {
        __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
        __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10, c13 = c10;
        __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20, c23 = c20;
        for (uint32_t k = 0; k < 128; ++k) {
            const float *brow = panel + static_cast<size_t>(k) * tn;
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
        __m256 s = _mm256_add_ps(_mm256_add_ps(c00, c01),
                                 _mm256_add_ps(c02, c03));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c10, c11),
                                           _mm256_add_ps(c12, c13)));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c20, c21),
                                           _mm256_add_ps(c22, c23)));
        float out[8];
        _mm256_storeu_ps(out, s);
        for (float v : out) sink += v;
    }
    return sink;
}

/* 3 broadcasts + 1 load: the B vector is reused by all four column blocks, so
 * this is the same FMA count with a quarter of the load traffic. */
__attribute__((target("avx2,fma"), noinline))
static float mix_oneload(long iters, const float *panel, const float *a0,
                          uint32_t tn) {
    float sink = 0.0f;
    for (long it = 0; it < iters; ++it) {
        __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
        __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10, c13 = c10;
        __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20, c23 = c20;
        for (uint32_t k = 0; k < 128; ++k) {
            const float *brow = panel + static_cast<size_t>(k) * tn;
            const __m256 va0 = _mm256_set1_ps(a0[k]);
            const __m256 va1 = _mm256_set1_ps(a0[(k + 1) & 127]);
            const __m256 va2 = _mm256_set1_ps(a0[(k + 2) & 127]);
            const __m256 bv = _mm256_loadu_ps(brow + 0);
            c00 = _mm256_fmadd_ps(va0, bv, c00);
            c10 = _mm256_fmadd_ps(va1, bv, c10);
            c20 = _mm256_fmadd_ps(va2, bv, c20);
            c01 = _mm256_fmadd_ps(va0, bv, c01);
            c11 = _mm256_fmadd_ps(va1, bv, c11);
            c21 = _mm256_fmadd_ps(va2, bv, c21);
            c02 = _mm256_fmadd_ps(va0, bv, c02);
            c12 = _mm256_fmadd_ps(va1, bv, c12);
            c22 = _mm256_fmadd_ps(va2, bv, c22);
            c03 = _mm256_fmadd_ps(va0, bv, c03);
            c13 = _mm256_fmadd_ps(va1, bv, c13);
            c23 = _mm256_fmadd_ps(va2, bv, c23);
        }
        __m256 s = _mm256_add_ps(_mm256_add_ps(c00, c01),
                                 _mm256_add_ps(c02, c03));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c10, c11),
                                           _mm256_add_ps(c12, c13)));
        s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(c20, c21),
                                           _mm256_add_ps(c22, c23)));
        float out[8];
        _mm256_storeu_ps(out, s);
        for (float v : out) sink += v;
    }
    return sink;
}

/* Streaming read bandwidth: the ceiling a GEMV runs into. 64 MiB so it cannot
 * live in L3 on the reference box (32 MiB). */
__attribute__((target("avx2,fma"), noinline))
static double stream_read_gbs(size_t bytes)
{
    std::vector<float> buf(bytes / sizeof(float), 1.0f);
    const size_t n = bytes / 32;
    const float *p = buf.data();
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now_ms();
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t i = 0; i < n; i += 4) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i * 8));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i * 8 + 8));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i * 8 + 16));
            a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i * 8 + 24));
        }
        const double ms = now_ms() - t0;
        __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        float out[8];
        _mm256_storeu_ps(out, s);
        if (out[0] == 12345.0f) printf(" ");
        best = std::min(best, ms);
    }
    return static_cast<double>(bytes) / (best * 1e6);
}

static double best_of_3(int threads, int kind, long iters, float *sink) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<std::thread> th;
        std::vector<float> part(static_cast<size_t>(threads));
        const double t0 = now_ms();
        for (int i = 0; i < threads; ++i) {
            th.emplace_back([&, i]() {
                static thread_local std::vector<float> panel(128 * 64, 1e-3f);
                static thread_local std::vector<float> av(3 * 128, 1.0f);
                if (kind == 0)
                    part[i] = reg_only(iters);
                else if (kind == 1)
                    part[i] = mix_shape(iters, panel.data(), av.data(),
                                        av.data() + 128, av.data() + 256, 64);
                else if (kind == 2)
                    part[i] = mix_nobcast(iters, panel.data(), av.data(), 64);
                else
                    part[i] = mix_oneload(iters, panel.data(), av.data(), 64);
            });
        }
        for (auto &t : th) t.join();
        const double dt = now_ms() - t0;
        for (float v : part) *sink += v;
        best = std::min(best, dt);
    }
    return best;
}

int main() {
    printf("%-6s %-10s %10s %12s\n", "threads", "kind", "ms", "GFLOP/s");
    for (int threads = 1; threads <= 2; ++threads) {
        float sink = 0.0f;
        const long iters = 400000;
        for (int kind = 0; kind < 4; ++kind) {
            const long it = (kind == 0) ? iters : iters / 64;
            const double ms = best_of_3(threads, kind, it, &sink);
            const double kmul = (kind == 0) ? 12.0 : 128.0 * 12.0;
            const double flops = static_cast<double>(threads) * it * kmul * 8 * 2;
            static const char *names[4] = {"reg", "mix", "mix-nobcast",
                                           "mix-1load"};
            printf("%-6d %-12s %10.3f %12.1f\n", threads, names[kind], ms,
                   flops / ms / 1e6);
        }
        if (sink == 12345.0f) printf(" ");
    }
    printf("%-6s %-12s %10s %12s\n", "-", "stream-read", "64 MiB", "GB/s");
    printf("%-6d %-12s %10s %12.1f\n", 1, "stream-read", "-",
           stream_read_gbs(64ull << 20));
    return 0;
}
