/*
 * bench_kernels.cpp - independent measurement harness for the virtual NPU.
 *
 * Purpose: compare every compute unit and ISA path the library exposes against
 * plain-CPU baselines that do NOT go through the runtime:
 *
 *   cpu_scalar    - naive m-n-k loop, vectorisation disabled (true scalar)
 *   cpu_autovec   - i-k-j loop written so the compiler vectorises it (-O3)
 *
 * Both baselines are timed with the same clock, warm-up and best-of-N policy as
 * the library paths, so the ratio is meaningful.
 *
 * Build: g++ -O2 -std=c++17 -Iinclude tools/bench_kernels.cpp \
 *            -Llib -lvnpu -Wl,-rpath,'$ORIGIN/../lib' -o bin/bench_kernels
 */
#include "vnpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ------------------------------------------------------------------------- */
/* timing                                                                    */
/* ------------------------------------------------------------------------- */

static double now_ms()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch()).count();
}

/* Best-of-N with warm-up.  Returns best milliseconds. */
template <typename F>
static double best_of(int iters, F fn)
{
    fn(); /* warm-up: page faults, cache fill, first dispatch */
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
/* plain-CPU baselines (no runtime)                                          */
/* ------------------------------------------------------------------------- */

#if defined(__GNUC__)
#  define NOVEC __attribute__((optimize("no-tree-vectorize")))
#else
#  define NOVEC
#endif

/* Textbook triple loop, exactly what a beginner writes. */
NOVEC
static void cpu_scalar_f32(const float *A, const float *B, float *C,
                           uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            C[(size_t)m * N + n] = acc;
        }
}

/* Same work, loop order that GCC/Clang auto-vectorise at -O3. */
static void cpu_autovec_f32(const float *A, const float *B, float *C,
                            uint32_t M, uint32_t K, uint32_t N)
{
    std::memset(C, 0, sizeof(float) * (size_t)M * N);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            const float a = A[(size_t)m * K + k];
            const float *brow = B + (size_t)k * N;
            float *crow = C + (size_t)m * N;
            for (uint32_t n = 0; n < N; ++n) crow[n] += a * brow[n];
        }
    }
}

NOVEC
static void cpu_scalar_i8(const int8_t *A, const int8_t *B, float *C,
                          uint32_t M, uint32_t K, uint32_t N,
                          const float *sa, const float *sb)
{
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; ++k)
                acc += (int32_t)A[(size_t)m * K + k] *
                       (int32_t)B[(size_t)k * N + n];
            C[(size_t)m * N + n] = (float)acc * sa[m] * sb[n];
        }
}

static void cpu_autovec_i8(const int8_t *A, const int8_t *B, float *C,
                           uint32_t M, uint32_t K, uint32_t N,
                           const float *sa, const float *sb)
{
    std::vector<int32_t> acc((size_t)M * N);
    std::memset(acc.data(), 0, sizeof(int32_t) * acc.size());
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t a = A[(size_t)m * K + k];
            const int8_t *brow = B + (size_t)k * N;
            int32_t *crow = acc.data() + (size_t)m * N;
            for (uint32_t n = 0; n < N; ++n)
                crow[n] += a * (int32_t)brow[n];
        }
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n)
            C[(size_t)m * N + n] =
                (float)acc[(size_t)m * N + n] * sa[m] * sb[n];
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

struct Shape { uint32_t M, K, N; };

static bool parse_shape(const char *s, Shape &out)
{
    return s && std::sscanf(s, "%ux%ux%u", &out.M, &out.K, &out.N) == 3 &&
           out.M && out.K && out.N;
}

static const char *unit_name(uint32_t u)
{
    switch (u) {
    case VNPU_UNIT_VOODOO:  return "voodoo";
    case VNPU_UNIT_WILDCAT: return "wildcat";
    case VNPU_UNIT_CELL:    return "cell";
    default:                return "auto";
    }
}

static float max_abs_diff(const std::vector<float> &a,
                          const std::vector<float> &b)
{
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

/* ------------------------------------------------------------------------- */
/* main                                                                      */
/* ------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    std::vector<Shape> shapes;
    std::vector<uint32_t> thread_list = {1, 2};
    std::vector<uint32_t> units = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                                   VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
    bool do_f32 = true, do_i8 = true;
    int iters = 5;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--shape") {
            Shape s{};
            if (!parse_shape(next(), s)) {
                std::fprintf(stderr, "bad shape\n");
                return 2;
            }
            shapes.push_back(s);
        } else if (a == "--threads") {
            thread_list.clear();
            const char *raw = next();
            std::string v = raw ? raw : "";
            size_t p = 0;
            while (p <= v.size()) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                if (q > p) thread_list.push_back(
                    (uint32_t)std::atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else if (a == "--iters") {
            iters = std::atoi(next());
        } else if (a == "--dtype") {
            const char *raw = next();
            std::string d = raw ? raw : "both";
            do_f32 = (d != "i8");
            do_i8 = (d != "f32");
        } else if (a == "--units") {
            units.clear();
            const char *raw = next();
            std::string d = raw ? raw : "all";
            if (d == "all") {
                units = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                         VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
            } else if (d == "auto") units = {VNPU_UNIT_AUTO};
            else if (d == "voodoo") units = {VNPU_UNIT_VOODOO};
            else if (d == "wildcat") units = {VNPU_UNIT_WILDCAT};
            else if (d == "cell") units = {VNPU_UNIT_CELL};
        } else if (a == "--list") {
            list_only = true;
        } else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--shape MxKxN]... [--threads 1,2] "
                        "[--iters N] [--dtype both|f32|i8] "
                        "[--units all|auto|voodoo|wildcat|cell] [--list]\n",
                        argv[0]);
            return 0;
        }
    }
    if (shapes.empty()) {
        shapes = {{64, 64, 64}, {128, 128, 128}, {256, 256, 256},
                  {512, 512, 512}, {1, 4096, 4096}, {32, 4096, 4096}};
    }

    if (list_only) {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.num_threads = 1;
        vnpu_device_t *dev = nullptr;
        vnpu_device_create(&cfg, &dev);
        vnpu_device_info_t info{};
        vnpu_device_get_info(dev, &info);
        std::printf("backend=%s isa_bits=%u threads=%u\n",
                    info.backend, info.simd_bits, info.num_threads);
        vnpu_device_destroy(dev);
        return 0;
    }

    std::printf("# virtual NPU - independent kernel matrix\n");
    std::printf("# baseline cpu_scalar = no vectorisation, "
                "cpu_autovec = compiler only (-O3)\n");
    std::printf("# all timings best-of-%d after one warm-up pass\n", iters);
    std::printf("%-12s %4s %-6s %-8s %10s %10s %8s %8s\n",
                "shape", "thr", "dtype", "path", "ms", "GFLOP/s",
                "vs scalar", "vs autovec");

    for (uint32_t threads : thread_list) {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.num_threads = threads;
        vnpu_device_t *dev = nullptr;
        vnpu_stream_t *stream = nullptr;
        if (vnpu_device_create(&cfg, &dev) != VNPU_OK ||
            vnpu_stream_create(dev, 0, &stream) != VNPU_OK) {
            std::fprintf(stderr, "device create failed\n");
            return 1;
        }
        vnpu_device_info_t info{};
        vnpu_device_get_info(dev, &info);
        std::printf("# device backend=%s simd_bits=%u threads=%u\n",
                    info.backend, info.simd_bits, info.num_threads);

        for (const Shape &s : shapes) {
            const uint32_t M = s.M, K = s.K, N = s.N;
            char label[64];
            std::snprintf(label, sizeof(label), "%ux%ux%u", M, K, N);
            const double flops = 2.0 * M * K * N;

            std::vector<float> A((size_t)M * K), B((size_t)K * N),
                               C((size_t)M * N), Cref((size_t)M * N);
            fill_f32(A, 7);
            fill_f32(B, 11);

            if (do_f32) {
                double t_scalar = best_of(iters, [&] {
                    cpu_scalar_f32(A.data(), B.data(), Cref.data(), M, K, N);
                });
                double t_auto = best_of(iters, [&] {
                    cpu_autovec_f32(A.data(), B.data(), Cref.data(), M, K, N);
                });
                std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8s %8s\n",
                            label, threads, "f32", "cpu_scal",
                            t_scalar, flops / (t_scalar * 1e6), "1.00", "-");
                std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8.2f %8s\n",
                            label, threads, "f32", "cpu_auto",
                            t_auto, flops / (t_auto * 1e6),
                            t_scalar / t_auto, "1.00");

                for (uint32_t unit : units) {
                    vnpu_matmul_f32_args_t args{};
                    args.A = A.data();
                    args.B = B.data();
                    args.C = C.data();
                    args.M = M; args.K = K; args.N = N;
                    args.alpha = 1.0f; args.beta = 0.0f;
                    args.unit = unit;
                    vnpu_status_t rc = VNPU_OK;
                    double t = best_of(iters, [&] {
                        rc = vnpu_matmul_f32(stream, &args);
                    });
                    if (rc != VNPU_OK) {
                        std::printf("%-12s %4u %-6s %-8s %10s %10s %8s %8s\n",
                                    label, threads, "f32", unit_name(unit),
                                    "ERR", "-", "-", "-");
                        continue;
                    }
                    float err = max_abs_diff(C, Cref);
                    double g = flops / (t * 1e6);
                    std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8.2f "
                                "%8.2f  err=%.3g\n",
                                label, threads, "f32", unit_name(unit), t, g,
                                t_scalar / t, t_auto / t, err);
                }
            }

            if (do_i8) {
                std::vector<int8_t> A8(A.size()), B8(B.size());
                std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
                for (size_t i = 0; i < A.size(); ++i)
                    A8[i] = (int8_t)std::lround(std::max(-127.0f,
                              std::min(127.0f, A[i] * 97.0f)));
                for (size_t i = 0; i < B.size(); ++i)
                    B8[i] = (int8_t)std::lround(std::max(-127.0f,
                              std::min(127.0f, B[i] * 97.0f)));

                vnpu_packed_weights_t *W = nullptr;
                if (vnpu_pack_weights_i8(dev, B8.data(), sb.data(), K, N,
                                         0, 0, &W) != VNPU_OK) {
                    std::fprintf(stderr, "pack failed %s\n", label);
                    continue;
                }

                double t_scalar = best_of(iters, [&] {
                    cpu_scalar_i8(A8.data(), B8.data(), Cref.data(),
                                  M, K, N, sa.data(), sb.data());
                });
                double t_auto = best_of(iters, [&] {
                    cpu_autovec_i8(A8.data(), B8.data(), Cref.data(),
                                   M, K, N, sa.data(), sb.data());
                });
                std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8s %8s\n",
                            label, threads, "i8", "cpu_scal", t_scalar,
                            flops / (t_scalar * 1e6), "1.00", "-");
                std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8.2f %8s\n",
                            label, threads, "i8", "cpu_auto", t_auto,
                            flops / (t_auto * 1e6), t_scalar / t_auto,
                            "1.00");

                for (uint32_t unit : units) {
                    vnpu_matmul_i8_args_t args{};
                    args.A = A8.data();
                    args.B = W;
                    args.scale_a = sa.data();
                    args.C = C.data();
                    args.M = M; args.K = K; args.N = N;
                    args.unit = unit;
                    vnpu_status_t rc = VNPU_OK;
                    double t = best_of(iters, [&] {
                        rc = vnpu_matmul_i8(stream, &args);
                    });
                    if (rc != VNPU_OK) {
                        std::printf("%-12s %4u %-6s %-8s %10s %10s %8s %8s\n",
                                    label, threads, "i8", unit_name(unit),
                                    "ERR", "-", "-", "-");
                        continue;
                    }
                    float err = max_abs_diff(C, Cref);
                    double g = flops / (t * 1e6);
                    std::printf("%-12s %4u %-6s %-8s %10.4f %10.2f %8.2f "
                                "%8.2f  err=%.3g\n",
                                label, threads, "i8", unit_name(unit), t, g,
                                t_scalar / t, t_auto / t, err);
                }
                vnpu_packed_weights_destroy(W);
            }
        }
        vnpu_stream_destroy(stream);
        vnpu_device_destroy(dev);
    }
    return 0;
}
