/*
 * vnpu-cli - info, correctness, benchmarking, autotuning, tracing and serving.
 */
#include "vnpu.h"
#include "vnpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

int vnpu_run_selftest(const char *filter);
int vnpu_run_cluster_demo(int argc, char **argv);
int vnpu_run_specdemo(int argc, char **argv);
int vnpu_run_llm_bench(int argc, char **argv);

namespace {

struct Shape {
    uint32_t M = 0, K = 0, N = 0;
    bool valid() const { return M && K && N; }
    std::string str() const {
        return std::to_string(M) + "x" + std::to_string(K) + "x" +
               std::to_string(N);
    }
};

/* A flat per-dimension cap rejected the shapes we care about most while still
 * admitting bad ones: it refused 1x64x262144, a decode row whose matrices are a
 * few hundred megabytes in total, and accepted 16384x16384x16384, which needs
 * several gigabytes.  Bound the f32 working set instead: A, B, C and the scalar
 * baseline copy. */
static bool shape_fits_memory(const Shape &s)
{
    const uint64_t mk = static_cast<uint64_t>(s.M) * s.K;
    const uint64_t kn = static_cast<uint64_t>(s.K) * s.N;
    const uint64_t mn = static_cast<uint64_t>(s.M) * s.N;
    /* Sample checks index with 32-bit products, and so do several kernels. */
    if (mk > 0xffffffffull || kn > 0xffffffffull || mn > 0xffffffffull)
        return false;
    const uint64_t bytes = sizeof(float) * (mk + kn + 2 * mn);
    return bytes <= (1ull << 30);
}

Shape parse_shape(const char *s)
{
    Shape out;
    if (!s) return out;
    char x = 0;
    if (std::sscanf(s, "%u%c%u%c%u", &out.M, &x, &out.K, &x, &out.N) != 5)
        return Shape{};
    if (!shape_fits_memory(out)) return Shape{};
    return out;
}

uint32_t parse_unit(const std::string &s)
{
    if (s == "voodoo" || s == "1") return VNPU_UNIT_VOODOO;
    if (s == "wildcat" || s == "2") return VNPU_UNIT_WILDCAT;
    if (s == "cell" || s == "3") return VNPU_UNIT_CELL;
    if (s == "all" || s == "255") return 255;
    return VNPU_UNIT_AUTO;
}

std::string unit_slug(uint32_t unit)
{
    switch (unit) {
    case VNPU_UNIT_VOODOO: return "voodoo";
    case VNPU_UNIT_WILDCAT: return "wildcat";
    case VNPU_UNIT_CELL: return "cell";
    default: return "auto";
    }
}

double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

uint32_t rng_state = 0x12345678u;
uint32_t xs()
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}
float rndf()
{
    return static_cast<float>(static_cast<int>(xs() % 2000) - 1000) /
           1000.0f;
}

void fill(std::vector<float> &v)
{
    for (auto &x : v) x = rndf();
}

struct BenchResult {
    std::string shape;
    std::string dtype;
    std::string unit;
    uint32_t threads = 0;
    uint32_t iterations = 0;
    double ms = 0.0;
    double gflops = 0.0;
    double baseline_ms = 0.0;      /* faster plain-C loop (no intrinsics) */
    double baseline_scalar_ms = 0.0;
    double baseline_compiler_ms = 0.0;
    double speedup = 0.0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t flops = 0;
    uint64_t tiles_dispatched = 0;
    uint64_t tiles_culled = 0;
    int correct = 0;
};

float reference_dot(const std::vector<float> &A,
                    const std::vector<float> &B,
                    uint32_t M, uint32_t K, uint32_t N,
                    uint32_t m, uint32_t n)
{
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; ++k)
        acc += A[static_cast<size_t>(m) * K + k] *
               B[static_cast<size_t>(k) * N + n];
    return acc;
}

bool check_samples(const std::vector<float> &C,
                   const std::vector<float> &A,
                   const std::vector<float> &B,
                   uint32_t M, uint32_t K, uint32_t N,
                   float atol, float rtol)
{
    uint32_t samples = std::min<uint32_t>(16, M * N);
    for (uint32_t s = 0; s < samples; ++s) {
        uint32_t idx = xs() % (M * N);
        uint32_t m = idx / N;
        uint32_t n = idx % N;
        float expected = reference_dot(A, B, M, K, N, m, n);
        float got = C[idx];
        if (std::fabs(got - expected) >
            atol + rtol * std::max(std::fabs(got), std::fabs(expected)))
            return false;
    }
    return true;
}

/* Keeps the baseline loops from being optimised away without putting a
 * volatile store inside the measured loop. */
volatile float g_bench_sink = 0.0f;

/* Naive scalar triple loop: no intrinsics, no blocking, the textbook m-n-k
 * order with B walked by rows of stride N.  Timed as the best of three runs
 * after a warm-up; the previous version took a single sample and put a volatile
 * store inside the loop, so it was timing the volatile pipeline as much as the
 * arithmetic. */
double scalar_baseline(const std::vector<float> &A,
                       const std::vector<float> &B,
                       std::vector<float> &C,
                       uint32_t M, uint32_t K, uint32_t N)
{
    double best = 1e30;
    for (uint32_t rep = 0; rep < 3; ++rep) {
        float sink = 0.0f;
        double t0 = now_ms();
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k)
                    acc += A[static_cast<size_t>(m) * K + k] *
                           B[static_cast<size_t>(k) * N + n];
                C[static_cast<size_t>(m) * N + n] = acc;
                sink += acc;
            }
        }
        g_bench_sink = sink;
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

/* The same arithmetic in the loop order the compiler can auto-vectorise (K
 * outer, C and B walked as rows).  Still no intrinsics and no blocking, so the
 * gap between this and the kernels is what the NPU layer actually buys.
 * Speedup columns use whichever plain-C baseline is faster, so the reported
 * factor can never be inflated by a slow reference loop. */
double compiler_baseline(const std::vector<float> &A,
                         const std::vector<float> &B,
                         std::vector<float> &C,
                         uint32_t M, uint32_t K, uint32_t N)
{
    double best = 1e30;
    for (uint32_t rep = 0; rep < 3; ++rep) {
        std::fill(C.begin(), C.end(), 0.0f);
        double t0 = now_ms();
        for (uint32_t m = 0; m < M; ++m) {
            const float *arow = &A[static_cast<size_t>(m) * K];
            float *crow = &C[static_cast<size_t>(m) * N];
            for (uint32_t k = 0; k < K; ++k) {
                const float a = arow[k];
                const float *brow = &B[static_cast<size_t>(k) * N];
                for (uint32_t n = 0; n < N; ++n) crow[n] += a * brow[n];
            }
        }
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

std::string results_to_json(const std::vector<BenchResult> &results,
                            const vnpu_device_info_t &info)
{
    std::ostringstream o;
    o << "{\n  \"version\": \"" << vnpu_version_string() << "\",\n";
    o << "  \"backend\": \"" << info.backend << "\",\n";
    o << "  \"simd_bits\": " << info.simd_bits << ",\n";
    o << "  \"threads\": " << info.num_threads << ",\n";
    o << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        o << "    {\"shape\":\"" << r.shape << "\",\"dtype\":\""
          << r.dtype << "\",\"unit\":\"" << r.unit << "\","
          << "\"threads\":" << r.threads << ","
          << "\"iterations\":" << r.iterations << ","
          << "\"ms\":" << std::fixed << std::setprecision(6) << r.ms << ","
          << "\"gflops\":" << r.gflops << ","
          << "\"baseline_ms\":" << r.baseline_ms << ","
          << "\"baseline_scalar_ms\":" << r.baseline_scalar_ms << ","
          << "\"baseline_compiler_ms\":" << r.baseline_compiler_ms << ","
          << "\"speedup\":" << r.speedup << ","
          << "\"bytes_read\":" << r.bytes_read << ","
          << "\"bytes_written\":" << r.bytes_written << ","
          << "\"flops\":" << r.flops << ","
          << "\"tiles_dispatched\":" << r.tiles_dispatched << ","
          << "\"tiles_culled\":" << r.tiles_culled << ","
          << "\"correct\":" << (r.correct ? "true" : "false") << "}";
        if (i + 1 < results.size()) o << ',';
        o << '\n';
    }
    o << "  ]\n}\n";
    return o.str();
}

void print_results(const std::vector<BenchResult> &results)
{
    std::printf("%-14s %-5s %-9s %10s %10s %10s %8s %s\n",
                "shape", "dtype", "unit", "ms", "GFLOP/s", "speedup",
                "iters", "check");
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("speedup = vs the faster of a naive scalar triple loop and the same\n"
                "          arithmetic in compiler-vectorisable order (plain C, no\n"
                "          intrinsics, best of 3)\n");
    for (const auto &r : results) {
        std::printf("%-14s %-5s %-9s %10.4f %10.3f %9.2fx %8u %s\n",
                    r.shape.c_str(), r.dtype.c_str(), r.unit.c_str(),
                    r.ms, r.gflops, r.speedup, r.iterations,
                    r.correct ? "OK" : "FAIL");
    }
}

int cmd_info(int argc, char **argv)
{
    bool json = false;
    for (int i = 0; i < argc; ++i)
        if (std::strcmp(argv[i], "--json") == 0) json = true;
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    vnpu_device_t *device = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK) return 1;
    vnpu_device_info_t info{};
    vnpu_device_get_info(device, &info);
    if (json) {
        std::printf("{\n");
        std::printf("  \"name\": \"%s\",\n", info.name);
        std::printf("  \"vendor\": \"%s\",\n", info.vendor);
        std::printf("  \"arch\": \"%s\",\n", info.cpu_arch);
        std::printf("  \"backend\": \"%s\",\n", info.backend);
        std::printf("  \"simd_bits\": %u,\n", info.simd_bits);
        std::printf("  \"threads\": %u,\n", info.num_threads);
        std::printf("  \"physical_cores\": %u,\n", info.physical_cores);
        std::printf("  \"compute_units\": %u,\n", info.num_compute_units);
        std::printf("  \"local_store_bytes\": %u,\n", info.local_store_bytes);
        std::printf("  \"tiles\": [%u,%u,%u],\n", info.tile_m, info.tile_k,
                    info.tile_n);
        std::printf("  \"features\": %llu,\n",
                    static_cast<unsigned long long>(info.features));
        std::printf("  \"l1d_bytes\": %llu,\n",
                    static_cast<unsigned long long>(info.l1d_bytes));
        std::printf("  \"l2_bytes\": %llu,\n",
                    static_cast<unsigned long long>(info.l2_bytes));
        std::printf("  \"l3_bytes\": %llu,\n",
                    static_cast<unsigned long long>(info.l3_bytes));
        std::printf("  \"memory_bytes\": %llu,\n",
                    static_cast<unsigned long long>(info.total_memory_bytes));
        std::printf("  \"abi\": %u,\n", info.abi_version);
        std::printf("  \"version\": \"%s\"\n}\n", vnpu_version_string());
    } else {
        std::printf("Virtual NPU %s\n", vnpu_version_string());
        std::printf("  name           : %s\n", info.name);
        std::printf("  backend        : %s (%u-bit SIMD)\n", info.backend,
                    info.simd_bits);
        std::printf("  CPU            : %s, %u units on %u physical cores\n",
                    info.cpu_arch, info.num_threads, info.physical_cores);
        std::printf("  compute units  : %u\n", info.num_compute_units);
        std::printf("  local store    : %u KiB/SPE\n",
                    info.local_store_bytes / 1024);
        std::printf("  default tiles  : %u x %u x %u\n", info.tile_m,
                    info.tile_k, info.tile_n);
        std::printf("  cache          : L1 %llu KiB, L2 %llu KiB, L3 %llu KiB\n",
                    static_cast<unsigned long long>(info.l1d_bytes / 1024),
                    static_cast<unsigned long long>(info.l2_bytes / 1024),
                    static_cast<unsigned long long>(info.l3_bytes / 1024));
        std::printf("  features       : 0x%llx\n",
                    static_cast<unsigned long long>(info.features));
    }
    vnpu_device_destroy(device);
    return 0;
}

int cmd_bench(int argc, char **argv)
{
    std::vector<Shape> shapes;
    uint32_t iterations = 5;
    uint32_t unit_filter = 255;
    uint32_t threads = 0;
    std::string dtype = "f32";
    std::string json_path;
    std::string trace_path;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--shape") {
            Shape s = parse_shape(next());
            if (!s.valid()) { std::fprintf(stderr, "invalid shape\n"); return 2; }
            shapes.push_back(s);
        } else if (a == "--iterations") iterations = std::atoi(next());
        else if (a == "--unit") unit_filter = parse_unit(next());
        else if (a == "--dtype") dtype = next();
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--json") json_path = next();
        else if (a == "--trace") trace_path = next();
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: vnpu-cli bench [--shape MxKxN]... [--iterations N] "
                        "[--unit auto|voodoo|wildcat|cell|all] "
                        "[--dtype f32|i8|both] [--threads N] [--json path] "
                        "[--trace trace.json]\n");
            return 0;
        }
    }
    if (shapes.empty()) {
        shapes = {Shape{64, 64, 64}, Shape{128, 128, 128},
                  Shape{256, 256, 256}, Shape{512, 512, 512}};
    }
    if (iterations == 0) iterations = 1;
    if (iterations > 100) iterations = 100;
    std::vector<std::string> dtypes;
    if (dtype == "both") dtypes = {"f32", "i8"};
    else if (dtype == "i8") dtypes = {"i8"};
    else dtypes = {"f32"};
    std::vector<uint32_t> units;
    if (unit_filter == 255)
        units = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                 VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
    else
        units = {unit_filter};

    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.num_threads = threads;
    if (!trace_path.empty()) {
        cfg.flags |= VNPU_CONFIG_TRACE | VNPU_CONFIG_TRACE_CHROME;
        cfg.trace_file = trace_path.c_str();
    }
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK ||
        vnpu_stream_create(device, 0, &stream) != VNPU_OK) {
        std::fprintf(stderr, "failed to create runtime\n");
        return 1;
    }
    vnpu_device_info_t info{};
    vnpu_device_get_info(device, &info);
    std::vector<BenchResult> results;
    for (const auto &s : shapes) {
        const uint32_t M = s.M, K = s.K, N = s.N;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> B(static_cast<size_t>(K) * N);
        std::vector<float> C(static_cast<size_t>(M) * N);
        std::vector<float> Cbase(C.size());
        fill(A); fill(B);
        double baseline_scalar = scalar_baseline(A, B, Cbase, M, K, N);
        double baseline_compiler = compiler_baseline(A, B, Cbase, M, K, N);
        double baseline = std::min(baseline_scalar, baseline_compiler);
        uint64_t flops = 2ull * M * K * N;
        uint64_t bytes_read = (static_cast<uint64_t>(M) * K +
                               static_cast<uint64_t>(K) * N) * sizeof(float);
        uint64_t bytes_written = static_cast<uint64_t>(M) * N * sizeof(float);
        for (const auto &dt : dtypes) {
            vnpu_packed_weights_t *W = nullptr;
            std::vector<int8_t> A8, B8;
            std::vector<float> scale_a, scale_b;
            if (dt == "i8") {
                A8.resize(A.size());
                B8.resize(B.size());
                scale_a.resize(M, 1.0f);
                scale_b.resize(N, 1.0f);
                vnpu_quantize_i8_args_t qa{};
                qa.X = A.data(); qa.Y = A8.data();
                qa.scales = scale_a.data(); qa.rows = M; qa.cols = K;
                qa.axis = VNPU_QUANT_PER_ROW;
                vnpu_quantize_i8(stream, &qa);
                vnpu_quantize_i8_args_t qb{};
                qb.X = B.data(); qb.Y = B8.data();
                qb.scales = scale_b.data(); qb.rows = K; qb.cols = N;
                qb.axis = VNPU_QUANT_PER_COLUMN;
                vnpu_quantize_i8(stream, &qb);
                vnpu_pack_weights_i8(device, B8.data(), scale_b.data(),
                                     K, N, 0, 0, &W);
            }
            for (uint32_t unit : units) {
                vnpu_status_t rc = VNPU_OK;
                for (uint32_t warm = 0; warm < 2; ++warm) {
                    if (dt == "f32") {
                        vnpu_matmul_f32_args_t args{};
                        args.A = A.data(); args.B = B.data();
                        args.C = C.data(); args.M = M; args.K = K; args.N = N;
                        args.alpha = 1.0f; args.beta = 0.0f; args.unit = unit;
                        rc = vnpu_matmul_f32(stream, &args);
                    } else {
                        vnpu_matmul_i8_args_t args{};
                        args.A = A8.data(); args.B = W;
                        args.scale_a = scale_a.data(); args.C = C.data();
                        args.M = M; args.K = K; args.N = N; args.unit = unit;
                        rc = vnpu_matmul_i8(stream, &args);
                    }
                    if (rc != VNPU_OK) break;
                }
                double ms = 0.0;
                if (rc == VNPU_OK) {
                    double t0 = now_ms();
                    for (uint32_t it = 0; it < iterations; ++it) {
                        if (dt == "f32") {
                            vnpu_matmul_f32_args_t args{};
                            args.A = A.data(); args.B = B.data();
                            args.C = C.data(); args.M = M; args.K = K;
                            args.N = N; args.alpha = 1.0f; args.beta = 0.0f;
                            args.unit = unit;
                            rc = vnpu_matmul_f32(stream, &args);
                        } else {
                            vnpu_matmul_i8_args_t args{};
                            args.A = A8.data(); args.B = W;
                            args.scale_a = scale_a.data(); args.C = C.data();
                            args.M = M; args.K = K; args.N = N;
                            args.unit = unit;
                            rc = vnpu_matmul_i8(stream, &args);
                        }
                        if (rc != VNPU_OK) break;
                    }
                    ms = (now_ms() - t0) / static_cast<double>(iterations);
                }
                bool correct = false;
                if (rc == VNPU_OK) {
                    if (dt == "f32")
                        correct = check_samples(C, A, B, M, K, N,
                                                4e-3f, 4e-3f);
                    else {
                        float max_abs = 0.0f;
                        for (uint32_t q = 0; q < 8; ++q) {
                            uint32_t idx = xs() % (M * N);
                            uint32_t m = idx / N, n = idx % N;
                            max_abs = std::max(max_abs,
                                std::fabs(reference_dot(A, B, M, K, N, m, n)));
                        }
                        correct = check_samples(C, A, B, M, K, N,
                            std::max(0.1f, 0.04f * max_abs), 0.06f);
                    }
                }
                BenchResult r;
                r.shape = s.str();
                r.dtype = dt;
                r.unit = unit_slug(unit);
                r.threads = info.num_threads;
                r.iterations = iterations;
                r.ms = ms;
                r.gflops = ms > 0.0 ?
                    static_cast<double>(flops) / (ms * 1e6) : 0.0;
                r.baseline_ms = baseline;
                r.baseline_scalar_ms = baseline_scalar;
                r.baseline_compiler_ms = baseline_compiler;
                r.speedup = ms > 0.0 ? baseline / ms : 0.0;
                r.bytes_read = bytes_read;
                r.bytes_written = bytes_written;
                r.flops = flops;
                r.correct = correct ? 1 : 0;
                vnpu_device_status_t st{};
                vnpu_device_get_status(device, &st);
                r.tiles_dispatched = st.tiles_dispatched;
                r.tiles_culled = st.tiles_culled;
                results.push_back(r);
            }
            if (W) vnpu_packed_weights_destroy(W);
        }
    }
    print_results(results);
    std::string json = results_to_json(results, info);
    if (!json_path.empty()) {
        std::ofstream out(json_path);
        if (out) out << json;
    }
    if (!trace_path.empty()) {
        vnpu_trace_dump(device, nullptr);
        vnpu_trace_write_chrome(device, trace_path.c_str());
        std::printf("Sherlock Chrome trace written to %s\n",
                    trace_path.c_str());
    }
    vnpu_stream_destroy(stream);
    vnpu_device_destroy(device);
    for (const auto &r : results)
        if (!r.correct) return 3;
    return 0;
}

int cmd_autotune(int argc, char **argv)
{
    Shape shape{256, 256, 256};
    uint32_t iterations = 3;
    uint32_t threads = 0;
    vnpu_dtype_t dtype = VNPU_DTYPE_F32;
    std::string json_path;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--shape") shape = parse_shape(next());
        else if (a == "--iterations") iterations = std::atoi(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--dtype") {
            std::string d = next();
            dtype = d == "i8" ? VNPU_DTYPE_I8 : VNPU_DTYPE_F32;
        } else if (a == "--json") json_path = next();
    }
    if (!shape.valid()) { std::fprintf(stderr, "invalid shape\n"); return 2; }
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.num_threads = threads;
    vnpu_device_t *device = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK) return 1;
    vnpu_tune_result_t r{};
    vnpu_status_t rc = vnpu_autotune_gemm(device, shape.M, shape.K, shape.N,
                                          iterations, dtype, &r);
    if (rc != VNPU_OK) {
        std::fprintf(stderr, "autotune failed: %s\n",
                     vnpu_status_string(rc));
        vnpu_device_destroy(device);
        return 1;
    }
    std::printf("Autotune %s (%s)\n", shape.str().c_str(),
                dtype == VNPU_DTYPE_I8 ? "i8" : "f32");
    std::printf("  unit           : %s\n", vnpu_unit_name(r.unit));
    std::printf("  tiles          : %ux%ux%u\n", r.tile_m, r.tile_k,
                r.tile_n);
    std::printf("  threads        : %u\n", r.threads);
    std::printf("  best time      : %.4f ms\n", r.best_ms);
    std::printf("  scalar baseline: %.4f ms\n", r.baseline_ms);
    std::printf("  speedup        : %.2fx\n", r.speedup);
    std::printf("  throughput     : %.3f GFLOP/s\n", r.gflops);
    if (!json_path.empty()) {
        std::ofstream out(json_path);
        if (out) {
            out << "{\"shape\":\"" << shape.str() << "\",\"unit\":\""
                << vnpu_unit_name(r.unit) << "\",\"tile_m\":" << r.tile_m
                << ",\"tile_k\":" << r.tile_k << ",\"tile_n\":" << r.tile_n
                << ",\"threads\":" << r.threads << ",\"best_ms\":"
                << r.best_ms << ",\"baseline_ms\":" << r.baseline_ms
                << ",\"speedup\":" << r.speedup << ",\"gflops\":"
                << r.gflops << "}\n";
        }
    }
    vnpu_device_destroy(device);
    return 0;
}

int cmd_trace(int argc, char **argv)
{
    Shape shape{256, 256, 256};
    std::string output = "vnpu-trace.json";
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--shape") shape = parse_shape(next());
        else if (a == "--output") output = next();
    }
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.flags |= VNPU_CONFIG_TRACE;
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK ||
        vnpu_stream_create(device, 0, &stream) != VNPU_OK)
        return 1;
    std::vector<float> A(shape.M * shape.K), B(shape.K * shape.N),
                       C(shape.M * shape.N);
    fill(A); fill(B);
    for (uint32_t unit : {VNPU_UNIT_VOODOO, VNPU_UNIT_WILDCAT,
                          VNPU_UNIT_CELL}) {
        vnpu_matmul_f32_args_t args{};
        args.A = A.data(); args.B = B.data(); args.C = C.data();
        args.M = shape.M; args.K = shape.K; args.N = shape.N;
        args.alpha = 1.0f; args.beta = 0.0f; args.unit = unit;
        vnpu_matmul_f32(stream, &args);
    }
    vnpu_trace_dump(device, nullptr);
    vnpu_trace_write_chrome(device, output.c_str());
    std::printf("Chrome trace: %s\n", output.c_str());
    vnpu_stream_destroy(stream);
    vnpu_device_destroy(device);
    return 0;
}

void signal_stop(int) { vnpu_node_stop(); }

int cmd_serve(int argc, char **argv)
{
    vnpu_node_config_t cfg;
    vnpu_node_config_init(&cfg);
    std::string bind = "0.0.0.0";
    std::string token;
    std::string name = "vnpu-node";
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--port") cfg.port = static_cast<uint16_t>(std::atoi(next()));
        else if (a == "--threads") cfg.num_threads = std::atoi(next());
        else if (a == "--bind") bind = next();
        else if (a == "--token") token = next();
        else if (a == "--name") name = next();
    }
    cfg.bind_address = bind.c_str();
    cfg.auth_token = token.empty() ? nullptr : token.c_str();
    cfg.node_name = name.c_str();
    std::signal(SIGINT, signal_stop);
    std::signal(SIGTERM, signal_stop);
    vnpu_status_t rc = vnpu_node_run(&cfg);
    if (rc != VNPU_OK)
        std::fprintf(stderr, "node failed: %s\n", vnpu_status_string(rc));
    return rc == VNPU_OK ? 0 : 1;
}

void usage()
{
    std::printf(
        "vnpu-cli %s - standalone virtual NPU\n"
        "Usage: vnpu-cli <command> [options]\n\n"
        "Commands:\n"
        "  info                      Show CPU/backend capabilities\n"
        "  selftest [filter]         Run correctness tests\n"
        "  bench                     Benchmark GEMM across units/dtypes\n"
        "  autotune                  Sweep tile sizes and compute units\n"
        "  trace                     Generate a Sherlock Chrome trace\n"
        "  serve                     Run a vnpu-node network server\n"
        "  cluster-demo              Start two local nodes and run a cluster job\n"
        "  specdemo                  Distributed draft/verify speculative demo\n"
        "  llm-bench                 Synthetic transformer block benchmark\n"
        "  help                      Show this message\n\n"
        "Examples:\n"
        "  vnpu-cli info --json\n"
        "  vnpu-cli bench --shape 512x512x512 --dtype both --iterations 5\n"
        "  vnpu-cli autotune --shape 1024x1024x1024 --dtype i8\n"
        "  vnpu-cli trace --shape 512x512x512 --output trace.json\n"
        "  vnpu-cli serve --port 9449 --token secret\n"
        "  vnpu-cli cluster-demo --shape 256x256x256\n",
        vnpu_version_string());
}

} /* namespace */

int main(int argc, char **argv)
{
    if (argc < 2) { usage(); return 0; }
    std::string cmd = argv[1];
    int sub_argc = argc - 2;
    char **sub_argv = argv + 2;
    if (cmd == "help" || cmd == "--help" || cmd == "-h") { usage(); return 0; }
    if (cmd == "info") return cmd_info(sub_argc, sub_argv);
    if (cmd == "selftest")
        return vnpu_run_selftest(sub_argc > 0 ? sub_argv[0] : nullptr);
    if (cmd == "bench") return cmd_bench(sub_argc, sub_argv);
    if (cmd == "autotune") return cmd_autotune(sub_argc, sub_argv);
    if (cmd == "trace") return cmd_trace(sub_argc, sub_argv);
    if (cmd == "serve") return cmd_serve(sub_argc, sub_argv);
    if (cmd == "cluster-demo") return vnpu_run_cluster_demo(sub_argc, sub_argv);
    if (cmd == "specdemo") return vnpu_run_specdemo(sub_argc, sub_argv);
    if (cmd == "llm-bench") return vnpu_run_llm_bench(sub_argc, sub_argv);
    std::fprintf(stderr, "unknown command: %s\n\n", cmd.c_str());
    usage();
    return 2;
}
