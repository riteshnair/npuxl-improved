/*
 * conformance.cpp - the "same answer everywhere" claim, tested.
 *
 * Every unit, thread count, structure (packed/unpacked B) and transposition is
 * run over a set of shapes and checked three ways:
 *
 *   1. against an independent plain-C reference written here, in double
 *      precision, so the check does not depend on any VNPU kernel;
 *   2. for thread-count invariance -- a GEMM's bytes must not change when the
 *      device is given 1 unit instead of 2, because each output element
 *      accumulates over the whole K inside one tile;
 *   3. for INT8, bit-identical results across units and thread counts, which
 *      integer accumulation gives for free if the tiling is correct.
 *
 * These are claims a user can rely on, so they are tested rather than asserted
 * in a README.  Exit status is non-zero if any cell fails.
 */
#include "vnpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;
bool g_verbose = false;

struct Rng {
    uint32_t s = 2463534242u;
    uint32_t next()
    {
        uint32_t x = s;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s = x;
        return x;
    }
    float f()
    {
        return static_cast<float>(static_cast<int>(next() % 2000) - 1000) /
               1000.0f;
    }
    int8_t i8() { return static_cast<int8_t>(next() % 255 - 127); }
};

struct Device {
    vnpu_device_t *dev = nullptr;
    vnpu_stream_t *stream = nullptr;

    bool create(uint32_t threads)
    {
        vnpu_config_t cfg{};
        cfg.num_threads = threads;
        if (vnpu_device_create(&cfg, &dev) != VNPU_OK) return false;
        if (vnpu_stream_create(dev, 0, &stream) != VNPU_OK) return false;
        return true;
    }
    ~Device()
    {
        if (stream) vnpu_stream_destroy(stream);
        if (dev) vnpu_device_destroy(dev);
    }
};

/* Independent reference: row-major A (M x K) times B (K x N), or B stored
 * transposed as N x K.  Accumulated in double so the tolerance below measures
 * the float kernel, not the reference. */
void ref_matmul(const std::vector<float> &A, const std::vector<float> &B,
                std::vector<double> &C, uint32_t M, uint32_t K, uint32_t N,
                bool trans_b)
{
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(m) * K + k];
            for (uint32_t n = 0; n < N; ++n) {
                const double b = trans_b
                    ? B[static_cast<size_t>(n) * K + k]
                    : B[static_cast<size_t>(k) * N + n];
                C[static_cast<size_t>(m) * N + n] += a * b;
            }
        }
    }
}

/* Relative error against the reference, scaled by the magnitude of the result
 * so a well-conditioned check does not depend on the shape's size. */
double rel_error(const std::vector<float> &C, const std::vector<double> &ref)
{
    double worst = 0.0, scale = 1.0;
    for (size_t i = 0; i < ref.size() && i < C.size(); ++i)
        scale = std::max(scale, std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size() && i < C.size(); ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(C[i]) - ref[i]));
    return worst / scale;
}

bool same_bytes(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void cell(const char *what, const std::string &detail, bool ok,
          const std::string &note)
{
    ++g_checks;
    if (!ok) ++g_failures;
    if (!ok || g_verbose)
        std::printf("  [%s] %-22s %-26s %s\n", ok ? "ok" : "FAIL", what,
                    detail.c_str(), note.c_str());
}

const char *unit_name(uint32_t u)
{
    switch (u) {
    case VNPU_UNIT_AUTO: return "auto";
    case VNPU_UNIT_VOODOO: return "voodoo";
    case VNPU_UNIT_WILDCAT: return "wildcat";
    case VNPU_UNIT_CELL: return "cell";
    default: return "?";
    }
}

struct Shape {
    uint32_t M, K, N;
};

std::string shape_name(const Shape &s)
{
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%ux%ux%u", s.M, s.K, s.N);
    return buf;
}

/* Run one F32 GEMM on a freshly created device; returns false on error. */
bool run_f32(Device &d, const Shape &s, uint32_t unit, bool trans_b,
             std::vector<float> &A, std::vector<float> &B,
             std::vector<float> &C)
{
    A.resize(static_cast<size_t>(s.M) * s.K);
    B.resize(static_cast<size_t>(s.K) * s.N);
    C.assign(static_cast<size_t>(s.M) * s.N, 0.0f);
    Rng rng;
    for (auto &x : A) x = rng.f();
    for (auto &x : B) x = rng.f();

    vnpu_matmul_f32_args_t args{};
    args.A = A.data();
    args.B = B.data();
    args.C = C.data();
    args.M = s.M;
    args.K = s.K;
    args.N = s.N;
    args.trans_b = trans_b ? 1u : 0u;
    args.alpha = 1.0f;
    args.beta = 0.0f;
    args.unit = unit;
    return vnpu_matmul_f32(d.stream, &args) == VNPU_OK;
}

void test_f32()
{
    const Shape shapes[] = {
        {1, 512, 1376},  /* decode: GEMV row against a 2.8 MB matrix */
        {33, 67, 129},   /* odd sizes: tails in every dimension */
        {128, 128, 128},
        {256, 256, 256},
        {17, 300, 200},  /* M not a multiple of the 3-row block */
    };
    const uint32_t units[] = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                              VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
    const uint32_t threads[] = {1, 2};
    const double kTol = 1e-4;

    std::printf("F32 vs plain-C reference (double accumulation, tol %.0e)\n",
                kTol);
    for (const auto &s : shapes) {
        bool have_single[2][4][2];   /* [trans_b][unit][pack] */
        std::vector<float> base[2][4][2];
        std::memset(have_single, 0, sizeof(have_single));
        std::string detail = shape_name(s);
        double worst_err = 0.0;
        bool ok = true;
        for (int pack = 1; pack >= 0; --pack) {
            setenv("VNPU_PACK", pack ? "1" : "0", 1);
            for (uint32_t ti = 0; ti < 2; ++ti) {
                Device d;
                if (!d.create(threads[ti])) {
                    cell("f32 device create", detail, false, "create failed");
                    return;
                }
                for (int tb = 0; tb < 2; ++tb) {
                    std::vector<double> ref;
                    for (uint32_t ui = 0; ui < 4; ++ui) {
                        std::vector<float> A, B, C;
                        if (!run_f32(d, s, units[ui], tb != 0, A, B, C)) {
                            cell("f32 run", detail, false, "API error");
                            return;
                        }
                        if (ref.empty())
                            ref_matmul(A, B, ref, s.M, s.K, s.N, tb != 0);
                        const double err = rel_error(C, ref);
                        worst_err = std::max(worst_err, err);
                        if (err > kTol) {
                            ok = false;
                            char note[64];
                            std::snprintf(note, sizeof(note), "err=%.2e", err);
                            std::string u =
                                std::string(unit_name(units[ui])) + "/t" +
                                std::to_string(threads[ti]) + "/p" +
                                std::to_string(pack) + "/tb" +
                                std::to_string(tb);
                            cell("f32 accuracy", detail + " " + u, false, note);
                        }
                        /* Thread-count invariance: same config, different unit
                         * count, must be bit-identical. */
                        const int ui_ix = static_cast<int>(ui);
                        if (ti == 0) {
                            base[tb][ui_ix][pack] = C;
                            have_single[tb][ui_ix][pack] = true;
                        } else if (have_single[tb][ui_ix][pack] &&
                                   !same_bytes(C, base[tb][ui_ix][pack])) {
                            ok = false;
                            std::string u = std::string(unit_name(units[ui])) +
                                            "/p" + std::to_string(pack) +
                                            "/tb" + std::to_string(tb);
                            cell("f32 unit invariance", detail + " " + u, false,
                                 "1 vs 2 units differ");
                        }
                    }
                }
            }
        }
        char note[64];
        std::snprintf(note, sizeof(note), "worst rel err %.2e", worst_err);
        cell("f32 shape", detail, ok, note);
    }
    std::printf("  unit-count invariance: 1 vs 2 units bit-identical\n");
}

/* INT8 needs packed weights; every unit and thread count must produce exactly
 * the same bytes because the accumulation is exact 32-bit integer. */
void test_i8()
{
    struct I8Shape {
        uint32_t M, K, N;
    };
    const I8Shape shapes[] = {{19, 47, 31}, {64, 256, 128}, {1, 512, 1376}};
    const uint32_t units[] = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                              VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
    const uint32_t threads[] = {1, 2};

    std::printf("INT8 bit-identical across units and thread counts\n");
    for (const auto &s : shapes) {
        Rng rng;
        std::vector<float> Af(static_cast<size_t>(s.M) * s.K);
        std::vector<float> Bf(static_cast<size_t>(s.K) * s.N);
        for (auto &x : Af) x = rng.f();
        for (auto &x : Bf) x = rng.f();

        std::vector<float> first;
        bool ok = true;
        /* Packed and unpacked B must agree bit for bit: the INT8 microkernel
         * keeps its int32 accumulators in registers across all of K, so panel
         * packing changes where B is read from, not what is computed. */
        for (int pack = 1; pack >= 0 && ok; --pack) {
        setenv("VNPU_PACK", pack ? "1" : "0", 1);
        for (uint32_t ti = 0; ti < 2; ++ti) {
            Device d;
            if (!d.create(threads[ti])) {
                cell("i8 device create", "shape", false, "create failed");
                return;
            }
            std::vector<int8_t> A8(Af.size()), B8(Bf.size());
            std::vector<float> scale_a(s.M, 1.0f), scale_b(s.N, 1.0f);
            vnpu_quantize_i8_args_t qa{};
            qa.X = Af.data();
            qa.Y = A8.data();
            qa.scales = scale_a.data();
            qa.rows = s.M;
            qa.cols = s.K;
            qa.axis = VNPU_QUANT_PER_ROW;
            if (vnpu_quantize_i8(d.stream, &qa) != VNPU_OK) {
                cell("i8 quantize", "shape", false, "API error");
                return;
            }
            vnpu_quantize_i8_args_t qb{};
            qb.X = Bf.data();
            qb.Y = B8.data();
            qb.scales = scale_b.data();
            qb.rows = s.K;
            qb.cols = s.N;
            qb.axis = VNPU_QUANT_PER_COLUMN;
            if (vnpu_quantize_i8(d.stream, &qb) != VNPU_OK) {
                cell("i8 quantize", "shape", false, "API error");
                return;
            }
            vnpu_packed_weights_t *W = nullptr;
            if (vnpu_pack_weights_i8(d.dev, B8.data(), scale_b.data(), s.K, s.N,
                                     0, 0, &W) != VNPU_OK) {
                cell("i8 pack", "shape", false, "API error");
                return;
            }
            for (uint32_t ui = 0; ui < 4; ++ui) {
                std::vector<float> C(static_cast<size_t>(s.M) * s.N);
                vnpu_matmul_i8_args_t args{};
                args.A = A8.data();
                args.B = W;
                args.scale_a = scale_a.data();
                args.C = C.data();
                args.M = s.M;
                args.K = s.K;
                args.N = s.N;
                args.unit = units[ui];
                if (vnpu_matmul_i8(d.stream, &args) != VNPU_OK) {
                    cell("i8 run", "shape", false, "API error");
                    vnpu_packed_weights_destroy(W);
                    return;
                }
                if (first.empty()) {
                    first = C;
                } else if (!same_bytes(C, first)) {
                    ok = false;
                    std::string u = std::string(unit_name(units[ui])) + "/t" +
                                    std::to_string(threads[ti]) + "/p" +
                                    std::to_string(pack);
                    cell("i8 invariance", u, false, "differs from first config");
                }
            }
            vnpu_packed_weights_destroy(W);
        }
        }
        char detail[32];
        std::snprintf(detail, sizeof(detail), "%ux%ux%u", s.M, s.K, s.N);
        cell("i8 shape", detail, ok, "bit-identical incl. pack on/off");
    }
}

/* The reference mode exists so a caller can isolate its own bugs from kernel
 * bugs: it must agree with the plain-C reference like any other config. */
void test_reference_mode()
{
    const Shape shapes[] = {{33, 67, 129}, {128, 128, 128}};
    std::printf("Reference mode (scalar, unpacked) vs plain-C reference\n");
    setenv("VNPU_MODE", "reference", 1);
    for (const auto &s : shapes) {
        Device d;
        bool ok = d.create(2);
        std::vector<float> A, B, C;
        if (ok) ok = run_f32(d, s, VNPU_UNIT_AUTO, false, A, B, C);
        double err = 1.0;
        if (ok) {
            std::vector<double> ref;
            ref_matmul(A, B, ref, s.M, s.K, s.N, false);
            err = rel_error(C, ref);
            ok = err <= 1e-4;
        }
        char note[64];
        std::snprintf(note, sizeof(note), "rel err %.2e", err);
        cell("reference mode", shape_name(s), ok, note);
    }
    unsetenv("VNPU_MODE");
}

}  // namespace

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0) g_verbose = true;
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("usage: %s [--verbose]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    test_f32();
    test_i8();
    test_reference_mode();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
