/*
 * selftest.cpp - correctness and smoke tests for the standalone virtual NPU.
 */
#include "vnpu.h"
#include "vnpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_tests = 0;
std::string g_filter;

bool enabled(const char *name)
{
    return g_filter.empty() ||
           std::string(name).find(g_filter) != std::string::npos;
}

void run_test(const char *name, const std::function<bool()> &fn)
{
    if (!enabled(name)) return;
    ++g_tests;
    bool ok = false;
    try {
        ok = fn();
    } catch (const std::exception &e) {
        std::fprintf(stderr, "  exception in %s: %s\n", name, e.what());
    }
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_failures;
}

uint32_t rng_state = 123456789u;
uint32_t xorshift()
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

float rnd()
{
    return static_cast<float>(static_cast<int>(xorshift() % 2000) - 1000) /
           1000.0f;
}

bool close(float a, float b, float atol = 2e-3f, float rtol = 2e-3f)
{
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    float d = std::fabs(a - b);
    return d <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

float gelu_ref(float x)
{
    return 0.5f * x *
           (1.0f + std::tanh(0.7978845608028654f *
                             (x + 0.035677408136300125f * x * x * x)));
}

void ref_matmul(const std::vector<float> &A,
                const std::vector<float> &B,
                const std::vector<float> &bias,
                std::vector<float> &C,
                uint32_t M, uint32_t K, uint32_t N,
                bool trans_b, uint32_t epilogue)
{
    C.assign(static_cast<size_t>(M) * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float b = trans_b ? B[static_cast<size_t>(n) * K + k] :
                                    B[static_cast<size_t>(k) * N + n];
                acc += A[static_cast<size_t>(m) * K + k] * b;
            }
            if (!bias.empty()) acc += bias[n];
            if (epilogue == VNPU_EPILOGUE_RELU) acc = std::max(0.0f, acc);
            else if (epilogue == VNPU_EPILOGUE_SILU) acc = silu_ref(acc);
            else if (epilogue == VNPU_EPILOGUE_GELU_TANH) acc = gelu_ref(acc);
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

bool compare_vectors(const std::vector<float> &a,
                     const std::vector<float> &b,
                     float atol = 2e-3f, float rtol = 2e-3f,
                     const char *label = "")
{
    if (a.size() != b.size()) {
        std::fprintf(stderr, "  %s size mismatch %zu vs %zu\n", label,
                     a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!close(a[i], b[i], atol, rtol)) {
            std::fprintf(stderr,
                         "  %s mismatch at %zu: got %.8f expected %.8f\n",
                         label, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

struct Runtime {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    Runtime() {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.node_name = "selftest";
        if (vnpu_device_create(&cfg, &device) != VNPU_OK ||
            vnpu_stream_create(device, 0, &stream) != VNPU_OK)
            throw std::runtime_error("runtime create failed");
    }
    ~Runtime() {
        if (stream) vnpu_stream_destroy(stream);
        if (device) vnpu_device_destroy(device);
    }
};

vnpu_status_t toy_logits(void *, const int32_t *tokens, uint32_t num_tokens,
                         uint32_t first_position, uint32_t num_positions,
                         float *logits, uint32_t vocab)
{
    if (!tokens || !num_tokens || !num_positions || !logits || !vocab)
        return VNPU_ERR_INVALID_ARGUMENT;
    (void)first_position;
    for (uint32_t p = 0; p < num_positions; ++p) {
        int32_t t = tokens[std::min<uint32_t>(num_tokens - 1, p)];
        uint32_t target = static_cast<uint32_t>((t + 1) % vocab);
        for (uint32_t v = 0; v < vocab; ++v)
            logits[static_cast<size_t>(p) * vocab + v] =
                -static_cast<float>(std::abs(static_cast<int>(v) -
                                             static_cast<int>(target)));
    }
    return VNPU_OK;
}

} /* namespace */

int vnpu_run_selftest(const char *filter)
{
    g_failures = 0;
    g_tests = 0;
    g_filter = filter ? filter : "";
    std::printf("Virtual NPU selftest (%s)\n", vnpu_version_string());
    std::printf("==========================================\n");

    run_test("backend_vtable", []() {
        const auto *b = vnpu_get_backend_v1();
        return b && b->abi_version == VNPU_ABI_VERSION &&
               b->device_create && b->matmul_f32 &&
               b->cluster_create && b->speculative_step;
    });

    run_test("device_info_and_mmio", []() {
        Runtime r;
        vnpu_device_info_t info{};
        if (vnpu_device_get_info(r.device, &info) != VNPU_OK) return false;
        if (info.abi_version != VNPU_ABI_VERSION) return false;
        if (vnpu_mmio_read32(r.device, VNPU_REG_ID) != VNPU_MAGIC)
            return false;
        vnpu_mmio_write32(r.device, VNPU_REG_IRQ_MASK, 0xffffffffu);
        vnpu_mmio_write32(r.device, VNPU_REG_DOORBELL, 1);
        vnpu_device_status_t s{};
        vnpu_device_get_status(r.device, &s);
        std::printf("  backend=%s simd=%u bits threads=%u tiles=%ux%ux%u\n",
                    info.backend, info.simd_bits, info.num_threads,
                    info.tile_m, info.tile_k, info.tile_n);
        return s.interrupts >= 1;
    });

    for (uint32_t unit : {VNPU_UNIT_VOODOO, VNPU_UNIT_WILDCAT,
                          VNPU_UNIT_CELL, VNPU_UNIT_AUTO}) {
        std::string name = "gemm_f32_unit_" + std::to_string(unit);
        run_test(name.c_str(), [unit]() {
            Runtime r;
            const uint32_t M = 37, K = 53, N = 29;
            std::vector<float> A(static_cast<size_t>(M) * K);
            std::vector<float> B(static_cast<size_t>(K) * N);
            std::vector<float> bias(N);
            std::vector<float> C, ref;
            for (auto &x : A) x = rnd();
            for (auto &x : B) x = rnd();
            for (auto &x : bias) x = rnd() * 0.1f;
            vnpu_matmul_f32_args_t args{};
            args.A = A.data(); args.B = B.data(); args.bias = bias.data();
            args.C = C.data(); // replaced below
            args.M = M; args.K = K; args.N = N;
            args.alpha = 1.0f; args.beta = 0.0f;
            args.unit = unit;
            C.resize(static_cast<size_t>(M) * N);
            args.C = C.data();
            if (vnpu_matmul_f32(r.stream, &args) != VNPU_OK) return false;
            ref_matmul(A, B, bias, ref, M, K, N, false,
                       VNPU_EPILOGUE_NONE);
            return compare_vectors(C, ref, 3e-3f, 3e-3f, "gemm");
        });
    }

    run_test("gemm_f32_transposed_b_and_epilogue", []() {
        Runtime r;
        const uint32_t M = 17, K = 41, N = 23;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> Bt(static_cast<size_t>(N) * K);
        std::vector<float> C(static_cast<size_t>(M) * N), ref;
        for (auto &x : A) x = rnd();
        for (auto &x : Bt) x = rnd();
        vnpu_matmul_f32_args_t args{};
        args.A = A.data(); args.B = Bt.data(); args.C = C.data();
        args.M = M; args.K = K; args.N = N;
        args.trans_b = 1;
        args.alpha = 1.0f; args.beta = 0.0f;
        args.epilogue = VNPU_EPILOGUE_RELU;
        args.unit = VNPU_UNIT_AUTO;
        if (vnpu_matmul_f32(r.stream, &args) != VNPU_OK) return false;
        ref_matmul(A, Bt, {}, ref, M, K, N, true, VNPU_EPILOGUE_RELU);
        return compare_vectors(C, ref, 3e-3f, 3e-3f, "gemm_nt");
    });

    /* Two streams on one device run two parallel_for calls at the same time.
     * Kernel scratch used to be indexed by the task slot a parallel_for hands
     * out, which is not a thread, so concurrent tiles overwrote each other's
     * packed panel: 10 of 12 runs of this shape were wrong before the scratch
     * became per-thread.  Nothing else in the suite overlaps two streams. */
    run_test("concurrent_streams_same_device", []() {
        Runtime r;
        vnpu_stream_t *s2 = nullptr;
        if (vnpu_stream_create(r.device, 0, &s2) != VNPU_OK) return false;
        const uint32_t M = 192, K = 256, N = 256;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> B(static_cast<size_t>(K) * N);
        for (auto &x : A) x = rnd();
        for (auto &x : B) x = rnd();
        std::vector<float> C1(static_cast<size_t>(M) * N);
        std::vector<float> C2(C1.size()), ref;

        ref_matmul(A, B, {}, ref, M, K, N, false, VNPU_EPILOGUE_NONE);
        bool ok = true;
        for (int rep = 0; rep < 4 && ok; ++rep) {
            std::fill(C1.begin(), C1.end(), 0.0f);
            std::fill(C2.begin(), C2.end(), 0.0f);
            vnpu_matmul_f32_args_t a1{};
            a1.A = A.data(); a1.B = B.data(); a1.C = C1.data();
            a1.M = M; a1.K = K; a1.N = N;
            a1.alpha = 1.0f; a1.unit = VNPU_UNIT_AUTO;
            vnpu_matmul_f32_args_t a2 = a1;
            a2.C = C2.data();
            std::thread t1([&]() { vnpu_matmul_f32(r.stream, &a1); });
            std::thread t2([&]() { vnpu_matmul_f32(s2, &a2); });
            t1.join();
            t2.join();
            vnpu_stream_sync(r.stream);
            vnpu_stream_sync(s2);
            ok = compare_vectors(C1, ref, 3e-3f, 3e-3f, "stream1") &&
                 compare_vectors(C2, ref, 3e-3f, 3e-3f, "stream2");
        }
        vnpu_stream_destroy(s2);
        return ok;
    });

    run_test("gemm_i8_vs_f32", []() {
        Runtime r;
        const uint32_t M = 19, K = 47, N = 31;
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> B(static_cast<size_t>(K) * N);
        std::vector<float> ref;
        for (auto &x : A) x = rnd();
        for (auto &x : B) x = rnd();
        std::vector<int8_t> A8(A.size()), B8(B.size());
        std::vector<float> scale_a(M, 1.0f), scale_b(N, 1.0f);
        vnpu_quantize_i8_args_t qa{};
        qa.X = A.data(); qa.Y = A8.data(); qa.scales = scale_a.data();
        qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
        if (vnpu_quantize_i8(r.stream, &qa) != VNPU_OK) return false;
        vnpu_quantize_i8_args_t qb{};
        qb.X = B.data(); qb.Y = B8.data(); qb.scales = scale_b.data();
        qb.rows = K; qb.cols = N; qb.axis = VNPU_QUANT_PER_COLUMN;
        if (vnpu_quantize_i8(r.stream, &qb) != VNPU_OK) return false;
        vnpu_packed_weights_t *W = nullptr;
        if (vnpu_pack_weights_i8(r.device, B8.data(), scale_b.data(),
                                 K, N, 0, 0, &W) != VNPU_OK)
            return false;
        std::vector<float> C(static_cast<size_t>(M) * N);
        vnpu_matmul_i8_args_t args{};
        args.A = A8.data(); args.B = W; args.scale_a = scale_a.data();
        args.C = C.data(); args.M = M; args.K = K; args.N = N;
        args.unit = VNPU_UNIT_AUTO;
        vnpu_status_t rc = vnpu_matmul_i8(r.stream, &args);
        vnpu_packed_weights_destroy(W);
        if (rc != VNPU_OK) return false;
        ref_matmul(A, B, {}, ref, M, K, N, false, VNPU_EPILOGUE_NONE);
        float max_abs = 0.0f;
        for (float x : ref) max_abs = std::max(max_abs, std::fabs(x));
        return compare_vectors(C, ref,
                               std::max(0.05f, 0.03f * max_abs),
                               0.05f, "gemm_i8");
    });

    run_test("rmsnorm_softmax_swiglu_gelu", []() {
        Runtime r;
        const uint32_t rows = 7, cols = 33;
        std::vector<float> X(rows * cols), W(cols), Y1(rows * cols),
                           Y2(rows * cols), Y3(rows * cols), Y4(rows * cols);
        for (auto &x : X) x = rnd();
        for (auto &x : W) x = 0.5f + rnd() * 0.5f;
        vnpu_rmsnorm_f32_args_t rn{};
        rn.X = X.data(); rn.W = W.data(); rn.Y = Y1.data();
        rn.rows = rows; rn.cols = cols; rn.eps = 1e-5f;
        if (vnpu_rmsnorm_f32(r.stream, &rn) != VNPU_OK) return false;
        for (uint32_t i = 0; i < rows; ++i) {
            float ss = 0.0f;
            for (uint32_t j = 0; j < cols; ++j)
                ss += X[i * cols + j] * X[i * cols + j];
            float inv = 1.0f / std::sqrt(ss / cols + rn.eps);
            for (uint32_t j = 0; j < cols; ++j) {
                float expected = X[i * cols + j] * inv * W[j];
                if (!close(Y1[i * cols + j], expected, 2e-4f, 2e-4f))
                    return false;
            }
        }
        vnpu_softmax_f32_args_t sm{};
        sm.X = X.data(); sm.Y = Y2.data();
        sm.rows = rows; sm.cols = cols; sm.temperature = 1.0f;
        if (vnpu_softmax_f32(r.stream, &sm) != VNPU_OK) return false;
        for (uint32_t i = 0; i < rows; ++i) {
            float maxv = *std::max_element(X.begin() + i * cols,
                                           X.begin() + (i + 1) * cols);
            float sum = 0.0f;
            std::vector<float> e(cols);
            for (uint32_t j = 0; j < cols; ++j) {
                e[j] = std::exp(X[i * cols + j] - maxv);
                sum += e[j];
            }
            for (uint32_t j = 0; j < cols; ++j)
                if (!close(Y2[i * cols + j], e[j] / sum, 2e-4f, 2e-4f))
                    return false;
        }
        vnpu_swiglu_f32_args_t sg{};
        sg.gate = X.data(); sg.up = W.data(); // broadcast-ish test data
        sg.Y = Y3.data(); sg.count = rows * cols; sg.alpha = 1.0f;
        // Use a properly sized up vector instead of W.
        std::vector<float> up(rows * cols);
        for (auto &x : up) x = rnd();
        sg.up = up.data();
        if (vnpu_swiglu_f32(r.stream, &sg) != VNPU_OK) return false;
        for (size_t i = 0; i < X.size(); ++i)
            if (!close(Y3[i], silu_ref(X[i]) * up[i], 3e-4f, 3e-4f))
                return false;
        vnpu_unary_f32_args_t gelu{};
        gelu.X = X.data(); gelu.Y = Y4.data(); gelu.count = X.size();
        if (vnpu_gelu_f32(r.stream, &gelu) != VNPU_OK) return false;
        for (size_t i = 0; i < X.size(); ++i)
            if (!close(Y4[i], gelu_ref(X[i]), 3e-4f, 3e-4f))
                return false;
        return true;
    });

    run_test("attention_small", []() {
        Runtime r;
        const uint32_t B = 1, H = 2, Sq = 5, Sk = 5, D = 16;
        size_t qn = static_cast<size_t>(B) * H * Sq * D;
        size_t kn = static_cast<size_t>(B) * H * Sk * D;
        std::vector<float> Q(qn), K(kn), V(kn), O(qn), Ref(qn, 0.0f);
        for (auto &x : Q) x = rnd() * 0.2f;
        for (auto &x : K) x = rnd() * 0.2f;
        for (auto &x : V) x = rnd() * 0.2f;
        vnpu_attention_f32_args_t a{};
        a.Q = Q.data(); a.K = K.data(); a.V = V.data(); a.O = O.data();
        a.batch = B; a.heads = H; a.seq_q = Sq; a.seq_k = Sk;
        a.head_dim = D; a.causal = 1; a.scale = 1.0f / std::sqrt((float)D);
        a.unit = VNPU_UNIT_AUTO;
        if (vnpu_attention_f32(r.stream, &a) != VNPU_OK) return false;
        /* Naive masked reference. */
        for (uint32_t b = 0; b < B; ++b)
        for (uint32_t h = 0; h < H; ++h)
        for (uint32_t i = 0; i < Sq; ++i) {
            size_t qbase = ((static_cast<size_t>(b) * H + h) * Sq + i) * D;
            size_t kbase = (static_cast<size_t>(b) * H + h) * Sk * D;
            std::vector<float> scores(Sk, -std::numeric_limits<float>::infinity());
            float maxv = -std::numeric_limits<float>::infinity();
            for (uint32_t j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; ++d)
                    dot += Q[qbase + d] * K[kbase + j * D + d];
                scores[j] = dot * a.scale;
                maxv = std::max(maxv, scores[j]);
            }
            float sum = 0.0f;
            for (uint32_t j = 0; j <= i; ++j) {
                scores[j] = std::exp(scores[j] - maxv);
                sum += scores[j];
            }
            for (uint32_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= i; ++j)
                    acc += scores[j] * V[kbase + j * D + d];
                Ref[qbase + d] = acc / sum;
            }
        }
        return compare_vectors(O, Ref, 2e-3f, 2e-3f, "attention");
    });

    run_test("rope_inplace", []() {
        Runtime r;
        const uint32_t batch = 1, heads = 2, seq = 4, dim = 16;
        std::vector<float> Q(batch * heads * seq * dim);
        std::vector<float> K = Q;
        for (size_t i = 0; i < Q.size(); ++i) {
            Q[i] = rnd();
            K[i] = rnd();
        }
        std::vector<float> q0 = Q, k0 = K;
        vnpu_rope_f32_args_t a{};
        a.Q = Q.data(); a.K = K.data();
        a.batch = batch; a.heads = heads;
        a.q_seq = seq; a.k_seq = seq; a.head_dim = dim;
        a.theta_base = 10000.0f; a.scaling = 1.0f;
        a.interleaved = 0;
        if (vnpu_rope_f32(r.stream, &a) != VNPU_OK) return false;
        bool changed = false;
        for (size_t i = 0; i < Q.size(); ++i) {
            if (!std::isfinite(Q[i]) || !std::isfinite(K[i])) return false;
            if (std::fabs(Q[i] - q0[i]) > 1e-6f ||
                std::fabs(K[i] - k0[i]) > 1e-6f) changed = true;
        }
        return changed;
    });

    run_test("mtp_head", []() {
        Runtime r;
        const uint32_t gamma = 3, H = 29, V = 17;
        std::vector<float> hidden(gamma * H), W(V * H), bias(V),
                           logits(gamma * V), ref;
        for (auto &x : hidden) x = rnd();
        for (auto &x : W) x = rnd();
        for (auto &x : bias) x = rnd() * 0.1f;
        vnpu_mtp_head_f32_args_t a{};
        a.hidden = hidden.data(); a.W = W.data(); a.bias = bias.data();
        a.logits = logits.data(); a.gamma = gamma; a.hidden_dim = H;
        a.vocab = V; a.trans_w = 1; a.unit = VNPU_UNIT_AUTO;
        if (vnpu_mtp_head_f32(r.stream, &a) != VNPU_OK) return false;
        ref_matmul(hidden, W, bias, ref, gamma, H, V, true,
                   VNPU_EPILOGUE_NONE);
        return compare_vectors(logits, ref, 3e-3f, 3e-3f, "mtp");
    });

    run_test("async_events_and_trace", []() {
        Runtime r;
        vnpu_event_t *event = nullptr;
        if (vnpu_event_create(&event) != VNPU_OK) return false;
        std::vector<vnpu_command_t> cmds(8);
        for (auto &c : cmds) {
            c = {};
            c.opcode = VNPU_OP_NOP;
            c.id = 1;
        }
        if (vnpu_stream_submit(r.stream, cmds.data(), cmds.size(), event) !=
            VNPU_OK) {
            vnpu_event_destroy(event);
            return false;
        }
        vnpu_status_t rc = vnpu_event_wait(event, 2000);
        vnpu_event_destroy(event);
        if (rc != VNPU_OK) return false;
        vnpu_trace_reset(r.device);
        if (vnpu_trace_enable(r.device, 1) != VNPU_OK) return false;
        const uint32_t M = 16, K = 16, N = 16;
        std::vector<float> A(M * K, 1.0f), B(K * N, 1.0f), C(M * N, 0.0f);
        vnpu_matmul_f32_args_t args{};
        args.A = A.data(); args.B = B.data(); args.C = C.data();
        args.M = M; args.K = K; args.N = N;
        args.alpha = 1.0f; args.beta = 0.0f; args.unit = VNPU_UNIT_AUTO;
        if (vnpu_matmul_f32(r.stream, &args) != VNPU_OK) return false;
        vnpu_trace_summary_t summary{};
        vnpu_trace_get_summary(r.device, &summary);
        bool all_ones = true;
        for (float x : C) if (!close(x, static_cast<float>(K))) all_ones = false;
        return summary.events > 0 && summary.flops > 0 && all_ones;
    });

    run_test("autotune_small", []() {
        Runtime r;
        vnpu_tune_result_t result{};
        vnpu_status_t rc = vnpu_autotune_gemm(r.device, 64, 64, 64, 1,
                                               VNPU_DTYPE_F32, &result);
        if (rc != VNPU_OK) return false;
        std::printf("  autotune: unit=%s tiles=%ux%ux%u %.3f ms %.2f GFLOP/s "
                    "%.2fx scalar\n",
                    vnpu_unit_name(result.unit), result.tile_m,
                    result.tile_k, result.tile_n, result.best_ms,
                    result.gflops, result.speedup);
        return result.tile_m && result.tile_k && result.tile_n &&
               result.best_ms > 0.0;
    });

    run_test("local_cluster", []() {
        Runtime r;
        vnpu_cluster_config_t cc;
        vnpu_cluster_config_init(&cc);
        cc.num_streams = 2;
        vnpu_cluster_t *cluster = nullptr;
        if (vnpu_cluster_create(&cc, &cluster) != VNPU_OK) return false;
        if (vnpu_cluster_add_device(cluster, r.device, 1) != VNPU_OK) {
            vnpu_cluster_destroy(cluster);
            return false;
        }
        const uint32_t M = 40, K = 23, N = 19;
        std::vector<float> A(M * K), B(K * N), C(M * N), ref;
        for (auto &x : A) x = rnd();
        for (auto &x : B) x = rnd();
        vnpu_weight_handle_t *wh = nullptr;
        if (vnpu_cluster_cache_weights_f32(cluster, "test", B.data(),
                                           K, N, 0, &wh) != VNPU_OK) {
            vnpu_cluster_destroy(cluster);
            return false;
        }
        vnpu_status_t rc = vnpu_cluster_matmul_f32(cluster, wh, A.data(),
                                                   C.data(), M, K, N,
                                                   VNPU_UNIT_AUTO);
        vnpu_cluster_stats_t stats{};
        vnpu_cluster_get_stats(cluster, &stats);
        vnpu_cluster_destroy(cluster);
        if (rc != VNPU_OK) return false;
        ref_matmul(A, B, {}, ref, M, K, N, false, VNPU_EPILOGUE_NONE);
        return compare_vectors(C, ref, 3e-3f, 3e-3f, "cluster") &&
               stats.total_jobs > 0;
    });

    run_test("speculative_step", []() {
        vnpu_speculative_config_t cfg;
        vnpu_speculative_config_init(&cfg);
        cfg.gamma = 4;
        cfg.vocab = 64;
        cfg.max_tokens = 8;
        cfg.eos_token = 999999;
        cfg.temperature = 0.0f;
        std::vector<int32_t> context = {3, 1, 4, 1, 5};
        vnpu_speculative_result_t result{};
        vnpu_status_t rc = vnpu_speculative_step(
            nullptr, &cfg, context.data(),
            static_cast<uint32_t>(context.size()),
            toy_logits, nullptr, toy_logits, nullptr, &result);
        return rc == VNPU_OK && result.produced >= 1 &&
               result.accepted <= cfg.gamma &&
               result.tokens_per_second > 0.0;
    });

    std::printf("==========================================\n");
    std::printf("%d tests, %d failures\n", g_tests, g_failures);
    return g_failures ? 1 : 0;
}
