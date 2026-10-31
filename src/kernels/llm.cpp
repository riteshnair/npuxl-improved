/*
 * llm.cpp - transformer operator kernels used by the virtual NPU.
 *
 * These are dependency-free, streaming kernels.  GEMM remains the dominant
 * cost, but RMSNorm, Softmax, RoPE, SwiGLU, GELU and attention are included so
 * an engine can keep an entire transformer block on the device.
 */
#include "../internal.h"

#if VNPU_X86 && (defined(__GNUC__) || defined(__clang__))
#  define VNPU_TARGET_AVX2   __attribute__((target("avx2,fma")))
#else
#  define VNPU_TARGET_AVX2
#endif

namespace vnpu {
namespace {

#if VNPU_X86
VNPU_TARGET_AVX2
static inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

VNPU_TARGET_AVX2
static inline __m256 exp256_ps(__m256 x)
{
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    const __m256 c5 = _mm256_set1_ps(0.0013333558146428443f);
    x = _mm256_max_ps(_mm256_set1_ps(-80.0f),
                      _mm256_min_ps(x, _mm256_set1_ps(80.0f)));
    __m256 z = _mm256_mul_ps(x, log2e);
    __m256 zi = _mm256_floor_ps(z);
    __m256 f = _mm256_sub_ps(z, zi);
    __m256 p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i n = _mm256_cvtps_epi32(zi);
    n = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    n = _mm256_slli_epi32(n, 23);
    __m256 pow2 = _mm256_castsi256_ps(n);
    return _mm256_mul_ps(p, pow2);
}

#endif /* VNPU_X86 */

/* ------------------------------------------------------------------------- */
/* RMSNorm                                                                   */
/* ------------------------------------------------------------------------- */

void rmsnorm_scalar_rows(const vnpu_rmsnorm_f32_args_t &a, uint32_t begin,
                         uint32_t end)
{
    float eps = a.eps > 0.0f ? a.eps : 1e-5f;
    for (uint32_t r = begin; r < end; ++r) {
        const float *x = a.X + static_cast<size_t>(r) * a.cols;
        float *y = a.Y + static_cast<size_t>(r) * a.cols;
        float ss = 0.0f;
        for (uint32_t j = 0; j < a.cols; ++j) ss += x[j] * x[j];
        float inv = 1.0f / std::sqrt(ss / static_cast<float>(a.cols) + eps);
        for (uint32_t j = 0; j < a.cols; ++j)
            y[j] = x[j] * inv * (a.W ? a.W[j] : 1.0f);
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static void rmsnorm_avx2_rows(const vnpu_rmsnorm_f32_args_t &a,
                              uint32_t begin, uint32_t end)
{
    float eps = a.eps > 0.0f ? a.eps : 1e-5f;
    for (uint32_t r = begin; r < end; ++r) {
        const float *x = a.X + static_cast<size_t>(r) * a.cols;
        float *y = a.Y + static_cast<size_t>(r) * a.cols;
        __m256 ss = _mm256_setzero_ps();
        uint32_t j = 0;
        for (; j + 8 <= a.cols; j += 8) {
            __m256 v = _mm256_loadu_ps(x + j);
            ss = _mm256_fmadd_ps(v, v, ss);
        }
        float sum = hsum256(ss);
        for (; j < a.cols; ++j) sum += x[j] * x[j];
        float inv = 1.0f / std::sqrt(sum / static_cast<float>(a.cols) + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        j = 0;
        for (; j + 8 <= a.cols; j += 8) {
            __m256 v = _mm256_loadu_ps(x + j);
            __m256 w = a.W ? _mm256_loadu_ps(a.W + j) :
                             _mm256_set1_ps(1.0f);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_mul_ps(v, w), vinv));
        }
        for (; j < a.cols; ++j)
            y[j] = x[j] * inv * (a.W ? a.W[j] : 1.0f);
    }
}

#endif

/* ------------------------------------------------------------------------- */
/* Softmax                                                                   */
/* ------------------------------------------------------------------------- */

void softmax_scalar_rows(const vnpu_softmax_f32_args_t &a, uint32_t begin,
                         uint32_t end)
{
    float inv_t = a.temperature > 0.0f ? 1.0f / a.temperature : 1.0f;
    for (uint32_t r = begin; r < end; ++r) {
        const float *x = a.X + static_cast<size_t>(r) * a.cols;
        float *y = a.Y + static_cast<size_t>(r) * a.cols;
        float maxv = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < a.cols; ++j) maxv = std::max(maxv, x[j]);
        float sum = 0.0f;
        for (uint32_t j = 0; j < a.cols; ++j) {
            float e = fast_expf((x[j] - maxv) * inv_t);
            y[j] = e;
            sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (uint32_t j = 0; j < a.cols; ++j) y[j] *= inv;
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static void softmax_avx2_rows(const vnpu_softmax_f32_args_t &a,
                              uint32_t begin, uint32_t end)
{
    float inv_t = a.temperature > 0.0f ? 1.0f / a.temperature : 1.0f;
    for (uint32_t r = begin; r < end; ++r) {
        const float *x = a.X + static_cast<size_t>(r) * a.cols;
        float *y = a.Y + static_cast<size_t>(r) * a.cols;
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        uint32_t j = 0;
        for (; j + 8 <= a.cols; j += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + j));
        float maxv = -std::numeric_limits<float>::infinity();
        {
            float tmp[8];
            _mm256_storeu_ps(tmp, vmax);
            for (float v : tmp) maxv = std::max(maxv, v);
        }
        for (; j < a.cols; ++j) maxv = std::max(maxv, x[j]);
        __m256 sumv = _mm256_setzero_ps();
        j = 0;
        for (; j + 8 <= a.cols; j += 8) {
            __m256 v = _mm256_loadu_ps(x + j);
            __m256 e = exp256_ps(_mm256_mul_ps(
                _mm256_sub_ps(v, _mm256_set1_ps(maxv)),
                _mm256_set1_ps(inv_t)));
            _mm256_storeu_ps(y + j, e);
            sumv = _mm256_add_ps(sumv, e);
        }
        float sum = hsum256(sumv);
        for (; j < a.cols; ++j) {
            float e = fast_expf((x[j] - maxv) * inv_t);
            y[j] = e;
            sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        __m256 vinv = _mm256_set1_ps(inv);
        j = 0;
        for (; j + 8 <= a.cols; j += 8)
            _mm256_storeu_ps(y + j,
                             _mm256_mul_ps(_mm256_loadu_ps(y + j), vinv));
        for (; j < a.cols; ++j) y[j] *= inv;
    }
}

#endif

/* ------------------------------------------------------------------------- */
/* RoPE                                                                      */
/* ------------------------------------------------------------------------- */

static void rotate_vector(float *x, uint32_t dim, float position,
                          float theta_base, float scaling, bool interleaved)
{
    if (!x || dim < 2) return;
    position *= scaling;
    if (interleaved) {
        for (uint32_t j = 0; j + 1 < dim; j += 2) {
            float inv = 1.0f / std::pow(theta_base,
                                        static_cast<float>(j) /
                                            static_cast<float>(dim));
            float angle = position * inv;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = x[j], x1 = x[j + 1];
            x[j] = x0 * c - x1 * s;
            x[j + 1] = x0 * s + x1 * c;
        }
    } else {
        uint32_t half = dim / 2;
        for (uint32_t j = 0; j < half; ++j) {
            float inv = 1.0f / std::pow(theta_base,
                                        2.0f * static_cast<float>(j) /
                                            static_cast<float>(dim));
            float angle = position * inv;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = x[j], x1 = x[j + half];
            x[j] = x0 * c - x1 * s;
            x[j + half] = x0 * s + x1 * c;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Elementwise kernels                                                       */
/* ------------------------------------------------------------------------- */

void add_scalar_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end)
{
    float alpha = a.alpha == 0.0f && a.beta == 0.0f ? 1.0f : a.alpha;
    float beta = a.alpha == 0.0f && a.beta == 0.0f ? 1.0f : a.beta;
    for (uint64_t i = begin; i < end; ++i)
        a.Y[i] = alpha * a.A[i] + beta * a.B[i];
}

void mul_scalar_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end)
{
    float alpha = a.alpha == 0.0f ? 1.0f : a.alpha;
    for (uint64_t i = begin; i < end; ++i)
        a.Y[i] = alpha * a.A[i] * a.B[i] + a.beta;
}

void swiglu_scalar_range(const vnpu_swiglu_f32_args_t &a, uint64_t begin,
                         uint64_t end)
{
    float alpha = a.alpha == 0.0f ? 1.0f : a.alpha;
    for (uint64_t i = begin; i < end; ++i)
        a.Y[i] = silu(a.gate[i]) * a.up[i] * alpha;
}

void gelu_scalar_range(const vnpu_unary_f32_args_t &a, uint64_t begin,
                       uint64_t end)
{
    for (uint64_t i = begin; i < end; ++i)
        a.Y[i] = gelu_tanh(a.X[i]);
}

#if VNPU_X86
VNPU_TARGET_AVX2
static void add_avx2_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                           uint64_t end)
{
    float alpha = a.alpha == 0.0f && a.beta == 0.0f ? 1.0f : a.alpha;
    float beta = a.alpha == 0.0f && a.beta == 0.0f ? 1.0f : a.beta;
    __m256 va = _mm256_set1_ps(alpha), vb = _mm256_set1_ps(beta);
    uint64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 x = _mm256_loadu_ps(a.A + i);
        __m256 y = _mm256_loadu_ps(a.B + i);
        _mm256_storeu_ps(a.Y + i,
                         _mm256_fmadd_ps(x, va, _mm256_mul_ps(y, vb)));
    }
    for (; i < end; ++i) a.Y[i] = alpha * a.A[i] + beta * a.B[i];
}

VNPU_TARGET_AVX2
static void mul_avx2_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                           uint64_t end)
{
    float alpha = a.alpha == 0.0f ? 1.0f : a.alpha;
    __m256 va = _mm256_set1_ps(alpha), vb = _mm256_set1_ps(a.beta);
    uint64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 x = _mm256_loadu_ps(a.A + i);
        __m256 y = _mm256_loadu_ps(a.B + i);
        _mm256_storeu_ps(a.Y + i,
                         _mm256_fmadd_ps(_mm256_mul_ps(x, y), va, vb));
    }
    for (; i < end; ++i) a.Y[i] = alpha * a.A[i] * a.B[i] + a.beta;
}

VNPU_TARGET_AVX2
static void swiglu_avx2_range(const vnpu_swiglu_f32_args_t &a,
                              uint64_t begin, uint64_t end)
{
    float alpha = a.alpha == 0.0f ? 1.0f : a.alpha;
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 va = _mm256_set1_ps(alpha);
    uint64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 g = _mm256_loadu_ps(a.gate + i);
        __m256 u = _mm256_loadu_ps(a.up + i);
        __m256 e = exp256_ps(_mm256_xor_ps(g, _mm256_set1_ps(-0.0f)));
        __m256 s = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(a.Y + i,
                         _mm256_mul_ps(_mm256_mul_ps(s, u), va));
    }
    for (; i < end; ++i) a.Y[i] = silu(a.gate[i]) * a.up[i] * alpha;
}

VNPU_TARGET_AVX2
static void gelu_avx2_range(const vnpu_unary_f32_args_t &a, uint64_t begin,
                            uint64_t end)
{
    const __m256 k0 = _mm256_set1_ps(0.7978845608028654f);
    const __m256 k1 = _mm256_set1_ps(0.035677408136300125f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    uint64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 x = _mm256_loadu_ps(a.X + i);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        __m256 inner = _mm256_mul_ps(k0, _mm256_fmadd_ps(k1, x3, x));
        __m256 e = exp256_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), inner));
        __m256 t = _mm256_sub_ps(_mm256_div_ps(two,
                                    _mm256_add_ps(one, e)), one);
        __m256 y = _mm256_mul_ps(_mm256_mul_ps(half, x),
                                 _mm256_add_ps(one, t));
        _mm256_storeu_ps(a.Y + i, y);
    }
    for (; i < end; ++i) a.Y[i] = gelu_tanh(a.X[i]);
}

VNPU_TARGET_AVX2
static float dot_avx2(const float *a, const float *b, uint32_t n)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                              _mm256_loadu_ps(b + i), acc);
    float sum = hsum256(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

VNPU_TARGET_AVX2
static void axpy_avx2(float *dst, const float *src, float scale, uint32_t n)
{
    __m256 v = _mm256_set1_ps(scale);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i,
            _mm256_fmadd_ps(_mm256_loadu_ps(src + i), v,
                            _mm256_loadu_ps(dst + i)));
    for (; i < n; ++i) dst[i] += scale * src[i];
}

#endif /* VNPU_X86 */

static float dot_dispatch(const float *a, const float *b, uint32_t n,
                          uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) return dot_avx2(a, b, n);
#else
    (void)isa;
#endif
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static void axpy_dispatch(float *dst, const float *src, float scale,
                          uint32_t n, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) { axpy_avx2(dst, src, scale, n); return; }
#else
    (void)isa;
#endif
    for (uint32_t i = 0; i < n; ++i) dst[i] += scale * src[i];
}

/* ------------------------------------------------------------------------- */
/* Quantization                                                              */
/* ------------------------------------------------------------------------- */

static inline int8_t quantize_value(float x, float inv_scale)
{
    float q = std::round(x * inv_scale);
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return static_cast<int8_t>(static_cast<int>(q));
}

void quantize_rows(const vnpu_quantize_i8_args_t &a, uint64_t begin,
                   uint64_t end)
{
    for (uint64_t r = begin; r < end; ++r) {
        const float *row = a.X + r * a.cols;
        int8_t *out = a.Y + r * a.cols;
        float maxabs = 0.0f;
        for (uint64_t j = 0; j < a.cols; ++j)
            maxabs = std::max(maxabs, std::fabs(row[j]));
        float scale = maxabs / 127.0f;
        if (scale <= 0.0f || !std::isfinite(scale)) scale = 1.0f;
        float inv = 1.0f / scale;
        a.scales[r] = scale;
        for (uint64_t j = 0; j < a.cols; ++j)
            out[j] = quantize_value(row[j], inv);
    }
}

void quantize_columns(const vnpu_quantize_i8_args_t &a, uint64_t begin,
                      uint64_t end)
{
    for (uint64_t c = begin; c < end; ++c) {
        float maxabs = 0.0f;
        for (uint64_t r = 0; r < a.rows; ++r)
            maxabs = std::max(maxabs, std::fabs(a.X[r * a.cols + c]));
        float scale = maxabs / 127.0f;
        if (scale <= 0.0f || !std::isfinite(scale)) scale = 1.0f;
        float inv = 1.0f / scale;
        a.scales[c] = scale;
        for (uint64_t r = 0; r < a.rows; ++r)
            a.Y[r * a.cols + c] =
                quantize_value(a.X[r * a.cols + c], inv);
    }
}

void quantize_tensor(const vnpu_quantize_i8_args_t &a, uint64_t begin,
                     uint64_t end)
{
    float inv = 1.0f / (a.scales[0] > 0.0f ? a.scales[0] : 1.0f);
    for (uint64_t i = begin; i < end; ++i)
        a.Y[i] = quantize_value(a.X[i], inv);
}

/* ------------------------------------------------------------------------- */
/* Streaming attention                                                       */
/* ------------------------------------------------------------------------- */

void attention_range(const vnpu_attention_f32_args_t &a, uint64_t begin,
                     uint64_t end, uint32_t isa)
{
    constexpr uint32_t BQ = 16;
    constexpr uint32_t BK = 64;
    const uint32_t D = a.head_dim;
    const uint64_t per_head_q = static_cast<uint64_t>(a.seq_q) * D;
    const uint64_t per_head_k = static_cast<uint64_t>(a.seq_k) * D;
    const uint64_t per_head_mask = static_cast<uint64_t>(a.seq_q) * a.seq_k;
    const uint64_t q_blocks = (a.seq_q + BQ - 1) / BQ;
    const float scale = a.scale > 0.0f ? a.scale :
                        1.0f / std::sqrt(static_cast<float>(D));

    thread_local std::vector<float> scores;
    thread_local std::vector<float> probs;
    thread_local std::vector<float> out_acc;
    thread_local std::vector<float> m;
    thread_local std::vector<float> l;
    scores.resize(BQ * BK);
    probs.resize(BQ * BK);
    out_acc.resize(BQ * D);
    m.resize(BQ);
    l.resize(BQ);

    for (uint64_t job = begin; job < end; ++job) {
        uint64_t qb = job % q_blocks;
        uint64_t bh = job / q_blocks;
        uint32_t b = static_cast<uint32_t>(bh / a.heads);
        uint32_t h = static_cast<uint32_t>(bh % a.heads);
        uint32_t q0 = static_cast<uint32_t>(qb * BQ);
        uint32_t qc = std::min<uint32_t>(BQ, a.seq_q - q0);
        const float *Qh = a.Q + (static_cast<uint64_t>(b) * a.heads + h) *
                                    per_head_q;
        const float *Kh = a.K + (static_cast<uint64_t>(b) * a.heads + h) *
                                    per_head_k;
        const float *Vh = a.V + (static_cast<uint64_t>(b) * a.heads + h) *
                                    per_head_k;
        const float *Mh = a.mask ? a.mask +
                          (static_cast<uint64_t>(b) * a.heads + h) *
                              per_head_mask : nullptr;
        float *Oh = a.O + (static_cast<uint64_t>(b) * a.heads + h) *
                              per_head_q;
        for (uint32_t i = 0; i < qc; ++i) {
            m[i] = -std::numeric_limits<float>::infinity();
            l[i] = 0.0f;
            std::fill(out_acc.begin() + i * D,
                      out_acc.begin() + (i + 1) * D, 0.0f);
        }
        const uint32_t causal_offset = a.seq_k > a.seq_q ?
                                       a.seq_k - a.seq_q : 0;
        for (uint32_t k0 = 0; k0 < a.seq_k; k0 += BK) {
            uint32_t kc = std::min<uint32_t>(BK, a.seq_k - k0);
            /* Scores. */
            for (uint32_t i = 0; i < qc; ++i) {
                uint32_t qi = q0 + i;
                const float *qrow = Qh + static_cast<uint64_t>(qi) * D;
                for (uint32_t j = 0; j < kc; ++j) {
                    uint32_t ki = k0 + j;
                    float s;
                    if (a.causal && ki > qi + causal_offset) {
                        s = -std::numeric_limits<float>::infinity();
                    } else {
                        s = dot_dispatch(qrow,
                            Kh + static_cast<uint64_t>(ki) * D, D, isa) *
                            scale;
                        if (Mh)
                            s += Mh[static_cast<uint64_t>(qi) * a.seq_k + ki];
                    }
                    scores[i * BK + j] = s;
                }
            }
            /* Online softmax and V accumulation. */
            for (uint32_t i = 0; i < qc; ++i) {
                float block_max = -std::numeric_limits<float>::infinity();
                for (uint32_t j = 0; j < kc; ++j)
                    block_max = std::max(block_max, scores[i * BK + j]);
                if (!std::isfinite(block_max)) continue;
                float new_m = std::max(m[i], block_max);
                float alpha = fast_expf(m[i] - new_m);
                float sum_p = 0.0f;
                for (uint32_t j = 0; j < kc; ++j) {
                    float p = fast_expf(scores[i * BK + j] - new_m);
                    probs[i * BK + j] = p;
                    sum_p += p;
                }
                l[i] = l[i] * alpha + sum_p;
                float *acc_row = out_acc.data() + i * D;
                for (uint32_t d = 0; d < D; ++d) acc_row[d] *= alpha;
                for (uint32_t j = 0; j < kc; ++j) {
                    float p = probs[i * BK + j];
                    if (p == 0.0f) continue;
                    axpy_dispatch(acc_row,
                        Vh + static_cast<uint64_t>(k0 + j) * D, p, D, isa);
                }
                m[i] = new_m;
            }
        }
        for (uint32_t i = 0; i < qc; ++i) {
            float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
            float *dst = Oh + static_cast<uint64_t>(q0 + i) * D;
            const float *src = out_acc.data() + i * D;
            for (uint32_t d = 0; d < D; ++d) dst[d] = src[d] * inv;
        }
    }
}

} /* anonymous namespace */

/* ------------------------------------------------------------------------- */
/* Public internal kernel entry points                                        */
/* ------------------------------------------------------------------------- */

void kernel_rmsnorm_rows(const vnpu_rmsnorm_f32_args_t &a, uint32_t begin,
                         uint32_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) { rmsnorm_avx2_rows(a, begin, end); return; }
#else
    (void)isa;
#endif
    rmsnorm_scalar_rows(a, begin, end);
}

void kernel_softmax_rows(const vnpu_softmax_f32_args_t &a, uint32_t begin,
                         uint32_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) { softmax_avx2_rows(a, begin, end); return; }
#else
    (void)isa;
#endif
    softmax_scalar_rows(a, begin, end);
}

void kernel_rope_tokens(const vnpu_rope_f32_args_t &a, uint32_t begin,
                        uint32_t end, uint32_t isa)
{
    (void)isa;
    if (!a.Q && !a.K) return;
    uint32_t seq = std::max(a.q_seq, a.k_seq);
    uint64_t per_head = static_cast<uint64_t>(a.batch) * a.heads * seq;
    float theta = a.theta_base > 0.0f ? a.theta_base : 10000.0f;
    float scaling = a.scaling > 0.0f ? a.scaling : 1.0f;
    for (uint64_t idx = begin; idx < end && idx < per_head; ++idx) {
        uint64_t bh = idx / seq;
        uint32_t t = static_cast<uint32_t>(idx % seq);
        float position = static_cast<float>(
            a.positions ? a.positions[t] :
            static_cast<int32_t>(t + a.position_offset));
        if (a.Q && t < a.q_seq) {
            size_t off = (bh * a.q_seq + t) * a.head_dim;
            rotate_vector(a.Q + off, a.head_dim, position, theta, scaling,
                          a.interleaved != 0);
        }
        if (a.K && t < a.k_seq) {
            size_t off = (bh * a.k_seq + t) * a.head_dim;
            rotate_vector(a.K + off, a.head_dim, position, theta, scaling,
                          a.interleaved != 0);
        }
    }
}

void kernel_swiglu_range(const vnpu_swiglu_f32_args_t &a, uint64_t begin,
                         uint64_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) {
        swiglu_avx2_range(a, begin, end);
        return;
    }
#else
    (void)isa;
#endif
    swiglu_scalar_range(a, begin, end);
}

void kernel_gelu_range(const vnpu_unary_f32_args_t &a, uint64_t begin,
                       uint64_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) {
        gelu_avx2_range(a, begin, end);
        return;
    }
#else
    (void)isa;
#endif
    gelu_scalar_range(a, begin, end);
}

void kernel_add_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) {
        add_avx2_range(a, begin, end);
        return;
    }
#else
    (void)isa;
#endif
    add_scalar_range(a, begin, end);
}

void kernel_mul_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end, uint32_t isa)
{
#if VNPU_X86
    if (isa == ISA_AVX2) {
        mul_avx2_range(a, begin, end);
        return;
    }
#else
    (void)isa;
#endif
    mul_scalar_range(a, begin, end);
}

void kernel_quantize_i8_range(const vnpu_quantize_i8_args_t &a,
                              uint64_t begin, uint64_t end, uint32_t isa)
{
    (void)isa;
    switch (a.axis) {
    case VNPU_QUANT_PER_ROW:
        quantize_rows(a, begin, end);
        break;
    case VNPU_QUANT_PER_COLUMN:
        quantize_columns(a, begin, end);
        break;
    case VNPU_QUANT_PER_TENSOR:
    default:
        quantize_tensor(a, begin, end);
        break;
    }
}

void kernel_attention_range(const vnpu_attention_f32_args_t &a,
                            uint64_t begin, uint64_t end, uint32_t isa)
{
    attention_range(a, begin, end, isa);
}

} /* namespace vnpu */
