/*
 * internal.h - shared C++ internals for the Virtual NPU runtime.
 * Not installed and not part of the stable ABI.
 */
#ifndef VNPU_INTERNAL_H
#define VNPU_INTERNAL_H

#include "vnpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#  include <immintrin.h>
#  define VNPU_X86 1
#else
#  define VNPU_X86 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define VNPU_NEON 1
#else
#  define VNPU_NEON 0
#endif

#if defined(__linux__)
#  include <unistd.h>
#  include <pthread.h>
#  include <sys/mman.h>
#endif

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#endif

namespace vnpu {

using Clock = std::chrono::steady_clock;

inline uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

inline uint64_t align_up(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) return value;
    return (value + alignment - 1) / alignment * alignment;
}

inline uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    return b ? (a + b - 1) / b : 0;
}

void *aligned_alloc_bytes(size_t alignment, size_t bytes);
void aligned_free_bytes(void *ptr);
void *reserve_virtual_memory(size_t bytes);
void release_virtual_memory(void *ptr, size_t bytes);

inline float fast_expf(float x)
{
    /* Fast exp approximation, adequate for GELU/SiLU/softmax when clamped. */
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) return std::numeric_limits<float>::infinity();
    const float LOG2E = 1.4426950408889634f;
    float z = x * LOG2E;
    float zi = std::floor(z);
    float f = z - zi;
    /* 2^f polynomial, minimax-ish over [0,1). */
    float p = 1.0f + f * (0.6931471805599453f +
              f * (0.2402265069591007f +
              f * (0.05550410866482158f +
              f * (0.009618129107628477f +
              f *  0.0013333558146428443f))));
#if defined(__x86_64__) || defined(__i386__)
    /* ldexp is exact for the integral exponent and avoids libm exp overhead. */
    return std::ldexp(p, static_cast<int>(zi));
#else
    return std::ldexp(p, static_cast<int>(zi));
#endif
}

inline float silu(float x) { return x / (1.0f + fast_expf(-x)); }

inline float gelu_tanh(float x)
{
    static const float k0 = 0.7978845608028654f;
    static const float k1 = 0.035677408136300125f;
    float x3 = x * x * x;
    float inner = k0 * (x + k1 * x3);
    float t = 2.0f / (1.0f + fast_expf(-2.0f * inner)) - 1.0f;
    return 0.5f * x * (1.0f + t);
}

inline float apply_epilogue(float value, uint32_t epilogue)
{
    switch (epilogue) {
    case VNPU_EPILOGUE_RELU:
        return value > 0.0f ? value : 0.0f;
    case VNPU_EPILOGUE_SILU:
        return silu(value);
    case VNPU_EPILOGUE_GELU_TANH:
    case VNPU_EPILOGUE_GELU_ERF:
        return gelu_tanh(value);
    case VNPU_EPILOGUE_NONE:
    default:
        return value;
    }
}

inline void debug_log(const char *fmt, ...)
{
    if (!std::getenv("VNPU_DEBUG")) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[vnpu] ");
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
}

/* Never allow a C++ exception to cross the C ABI. */
template <typename F>
inline vnpu_status_t guard(F &&fn) noexcept
{
    try {
        return fn();
    } catch (const std::bad_alloc &) {
        return VNPU_ERR_OUT_OF_MEMORY;
    } catch (const std::exception &e) {
        debug_log("exception: %s", e.what());
        return VNPU_ERR_INTERNAL;
    } catch (...) {
        return VNPU_ERR_INTERNAL;
    }
}

/* ------------------------------------------------------------------------- */
/* CPU feature detection                                                     */
/* ------------------------------------------------------------------------- */

enum IsaKind : uint32_t {
    ISA_SCALAR = 0,
    ISA_AVX2   = 1,
    ISA_AVX512 = 2,
    ISA_NEON   = 3
};

struct CpuFeatures {
    uint32_t isa = ISA_SCALAR;
    uint64_t bits = VNPU_FEATURE_SCALAR;
    uint32_t simd_bits = 0;
    uint32_t logical_cpus = 1;
    /* Physical cores.  SMT siblings share L1/L2, and these kernels depend on
     * L1 residency, so the default unit count follows this rather than the
     * logical count (measured 2/2 on the reference VM; on a 14c/18t hybrid
     * laptop it keeps the device from oversubscribing). */
    uint32_t physical_cpus = 1;
    uint64_t l1d = 32 * 1024;
    uint64_t l2 = 256 * 1024;
    uint64_t l3 = 8 * 1024 * 1024;
    uint64_t total_memory = 0;
    std::string arch;
    std::string backend;
};

CpuFeatures detect_cpu_features();
const char *isa_name(uint32_t isa);

/* ------------------------------------------------------------------------- */
/* Thread pool with a deadlock-free parallel_for                              */
/* ------------------------------------------------------------------------- */

struct ParallelState {
    uint64_t n = 0;
    uint64_t grain = 1;
    uint64_t chunks = 0;
    uint32_t worker_count = 0;
    std::atomic<uint64_t> next{0};
    std::atomic<uint64_t> remaining{0};
    std::atomic<uint64_t> workers_done{0};
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    std::mutex error_mutex;
    std::exception_ptr first_error;
};

class ThreadPool {
public:
    explicit ThreadPool(uint32_t num_threads, bool pin_threads);
    ~ThreadPool();

    uint32_t size() const { return static_cast<uint32_t>(workers_.size()); }
    void enqueue(std::function<void()> task);

    /* Batch lease.
     *
     * A batch runs its commands back to back, and every command ends in its own
     * parallel_for -- a sleep/wake round trip per command.  While a lease is
     * held, workers spin briefly for the next task instead of blocking, so the
     * gap between two commands of one batch costs a few pauses rather than a
     * futex sleep.  Bounded by kLeaseSpinUs, so an idle pool still parks. */
    static constexpr uint32_t kLeaseSpinUs = 8;

    void lease_acquire()
    {
        lease_depth_.fetch_add(1, std::memory_order_acq_rel);
    }

    void lease_release()
    {
        if (lease_depth_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            /* Last lease gone: let spinning workers fall through to the wait
             * instead of finishing their remaining spin budget. */
            std::lock_guard<std::mutex> lock(mutex_);
            cv_.notify_all();
        }
    }

    struct ScopedLease {
        ThreadPool *pool;
        explicit ScopedLease(ThreadPool *p) : pool(p) { pool->lease_acquire(); }
        ~ScopedLease() { pool->lease_release(); }
        ScopedLease(const ScopedLease &) = delete;
        ScopedLease &operator=(const ScopedLease &) = delete;
    };

    /* Per-unit speed weights (index 0 is the issuing thread).  Empty or uniform
     * means every unit runs at the same rate, so dynamic chunk stealing is
     * used unchanged. */
    void set_unit_weights(std::vector<float> w)
    {
        unit_weights_ = std::move(w);
        weighted_ = false;
        if (unit_weights_.empty()) return;
        float lo = unit_weights_[0], hi = unit_weights_[0];
        for (float x : unit_weights_) {
            if (!(x > 0.0f)) x = 1.0f;
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        weighted_ = (hi > lo * 1.05f);
    }

    template <typename Fn>
    void parallel_for(uint64_t n, uint64_t grain, Fn fn)
    {
        if (n == 0) return;
        grain = std::max<uint64_t>(1, grain);
        const uint64_t chunks = (n + grain - 1) / grain;
        /* A single chunk is pure launch cost if it is handed to a worker:
         * decoding issues many tiny commands, and each one used to pay a
         * condvar wake (~15-35 us) instead of running inline. */
        if (workers_.empty() || chunks <= 1) {
            fn(0, n, 0);
            return;
        }
        auto state = std::make_shared<ParallelState>();
        state->n = n;
        state->grain = grain;
        state->chunks = chunks;
        /* The calling thread is itself a compute unit, so only the remaining
         * units are woken: a 2-unit device runs two threads, not three. */
        state->worker_count = static_cast<uint32_t>(
            std::min<uint64_t>(workers_.size() - 1, chunks - 1));
        state->remaining.store(state->chunks);

        /* Heterogeneous units: split the range once, proportionally to each
         * unit's speed, instead of letting fast units race ahead and then idle
         * at the barrier.  That only pays for a coarse partition; with many
         * chunks the atomic counter already hands the fast unit proportionally
         * more work, and a fixed ratio would cap it (measured: a 2:1 split of
         * 32 tiles cost 1.33x on identical units). */
        if (weighted_ && state->worker_count > 0 &&
            chunks <= 2 * (state->worker_count + 1)) {
            const uint32_t units = state->worker_count + 1;
            std::vector<float> w(units, 1.0f);
            for (uint32_t u = 0; u < units && u < unit_weights_.size(); ++u)
                if (unit_weights_[u] > 0.0f) w[u] = unit_weights_[u];
            double total = 0.0;
            for (float x : w) total += x;
            for (uint32_t u = 1; u < units; ++u) {
                double before = 0.0, after = 0.0;
                for (uint32_t i = 0; i < units; ++i) {
                    if (i < u) before += w[i];
                    if (i <= u) after += w[i];
                }
                uint64_t lo = static_cast<uint64_t>(
                    (before / total) * static_cast<double>(n));
                uint64_t hi = (u + 1 == units) ? n
                    : static_cast<uint64_t>((after / total) *
                                            static_cast<double>(n));
                /* Worker ids stay in [0, worker_count) so the issuing thread
                 * keeps owning id worker_count; otherwise the two collide on
                 * the per-unit Cell scratch buffer. */
                enqueue([state, &fn, u, lo, hi]() {
                    try {
                        if (lo < hi) fn(lo, hi, u - 1);
                    } catch (...) {
                        std::lock_guard<std::mutex> guard(state->error_mutex);
                        if (!state->first_error)
                            state->first_error = std::current_exception();
                    }
                    state->workers_done.fetch_add(1,
                        std::memory_order_acq_rel);
                    state->wait_cv.notify_all();
                });
            }
            /* hi(u-1) and lo(u) are the same expression, so truncation can
             * never open a gap between the issuer's range and the workers'. */
            uint64_t c_hi = (units == 1) ? n : static_cast<uint64_t>(
                (static_cast<double>(w[0]) / total) * static_cast<double>(n));
            if (c_hi > 0) {
                try {
                    fn(0, c_hi, state->worker_count);
                } catch (...) {
                    std::lock_guard<std::mutex> guard(state->error_mutex);
                    if (!state->first_error)
                        state->first_error = std::current_exception();
                }
            }
            std::unique_lock<std::mutex> lock(state->wait_mutex);
            state->wait_cv.wait(lock, [&]() {
                return state->workers_done.load(std::memory_order_acquire) ==
                       state->worker_count;
            });
            if (state->first_error)
                std::rethrow_exception(state->first_error);
            return;
        }

        for (uint32_t w = 0; w < state->worker_count; ++w) {
            enqueue([state, &fn, w]() {
                for (;;) {
                    uint64_t begin = state->next.fetch_add(
                        state->grain, std::memory_order_relaxed);
                    if (begin >= state->n) break;
                    uint64_t end = std::min<uint64_t>(state->n,
                                                      begin + state->grain);
                    try {
                        fn(begin, end, w);
                    } catch (...) {
                        std::lock_guard<std::mutex> guard(state->error_mutex);
                        if (!state->first_error)
                            state->first_error = std::current_exception();
                    }
                    {
                        std::lock_guard<std::mutex> guard(state->wait_mutex);
                        state->remaining.fetch_sub(1,
                            std::memory_order_acq_rel);
                    }
                    state->wait_cv.notify_all();
                }
                {
                    std::lock_guard<std::mutex> guard(state->wait_mutex);
                    state->workers_done.fetch_add(1,
                        std::memory_order_acq_rel);
                }
                state->wait_cv.notify_all();
            });
        }
        /* The issuing thread drains chunks too: it starts on real work at once
         * instead of blocking on a condvar wake. */
        for (;;) {
            uint64_t begin = state->next.fetch_add(state->grain,
                                                  std::memory_order_relaxed);
            if (begin >= state->n) break;
            uint64_t end = std::min<uint64_t>(state->n, begin + state->grain);
            try {
                fn(begin, end, state->worker_count);
            } catch (...) {
                std::lock_guard<std::mutex> guard(state->error_mutex);
                if (!state->first_error)
                    state->first_error = std::current_exception();
                state->remaining.fetch_sub(1, std::memory_order_acq_rel);
                continue;
            }
            {
                std::lock_guard<std::mutex> guard(state->wait_mutex);
                state->remaining.fetch_sub(1, std::memory_order_acq_rel);
            }
            state->wait_cv.notify_all();
        }
        std::unique_lock<std::mutex> lock(state->wait_mutex);
        state->wait_cv.wait(lock, [&]() {
            return state->remaining.load(std::memory_order_acquire) == 0 &&
                   state->workers_done.load(std::memory_order_acquire) ==
                       state->worker_count;
        });
        if (state->first_error) std::rethrow_exception(state->first_error);
    }

private:
    void worker_loop(uint32_t index);
    /* Try to take a task without sleeping.  Returns false after the spin
     * budget is spent, leaving the caller to block on the condvar. */
    bool spin_for_task(std::function<void()> &task);

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    mutable    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    bool pin_threads_ = false;
    std::vector<float> unit_weights_;
    bool weighted_ = false;
    /* Mirrors tasks_.size() so the lease spin can poll without the mutex. */
    std::atomic<uint32_t> pending_tasks_{0};
    std::atomic<uint32_t> lease_depth_{0};
};

/* ------------------------------------------------------------------------- */
/* Sherlock tracing                                                          */
/* ------------------------------------------------------------------------- */

struct TraceEvent {
    std::string name;
    std::string category;
    uint32_t opcode = 0;
    uint32_t unit = 0;
    uint32_t thread_id = 0;
    uint64_t start_ns = 0;
    uint64_t duration_ns = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t flops = 0;
};

struct TraceState {
    static constexpr size_t kMaxEvents = 200000;
    std::atomic<bool> enabled{false};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> dispatch_ns{0};
    std::atomic<uint64_t> memory_ns{0};
    std::atomic<uint64_t> compute_ns{0};
    std::atomic<uint64_t> sync_ns{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> flops{0};
    std::atomic<uint64_t> overhead_ns{0};
    std::array<std::atomic<uint64_t>, VNPU_OP_COUNT> opcode_ns{};
    std::array<std::atomic<uint64_t>, VNPU_OP_COUNT> opcode_count{};
    std::vector<TraceEvent> events;
    mutable std::mutex mutex;
    std::string trace_file;

    TraceState() {
        for (auto &x : opcode_ns) x.store(0);
        for (auto &x : opcode_count) x.store(0);
    }

    void reset();
    void record(const TraceEvent &event);
    void get_summary(vnpu_trace_summary_t *summary) const;
    vnpu_status_t dump_text(const char *path) const;
    vnpu_status_t write_chrome(const char *path) const;
};

class ScopedTrace {
public:
    ScopedTrace(TraceState *trace, std::string name, uint32_t opcode,
                uint32_t unit, uint32_t category = 0);
    ~ScopedTrace();
    void add_bytes(uint64_t read_bytes, uint64_t write_bytes);
    void add_flops(uint64_t value);
    void set_compute();
    void set_memory();
    void set_sync();
    void set_dispatch();
    void cancel();

private:
    TraceState *trace_;
    TraceEvent event_;
    uint64_t start_ns_;
    uint64_t category_;
    bool active_;
};

/* ------------------------------------------------------------------------- */
/* Kernel data structures                                                    */
/* ------------------------------------------------------------------------- */

struct TileJob {
    uint32_t m0 = 0;
    uint32_t n0 = 0;
    uint32_t tm = 0;
    uint32_t tn = 0;
    uint32_t tile_m = 0;
    uint32_t tile_n = 0;
    uint32_t tile_k = 0;
    uint8_t cull = 0;
    uint8_t pad[3] = {0, 0, 0};
};

struct GemmF32Params {
    const float *A = nullptr;
    const float *B = nullptr;
    const float *bias = nullptr;
    float *C = nullptr;
    uint32_t M = 0, K = 0, N = 0;
    uint32_t lda = 0, ldb = 0, ldc = 0;
    uint32_t trans_a = 0, trans_b = 0;
    float alpha = 1.0f, beta = 0.0f;
    uint32_t epilogue = 0;
    uint32_t unit = 0;
    uint32_t isa = ISA_SCALAR;
    uint32_t tile_m = 0, tile_k = 0, tile_n = 0;
    bool pack_b = true;
    const uint8_t *tile_mask = nullptr;
    uint32_t mask_tile_m = 0, mask_tile_n = 0;
    const std::vector<TileJob> *tiles = nullptr;
    /* Cell local-store scratch, one buffer per worker. */
    std::vector<std::vector<uint8_t>> *local_stores = nullptr;
    uint64_t local_store_bytes = 0;
};

struct PackedI8View {
    const uint8_t *packed = nullptr;      /* VNNI layout [n_block][k4][16][4] */
    const int8_t *original = nullptr;     /* canonical row-major KxN */
    const float *scales = nullptr;        /* length N */
    const int32_t *column_sums = nullptr; /* length N */
    uint32_t K = 0, N = 0;
    uint32_t Kpad = 0, Npad = 0;
    uint64_t packed_bytes = 0;
};

struct GemmI8Params {
    const int8_t *A = nullptr;
    const float *scale_a = nullptr;
    const float *bias = nullptr;
    float *C = nullptr;
    uint32_t M = 0, K = 0, N = 0;
    uint32_t lda = 0, ldc = 0;
    uint32_t epilogue = 0;
    uint32_t unit = 0;
    uint32_t isa = ISA_SCALAR;
    bool has_vnni = false;
    uint32_t tile_m = 0, tile_k = 0, tile_n = 0;
    /* Panel-packed B for the AVX2 path (VNPU_PACK=0 selects the row-major
     * reads, which is the A/B control). */
    bool pack_b = true;
    PackedI8View B;
    const std::vector<TileJob> *tiles = nullptr;
    std::vector<std::vector<uint8_t>> *local_stores = nullptr;
    uint64_t local_store_bytes = 0;
};

/* kernels.cpp */
void kernel_gemm_f32_range(const GemmF32Params &p, size_t begin, size_t end,
                           uint32_t worker_id);
void kernel_gemm_i8_range(const GemmI8Params &p, size_t begin, size_t end,
                          uint32_t worker_id);

/* llm.cpp */
void kernel_rmsnorm_rows(const vnpu_rmsnorm_f32_args_t &a, uint32_t begin,
                         uint32_t end, uint32_t isa);
void kernel_softmax_rows(const vnpu_softmax_f32_args_t &a, uint32_t begin,
                         uint32_t end, uint32_t isa);
void kernel_rope_tokens(const vnpu_rope_f32_args_t &a, uint32_t begin,
                        uint32_t end, uint32_t isa);
void kernel_swiglu_range(const vnpu_swiglu_f32_args_t &a, uint64_t begin,
                         uint64_t end, uint32_t isa);
void kernel_gelu_range(const vnpu_unary_f32_args_t &a, uint64_t begin,
                       uint64_t end, uint32_t isa);
void kernel_add_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end, uint32_t isa);
void kernel_mul_range(const vnpu_binary_f32_args_t &a, uint64_t begin,
                      uint64_t end, uint32_t isa);
void kernel_quantize_i8_range(const vnpu_quantize_i8_args_t &a, uint64_t begin,
                              uint64_t end, uint32_t isa);
void kernel_attention_range(const vnpu_attention_f32_args_t &a, uint64_t begin,
                            uint64_t end, uint32_t isa);

/* scheduler.cpp */
vnpu_status_t schedule_command(vnpu_device_t *device, vnpu_stream_t *stream,
                               const vnpu_command_t &cmd, uint32_t flags);
vnpu_status_t build_gemm_tiles(vnpu_device_t *device,
                               const vnpu_matmul_f32_args_t &args,
                               std::vector<TileJob> &tiles);

/* cluster.cpp */
void cluster_register_builtins();

} /* namespace vnpu */

/* ------------------------------------------------------------------------- */
/* Opaque C objects are C++ structs with the same tag names.                  */
/* ------------------------------------------------------------------------- */

struct vnpu_event {
    std::mutex mutex;
    std::condition_variable cv;
    bool signaled = false;
    uint32_t generation = 0;
    uint64_t command_count = 0;
    vnpu_status_t status = VNPU_OK;

    void signal(vnpu_status_t result = VNPU_OK);
    vnpu_status_t wait(uint32_t timeout_ms);
};

struct vnpu_buffer {
    vnpu_device_t *device = nullptr;
    void *host_ptr = nullptr;
    uint64_t bytes = 0;
    uint64_t handle = 0;
    bool owned = true;
};

struct vnpu_packed_weights {
    vnpu_device_t *device = nullptr;
    std::vector<int8_t> original;      /* canonical KxN */
    std::vector<uint8_t> packed;       /* VNNI layout */
    std::vector<float> scales;         /* length N */
    std::vector<int32_t> column_sums;  /* length N */
    uint32_t K = 0;
    uint32_t N = 0;
    uint32_t Kpad = 0;
    uint32_t Npad = 0;
    uint32_t layout = 0;
    std::mutex mutex;
};

struct CommandBatch {
    std::vector<vnpu_command_t> commands;
    vnpu_event_t *event = nullptr; /* borrowed */
    uint32_t flags = 0;
    uint64_t batch_id = 0;
    uint64_t queued_ns = 0;
};

struct vnpu_stream {
    vnpu_device_t *device = nullptr;
    int priority = 0;
    std::thread worker;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<CommandBatch> queue;
    bool stopping = false;
    uint64_t next_batch_id = 1;
    uint64_t completed_batches = 0;
    std::atomic<uint64_t> pending_commands{0};
    std::atomic<uint64_t> last_error{0};

    void run();
    void submit(CommandBatch batch);
    vnpu_status_t sync();
};

struct vnpu_device {
    vnpu_config_t config{};
    vnpu_device_info_t info{};
    vnpu::CpuFeatures cpu;
    std::unique_ptr<vnpu::ThreadPool> pool;
    vnpu::TraceState trace;
    std::array<std::atomic<uint32_t>, VNPU_REG_COUNT> regs{};
    std::atomic<uint64_t> commands_submitted{0};
    std::atomic<uint64_t> commands_completed{0};
    std::atomic<uint64_t> kernel_ns{0};
    std::atomic<uint64_t> dispatch_ns{0};
    std::atomic<uint64_t> memory_ns{0};
    std::atomic<uint64_t> sync_ns{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> flops{0};
    std::atomic<uint64_t> tiles_dispatched{0};
    std::atomic<uint64_t> tiles_culled{0};
    std::atomic<uint64_t> dma_bytes{0};
    std::atomic<uint64_t> interrupts{0};
    std::atomic<uint32_t> busy{0};
    std::atomic<uint32_t> error{0};
    std::atomic<uint32_t> active_workers{0};
    std::atomic<uint32_t> last_opcode{0};
    std::atomic<uint32_t> tile_m{0};
    std::atomic<uint32_t> tile_k{0};
    std::atomic<uint32_t> tile_n{0};
    /* Panel-packed f32 NN path (see kernels.cpp pack_b_panel).  VNPU_PACK=0
     * disables it, which exists so the two structures can be A/B measured on
     * the same binary. */
    bool pack_b = true;
    /* Hold the thread pool hot for the duration of a submitted batch
     * (VNPU_LEASE=0 disables, for A/B measurement). */
    bool batch_lease = true;
    std::vector<std::vector<uint8_t>> local_stores;
    std::vector<vnpu_stream_t *> streams;
    vnpu_interrupt_cb_t irq_cb = nullptr;
    void *irq_user = nullptr;
    std::mutex irq_mutex;
    std::mutex stream_mutex;
    std::string name;
    std::string trace_file;

    uint32_t isa() const { return cpu.isa; }
    /* No VNNI kernel is compiled: AVX2+FMA (Zen 3) is the top rung. */
    bool has_vnni() const { return false; }
    void raise_irq(uint32_t bits);
};

/* Cluster objects.  Full definitions live here so all TUs can see them. */
struct RemoteConn {
    intptr_t fd = -1;          /* socket_t stored as an integer */
    std::mutex mutex;
    bool healthy = true;
};

struct vnpu_remote_node {
    std::string host;
    uint16_t port = 0;
    std::string auth_token;
    uint32_t weight = 1;
    uint32_t flags = 0;
    std::vector<std::unique_ptr<RemoteConn>> conns;
    std::atomic<uint32_t> next_conn{0};
    std::mutex mutex;
    uint64_t jobs = 0;
    uint64_t failures = 0;
    double time_ns = 0.0;
    bool connected = false;
};

struct vnpu_weight_handle {
    std::string key;
    uint64_t id = 0;            /* stable network weight id */
    std::vector<float> B;       /* canonical KxN */
    uint32_t K = 0;
    uint32_t N = 0;
    uint32_t trans_b = 0;
    std::mutex mutex;
};

struct ClusterWorker {
    enum Kind { LOCAL, REMOTE } kind = LOCAL;
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    vnpu_remote_node_t *remote = nullptr;
    RemoteConn *conn = nullptr;
    uint32_t connection_index = 0;
    uint32_t weight = 1;
    uint64_t jobs = 0;
    uint64_t failures = 0;
    double time_ns = 0.0;
    std::mutex mutex;
};

struct vnpu_cluster {
    vnpu_cluster_config_t config{};
    std::string name;
    std::string auth_token;
    std::vector<std::unique_ptr<ClusterWorker>> workers;
    std::vector<std::unique_ptr<vnpu_remote_node_t>> remotes;
    std::vector<std::unique_ptr<vnpu_weight_handle_t>> weights;
    std::vector<vnpu_stream_t *> owned_streams;
    std::vector<vnpu_event_t *> owned_events;
    std::atomic<uint64_t> local_jobs{0};
    std::atomic<uint64_t> remote_jobs{0};
    std::atomic<uint64_t> total_jobs{0};
    std::atomic<uint64_t> failed_jobs{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> culled_tiles{0};
    std::atomic<uint64_t> local_time_ns{0};
    std::atomic<uint64_t> remote_time_ns{0};
    std::atomic<uint64_t> last_speedup_x1000{0};
    std::atomic<uint64_t> last_wall_time_ns{0};
    std::mutex mutex;
};

#endif /* VNPU_INTERNAL_H */
