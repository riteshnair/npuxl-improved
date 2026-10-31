/*
 * runtime.cpp - device lifecycle, virtual MMIO, streams, events, buffers and
 * Sherlock tracing.  No third-party dependencies.
 */
#include "../internal.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace vnpu {

/* ------------------------------------------------------------------------- */
/* Platform helpers                                                          */
/* ------------------------------------------------------------------------- */

void *aligned_alloc_bytes(size_t alignment, size_t bytes)
{
    if (bytes == 0) return nullptr;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    /* posix_memalign requires a power-of-two multiple of sizeof(void*). */
    size_t a = alignment;
    while ((a & (a - 1)) != 0) a++;
#if defined(_WIN32)
    return _aligned_malloc(bytes, a);
#else
    void *ptr = nullptr;
    if (posix_memalign(&ptr, a, bytes) != 0) return nullptr;
    return ptr;
#endif
}

void aligned_free_bytes(void *ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void *reserve_virtual_memory(size_t bytes)
{
#if defined(__linux__)
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    return p;
#else
    return calloc(1, bytes);
#endif
}

void release_virtual_memory(void *ptr, size_t bytes)
{
    if (!ptr) return;
#if defined(__linux__)
    munmap(ptr, bytes);
#else
    (void)bytes;
    free(ptr);
#endif
}

CpuFeatures detect_cpu_features()
{
    CpuFeatures f;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    f.arch = "x86";
    f.bits |= VNPU_FEATURE_SSE2;
    f.simd_bits = 128;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        f.isa = ISA_AVX2;
        f.simd_bits = 256;
        f.bits |= VNPU_FEATURE_AVX2_FMA;
    }
    /* AVX2+FMA (Zen 3) is the top of the build's instruction ladder.  Wide
     * AVX-512/VNNI paths were dropped: without hardware to measure them on,
     * they were unverifiable code that also downclocked the target laptop. */
#endif
#elif defined(__aarch64__)
    f.arch = "aarch64";
#if VNPU_NEON
    f.isa = ISA_NEON;
    f.simd_bits = 128;
    f.bits |= VNPU_FEATURE_NEON;
#endif
#else
    f.arch = "generic";
#endif
    f.bits |= VNPU_FEATURE_THREADS | VNPU_FEATURE_CLUSTER |
              VNPU_FEATURE_REMOTE | VNPU_FEATURE_TRACE;
    if (f.bits & VNPU_FEATURE_BF16) { /* already set */ }

    unsigned logical = std::thread::hardware_concurrency();
    f.logical_cpus = logical ? logical : 1;
    f.physical_cpus = f.logical_cpus;
#if defined(__linux__)
    /* Count distinct (package, core) pairs: hyperthread siblings share the
     * same pair, so this is the real core count behind the logical one. */
    {
        std::vector<std::pair<std::string, std::string>> cores;
        for (uint32_t cpu = 0; cpu < 4096; ++cpu) {
            char base[128];
            std::snprintf(base, sizeof(base),
                          "/sys/devices/system/cpu/cpu%u/topology/", cpu);
            auto read_one = [&](const char *name, std::string &out) {
                std::string path = std::string(base) + name;
                std::FILE *fp = std::fopen(path.c_str(), "r");
                if (!fp) return false;
                char buf[64] = {0};
                size_t got = std::fread(buf, 1, sizeof(buf) - 1, fp);
                std::fclose(fp);
                if (!got) return false;
                out.assign(buf);
                while (!out.empty() &&
                       (out.back() == '\n' || out.back() == '\r' ||
                        out.back() == ' '))
                    out.pop_back();
                return true;
            };
            /* `core_cpus_list` names the sibling set, which is the reliable
             * key: some hypervisors report the same core_id for every CPU in
             * the guest (this reference VM does), which would collapse them
             * all into one core.  Fall back to (package, core) when absent. */
            std::string siblings;
            if (read_one("core_cpus_list", siblings) && !siblings.empty()) {
                std::pair<std::string, std::string> key{"siblings", siblings};
                if (std::find(cores.begin(), cores.end(), key) == cores.end())
                    cores.push_back(key);
                continue;
            }
            std::string pkg, core;
            if (!read_one("physical_package_id", pkg)) break;
            if (!read_one("core_id", core)) break;
            std::pair<std::string, std::string> key{pkg, core};
            if (std::find(cores.begin(), cores.end(), key) == cores.end())
                cores.push_back(key);
        }
        if (!cores.empty())
            f.physical_cpus = static_cast<uint32_t>(cores.size());
    }
#elif defined(_WIN32)
    {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len) {
            std::vector<uint8_t> buf(len);
            if (GetLogicalProcessorInformationEx(
                    RelationProcessorCore,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                        buf.data()), &len)) {
                uint32_t count = 0;
                DWORD off = 0;
                auto *p = reinterpret_cast<
                    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
                while (off < len) {
                    ++count;
                    off += p->Size;
                    p = reinterpret_cast<
                        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                            buf.data() + off);
                }
                if (count) f.physical_cpus = count;
            }
        }
    }
#endif
#if defined(__linux__)
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (l1 > 0) f.l1d = static_cast<uint64_t>(l1);
    if (l2 > 0) f.l2 = static_cast<uint64_t>(l2);
    if (l3 > 0) f.l3 = static_cast<uint64_t>(l3);
    if (pages > 0 && page_size > 0)
        f.total_memory = static_cast<uint64_t>(pages) *
                         static_cast<uint64_t>(page_size);
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) f.logical_cpus = static_cast<uint32_t>(online);
#elif defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    f.logical_cpus = si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1;
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) f.total_memory = ms.ullTotalPhys;
#endif
    std::string backend = isa_name(f.isa);
    f.backend = backend;
    return f;
}

const char *isa_name(uint32_t isa)
{
    switch (isa) {
    case ISA_AVX2: return "AVX2+FMA";
    case ISA_AVX512: return "AVX512F";
    case ISA_NEON: return "NEON";
    case ISA_SCALAR:
    default: return "scalar";
    }
}

/* ------------------------------------------------------------------------- */
/* Thread pool                                                               */
/* ------------------------------------------------------------------------- */

ThreadPool::ThreadPool(uint32_t num_threads, bool pin_threads)
    : pin_threads_(pin_threads)
{
    num_threads = std::max<uint32_t>(1, num_threads);
    for (uint32_t i = 0; i < num_threads; ++i)
        workers_.emplace_back(&ThreadPool::worker_loop, this, i);
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto &worker : workers_)
        if (worker.joinable()) worker.join();
}

void ThreadPool::worker_loop(uint32_t index)
{
#if defined(__linux__) && defined(VNPU_CONFIG_PIN_THREADS)
    (void)index;
#else
    (void)index;
#endif
    if (pin_threads_) {
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        unsigned n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
        if (n > CPU_SETSIZE) n = CPU_SETSIZE;
        CPU_SET(index % n, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
        SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << (index % (sizeof(DWORD_PTR) * 8)));
#endif
    }
    for (;;) {
        std::function<void()> task;
        /* Inside a batch lease, poll for the next task for a short while
         * before blocking: the next command's parallel_for is microseconds
         * away, and a futex sleep/wake costs more than the spin. */
        bool have_task = false;
        if (lease_depth_.load(std::memory_order_acquire) != 0)
            have_task = spin_for_task(task);
        if (!have_task) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
            pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
        }
        try {
            task();
        } catch (const std::exception &e) {
            debug_log("worker exception: %s", e.what());
        } catch (...) {
            debug_log("worker unknown exception");
        }
    }
}

namespace {
/* Spin hint for the lease path. */
inline void relax_cpu()
{
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}
}  // namespace

void ThreadPool::enqueue(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return;
        tasks_.push_back(std::move(task));
        pending_tasks_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_one();
}

bool ThreadPool::spin_for_task(std::function<void()> &task)
{
    const uint64_t deadline =
        vnpu::now_ns() + static_cast<uint64_t>(kLeaseSpinUs) * 1000ull;
    uint32_t iter = 0;
    while (pending_tasks_.load(std::memory_order_acquire) == 0) {
        if ((++iter & 63u) == 0 && vnpu::now_ns() >= deadline) return false;
        relax_cpu();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) return false;
    task = std::move(tasks_.front());
    tasks_.pop_front();
    pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

/* ------------------------------------------------------------------------- */
/* Sherlock tracing                                                          */
/* ------------------------------------------------------------------------- */

void TraceState::reset()
{
    std::lock_guard<std::mutex> lock(mutex);
    events.clear();
    total_ns.store(0);
    dispatch_ns.store(0);
    memory_ns.store(0);
    compute_ns.store(0);
    sync_ns.store(0);
    bytes_read.store(0);
    bytes_written.store(0);
    flops.store(0);
    for (auto &x : opcode_ns) x.store(0);
    for (auto &x : opcode_count) x.store(0);
    uint64_t t0 = now_ns();
    uint64_t sink = 0;
    for (int i = 0; i < 64; ++i) sink += now_ns();
    uint64_t t1 = now_ns();
    overhead_ns.store((t1 - t0) / 64 + (sink & 1));
}

void TraceState::record(const TraceEvent &event)
{
    if (!enabled.load(std::memory_order_relaxed)) return;
    total_ns.fetch_add(event.duration_ns, std::memory_order_relaxed);
    bytes_read.fetch_add(event.bytes_read, std::memory_order_relaxed);
    bytes_written.fetch_add(event.bytes_written, std::memory_order_relaxed);
    flops.fetch_add(event.flops, std::memory_order_relaxed);
    if (event.category == "memory")
        memory_ns.fetch_add(event.duration_ns, std::memory_order_relaxed);
    else if (event.category == "dispatch")
        dispatch_ns.fetch_add(event.duration_ns, std::memory_order_relaxed);
    else if (event.category == "sync")
        sync_ns.fetch_add(event.duration_ns, std::memory_order_relaxed);
    else
        compute_ns.fetch_add(event.duration_ns, std::memory_order_relaxed);
    uint32_t op = std::min<uint32_t>(event.opcode, VNPU_OP_COUNT - 1);
    opcode_ns[op].fetch_add(event.duration_ns, std::memory_order_relaxed);
    opcode_count[op].fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    if (events.size() < kMaxEvents) events.push_back(event);
}

void TraceState::get_summary(vnpu_trace_summary_t *s) const
{
    if (!s) return;
    std::memset(s, 0, sizeof(*s));
    s->events = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        s->events = events.size();
    }
    s->total_time_ns = total_ns.load();
    s->dispatch_time_ns = dispatch_ns.load();
    s->memory_time_ns = memory_ns.load();
    s->compute_time_ns = compute_ns.load();
    s->sync_time_ns = sync_ns.load();
    s->bytes_read = bytes_read.load();
    s->bytes_written = bytes_written.load();
    s->flops = flops.load();
    s->achieved_gflops = static_cast<double>(s->flops) /
                         static_cast<double>(std::max<uint64_t>(1, s->compute_time_ns));
    s->achieved_gbytes_s = static_cast<double>(s->bytes_read + s->bytes_written) /
                           static_cast<double>(std::max<uint64_t>(1, s->memory_time_ns));
    s->trace_overhead_ns = static_cast<double>(overhead_ns.load());
    s->hot_time_ns = 0;
    s->hot_opcode = VNPU_OP_NOP;
    for (uint32_t i = 0; i < VNPU_OP_COUNT; ++i) {
        uint64_t t = opcode_ns[i].load();
        if (t > s->hot_time_ns) {
            s->hot_time_ns = t;
            s->hot_opcode = i;
        }
    }
}

static const char *category_bucket(uint64_t compute, uint64_t memory,
                                   uint64_t dispatch, uint64_t sync)
{
    uint64_t total = compute + memory + dispatch + sync;
    if (total == 0) return "idle";
    double c = 100.0 * compute / total;
    double m = 100.0 * memory / total;
    double d = 100.0 * dispatch / total;
    if (d > 15.0) return "dispatch-bound (fuse commands / enlarge tiles)";
    if (m > c * 1.3) return "memory-bound (quantize / cache weights / enlarge K tile)";
    if (c > m * 1.3) return "compute-bound (use VNNI / more workers / larger tiles)";
    return "balanced";
}

vnpu_status_t TraceState::dump_text(const char *path) const
{
    return guard([&]() -> vnpu_status_t {
        vnpu_trace_summary_t s;
        get_summary(&s);
        std::FILE *fp = stderr;
        bool close = false;
        if (path && path[0]) {
            fp = std::fopen(path, "w");
            if (!fp) return VNPU_ERR_IO;
            close = true;
        }
        std::fprintf(fp, "=== SHERLOCK VIRTUAL NPU TRACE ===\n");
        std::fprintf(fp, "events              : %llu\n",
                     static_cast<unsigned long long>(s.events));
        std::fprintf(fp, "total               : %.6f ms\n", s.total_time_ns / 1e6);
        std::fprintf(fp, "compute             : %.6f ms (%.1f%%)\n",
                     s.compute_time_ns / 1e6,
                     100.0 * s.compute_time_ns /
                         std::max<uint64_t>(1, s.total_time_ns));
        std::fprintf(fp, "memory              : %.6f ms (%.1f%%)\n",
                     s.memory_time_ns / 1e6,
                     100.0 * s.memory_time_ns /
                         std::max<uint64_t>(1, s.total_time_ns));
        std::fprintf(fp, "dispatch            : %.6f ms (%.1f%%)\n",
                     s.dispatch_time_ns / 1e6,
                     100.0 * s.dispatch_time_ns /
                         std::max<uint64_t>(1, s.total_time_ns));
        std::fprintf(fp, "sync                : %.6f ms (%.1f%%)\n",
                     s.sync_time_ns / 1e6,
                     100.0 * s.sync_time_ns /
                         std::max<uint64_t>(1, s.total_time_ns));
        std::fprintf(fp, "flops               : %llu\n",
                     static_cast<unsigned long long>(s.flops));
        std::fprintf(fp, "compute throughput  : %.3f GFLOP/s\n", s.achieved_gflops);
        std::fprintf(fp, "memory throughput   : %.3f GB/s\n", s.achieved_gbytes_s);
        std::fprintf(fp, "bytes read/written  : %llu / %llu\n",
                     static_cast<unsigned long long>(s.bytes_read),
                     static_cast<unsigned long long>(s.bytes_written));
        std::fprintf(fp, "arithmetic intensity: %.3f FLOP/byte\n",
                     static_cast<double>(s.flops) /
                         static_cast<double>(std::max<uint64_t>(
                             1, s.bytes_read + s.bytes_written)));
        std::fprintf(fp, "hot opcode          : %s (%.3f ms)\n",
                     vnpu_opcode_name(s.hot_opcode), s.hot_time_ns / 1e6);
        std::fprintf(fp, "roofline verdict    : %s\n",
                     category_bucket(s.compute_time_ns, s.memory_time_ns,
                                     s.dispatch_time_ns, s.sync_time_ns));
        std::fprintf(fp, "per-opcode time:\n");
        for (uint32_t i = 0; i < VNPU_OP_COUNT; ++i) {
            uint64_t t = opcode_ns[i].load();
            uint64_t c = opcode_count[i].load();
            if (!c) continue;
            std::fprintf(fp, "  %-22s %8llu calls %10.3f ms  %8.3f us/call\n",
                         vnpu_opcode_name(i),
                         static_cast<unsigned long long>(c), t / 1e6,
                         static_cast<double>(t) / 1000.0 /
                             static_cast<double>(c));
        }
        std::vector<TraceEvent> slow;
        {
            std::lock_guard<std::mutex> lock(mutex);
            slow = events;
        }
        std::sort(slow.begin(), slow.end(),
                  [](const TraceEvent &a, const TraceEvent &b) {
                      return a.duration_ns > b.duration_ns;
                  });
        std::fprintf(fp, "slowest events:\n");
        for (size_t i = 0; i < std::min<size_t>(12, slow.size()); ++i) {
            const auto &e = slow[i];
            std::fprintf(fp,
                         "  %-28s op=%-18s unit=%-8s tid=%-3u %9.3f us "
                         "%llu FLOP %llu B\n",
                         e.name.c_str(), vnpu_opcode_name(e.opcode),
                         vnpu_unit_name(e.unit), e.thread_id,
                         e.duration_ns / 1000.0,
                         static_cast<unsigned long long>(e.flops),
                         static_cast<unsigned long long>(e.bytes_read +
                                                          e.bytes_written));
        }
        std::fprintf(fp, "trace overhead      : %.1f ns/event\n",
                     s.trace_overhead_ns);
        std::fprintf(fp, "===================================\n");
        if (close) std::fclose(fp);
        return VNPU_OK;
    });
}

static std::string json_escape(const std::string &s)
{
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4)
                    << std::setfill('0') << static_cast<int>(c)
                    << std::dec;
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

vnpu_status_t TraceState::write_chrome(const char *path) const
{
    return guard([&]() -> vnpu_status_t {
        if (!path || !path[0]) return VNPU_ERR_INVALID_ARGUMENT;
        std::ofstream out(path, std::ios::trunc);
        if (!out) return VNPU_ERR_IO;
        std::vector<TraceEvent> copy;
        {
            std::lock_guard<std::mutex> lock(mutex);
            copy = events;
        }
        // Timestamps are written RELATIVE to the first event and with fixed
        // precision. Both halves matter, and the default was wrong in a way
        // that looked like a coarse clock: this stream inherits ostream's
        // default 6 significant digits, so an absolute monotonic microsecond
        // stamp near 1.2e9 was emitted as e.g. "1.1953e+09" -- a 10 ms grid.
        // Every command inside one grid cell shared an instant, so a trace
        // could not answer gap or duty-cycle questions at all, while durations
        // (small numbers, same 6 digits) stayed exact and looked trustworthy.
        // Rebasing removes the magnitude; fixed(3) pins 1 ns of resolution.
        uint64_t t0 = copy.empty() ? 0u : copy[0].start_ns;
        for (const auto &e : copy) t0 = std::min(t0, e.start_ns);
        out << std::fixed << std::setprecision(3);
        out << "{\n\"traceEvents\":[\n";
        for (size_t i = 0; i < 8; ++i) {
            out << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":"
                << i << ",\"args\":{\"name\":\"VNPU worker " << i
                << "\"}},\n";
        }
        for (size_t i = 0; i < copy.size(); ++i) {
            const auto &e = copy[i];
            uint32_t tid = std::min<uint32_t>(e.thread_id, 7);
            out << "{\"name\":\"" << json_escape(e.name)
                << "\",\"cat\":\"" << json_escape(e.category)
                << "\",\"ph\":\"X\",\"ts\":" << ((e.start_ns - t0) / 1000.0)
                << ",\"dur\":" << (e.duration_ns / 1000.0)
                << ",\"pid\":1,\"tid\":" << tid
                << ",\"args\":{\"op\":\"" << vnpu_opcode_name(e.opcode)
                << "\",\"unit\":\"" << vnpu_unit_name(e.unit)
                << "\",\"bytes_read\":" << e.bytes_read
                << ",\"bytes_written\":" << e.bytes_written
                << ",\"flops\":" << e.flops << "}}";
            if (i + 1 < copy.size()) out << ',';
            out << '\n';
        }
        out << "],\n\"displayTimeUnit\":\"ns\",\"otherData\":{"
            << "\"version\":\"" << vnpu_version_string() << "\"}}\n";
        return VNPU_OK;
    });
}

ScopedTrace::ScopedTrace(TraceState *trace, std::string name, uint32_t opcode,
                         uint32_t unit, uint32_t category)
    : trace_(trace), category_(category), active_(false)
{
    if (!trace_ || !trace_->enabled.load(std::memory_order_relaxed)) return;
    active_ = true;
    start_ns_ = now_ns();
    event_.name = std::move(name);
    event_.opcode = opcode;
    event_.unit = unit;
    event_.thread_id = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) % 8u);
    event_.start_ns = start_ns_;
}

ScopedTrace::~ScopedTrace()
{
    if (!active_) return;
    event_.duration_ns = now_ns() - start_ns_;
    switch (category_) {
    case 1: event_.category = "memory"; break;
    case 2: event_.category = "sync"; break;
    case 3: event_.category = "dispatch"; break;
    default: event_.category = "compute"; break;
    }
    trace_->record(event_);
}

void ScopedTrace::add_bytes(uint64_t r, uint64_t w)
{
    if (!active_) return;
    event_.bytes_read += r;
    event_.bytes_written += w;
}

void ScopedTrace::add_flops(uint64_t v)
{
    if (!active_) return;
    event_.flops += v;
}

void ScopedTrace::set_compute() { category_ = 0; }
void ScopedTrace::set_memory() { category_ = 1; }
void ScopedTrace::set_sync() { category_ = 2; }
void ScopedTrace::set_dispatch() { category_ = 3; }
void ScopedTrace::cancel() { active_ = false; }

} /* namespace vnpu */

/* ------------------------------------------------------------------------- */
/* Small public helpers                                                      */
/* ------------------------------------------------------------------------- */

extern "C" {

VNPU_API void VNPU_CALL vnpu_config_init(vnpu_config_t *config)
{
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->abi_version = VNPU_ABI_VERSION;
    const char *threads = std::getenv("VNPU_THREADS");
    if (threads) config->num_threads = static_cast<uint32_t>(std::atoi(threads));
    const char *trace = std::getenv("VNPU_TRACE");
    if (trace && std::atoi(trace)) config->flags |= VNPU_CONFIG_TRACE;
    const char *trace_file = std::getenv("VNPU_TRACE_FILE");
    if (trace_file) config->trace_file = trace_file;
}

VNPU_API const char * VNPU_CALL vnpu_status_string(vnpu_status_t status)
{
    switch (status) {
    case VNPU_OK: return "OK";
    case VNPU_ERR_INVALID_ARGUMENT: return "invalid argument";
    case VNPU_ERR_OUT_OF_MEMORY: return "out of memory";
    case VNPU_ERR_UNSUPPORTED: return "unsupported";
    case VNPU_ERR_INVALID_COMMAND: return "invalid command";
    case VNPU_ERR_DEVICE_BUSY: return "device busy";
    case VNPU_ERR_DEVICE_ERROR: return "device error";
    case VNPU_ERR_TIMEOUT: return "timeout";
    case VNPU_ERR_NETWORK: return "network error";
    case VNPU_ERR_AUTH: return "authentication error";
    case VNPU_ERR_PROTOCOL: return "protocol error";
    case VNPU_ERR_NOT_FOUND: return "not found";
    case VNPU_ERR_IO: return "I/O error";
    case VNPU_ERR_INTERNAL:
    default: return "internal error";
    }
}

VNPU_API const char * VNPU_CALL vnpu_version_string(void)
{
    return "vnpu 1.0.0";
}

VNPU_API uint32_t VNPU_CALL vnpu_abi_version(void) { return VNPU_ABI_VERSION; }

VNPU_API const char * VNPU_CALL vnpu_opcode_name(uint32_t opcode)
{
    static const char *names[] = {
        "NOP", "MATMUL_F32", "MATMUL_I8", "RMSNORM_F32", "SOFTMAX_F32",
        "ROPE_F32", "SWIGLU_F32", "GELU_F32", "ADD_F32", "MUL_F32",
        "ATTENTION_F32", "QUANTIZE_I8", "PACK_WEIGHTS_I8", "MTP_HEAD_F32",
        "COPY", "MEMSET", "BARRIER"
    };
    return opcode < (sizeof(names) / sizeof(names[0])) ? names[opcode] : "UNKNOWN";
}

VNPU_API const char * VNPU_CALL vnpu_unit_name(uint32_t unit)
{
    switch (unit) {
    case VNPU_UNIT_AUTO: return "AUTO";
    case VNPU_UNIT_VOODOO: return "VOODOO";
    case VNPU_UNIT_WILDCAT: return "WILDCAT";
    case VNPU_UNIT_CELL: return "CELL";
    case VNPU_UNIT_GPU: return "GPU";
    default: return "UNKNOWN";
    }
}

VNPU_API uint16_t VNPU_CALL vnpu_float_to_bf16(float value)
{
    uint32_t u;
    std::memcpy(&u, &value, 4);
    uint32_t rounding_bias = 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

VNPU_API float VNPU_CALL vnpu_bf16_to_float(uint16_t value)
{
    uint32_t u = static_cast<uint32_t>(value) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

VNPU_API uint16_t VNPU_CALL vnpu_float_to_f16(float value)
{
    uint32_t f;
    std::memcpy(&f, &value, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = f & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u |
                                    (mant ? 0x200u : 0u));
    }
    return static_cast<uint16_t>(sign |
                                (static_cast<uint32_t>(exp) << 10) |
                                ((mant + 0x1000u) >> 13));
}

VNPU_API float VNPU_CALL vnpu_f16_to_float(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 113;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

VNPU_API void VNPU_CALL vnpu_cluster_config_init(vnpu_cluster_config_t *config)
{
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->abi_version = VNPU_ABI_VERSION;
    config->num_streams = 1;
    config->request_timeout_ms = 30000;
    config->cluster_name = "vnpu-cluster";
}

VNPU_API void VNPU_CALL vnpu_speculative_config_init(vnpu_speculative_config_t *config)
{
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->abi_version = VNPU_ABI_VERSION;
    config->gamma = 4;
    config->vocab = 32000;
    config->max_tokens = 16;
    config->eos_token = 2;
    config->seed = 1;
}

VNPU_API void VNPU_CALL vnpu_node_config_init(vnpu_node_config_t *config)
{
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->abi_version = VNPU_ABI_VERSION;
    config->port = 9449;
    config->num_threads = 0;
    config->bind_address = "0.0.0.0";
    config->node_name = "vnpu-node";
}

/* ------------------------------------------------------------------------- */
/* Device MMIO and lifecycle                                                  */
/* ------------------------------------------------------------------------- */

void vnpu_device::raise_irq(uint32_t bits)
{
    uint32_t old = regs[VNPU_REG_IRQ_STATUS / 4].fetch_or(bits);
    uint32_t mask = regs[VNPU_REG_IRQ_MASK / 4].load();
    if ((old & bits) == 0 && (mask & bits)) {
        interrupts.fetch_add(1);
        vnpu_interrupt_cb_t cb;
        void *user;
        {
            std::lock_guard<std::mutex> lock(irq_mutex);
            cb = irq_cb;
            user = irq_user;
        }
        if (cb) cb(this, bits, user);
    }
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_create(const vnpu_config_t *config,
                                                    vnpu_device_t **out_device)
{
    if (!out_device) return VNPU_ERR_INVALID_ARGUMENT;
    *out_device = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        if (config) cfg = *config;
        if (cfg.abi_version == 0) cfg.abi_version = VNPU_ABI_VERSION;
        if (cfg.abi_version != VNPU_ABI_VERSION) return VNPU_ERR_UNSUPPORTED;

        auto *device = new vnpu_device_t();
        device->config = cfg;
        device->cpu = vnpu::detect_cpu_features();
        /* VNPU_FORCE_ISA=scalar|avx2|avx512 is useful for sanitizer builds and
         * deterministic cross-machine comparisons. */
        const char *forced_isa = std::getenv("VNPU_FORCE_ISA");
        if (forced_isa && forced_isa[0]) {
            std::string forced(forced_isa);
            if (forced == "scalar") {
                device->cpu.isa = vnpu::ISA_SCALAR;
                device->cpu.simd_bits = 0;
                device->cpu.bits &= ~(VNPU_FEATURE_AVX2_FMA |
                                      VNPU_FEATURE_AVX512F |
                                      VNPU_FEATURE_AVX512VNNI);
                device->cpu.backend = "scalar(forced)";
            } else if (forced == "avx2") {
                device->cpu.isa = vnpu::ISA_AVX2;
                device->cpu.simd_bits = 256;
                device->cpu.bits |= VNPU_FEATURE_AVX2_FMA;
                device->cpu.bits &= ~(VNPU_FEATURE_AVX512F |
                                      VNPU_FEATURE_AVX512VNNI);
                device->cpu.backend = "AVX2+FMA(forced)";
            } else {
                /* Refuse rather than execute instructions this CPU may not
                 * implement: forcing an unbuilt ISA used to SIGILL. */
                vnpu::debug_log("VNPU_FORCE_ISA=%s ignored: only scalar and "
                                "avx2 are built", forced.c_str());
            }
        }
        /* Default to physical cores, not logical ones.  SMT siblings share the
         * L1/L2 these kernels depend on, and a multiply-add-bound tile gains
         * little from a second thread on the same core while losing cache.  On
         * a 14c/18t hybrid laptop that is 14 units instead of 18. */
        uint32_t def_units = device->cpu.physical_cpus ?
                             device->cpu.physical_cpus :
                             device->cpu.logical_cpus;
        uint32_t threads = cfg.num_threads ? cfg.num_threads : def_units;
        threads = std::min<uint32_t>(threads, std::max<uint32_t>(1, device->cpu.logical_cpus));
        device->pool = std::make_unique<vnpu::ThreadPool>(
            threads, (cfg.flags & VNPU_CONFIG_PIN_THREADS) != 0);
        /* VNPU_UNIT_WEIGHTS=2,2,1,1,1,1,1,1,1 declares each unit's relative
         * speed (index 0 is the issuing thread).  On a hybrid CPU the P units
         * then get proportionally more tiles, which is what stops the E units
         * from setting the pace at every barrier.  Uniform weights keep
         * dynamic chunk stealing, which is better whenever the partition is
         * fine-grained. */
        /* Reference mode: the slowest, simplest path this library can produce
         * (scalar kernels, unpacked B, no pool lease).  Anything that still
         * reproduces here is a caller bug rather than a fast-kernel bug -- the
         * same isolation a software rasterizer gives a graphics driver. */
        const char *mode = std::getenv("VNPU_MODE");
        const bool reference = mode && std::string(mode) == "reference";
        if (reference) {
            device->cpu.isa = vnpu::ISA_SCALAR;
            device->cpu.simd_bits = 0;
            device->cpu.bits &= ~(VNPU_FEATURE_AVX2_FMA |
                                  VNPU_FEATURE_AVX512F |
                                  VNPU_FEATURE_AVX512VNNI);
            device->cpu.backend = "scalar(reference)";
        }
        /* VNPU_PACK=0 selects the unpacked row-major tile path.  Kept so the
         * two B-access structures can be A/B measured from one binary. */
        device->pack_b = !reference;
        device->batch_lease = false;
        if (const char *penv = std::getenv("VNPU_PACK")) {
            device->pack_b = !(penv[0] == '0' && penv[1] == '\0');
        }
        if (const char *lenv = std::getenv("VNPU_LEASE")) {
            device->batch_lease = !(lenv[0] == '0' && lenv[1] == '\0');
        }
        if (reference) {
            /* Reference mode wins over the A/B switches. */
            device->pack_b = false;
            device->batch_lease = false;
        }
        if (const char *wenv = std::getenv("VNPU_UNIT_WEIGHTS")) {
            std::vector<float> w;
            const char *q = wenv;
            while (*q) {
                char *end = nullptr;
                float v = std::strtof(q, &end);
                if (end == q) break;
                w.push_back(v);
                q = end;
                while (*q == ',' || *q == ' ') ++q;
            }
            if (!w.empty()) device->pool->set_unit_weights(std::move(w));
        }
        /* One extra slot: the issuing thread executes tiles too and needs a
         * scratch buffer of its own (worker id == num_threads - 1). */
        device->local_stores.resize(threads + 1);
        for (auto &store : device->local_stores)
            store.resize(VNPU_LOCAL_STORE_BYTES + VNPU_CACHELINE_BYTES);

        uint32_t tm = cfg.tile_m, tk = cfg.tile_k, tn = cfg.tile_n;
        if (!tm || !tk || !tn) {
            if (device->cpu.isa == vnpu::ISA_AVX2 ||
                device->cpu.isa == vnpu::ISA_NEON) {
                /* Measured best on a 256-bit core: 128x128x64 beat the old
                 * 64x128x64 default on every dense shape tested. */
                tm = 128; tk = 128; tn = 64;
            } else {
                tm = 32; tk = 64; tn = 32;
            }
        }
        device->tile_m.store(tm);
        device->tile_k.store(tk);
        device->tile_n.store(tn);

        device->name = cfg.node_name ? cfg.node_name : "vnpu0";
        device->trace_file = cfg.trace_file ? cfg.trace_file :
                             (std::getenv("VNPU_TRACE_FILE") ?
                              std::getenv("VNPU_TRACE_FILE") : "");
        bool trace_enabled = (cfg.flags & VNPU_CONFIG_TRACE) != 0;
        device->trace.enabled.store(trace_enabled);
        device->trace.trace_file = device->trace_file;
        device->trace.reset();

        std::memset(&device->info, 0, sizeof(device->info));
        device->info.abi_version = VNPU_ABI_VERSION;
        std::snprintf(device->info.name, sizeof(device->info.name), "%s",
                      device->name.c_str());
        std::snprintf(device->info.vendor, sizeof(device->info.vendor),
                      "Arena Virtual NPU");
        std::snprintf(device->info.cpu_arch, sizeof(device->info.cpu_arch),
                      "%s", device->cpu.arch.c_str());
        std::snprintf(device->info.backend, sizeof(device->info.backend),
                      "%s", device->cpu.backend.c_str());
        device->info.simd_bits = device->cpu.simd_bits;
        device->info.num_threads = threads;
        device->info.physical_cores = device->cpu.physical_cpus;
        device->info.num_compute_units = cfg.num_compute_units ?
                                         cfg.num_compute_units : threads;
        device->info.local_store_bytes = VNPU_LOCAL_STORE_BYTES;
        device->info.tile_m = tm;
        device->info.tile_k = tk;
        device->info.tile_n = tn;
        device->info.features = device->cpu.bits;
        device->info.l1d_bytes = device->cpu.l1d;
        device->info.l2_bytes = device->cpu.l2;
        device->info.l3_bytes = device->cpu.l3;
        device->info.total_memory_bytes = device->cpu.total_memory;

        for (auto &r : device->regs) r.store(0);
        device->regs[VNPU_REG_ID / 4].store(VNPU_MAGIC);
        device->regs[VNPU_REG_VERSION / 4].store(VNPU_ABI_VERSION);
        device->regs[VNPU_REG_CONTROL / 4].store(1);
        device->regs[VNPU_REG_STATUS / 4].store(1);
        device->regs[VNPU_REG_ACTIVE_CUS / 4].store(device->info.num_compute_units);
        device->regs[VNPU_REG_CLOCK_MHZ / 4].store(0);
        device->regs[VNPU_REG_FEATURE_LO / 4].store(
            static_cast<uint32_t>(device->cpu.bits & 0xffffffffu));
        device->regs[VNPU_REG_FEATURE_HI / 4].store(
            static_cast<uint32_t>(device->cpu.bits >> 32));
        device->regs[VNPU_REG_IRQ_MASK / 4].store(0xffffffffu);

        vnpu::debug_log("created %s: %s, %u threads, tiles %ux%ux%u",
                        device->name.c_str(), device->info.backend,
                        threads, tm, tk, tn);
        *out_device = device;
        return VNPU_OK;
    });
}

static void destroy_stream_locked(vnpu_stream_t *stream)
{
    if (!stream) return;
    {
        std::lock_guard<std::mutex> lock(stream->mutex);
        stream->stopping = true;
    }
    stream->cv.notify_all();
    if (stream->worker.joinable()) stream->worker.join();
    delete stream;
}

VNPU_API void VNPU_CALL vnpu_device_destroy(vnpu_device_t *device)
{
    if (!device) return;
    std::vector<vnpu_stream_t *> streams;
    {
        std::lock_guard<std::mutex> lock(device->stream_mutex);
        streams = device->streams;
        device->streams.clear();
    }
    for (auto *stream : streams) {
        if (stream) stream->sync();
        destroy_stream_locked(stream);
    }
    if ((device->config.flags & VNPU_CONFIG_TRACE_CHROME) &&
        !device->trace_file.empty()) {
        device->trace.write_chrome(device->trace_file.c_str());
    }
    delete device;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_get_info(const vnpu_device_t *device,
                                                      vnpu_device_info_t *info)
{
    if (!device || !info) return VNPU_ERR_INVALID_ARGUMENT;
    *info = device->info;
    info->tile_m = device->tile_m.load();
    info->tile_k = device->tile_k.load();
    info->tile_n = device->tile_n.load();
    return VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_get_status(const vnpu_device_t *device,
                                                        vnpu_device_status_t *status)
{
    if (!device || !status) return VNPU_ERR_INVALID_ARGUMENT;
    std::memset(status, 0, sizeof(*status));
    status->commands_submitted = device->commands_submitted.load();
    status->commands_completed = device->commands_completed.load();
    status->kernel_time_ns = device->kernel_ns.load();
    status->dispatch_time_ns = device->dispatch_ns.load();
    status->memory_time_ns = device->memory_ns.load();
    status->sync_time_ns = device->sync_ns.load();
    status->bytes_read = device->bytes_read.load();
    status->bytes_written = device->bytes_written.load();
    status->flops = device->flops.load();
    status->tiles_dispatched = device->tiles_dispatched.load();
    status->tiles_culled = device->tiles_culled.load();
    status->dma_bytes = device->dma_bytes.load();
    status->interrupts = device->interrupts.load();
    status->busy = device->busy.load();
    status->error = device->error.load();
    status->last_opcode = device->last_opcode.load();
    status->active_workers = device->active_workers.load();
    return VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_reset(vnpu_device_t *device)
{
    if (!device) return VNPU_ERR_INVALID_ARGUMENT;
    if (device->busy.load()) return VNPU_ERR_DEVICE_BUSY;
    device->commands_submitted.store(0);
    device->commands_completed.store(0);
    device->kernel_ns.store(0);
    device->dispatch_ns.store(0);
    device->memory_ns.store(0);
    device->sync_ns.store(0);
    device->bytes_read.store(0);
    device->bytes_written.store(0);
    device->flops.store(0);
    device->tiles_dispatched.store(0);
    device->tiles_culled.store(0);
    device->dma_bytes.store(0);
    device->interrupts.store(0);
    device->error.store(0);
    device->last_opcode.store(0);
    device->regs[VNPU_REG_ERROR / 4].store(0);
    device->regs[VNPU_REG_IRQ_STATUS / 4].store(0);
    device->trace.reset();
    return VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_set_tiles(vnpu_device_t *device,
                                                       uint32_t tile_m,
                                                       uint32_t tile_k,
                                                       uint32_t tile_n)
{
    if (!device || !tile_m || !tile_k || !tile_n)
        return VNPU_ERR_INVALID_ARGUMENT;
    device->tile_m.store(tile_m);
    device->tile_k.store(tile_k);
    device->tile_n.store(tile_n);
    device->info.tile_m = tile_m;
    device->info.tile_k = tile_k;
    device->info.tile_n = tile_n;
    return VNPU_OK;
}

VNPU_API uint32_t VNPU_CALL vnpu_mmio_read32(vnpu_device_t *device, uint32_t offset)
{
    if (!device || (offset & 3u)) return 0xffffffffu;
    uint32_t idx = offset / 4;
    if (idx >= VNPU_REG_COUNT) return 0xffffffffu;
    switch (offset) {
    case VNPU_REG_STATUS:
        return 1u | (device->busy.load() ? 2u : 0u);
    case VNPU_REG_ERROR:
        return device->regs[idx].load() | (device->error.load() ? 1u : 0u);
    case VNPU_REG_CMD_HEAD:
        return static_cast<uint32_t>(device->commands_completed.load());
    case VNPU_REG_CMD_TAIL:
        return static_cast<uint32_t>(device->commands_submitted.load());
    case VNPU_REG_ACTIVE_CUS:
        return device->active_workers.load();
    case VNPU_REG_THERMAL:
        return 0;
    default:
        return device->regs[idx].load();
    }
}

VNPU_API void VNPU_CALL vnpu_mmio_write32(vnpu_device_t *device, uint32_t offset,
                                          uint32_t value)
{
    if (!device || (offset & 3u)) return;
    uint32_t idx = offset / 4;
    if (idx >= VNPU_REG_COUNT) return;
    device->regs[idx].store(value);
    if (offset == VNPU_REG_DOORBELL && value)
        device->raise_irq(1u << 0);
    else if (offset == VNPU_REG_CONTROL && (value & 2u))
        vnpu_device_reset(device);
    else if (offset == VNPU_REG_IRQ_STATUS)
        device->regs[idx].store(0);
}

VNPU_API void VNPU_CALL vnpu_device_set_interrupt_callback(vnpu_device_t *device,
                                                           vnpu_interrupt_cb_t cb,
                                                           void *user_data)
{
    if (!device) return;
    std::lock_guard<std::mutex> lock(device->irq_mutex);
    device->irq_cb = cb;
    device->irq_user = user_data;
}

/* ------------------------------------------------------------------------- */
/* Events                                                                    */
/* ------------------------------------------------------------------------- */

void vnpu_event::signal(vnpu_status_t result)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        status = result;
        signaled = true;
        generation++;
    }
    cv.notify_all();
}

vnpu_status_t vnpu_event::wait(uint32_t timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (timeout_ms == 0) {
        if (!cv.wait_for(lock, std::chrono::milliseconds(0),
                         [&]() { return signaled; }))
            return VNPU_ERR_TIMEOUT;
    } else {
        if (!cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [&]() { return signaled; }))
            return VNPU_ERR_TIMEOUT;
    }
    return status;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_event_create(vnpu_event_t **out_event)
{
    if (!out_event) return VNPU_ERR_INVALID_ARGUMENT;
    *out_event = nullptr;
    try {
        *out_event = new vnpu_event_t();
        return VNPU_OK;
    } catch (...) {
        return VNPU_ERR_OUT_OF_MEMORY;
    }
}

VNPU_API void VNPU_CALL vnpu_event_destroy(vnpu_event_t *event)
{
    delete event;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_event_wait(vnpu_event_t *event,
                                                 uint32_t timeout_ms)
{
    return event ? event->wait(timeout_ms) : VNPU_ERR_INVALID_ARGUMENT;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_event_query(const vnpu_event_t *event)
{
    if (!event) return VNPU_ERR_INVALID_ARGUMENT;
    auto *mutable_event = const_cast<vnpu_event_t *>(event);
    std::lock_guard<std::mutex> lock(mutable_event->mutex);
    return mutable_event->signaled ? VNPU_OK : VNPU_ERR_DEVICE_BUSY;
}

/* ------------------------------------------------------------------------- */
/* Streams                                                                   */
/* ------------------------------------------------------------------------- */

void vnpu_stream::run()
{
    for (;;) {
        CommandBatch batch;
        bool more_queued = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() { return stopping || !queue.empty(); });
            if (stopping && queue.empty()) return;
            batch = std::move(queue.front());
            queue.pop_front();
            /* Only worth keeping the pool hot if the next command is already
             * queued: otherwise the workers are spinning on a core some other
             * thread could be using (measured: spinning without pending work
             * cost 3% of prefill on a 2-vCPU box). */
            more_queued = !queue.empty();
        }
        vnpu_status_t batch_status = VNPU_OK;
        device->busy.fetch_add(1);
        device->active_workers.fetch_add(1);
        std::unique_ptr<vnpu::ThreadPool::ScopedLease> lease;
        if (device->batch_lease && device->pool && more_queued)
            lease.reset(new vnpu::ThreadPool::ScopedLease(device->pool.get()));
        for (const auto &cmd : batch.commands) {
            uint64_t cmd_begin = vnpu::now_ns();
            if (batch.queued_ns && cmd_begin > batch.queued_ns)
                device->dispatch_ns.fetch_add(cmd_begin - batch.queued_ns);
            vnpu_status_t rc = vnpu::schedule_command(device, this, cmd,
                                                      batch.flags);
            device->commands_completed.fetch_add(1);
            pending_commands.fetch_sub(1);
            device->regs[VNPU_REG_CMD_HEAD / 4].store(
                static_cast<uint32_t>(device->commands_completed.load()));
            if (rc != VNPU_OK) {
                batch_status = rc;
                device->error.store(static_cast<uint32_t>(-rc));
                device->regs[VNPU_REG_ERROR / 4].store(
                    static_cast<uint32_t>(-rc));
                last_error.store(static_cast<uint32_t>(-rc));
            }
        }
        device->active_workers.fetch_sub(1);
        device->busy.fetch_sub(1);
        uint64_t completion_begin = vnpu::now_ns();
        device->sync_ns.fetch_add(vnpu::now_ns() - completion_begin);
        device->raise_irq(1u << 1);
        if (batch.event) batch.event->signal(batch_status);
        {
            std::lock_guard<std::mutex> lock(mutex);
            completed_batches++;
        }
        cv.notify_all();
    }
}

void vnpu_stream::submit(CommandBatch batch)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) return;
        batch.batch_id = next_batch_id++;
        queue.push_back(std::move(batch));
    }
    cv.notify_one();
}

vnpu_status_t vnpu_stream::sync()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&]() {
        return queue.empty() && pending_commands.load() == 0;
    });
    uint32_t err = last_error.load();
    return err ? VNPU_ERR_DEVICE_ERROR : VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_create(vnpu_device_t *device,
                                                    int priority,
                                                    vnpu_stream_t **out_stream)
{
    if (!device || !out_stream) return VNPU_ERR_INVALID_ARGUMENT;
    *out_stream = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        auto *stream = new vnpu_stream_t();
        stream->device = device;
        stream->priority = priority;
        {
            std::lock_guard<std::mutex> lock(device->stream_mutex);
            device->streams.push_back(stream);
        }
        stream->worker = std::thread(&vnpu_stream::run, stream);
        *out_stream = stream;
        return VNPU_OK;
    });
}

VNPU_API void VNPU_CALL vnpu_stream_destroy(vnpu_stream_t *stream)
{
    if (!stream) return;
    vnpu_device_t *device = stream->device;
    if (device) {
        std::lock_guard<std::mutex> lock(device->stream_mutex);
        auto &v = device->streams;
        v.erase(std::remove(v.begin(), v.end(), stream), v.end());
    }
    stream->sync();
    destroy_stream_locked(stream);
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_sync(vnpu_stream_t *stream)
{
    return stream ? stream->sync() : VNPU_ERR_INVALID_ARGUMENT;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_submit(vnpu_stream_t *stream,
                                                    const vnpu_command_t *commands,
                                                    uint64_t count,
                                                    vnpu_event_t *event)
{
    if (!stream || (!commands && count) || count > (1u << 24))
        return VNPU_ERR_INVALID_ARGUMENT;
    if (count == 0) {
        if (event) event->signal(VNPU_OK);
        return VNPU_OK;
    }
    return vnpu::guard([&]() -> vnpu_status_t {
        CommandBatch batch;
        batch.commands.assign(commands, commands + count);
        batch.event = event;
        batch.queued_ns = vnpu::now_ns();
        uint32_t flags = 0;
        for (const auto &cmd : batch.commands) flags |= cmd.flags;
        batch.flags = flags;
        if (event) {
            std::lock_guard<std::mutex> lock(event->mutex);
            event->signaled = false;
            event->status = VNPU_OK;
            event->command_count = count;
        }
        stream->pending_commands.fetch_add(count);
        stream->device->commands_submitted.fetch_add(count);
        stream->device->regs[VNPU_REG_CMD_TAIL / 4].store(
            static_cast<uint32_t>(stream->device->commands_submitted.load()));
        stream->submit(std::move(batch));
        return VNPU_OK;
    });
}

/* ------------------------------------------------------------------------- */
/* Buffers and DMA                                                           */
/* ------------------------------------------------------------------------- */

VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_alloc(vnpu_device_t *device,
                                                   uint64_t bytes,
                                                   vnpu_buffer_t **out_buffer)
{
    if (!device || !bytes || !out_buffer ||
        bytes > std::numeric_limits<size_t>::max())
        return VNPU_ERR_INVALID_ARGUMENT;
    *out_buffer = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        void *ptr = vnpu::aligned_alloc_bytes(VNPU_CACHELINE_BYTES,
                                              static_cast<size_t>(bytes));
        if (!ptr) return VNPU_ERR_OUT_OF_MEMORY;
        std::memset(ptr, 0, static_cast<size_t>(bytes));
        auto *buffer = new vnpu_buffer_t();
        buffer->device = device;
        buffer->host_ptr = ptr;
        buffer->bytes = bytes;
        static std::atomic<uint64_t> next_handle{1};
        buffer->handle = next_handle.fetch_add(1);
        *out_buffer = buffer;
        return VNPU_OK;
    });
}

VNPU_API void VNPU_CALL vnpu_buffer_free(vnpu_buffer_t *buffer)
{
    if (!buffer) return;
    if (buffer->owned) vnpu::aligned_free_bytes(buffer->host_ptr);
    delete buffer;
}

VNPU_API void * VNPU_CALL vnpu_buffer_host_ptr(vnpu_buffer_t *buffer)
{
    return buffer ? buffer->host_ptr : nullptr;
}

VNPU_API uint64_t VNPU_CALL vnpu_buffer_size(const vnpu_buffer_t *buffer)
{
    return buffer ? buffer->bytes : 0;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_copy(vnpu_stream_t *stream,
                                                  void *dst,
                                                  const void *src,
                                                  uint64_t bytes,
                                                  uint32_t direction)
{
    if (!stream || !dst || !src || !bytes) return VNPU_ERR_INVALID_ARGUMENT;
    vnpu_command_t cmd{};
    cmd.opcode = VNPU_OP_COPY;
    cmd.args.copy.dst = dst;
    cmd.args.copy.src = src;
    cmd.args.copy.bytes = bytes;
    cmd.args.copy.direction = direction;
    vnpu_status_t rc = vnpu_stream_submit(stream, &cmd, 1, nullptr);
    if (rc != VNPU_OK) return rc;
    return vnpu_stream_sync(stream);
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_memset(vnpu_stream_t *stream,
                                                    void *dst,
                                                    uint32_t value,
                                                    uint64_t bytes)
{
    if (!stream || !dst || !bytes) return VNPU_ERR_INVALID_ARGUMENT;
    vnpu_command_t cmd{};
    cmd.opcode = VNPU_OP_MEMSET;
    cmd.args.memset.dst = dst;
    cmd.args.memset.value = value;
    cmd.args.memset.bytes = bytes;
    vnpu_status_t rc = vnpu_stream_submit(stream, &cmd, 1, nullptr);
    if (rc != VNPU_OK) return rc;
    return vnpu_stream_sync(stream);
}

/* ------------------------------------------------------------------------- */
/* Tracing API                                                               */
/* ------------------------------------------------------------------------- */

VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_enable(vnpu_device_t *device, int enable)
{
    if (!device) return VNPU_ERR_INVALID_ARGUMENT;
    device->trace.enabled.store(enable != 0);
    if (enable) device->trace.reset();
    return VNPU_OK;
}

VNPU_API void VNPU_CALL vnpu_trace_reset(vnpu_device_t *device)
{
    if (device) device->trace.reset();
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_get_summary(const vnpu_device_t *device,
                                                        vnpu_trace_summary_t *summary)
{
    if (!device || !summary) return VNPU_ERR_INVALID_ARGUMENT;
    device->trace.get_summary(summary);
    return VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_dump(const vnpu_device_t *device,
                                                 const char *path)
{
    if (!device) return VNPU_ERR_INVALID_ARGUMENT;
    return device->trace.dump_text(path);
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_write_chrome(const vnpu_device_t *device,
                                                         const char *path)
{
    if (!device || !path) return VNPU_ERR_INVALID_ARGUMENT;
    return device->trace.write_chrome(path);
}

} /* extern "C" */
