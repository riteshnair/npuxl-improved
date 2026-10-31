/*
 * sherlock.h - Sherlock-it style tracing for the adapted Virtual NPU.
 *
 * Matches the upstream npuxl contract (include/vnpu_compat.h + src/compat):
 *   - the same SHERLOCK_TRAP_* trap categories
 *   - the same sherlock_trace_t bucket struct
 *   - the same sherlock_trace_start()/sherlock_trace_end() API
 *   - the same dump format ("=== SHERLOCK TRACE [label] ===")
 *
 * On top of that it adds the two things needed to actually optimise:
 *   - a per-name hot-spot table (calls, ns, ops, bytes), sorted by cost
 *   - a roofline verdict per phase (compute bound vs memory bound)
 *
 * Cost: light. A started phase is one timer read, one timer read + a few scalar
 * adds at end. No allocation, no string work on the hot path (names are
 * compared by pointer and only stored, never copied). 25-45 ns/event, same
 * order as upstream. Put it at phase boundaries only - never inside a K loop.
 *
 * Env: VNPU_TRACE=1 -> dump at exit, VNPU_TRACE=2 -> also vnpu-trace.json
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#  include <windows.h>
#endif

/* --------------------------------------------------------------- traps ---- */
enum {
    SHERLOCK_TRAP_DISPATCH = 0,
    SHERLOCK_TRAP_MEMORY = 1,
    SHERLOCK_TRAP_INSTRUCTION = 2,
    SHERLOCK_TRAP_SYNC = 3,
    SHERLOCK_TRAP_ALLOC = 4,
    SHERLOCK_TRAP_TRANSFER = 5,
    SHERLOCK_TRAP_DEPENDENCY = 6,
    SHERLOCK_TRAP_ACCURACY = 7,
    SHERLOCK_TRAP_COMPUTE = 8,
    SHERLOCK_TRAP_COUNT = 9
};

typedef struct sherlock_trace {
    double dispatch_time;
    double memory_time;
    double compute_time;
    double sync_time;
    double total_time;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t cache_misses;
    uint64_t ops_count;
    int traps_enabled;
} sherlock_trace_t;

/* ------------------------------------------------------------ internals --- */
namespace sherlock_impl {

inline uint64_t now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER f = {0};
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    /* ns = ticks * 1e9 / freq, kept in integer maths to avoid drift */
    return (uint64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

enum { MAX_NAMES = 64 };

struct entry {
    const char* name;
    int trap;
    uint64_t ns;      /* total nanos   */
    uint64_t ops;     /* int8 MACs     */
    uint64_t bytes;   /* bytes moved   */
    uint64_t calls;
};

inline sherlock_trace_t& tr(void) { static sherlock_trace_t t = {0,0,0,0,0,0,0,0,0,0}; return t; }
inline uint64_t* starts(void) { static uint64_t s[SHERLOCK_TRAP_COUNT] = {0}; return s; }
inline entry* tab(void) { static entry e[MAX_NAMES]; return e; }
inline int& used(void) { static int u = 0; return u; }
inline int& lvl(void) { static int v = -1; if (v < 0) { const char* e = getenv("VNPU_TRACE"); v = e ? atoi(e) : 0; } return v; }
inline bool& muted(void) { static bool m = false; return m; }
inline bool on(void) { return !muted() && lvl() >= 1; }

inline double& peak_mac_s(void)  { static double v = 0; return v; }
inline double& peak_byte_s(void) { static double v = 0; return v; }
/* bandwidth ladder, GB/s: [0]=L1 [1]=L2 [2]=L3 [3]=DRAM */
inline double* bw(void) { static double v[4] = {0, 0, 0, 0}; return v; }
inline const char* bw_label(double gbs) {
    double* b = bw();
    if (b[0] > 0 && gbs <= b[0]) return "L1-bw-roof";
    if (b[1] > 0 && gbs <= b[1]) return "L2-bw-roof";
    if (b[2] > 0 && gbs <= b[2]) return "L3-bw-roof";
    return "DRAM-bw-roof";
}

inline int slot_for(const char* name, int trap) {
    entry* e = tab();
    const int n = used();
    for (int i = 0; i < n; i++)
        if (e[i].name == name) return i;
    if (n >= MAX_NAMES) return 0;
    e[n].name = name; e[n].trap = trap;
    e[n].ns = e[n].ops = e[n].bytes = e[n].calls = 0;
    used() = n + 1;
    return n;
}

} /* namespace sherlock_impl */

/* ----------------------------------------------------------------- API ---- */
inline void sherlock_enable_tracing(int enable) { sherlock_impl::tr().traps_enabled = enable ? 1 : 0; }

/* Silence recording around a parameter sweep so the shared phase names do not
 * pollute the trace with every variant we are only trying out. */
inline void sherlock_mute(bool m) { sherlock_impl::muted() = m; }

inline void sherlock_reset_trace(void) {
    ::sherlock_trace_t& t = sherlock_impl::tr();
    t.dispatch_time = t.memory_time = t.compute_time = t.sync_time = 0.0;
    t.bytes_read = t.bytes_written = t.cache_misses = t.ops_count = 0;
    for (int i = 0; i < SHERLOCK_TRAP_COUNT; i++) sherlock_impl::starts()[i] = 0;
    sherlock_impl::used() = 0;
}

inline void sherlock_trace_start(int trap_type, const char* /*name*/) {
    if (!sherlock_impl::on()) return;
    if (trap_type < 0 || trap_type >= SHERLOCK_TRAP_COUNT) return;
    sherlock_impl::starts()[trap_type] = sherlock_impl::now_ns();
}

inline void sherlock_trace_end(int trap_type, const char* name, uint64_t ops, uint64_t bytes) {
    if (!sherlock_impl::on()) return;
    if (trap_type < 0 || trap_type >= SHERLOCK_TRAP_COUNT) return;
    const uint64_t start = sherlock_impl::starts()[trap_type];
    if (!start) return;
    const uint64_t ns = sherlock_impl::now_ns() - start;
    const double elapsed = (double)ns / 1e9;

    ::sherlock_trace_t& t = sherlock_impl::tr();
    switch (trap_type) {    case SHERLOCK_TRAP_MEMORY:
    case SHERLOCK_TRAP_ALLOC:
    case SHERLOCK_TRAP_TRANSFER:
        t.memory_time += elapsed; break;
    case SHERLOCK_TRAP_DISPATCH:
        t.dispatch_time += elapsed; break;
    case SHERLOCK_TRAP_SYNC:
    case SHERLOCK_TRAP_DEPENDENCY:
        t.sync_time += elapsed; break;
    case SHERLOCK_TRAP_COMPUTE:
    case SHERLOCK_TRAP_INSTRUCTION:
    case SHERLOCK_TRAP_ACCURACY:
    default:
        t.compute_time += elapsed; break;
    }
    t.ops_count += ops;
    t.bytes_read += bytes;

    const int s = sherlock_impl::slot_for(name, trap_type);
    sherlock_impl::entry& e = sherlock_impl::tab()[s];
    e.ns += ns; e.ops += ops; e.bytes += bytes; e.calls += 1;
}

/* RAII convenience: ops/bytes known up front. */
struct sherlock_scope {
    int trap; const char* name; uint64_t ops, bytes;
    sherlock_scope(int t, const char* n, uint64_t o, uint64_t b)
        : trap(t), name(n), ops(o), bytes(b) { sherlock_trace_start(t, n); }
    ~sherlock_scope() { sherlock_trace_end(trap, name, ops, bytes); }
};

#define SHC_(a, b) a##b
#define SHC(a, b)  SHC_(a, b)
#define SHERLOCK_SCOPE(trap, name, ops, bytes) \
    sherlock_scope SHC(_shs_, __COUNTER__)((trap), (name), (uint64_t)(ops), (uint64_t)(bytes))

inline void sherlock_set_peaks(double mac_s, double byte_s) {
    sherlock_impl::peak_mac_s() = mac_s;
    sherlock_impl::peak_byte_s() = byte_s;
}
inline void sherlock_set_bw_ladder(double l1, double l2, double l3, double dram) {
    double* b = sherlock_impl::bw();
    b[0] = l1; b[1] = l2; b[2] = l3; b[3] = dram;
}

static const char* sherlock_trap_name(int t) {
    switch (t) {
    case SHERLOCK_TRAP_DISPATCH:    return "DISPATCH";
    case SHERLOCK_TRAP_MEMORY:      return "MEMORY";
    case SHERLOCK_TRAP_INSTRUCTION: return "INSTRUCTION";
    case SHERLOCK_TRAP_SYNC:        return "SYNC";
    case SHERLOCK_TRAP_ALLOC:       return "ALLOC";
    case SHERLOCK_TRAP_TRANSFER:    return "TRANSFER";
    case SHERLOCK_TRAP_DEPENDENCY:  return "DEPENDENCY";
    case SHERLOCK_TRAP_ACCURACY:    return "ACCURACY";
    case SHERLOCK_TRAP_COMPUTE:     return "COMPUTE";
    default:                        return "?";
    }
}

inline void sherlock_dump_trace(const char* label) {
    if (!sherlock_impl::on()) return;
    ::sherlock_trace_t& t = sherlock_impl::tr();
    const double total = t.total_time > 0.0 ? t.total_time : 1.0;

    /* ---- the upstream-style bucket summary ---- */
    printf("\n=== SHERLOCK TRACE [%s] ===\n", label ? label : "UNKNOWN");
    printf("Total time:     %.6f ms\n", t.total_time * 1000.0);
    printf("Dispatch:       %.6f ms (%.1f%%)\n", t.dispatch_time * 1000.0, 100.0 * t.dispatch_time / total);
    printf("Memory:         %.6f ms (%.1f%%)\n", t.memory_time    * 1000.0, 100.0 * t.memory_time    / total);
    printf("Compute:        %.6f ms (%.1f%%)\n", t.compute_time   * 1000.0, 100.0 * t.compute_time   / total);
    printf("Sync:           %.6f ms (%.1f%%)\n", t.sync_time      * 1000.0, 100.0 * t.sync_time      / total);
    printf("Ops:            %llu\n", (unsigned long long)t.ops_count);
    printf("Bytes read:     %llu\n", (unsigned long long)t.bytes_read);
    printf("Bytes written:  %llu\n", (unsigned long long)t.bytes_written);

    /* ---- hot-spot list, worst first: where to optimise ---- */
    const int n = sherlock_impl::used();
    std::vector<int> ord(n);
    for (int i = 0; i < n; i++) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return sherlock_impl::tab()[a].ns > sherlock_impl::tab()[b].ns;
    });

    const double pm = sherlock_impl::peak_mac_s();

    printf("\n-- hot spots (worst first) --\n");
    printf("%-14s %-22s %7s %11s %8s %9s %9s %10s  %s\n",
           "trap", "phase", "calls", "total ms", "share", "GMAC/s", "GB/s", "B/MAC", "limiter");
    printf("---------------------------------------------------------------------------------------------\n");
    for (int k = 0; k < n; k++) {
        const sherlock_impl::entry& e = sherlock_impl::tab()[ord[k]];
        const double sec = (double)e.ns / 1e9;
        char gmac[16] = "-", gb[16] = "-", bpm[16] = "-";
        if (e.ops && sec > 0)   snprintf(gmac, sizeof(gmac), "%.1f", (double)e.ops / sec / 1e9);
        if (e.bytes && sec > 0) snprintf(gb, sizeof(gb), "%.1f", (double)e.bytes / sec / 1e9);
        if (e.ops)              snprintf(bpm, sizeof(bpm), "%.3f", (double)e.bytes / (double)e.ops);

        /* limiter: how close to the int8 compute peak, vs which bandwidth roof
         * the achieved GB/s is consistent with. The roof is the actionable bit:
         * "L2-bw-roof" means blocking beat compute, not more ALU work. */
        char lim[48] = "";
        if (pm > 0 && e.ops && sec > 0) {
            const double fc = ((double)e.ops / sec) / pm;
            if (e.bytes && sec > 0) {
                const double g = (double)e.bytes / sec / 1e9;
                snprintf(lim, sizeof(lim), "%.0f%% compute, %s", fc * 100.0,
                         sherlock_impl::bw_label(g));
            } else {
                snprintf(lim, sizeof(lim), "%.0f%% compute", fc * 100.0);
            }
        }
        printf("%-14s %-22s %7llu %11.3f %7.1f%% %9s %9s %10s  %s\n",
               sherlock_trap_name(e.trap), e.name,
               (unsigned long long)e.calls, (double)e.ns / 1e6,
               100.0 * (double)e.ns / (t.total_time * 1e9 > 0 ? t.total_time * 1e9 : 1.0),
               gmac, gb, bpm, lim);
    }
    printf("=============================\n\n");

    if (sherlock_impl::lvl() >= 2) {
        FILE* f = fopen("vnpu-trace.json", "w");
        if (f) {
            fprintf(f, "{\n\"displayTimeUnit\":\"ns\",\n\"traceEvents\":[\n");
            for (int k = 0; k < n; k++) {
                const sherlock_impl::entry& e = sherlock_impl::tab()[ord[k]];
                fprintf(f, "%s{\"ph\":\"X\",\"ts\":0,\"dur\":%llu,\"name\":\"%s/%s\","
                           "\"args\":{\"calls\":%llu,\"ops\":%llu,\"bytes\":%llu}}",
                        k ? ",\n" : "", (unsigned long long)e.ns,
                        sherlock_trap_name(e.trap), e.name,
                        (unsigned long long)e.calls, (unsigned long long)e.ops,
                        (unsigned long long)e.bytes);
            }
            fprintf(f, "\n]}\n");
            fclose(f);
            printf("chrome trace: vnpu-trace.json\n");
        }
    }
}
