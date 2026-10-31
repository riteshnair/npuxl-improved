/*
 * scheduler.cpp - virtual command processor and compute-unit dispatch.
 *
 * Voodoo, Wildcat and Cell are distinct schedulers over the same SIMD
 * microkernels.  They differ in tile geometry, tile order, culling and local
 * store/DMA behavior, which is what makes them useful for different LLM shapes.
 */
#include "../internal.h"

#include <sstream>

namespace vnpu {

struct ExecCtx {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    const vnpu_command_t *cmd = nullptr;
    uint32_t flags = 0;
    ScopedTrace *trace = nullptr;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t flops = 0;
    uint64_t dma_bytes = 0;
    uint64_t tiles_dispatched = 0;
    uint64_t tiles_culled = 0;

    void add_bytes(uint64_t r, uint64_t w) {
        bytes_read += r;
        bytes_written += w;
    }
    void add_flops(uint64_t f) { flops += f; }
};

static inline bool mul_overflow(uint64_t a, uint64_t b, uint64_t c)
{
    if (a == 0 || b == 0 || c == 0) return false;
    uint64_t ab = a * b;
    if (b != 0 && ab / b != a) return true;
    if (c != 0 && ab > std::numeric_limits<uint64_t>::max() / c) return true;
    return false;
}

static uint32_t choose_unit(const vnpu_device_t *device,
                            const vnpu_matmul_f32_args_t &a,
                            uint32_t flags)
{
    if (a.unit == VNPU_UNIT_GPU) return VNPU_UNIT_GPU;
    if (flags & VNPU_CMD_FLAG_CELL_DMA) return VNPU_UNIT_CELL;
    if (flags & VNPU_CMD_FLAG_VOODOO_FIFO) return VNPU_UNIT_VOODOO;
    if (a.unit != VNPU_UNIT_AUTO) return a.unit;
    if (a.tile_mask || (flags & VNPU_CMD_FLAG_WILDCAT_CULL))
        return VNPU_UNIT_WILDCAT;
    /* CELL is never chosen automatically.  Its staged local-store loop is
     * single-threaded scalar, and it measured 0.3-0.6x a plain -O3 loop for
     * every shape tested, including the tiny decode shapes (M<=4) and small
     * K,N shapes it used to be selected for.  It stays available as an
     * explicit unit. */
    return VNPU_UNIT_VOODOO;
}

/* Widest useful column tile for the M=1 decode path, in columns.
 *
 * The rank-1 decoder stages one int32 accumulator per column of the tile (4
 * bytes) and re-reads that staging row on every k step, so it competes with the
 * B stream for L2.  Giving it half of L2 is what the measurements support,
 * 2 threads, int8, best of 7 interleaved cells:
 *
 *   K=64, N=262144:   65536 cols 0.86 ms (19.6 GB/s)   131072 cols 1.86 ms
 *                     (9.0 GB/s)   whole row 3.50 ms (4.8 GB/s)
 *   K=64, N=1048576:  65536 cols 3.59 ms (18.7 GB/s)  524288 cols 9.0 ms
 *
 * 4 bytes x 131072 = 512 KiB is the whole of this L2, and the staging row then
 * thrashes against the stream; at 256 KiB it still fits alongside it.  The
 * scratch buffer is the second limit: the kernel falls back to the old blocked
 * walk if the staging row does not fit VNPU_LOCAL_STORE_BYTES. */
static uint32_t decode_max_tile_columns(const vnpu_device_t *device,
                                       uint32_t rows)
{
    uint64_t l2 = device->info.l2_bytes ? device->info.l2_bytes
                                        : (512ull * 1024ull);
    /* Each staged accumulator row costs sizeof(int32_t) per column and they all
     * have to share half of L2 with the B stream, so the column budget divides
     * by the row count. */
    uint64_t cols = l2 / (2ull * sizeof(int32_t) * rows);
    /* thread_scratch() hands out VNPU_LOCAL_STORE_BYTES + one cache line, so a
     * staging row of exactly VNPU_LOCAL_STORE_BYTES still fits after alignment. */
    const uint64_t scratch_cols =
        (uint64_t)VNPU_LOCAL_STORE_BYTES / (sizeof(int32_t) * rows);
    if (cols > scratch_cols) cols = scratch_cols;
    cols &= ~(uint64_t)31u;
    if (cols < 256u) cols = 256u;
    if (cols > 0x7fffffffu) cols = 0x7fffffffu;
    return static_cast<uint32_t>(cols);
}

static void tile_geometry(const vnpu_device_t *device, uint32_t unit,
                          uint32_t M, uint32_t N, bool int8,
                          uint32_t &tm, uint32_t &tk, uint32_t &tn)
{
    tm = std::max<uint32_t>(1, device->tile_m.load());
    tk = std::max<uint32_t>(1, device->tile_k.load());
    tn = std::max<uint32_t>(1, device->tile_n.load());
    /* Decode (M<=8): one row block already covers all of M, so N is the only
     * axis that can be spread across units.  Keeping the prefill tile width
     * here gave a handful of very wide tiles, i.e. one unit did the token. */
    if (M <= 8 && (unit == VNPU_UNIT_VOODOO || unit == VNPU_UNIT_AUTO)) {
        tm = M;
        /* From M=4 up the column strip is reused by every row of the tile, so
         * it has to stay small enough to live in L2 and the narrow tile is
         * kept: the wide split dropped 4x4096x4096 int8 from 36 to 21 GFLOP/s,
         * and re-measuring with the rank-R stream confirmed it -- four staged
         * rows cost 1.98 -> 2.32 ms there.  Only one to three rows have no
         * reuse partner. */
        if (M > 3) {
            tn = std::min<uint32_t>(std::max<uint32_t>(32, tn), 64);
            if (tn > N) tn = N;
            return;
        }
        /* Two rows are the awkward middle: too many accumulators to keep a wide
         * column block in registers, too few to amortise the packed panel that
         * M>=3 uses.  With the 64-column tile they re-read B as strided 64-byte
         * runs, which measured 1.50 ms at 2x4096x4096 int8 against 0.65 ms for
         * one row.  The int8 kernel stages both int32 rows instead, so the same
         * width rule applies with half the column budget per row.
         *
         * That stream doubles the accumulator traffic, so it only pays while a
         * unit still owns a long run: 2x4096x4096 (2048 columns each) went 1.63
         * -> 1.05 ms, but 2x4096x1024 (512 columns each) went 0.28 -> 0.53 ms
         * and keeps the narrow tile.  The f32 kernel has no such stream, and a
         * wide tile there cost 4.9 -> 6.0 ms at 2x4096x4096. */
        const uint32_t units = std::max<uint32_t>(1, device->info.num_threads);
        if (M <= 3 && (!int8 || N / units < 1024u)) {
            tn = std::min<uint32_t>(std::max<uint32_t>(32, tn), 64);
            if (tn > N) tn = N;
            return;
        }
        /* A single row touches every weight byte exactly once, so the column
         * tile width decides whether B is read as a stream or as a stride-N
         * walk.  One contiguous slice of N per unit is the rule, and every
         * weight byte is then read once, in order, at the longest run of
         * columns that unit's share of N allows: 1x4096x4096 int8, 2 threads,
         * best of 7 interleaved rounds, 2048 cols (one slice each) 0.99 ms,
         * 4096 cols (one tile, a unit left idle) 1.13, 1024 cols 1.18, 512 cols
         * 1.20, and the old fixed 64 cols 2.85 ms.
         *
         * Two limits shape it.  A slice narrower than 256 columns is not worth
         * a second unit, because the run per row gets too short to pay for it:
         * 1x4096x128 and 1x4096x256 both measured faster unsplit (0.053 vs
         * 0.063 ms and 0.069 vs 0.078).  And a tile may not exceed what L2 can
         * hold next to the stream, see decode_max_tile_columns; when that limit
         * binds, the tile count is rounded up to a multiple of units so the
         * share stays even. */
        const uint32_t rows = M;
        uint32_t tiles = std::min<uint32_t>(units, std::max<uint32_t>(1, N / 256u));
        const uint32_t cap = decode_max_tile_columns(device, rows);
        if (ceil_div_u32(N, tiles) > cap) {
            tiles = ceil_div_u32(N, cap);
            /* Once the cache limit forces extra tiles, round their count up to a
             * multiple of units so every unit gets the same number of them: five
             * tiles over two units (3+2) measured 1.22 ms on K=64,N=262144 int8
             * against 0.99 ms for four (2+2). */
            tiles = ceil_div_u32(tiles, units) * units;
        }
        uint32_t wide = (ceil_div_u32(N, tiles) + 31u) & ~31u;
        tn = std::min<uint32_t>(N, std::max<uint32_t>(1, wide));
        return;
    }
    /* Panel-packed path: one tile spans all of M so its B panel is packed once
     * for the whole GEMM instead of once per M tile.  Only when N alone gives
     * the units enough tiles to stay busy. */
    if (device->pack_b &&
        (unit == VNPU_UNIT_VOODOO || unit == VNPU_UNIT_AUTO)) {
        const uint32_t units = std::max<uint32_t>(1, device->info.num_threads);
        const uint32_t nt64 = ceil_div_u32(N, 64);
        if (M >= 3 && nt64 >= 2 * units) {
            tm = M;
            tn = 64;
            return;
        }
    }
    switch (unit) {
    case VNPU_UNIT_WILDCAT:
        tm = std::min<uint32_t>(tm, 32);
        tn = std::min<uint32_t>(tn, 64);
        tk = std::min<uint32_t>(tk, 128);
        break;
    case VNPU_UNIT_CELL:
        tm = std::min<uint32_t>(tm, 16);
        tn = std::min<uint32_t>(tn, 64);
        tk = std::min<uint32_t>(tk, 64);
        break;
    case VNPU_UNIT_VOODOO:
    default:
        break;
    }
    /* Avoid pathological numbers of micro-tasks on large cloud VMs. */
    uint64_t tiles64 = static_cast<uint64_t>(ceil_div_u32(M, tm)) *
                       static_cast<uint64_t>(ceil_div_u32(N, tn));
    uint64_t target = static_cast<uint64_t>(device->info.num_threads) * 64;
    while (tiles64 > target && (tm < M || tn < N)) {
        if (tm < M) tm = std::min<uint32_t>(M, tm * 2);
        if (tn < N) tn = std::min<uint32_t>(N, tn * 2);
        tiles64 = static_cast<uint64_t>(ceil_div_u32(M, tm)) *
                  static_cast<uint64_t>(ceil_div_u32(N, tn));
    }
}

static vnpu_status_t build_tiles_generic(const vnpu_device_t *device,
                                         uint32_t M, uint32_t N,
                                         uint32_t unit,
                                         const uint8_t *tile_mask,
                                         uint32_t mask_tile_m,
                                         uint32_t mask_tile_n,
                                         bool int8,
                                         std::vector<TileJob> &tiles)
{
    if (!device || !M || !N) return VNPU_ERR_INVALID_ARGUMENT;
    uint32_t tm, tk, tn;
    tile_geometry(device, unit, M, N, int8, tm, tk, tn);
    uint32_t mt = ceil_div_u32(M, tm);
    uint32_t nt = ceil_div_u32(N, tn);
    tiles.clear();
    tiles.reserve(static_cast<size_t>(mt) * nt);
    auto make_tile = [&](uint32_t mi, uint32_t ni) {
        TileJob t{};
        t.m0 = mi * tm;
        t.n0 = ni * tn;
        t.tm = std::min<uint32_t>(tm, M - t.m0);
        t.tn = std::min<uint32_t>(tn, N - t.n0);
        t.tile_m = tm;
        t.tile_n = tn;
        t.tile_k = tk;
        t.cull = 0;
        if (tile_mask && mask_tile_m && mask_tile_n) {
            uint32_t mm = std::min<uint32_t>(mask_tile_m - 1,
                static_cast<uint32_t>((static_cast<uint64_t>(mi) *
                                       mask_tile_m) / mt));
            uint32_t nn = std::min<uint32_t>(mask_tile_n - 1,
                static_cast<uint32_t>((static_cast<uint64_t>(ni) *
                                       mask_tile_n) / nt));
            if (tile_mask[static_cast<size_t>(mm) * mask_tile_n + nn] == 0)
                t.cull = 1;
        }
        tiles.push_back(t);
    };
    if (unit == VNPU_UNIT_WILDCAT) {
        /* Interleaved order gives work-stealing workers a mix of A/B panels. */
        for (uint32_t ni = 0; ni < nt; ++ni)
            for (uint32_t mi = 0; mi < mt; ++mi)
                make_tile(mi, ni);
    } else if (unit == VNPU_UNIT_CELL) {
        /* Group four M tiles for each N panel to reuse the DMA'd B panel. */
        for (uint32_t ni = 0; ni < nt; ++ni)
            for (uint32_t mi = 0; mi < mt; ++mi)
                make_tile(mi, ni);
    } else {
        for (uint32_t mi = 0; mi < mt; ++mi)
            for (uint32_t ni = 0; ni < nt; ++ni)
                make_tile(mi, ni);
    }
    return VNPU_OK;
}

vnpu_status_t build_gemm_tiles(vnpu_device_t *device,
                               const vnpu_matmul_f32_args_t &args,
                               std::vector<TileJob> &tiles)
{
    uint32_t unit = choose_unit(device, args, 0);
    return build_tiles_generic(device, args.M, args.N, unit,
                               args.tile_mask, args.mask_tile_m,
                               args.mask_tile_n, false, tiles);
}

static vnpu_status_t prepare_matmul_f32(const vnpu_matmul_f32_args_t &in,
                                        uint32_t flags,
                                        const vnpu_device_t *device,
                                        GemmF32Params &p,
                                        uint32_t &unit)
{
    if (!in.A || !in.B || !in.C || !in.M || !in.K || !in.N)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (mul_overflow(in.M, in.K, in.N))
        return VNPU_ERR_INVALID_ARGUMENT;
    if (in.trans_a > 1 || in.trans_b > 1 || in.epilogue > VNPU_EPILOGUE_GELU_ERF)
        return VNPU_ERR_INVALID_ARGUMENT;
    unit = choose_unit(device, in, flags);
    if (unit == VNPU_UNIT_GPU) return VNPU_ERR_UNSUPPORTED;
    p.A = in.A;
    p.B = in.B;
    p.bias = in.bias;
    p.C = in.C;
    p.M = in.M; p.K = in.K; p.N = in.N;
    p.lda = in.lda ? in.lda : (in.trans_a ? in.M : in.K);
    p.ldb = in.ldb ? in.ldb : (in.trans_b ? in.K : in.N);
    p.ldc = in.ldc ? in.ldc : in.N;
    if (p.lda < (in.trans_a ? in.M : in.K) ||
        p.ldb < (in.trans_b ? in.K : in.N) ||
        p.ldc < in.N)
        return VNPU_ERR_INVALID_ARGUMENT;
    p.trans_a = in.trans_a;
    p.trans_b = in.trans_b;
    p.alpha = in.alpha == 0.0f ? 1.0f : in.alpha;
    p.beta = in.beta;
    p.epilogue = in.epilogue;
    p.unit = unit;
    p.isa = device->cpu.isa;
    p.tile_m = device->tile_m.load();
    p.tile_k = device->tile_k.load();
    p.tile_n = device->tile_n.load();
    p.pack_b = device->pack_b;
    p.tile_mask = in.tile_mask;
    p.mask_tile_m = in.mask_tile_m;
    p.mask_tile_n = in.mask_tile_n;
    p.local_stores = const_cast<std::vector<std::vector<uint8_t>> *>(
        &device->local_stores);
    p.local_store_bytes = VNPU_LOCAL_STORE_BYTES;
    return VNPU_OK;
}

static void execute_matmul_f32(ExecCtx &ctx)
{
    const auto &in = ctx.cmd->args.matmul_f32;
    GemmF32Params p;
    uint32_t unit = VNPU_UNIT_VOODOO;
    vnpu_status_t rc = prepare_matmul_f32(in, ctx.flags, ctx.device, p, unit);
    if (rc != VNPU_OK) throw std::runtime_error(vnpu_status_string(rc));
    std::vector<TileJob> tiles;
    rc = build_tiles_generic(ctx.device, in.M, in.N, unit,
                             in.tile_mask, in.mask_tile_m,
                             in.mask_tile_n, false, tiles);
    if (rc != VNPU_OK) throw std::runtime_error(vnpu_status_string(rc));
    p.tiles = &tiles;
    uint64_t active_tiles = 0;
    for (const auto &t : tiles) if (!t.cull) active_tiles++;
    ctx.tiles_dispatched += active_tiles;
    ctx.tiles_culled += tiles.size() - active_tiles;
    uint64_t grain = std::max<uint64_t>(1,
        ceil_div_u32(static_cast<uint32_t>(tiles.size()),
                     static_cast<uint32_t>(ctx.device->info.num_threads * 4)));
    ctx.device->pool->parallel_for(tiles.size(), grain,
        [&](uint64_t begin, uint64_t end, uint32_t worker) {
            kernel_gemm_f32_range(p, begin, end, worker);
        });
    uint64_t a_bytes = static_cast<uint64_t>(in.M) * in.K * sizeof(float);
    uint64_t b_bytes = static_cast<uint64_t>(in.K) * in.N * sizeof(float);
    uint64_t c_bytes = static_cast<uint64_t>(in.M) * in.N * sizeof(float);
    uint64_t bias_bytes = in.bias ? static_cast<uint64_t>(in.N) * sizeof(float) : 0;
    ctx.add_bytes(a_bytes + b_bytes + bias_bytes +
                  (in.beta != 0.0f ? c_bytes : 0), c_bytes);
    ctx.add_flops(2ull * in.M * in.K * in.N);
    if (unit == VNPU_UNIT_CELL)
        ctx.dma_bytes += a_bytes + b_bytes;
}

static PackedI8View make_packed_view(const vnpu_packed_weights_t *w)
{
    PackedI8View v{};
    v.packed = w->packed.data();
    v.original = w->original.data();
    v.scales = w->scales.data();
    v.column_sums = w->column_sums.data();
    v.K = w->K;
    v.N = w->N;
    v.Kpad = w->Kpad;
    v.Npad = w->Npad;
    v.packed_bytes = w->packed.size();
    return v;
}

static void execute_matmul_i8(ExecCtx &ctx)
{
    const auto &in = ctx.cmd->args.matmul_i8;
    auto *w = in.B;
    if (!in.A || !w || !in.C || !in.M || !in.K || !in.N)
        throw std::runtime_error("invalid matmul_i8 arguments");
    if (w->K != in.K || w->N != in.N)
        throw std::runtime_error("packed weight shape mismatch");
    uint32_t unit = choose_unit(ctx.device,
        vnpu_matmul_f32_args_t{nullptr, nullptr, nullptr, nullptr,
                               in.M, in.K, in.N, 0, 0, 0, 0, 0,
                               1.0f, 0.0f, in.epilogue, in.unit, 0,
                               nullptr, 0, 0}, ctx.flags);
    if (unit == VNPU_UNIT_GPU) throw std::runtime_error("GPU backend not enabled");
    std::vector<TileJob> tiles;
    build_tiles_generic(ctx.device, in.M, in.N, unit, nullptr, 0, 0, true,
                        tiles);
    GemmI8Params p;
    p.A = in.A;
    p.scale_a = in.scale_a;
    p.bias = in.bias;
    p.C = in.C;
    p.M = in.M; p.K = in.K; p.N = in.N;
    p.lda = in.lda ? in.lda : in.K;
    p.ldc = in.ldc ? in.ldc : in.N;
    if (p.lda < in.K || p.ldc < in.N)
        throw std::runtime_error("invalid int8 leading dimensions");
    p.epilogue = in.epilogue;
    p.unit = unit;
    p.isa = ctx.device->cpu.isa;
    p.has_vnni = false;   /* no VNNI path is built */
    p.tile_m = ctx.device->tile_m.load();
    p.tile_k = ctx.device->tile_k.load();
    p.tile_n = ctx.device->tile_n.load();
    p.pack_b = ctx.device->pack_b;
    p.B = make_packed_view(w);
    p.tiles = &tiles;
    p.local_stores = &ctx.device->local_stores;
    p.local_store_bytes = VNPU_LOCAL_STORE_BYTES;
    uint64_t active = 0;
    for (const auto &t : tiles) if (!t.cull) active++;
    ctx.tiles_dispatched += active;
    ctx.tiles_culled += tiles.size() - active;
    uint64_t grain = std::max<uint64_t>(1,
        ceil_div_u32(static_cast<uint32_t>(tiles.size()),
                     static_cast<uint32_t>(ctx.device->info.num_threads * 4)));
    ctx.device->pool->parallel_for(tiles.size(), grain,
        [&](uint64_t begin, uint64_t end, uint32_t worker) {
            kernel_gemm_i8_range(p, begin, end, worker);
        });
    uint64_t a_bytes = static_cast<uint64_t>(in.M) * in.K;
    uint64_t b_bytes = static_cast<uint64_t>(in.K) * in.N;
    uint64_t c_bytes = static_cast<uint64_t>(in.M) * in.N * sizeof(float);
    uint64_t scale_bytes = static_cast<uint64_t>(in.N + (in.scale_a ? in.M : 0)) *
                           sizeof(float);
    ctx.add_bytes(a_bytes + b_bytes + scale_bytes, c_bytes);
    ctx.add_flops(2ull * in.M * in.K * in.N);
    if (unit == VNPU_UNIT_CELL) ctx.dma_bytes += a_bytes + b_bytes;
}

static void execute_quantize_i8(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.quantize_i8;
    if (!a.X || !a.Y || !a.scales || !a.rows || !a.cols ||
        a.axis > VNPU_QUANT_PER_TENSOR)
        throw std::runtime_error("invalid quantize_i8 arguments");
    if (mul_overflow(a.rows, a.cols, 1))
        throw std::runtime_error("quantize shape overflow");
    uint64_t total = a.rows * a.cols;
    uint64_t grain = std::max<uint64_t>(1, total /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    if (a.axis == VNPU_QUANT_PER_TENSOR) {
        std::vector<float> partial;
        uint64_t chunks = (total + grain - 1) / grain;
        partial.resize(chunks, 0.0f);
        ctx.device->pool->parallel_for(total, grain,
            [&](uint64_t begin, uint64_t end, uint32_t) {
                uint64_t c = begin / grain;
                float m = 0.0f;
                for (uint64_t i = begin; i < end; ++i)
                    m = std::max(m, std::fabs(a.X[i]));
                if (c < partial.size()) partial[c] = m;
            });
        float maxabs = 0.0f;
        for (float x : partial) maxabs = std::max(maxabs, x);
        float scale = maxabs / 127.0f;
        if (scale <= 0.0f || !std::isfinite(scale)) scale = 1.0f;
        a.scales[0] = scale;
        ctx.device->pool->parallel_for(total, grain,
            [&](uint64_t begin, uint64_t end, uint32_t) {
                kernel_quantize_i8_range(a, begin, end, ctx.device->cpu.isa);
            });
    } else {
        uint64_t jobs = (a.axis == VNPU_QUANT_PER_ROW) ? a.rows : a.cols;
        uint64_t jgrain = std::max<uint64_t>(1, jobs /
            static_cast<uint64_t>(ctx.device->info.num_threads * 4));
        ctx.device->pool->parallel_for(jobs, jgrain,
            [&](uint64_t begin, uint64_t end, uint32_t) {
                kernel_quantize_i8_range(a, begin, end, ctx.device->cpu.isa);
            });
    }
    ctx.add_bytes(total * sizeof(float), total);
    ctx.add_flops(total * 3);
}

static void execute_rmsnorm(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.rmsnorm;
    if (!a.X || !a.Y || !a.rows || !a.cols)
        throw std::runtime_error("invalid rmsnorm arguments");
    uint64_t grain = std::max<uint64_t>(1, a.rows /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(a.rows, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_rmsnorm_rows(a, static_cast<uint32_t>(begin),
                                static_cast<uint32_t>(end),
                                ctx.device->cpu.isa);
        });
    uint64_t n = static_cast<uint64_t>(a.rows) * a.cols;
    ctx.add_bytes(n * sizeof(float) + (a.W ? a.cols * sizeof(float) : 0),
                  n * sizeof(float));
    ctx.add_flops(n * 5);
}

static void execute_softmax(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.softmax;
    if (!a.X || !a.Y || !a.rows || !a.cols)
        throw std::runtime_error("invalid softmax arguments");
    uint64_t grain = std::max<uint64_t>(1, a.rows /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(a.rows, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_softmax_rows(a, static_cast<uint32_t>(begin),
                                static_cast<uint32_t>(end),
                                ctx.device->cpu.isa);
        });
    uint64_t n = static_cast<uint64_t>(a.rows) * a.cols;
    ctx.add_bytes(n * sizeof(float), n * sizeof(float));
    ctx.add_flops(n * 6);
}

static void execute_rope(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.rope;
    if ((!a.Q && !a.K) || !a.batch || !a.heads || !a.head_dim ||
        (!a.q_seq && !a.k_seq) || (a.head_dim & 1u))
        throw std::runtime_error("invalid rope arguments");
    uint64_t seq = std::max(a.q_seq, a.k_seq);
    uint64_t jobs = static_cast<uint64_t>(a.batch) * a.heads * seq;
    uint64_t grain = std::max<uint64_t>(1, jobs /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(jobs, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_rope_tokens(a, static_cast<uint32_t>(begin),
                               static_cast<uint32_t>(end),
                               ctx.device->cpu.isa);
        });
    uint64_t qn = static_cast<uint64_t>(a.batch) * a.heads * a.q_seq *
                  a.head_dim;
    uint64_t kn = static_cast<uint64_t>(a.batch) * a.heads * a.k_seq *
                  a.head_dim;
    ctx.add_bytes((qn + kn) * sizeof(float), (qn + kn) * sizeof(float));
    ctx.add_flops((qn + kn) * 8);
}

static void execute_swiglu(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.swiglu;
    if (!a.gate || !a.up || !a.Y || !a.count)
        throw std::runtime_error("invalid swiglu arguments");
    uint64_t grain = std::max<uint64_t>(4096, a.count /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(a.count, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_swiglu_range(a, begin, end, ctx.device->cpu.isa);
        });
    ctx.add_bytes(a.count * 2 * sizeof(float), a.count * sizeof(float));
    ctx.add_flops(a.count * 8);
}

static void execute_gelu(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.unary;
    if (!a.X || !a.Y || !a.count)
        throw std::runtime_error("invalid gelu arguments");
    uint64_t grain = std::max<uint64_t>(4096, a.count /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(a.count, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_gelu_range(a, begin, end, ctx.device->cpu.isa);
        });
    ctx.add_bytes(a.count * sizeof(float), a.count * sizeof(float));
    ctx.add_flops(a.count * 12);
}

static void execute_binary(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.binary;
    if (!a.A || !a.B || !a.Y || !a.count)
        throw std::runtime_error("invalid binary arguments");
    uint64_t grain = std::max<uint64_t>(4096, a.count /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    void (*fn)(const vnpu_binary_f32_args_t &, uint64_t, uint64_t, uint32_t) =
        ctx.cmd->opcode == VNPU_OP_ADD_F32 ? kernel_add_range : kernel_mul_range;
    ctx.device->pool->parallel_for(a.count, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            fn(a, begin, end, ctx.device->cpu.isa);
        });
    ctx.add_bytes(a.count * 2 * sizeof(float), a.count * sizeof(float));
    ctx.add_flops(a.count * 3);
}

static void execute_attention(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.attention;
    if (!a.Q || !a.K || !a.V || !a.O || !a.batch || !a.heads ||
        !a.seq_q || !a.seq_k || !a.head_dim)
        throw std::runtime_error("invalid attention arguments");
    constexpr uint32_t BQ = 16;
    uint64_t q_blocks = (a.seq_q + BQ - 1) / BQ;
    uint64_t jobs = static_cast<uint64_t>(a.batch) * a.heads * q_blocks;
    uint64_t grain = std::max<uint64_t>(1, jobs /
        static_cast<uint64_t>(ctx.device->info.num_threads * 4));
    ctx.device->pool->parallel_for(jobs, grain,
        [&](uint64_t begin, uint64_t end, uint32_t) {
            kernel_attention_range(a, begin, end, ctx.device->cpu.isa);
        });
    uint64_t qn = static_cast<uint64_t>(a.batch) * a.heads * a.seq_q *
                  a.head_dim;
    uint64_t kn = static_cast<uint64_t>(a.batch) * a.heads * a.seq_k *
                  a.head_dim;
    uint64_t mask_n = a.mask ? static_cast<uint64_t>(a.batch) * a.heads *
                               a.seq_q * a.seq_k : 0;
    ctx.add_bytes((qn + 2 * kn + mask_n) * sizeof(float),
                  qn * sizeof(float));
    ctx.add_flops(4ull * a.batch * a.heads * a.seq_q * a.seq_k *
                  a.head_dim);
}

static void execute_mtp_head(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.mtp_head;
    if (!a.hidden || !a.W || !a.logits || !a.gamma ||
        !a.hidden_dim || !a.vocab)
        throw std::runtime_error("invalid mtp_head arguments");
    vnpu_command_t gemm{};
    gemm.opcode = VNPU_OP_MATMUL_F32;
    gemm.flags = ctx.flags;
    gemm.id = ctx.cmd->id;
    auto &g = gemm.args.matmul_f32;
    g.A = a.hidden;
    g.B = a.W;
    g.bias = a.bias;
    g.C = a.logits;
    g.M = a.gamma;
    g.K = a.hidden_dim;
    g.N = a.vocab;
    g.trans_a = 0;
    g.trans_b = a.trans_w;
    g.alpha = 1.0f;
    g.beta = 0.0f;
    g.unit = a.unit;
    ExecCtx nested = ctx;
    nested.cmd = &gemm;
    execute_matmul_f32(nested);
    ctx.add_bytes(nested.bytes_read, nested.bytes_written);
    ctx.add_flops(nested.flops);
    ctx.tiles_dispatched += nested.tiles_dispatched;
    ctx.tiles_culled += nested.tiles_culled;
    ctx.dma_bytes += nested.dma_bytes;
}

static void execute_copy(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.copy;
    if (!a.dst || !a.src || !a.bytes)
        throw std::runtime_error("invalid copy arguments");
    std::memmove(a.dst, a.src, static_cast<size_t>(a.bytes));
    ctx.add_bytes(a.bytes, a.bytes);
    ctx.dma_bytes += a.bytes;
}

static void execute_memset(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.memset;
    if (!a.dst || !a.bytes)
        throw std::runtime_error("invalid memset arguments");
    std::memset(a.dst, static_cast<int>(a.value & 0xffu),
                static_cast<size_t>(a.bytes));
    ctx.add_bytes(0, a.bytes);
    ctx.dma_bytes += a.bytes;
}

static void execute_pack_weights(ExecCtx &ctx)
{
    const auto &a = ctx.cmd->args.pack_weights_i8;
    if (!a.B || !a.out || !a.K || !a.N)
        throw std::runtime_error("invalid pack_weights arguments");
    vnpu_packed_weights_t *w = nullptr;
    vnpu_status_t rc = vnpu_pack_weights_i8(ctx.device, a.B, a.scales,
                                            a.K, a.N, a.ldb, a.trans_b, &w);
    if (rc != VNPU_OK) throw std::runtime_error(vnpu_status_string(rc));
    *a.out = w;
    ctx.add_bytes(static_cast<uint64_t>(a.K) * a.N * 2,
                  w->packed.size());
    ctx.dma_bytes += static_cast<uint64_t>(a.K) * a.N * 2;
}

vnpu_status_t schedule_command(vnpu_device_t *device, vnpu_stream_t *stream,
                               const vnpu_command_t &cmd, uint32_t flags)
{
    if (!device || !stream || cmd.opcode >= VNPU_OP_COUNT)
        return VNPU_ERR_INVALID_COMMAND;
    uint64_t start = now_ns();
    uint32_t unit = VNPU_UNIT_AUTO;
    switch (cmd.opcode) {
    case VNPU_OP_MATMUL_F32: unit = cmd.args.matmul_f32.unit; break;
    case VNPU_OP_MATMUL_I8: unit = cmd.args.matmul_i8.unit; break;
    case VNPU_OP_MTP_HEAD_F32: unit = cmd.args.mtp_head.unit; break;
    default: break;
    }
    std::string trace_name = std::string(vnpu_opcode_name(cmd.opcode)) +
                             "#" + std::to_string(cmd.id);
    uint32_t category = 0; /* compute */
    if (cmd.opcode == VNPU_OP_COPY || cmd.opcode == VNPU_OP_MEMSET ||
        cmd.opcode == VNPU_OP_PACK_WEIGHTS_I8)
        category = 1; /* memory */
    else if (cmd.opcode == VNPU_OP_BARRIER || cmd.opcode == VNPU_OP_NOP)
        category = 2; /* sync */
    ScopedTrace trace(&device->trace, trace_name, cmd.opcode, unit, category);
    ExecCtx ctx;
    ctx.device = device;
    ctx.stream = stream;
    ctx.cmd = &cmd;
    ctx.flags = flags;
    ctx.trace = &trace;
    device->last_opcode.store(cmd.opcode);
    vnpu_status_t rc = VNPU_OK;
    try {
        switch (cmd.opcode) {
        case VNPU_OP_NOP:
        case VNPU_OP_BARRIER:
            break;
        case VNPU_OP_MATMUL_F32: execute_matmul_f32(ctx); break;
        case VNPU_OP_MATMUL_I8: execute_matmul_i8(ctx); break;
        case VNPU_OP_RMSNORM_F32: execute_rmsnorm(ctx); break;
        case VNPU_OP_SOFTMAX_F32: execute_softmax(ctx); break;
        case VNPU_OP_ROPE_F32: execute_rope(ctx); break;
        case VNPU_OP_SWIGLU_F32: execute_swiglu(ctx); break;
        case VNPU_OP_GELU_F32: execute_gelu(ctx); break;
        case VNPU_OP_ADD_F32:
        case VNPU_OP_MUL_F32: execute_binary(ctx); break;
        case VNPU_OP_ATTENTION_F32: execute_attention(ctx); break;
        case VNPU_OP_QUANTIZE_I8: execute_quantize_i8(ctx); break;
        case VNPU_OP_PACK_WEIGHTS_I8: execute_pack_weights(ctx); break;
        case VNPU_OP_MTP_HEAD_F32: execute_mtp_head(ctx); break;
        case VNPU_OP_COPY: execute_copy(ctx); break;
        case VNPU_OP_MEMSET: execute_memset(ctx); break;
        default: rc = VNPU_ERR_INVALID_COMMAND; break;
        }
    } catch (const std::exception &e) {
        debug_log("command %s failed: %s", vnpu_opcode_name(cmd.opcode),
                  e.what());
        rc = VNPU_ERR_INTERNAL;
    } catch (...) {
        rc = VNPU_ERR_INTERNAL;
    }
    uint64_t duration = now_ns() - start;
    device->kernel_ns.fetch_add(duration);
    if (category == 1) device->memory_ns.fetch_add(duration);
    else if (category == 2) device->sync_ns.fetch_add(duration);
    device->commands_completed.fetch_add(0); /* status is updated by stream */
    device->bytes_read.fetch_add(ctx.bytes_read);
    device->bytes_written.fetch_add(ctx.bytes_written);
    device->flops.fetch_add(ctx.flops);
    device->dma_bytes.fetch_add(ctx.dma_bytes);
    device->tiles_dispatched.fetch_add(ctx.tiles_dispatched);
    device->tiles_culled.fetch_add(ctx.tiles_culled);
    trace.add_bytes(ctx.bytes_read, ctx.bytes_written);
    trace.add_flops(ctx.flops);
    if (rc != VNPU_OK) trace.cancel();
    return rc;
}

} /* namespace vnpu */

/* ------------------------------------------------------------------------- */
/* Packed INT8 weights                                                       */
/* ------------------------------------------------------------------------- */

extern "C" {

VNPU_API vnpu_status_t VNPU_CALL vnpu_pack_weights_i8(
    vnpu_device_t *device,
    const int8_t *B,
    const float *scales,
    uint32_t K,
    uint32_t N,
    uint32_t ldb,
    uint32_t trans_b,
    vnpu_packed_weights_t **out_weights)
{
    if (!device || !B || !out_weights || !K || !N || trans_b > 1)
        return VNPU_ERR_INVALID_ARGUMENT;
    *out_weights = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        uint32_t ldb_eff = ldb ? ldb : (trans_b ? K : N);
        uint32_t min_ldb = trans_b ? K : N;
        if (ldb_eff < min_ldb) return VNPU_ERR_INVALID_ARGUMENT;
        uint32_t Kpad = static_cast<uint32_t>(
            vnpu::align_up(K, 4));
        uint32_t Npad = static_cast<uint32_t>(
            vnpu::align_up(N, 16));
        uint32_t k4 = Kpad / 4;
        uint64_t packed_bytes = static_cast<uint64_t>(Npad / 16) * k4 * 64;
        auto *w = new vnpu_packed_weights_t();
        w->device = device;
        w->K = K;
        w->N = N;
        w->Kpad = Kpad;
        w->Npad = Npad;
        w->layout = 1; /* VNNI [N/16][K/4][16][4] */
        w->original.resize(static_cast<size_t>(K) * N);
        w->packed.resize(static_cast<size_t>(packed_bytes), 0);
        w->scales.resize(Npad, 1.0f);
        w->column_sums.resize(Npad, 0);
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t n = 0; n < N; ++n) {
                int8_t v = trans_b ?
                    B[static_cast<size_t>(n) * ldb_eff + k] :
                    B[static_cast<size_t>(k) * ldb_eff + n];
                w->original[static_cast<size_t>(k) * N + n] = v;
                w->column_sums[n] += static_cast<int32_t>(v);
            }
        }
        if (scales) {
            for (uint32_t n = 0; n < N; ++n) w->scales[n] = scales[n];
        }
        /* Parallel packing by 16-column blocks. */
        uint32_t blocks = Npad / 16;
        device->pool->parallel_for(blocks, 1,
            [&](uint64_t begin, uint64_t end, uint32_t) {
                for (uint64_t nb = begin; nb < end; ++nb) {
                    for (uint32_t g = 0; g < k4; ++g) {
                        uint8_t *dst = w->packed.data() +
                            (nb * static_cast<uint64_t>(k4) + g) * 64;
                        for (uint32_t lane = 0; lane < 16; ++lane) {
                            uint32_t n = static_cast<uint32_t>(nb * 16 + lane);
                            for (uint32_t kk = 0; kk < 4; ++kk) {
                                uint32_t k = g * 4 + kk;
                                int8_t v = 0;
                                if (n < N && k < K)
                                    v = w->original[
                                        static_cast<size_t>(k) * N + n];
                                dst[lane * 4 + kk] =
                                    static_cast<uint8_t>(v);
                            }
                        }
                    }
                }
            });
        device->dma_bytes.fetch_add(static_cast<uint64_t>(K) * N * 2);
        *out_weights = w;
        return VNPU_OK;
    });
}

VNPU_API void VNPU_CALL vnpu_packed_weights_destroy(vnpu_packed_weights_t *weights)
{
    delete weights;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_packed_weights_info(
    const vnpu_packed_weights_t *weights,
    uint32_t *K,
    uint32_t *N,
    uint64_t *bytes,
    const float **scales,
    const int32_t **column_sums)
{
    if (!weights) return VNPU_ERR_INVALID_ARGUMENT;
    if (K) *K = weights->K;
    if (N) *N = weights->N;
    if (bytes) *bytes = weights->packed.size();
    if (scales) *scales = weights->scales.data();
    if (column_sums) *column_sums = weights->column_sums.data();
    return VNPU_OK;
}

/* ------------------------------------------------------------------------- */
/* Synchronous convenience wrappers                                          */
/* ------------------------------------------------------------------------- */

#define VNPU_SIMPLE_WRAPPER(fn_name, op_code, arg_field, arg_type)            \
    VNPU_API vnpu_status_t VNPU_CALL fn_name(vnpu_stream_t *stream,           \
                                            const arg_type *args)            \
    {                                                                          \
        if (!stream || !args) return VNPU_ERR_INVALID_ARGUMENT;               \
        vnpu_command_t cmd{};                                                  \
        cmd.opcode = op_code;                                                  \
        cmd.args.arg_field = *args;                                            \
        vnpu_status_t rc = vnpu_stream_submit(stream, &cmd, 1, nullptr);      \
        if (rc != VNPU_OK) return rc;                                          \
        return vnpu_stream_sync(stream);                                        \
    }

VNPU_SIMPLE_WRAPPER(vnpu_matmul_f32, VNPU_OP_MATMUL_F32, matmul_f32,
                    vnpu_matmul_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_matmul_i8, VNPU_OP_MATMUL_I8, matmul_i8,
                    vnpu_matmul_i8_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_quantize_i8, VNPU_OP_QUANTIZE_I8, quantize_i8,
                    vnpu_quantize_i8_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_rmsnorm_f32, VNPU_OP_RMSNORM_F32, rmsnorm,
                    vnpu_rmsnorm_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_softmax_f32, VNPU_OP_SOFTMAX_F32, softmax,
                    vnpu_softmax_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_rope_f32, VNPU_OP_ROPE_F32, rope,
                    vnpu_rope_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_swiglu_f32, VNPU_OP_SWIGLU_F32, swiglu,
                    vnpu_swiglu_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_gelu_f32, VNPU_OP_GELU_F32, unary,
                    vnpu_unary_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_add_f32, VNPU_OP_ADD_F32, binary,
                    vnpu_binary_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_mul_f32, VNPU_OP_MUL_F32, binary,
                    vnpu_binary_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_attention_f32, VNPU_OP_ATTENTION_F32, attention,
                    vnpu_attention_f32_args_t)
VNPU_SIMPLE_WRAPPER(vnpu_mtp_head_f32, VNPU_OP_MTP_HEAD_F32, mtp_head,
                    vnpu_mtp_head_f32_args_t)

#undef VNPU_SIMPLE_WRAPPER

} /* extern "C" */
