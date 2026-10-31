# Virtual NPU: what is actually in here

> **Status: separate prototype, not part of `libvnpu`.**
> This file documents `tools/laptop/vnpu_laptop.cpp`, a Windows/Meteor Lake
> experiment. It is not in `LIB_SRCS` in the `Makefile` and is not built by
> `make` or `CMakeLists.txt`, so **none of the numbers below describe the
> shipped library**. The library has no AVX-512 and no VNNI kernel: it runs
> AVX2+FMA (`src/kernels/kernels.cpp`), where INT8 has no dot-product
> instruction. Library figures are in `README.md` and `docs/PERFORMANCE.md`,
> measured against a compiler-vectorised plain-C baseline.

Source of truth: `vnpu_laptop.cpp` (kernels, pools, benchmarks) and `sherlock.h`
(tracing). Target: Intel Core Ultra 5 135H (Meteor Lake), AVX2 + FMA3 +
AVX-VNNI (VEX-256), no AVX-512. The baseline it must beat is a plain
compiler-vectorisable triple loop (`gemm_i8_rawcpu`) and a scalar oracle.

## Kernels

Every kernel below is verified exact against the scalar oracle (max abs diff 0).

| kernel | technique |
|---|---|
| `gemm_i8_rawcpu` | scalar baseline, the competition |
| `gemm_i8_scalar` | correctness oracle |
| `gemm_i8_avx2` | `maddubs` + `mul_sum_i8_pairs` + packed `hsum_epi32` |
| `gemm_i8_vnni_blk` | VEX-256 `VPDPBUSD`, `[colblock][k4]` layout |
| `gemm_i8_vnni_pan8` | panel-packed, 8 accumulator blocks, K unroll 2, prefetch |
| `gemm_i8_vnni_rowblk<MR>` | M-blocked (1/2/4 rows), the prefill lever |
| `gemm_i8_vnni_stage<MODE,PF>` | Cell-style staging tile, double-buffered, see STAGING_TILE.md |

Weight layout: `PackedI8` keeps a panel-packed `[panel][k4][8][32]` copy so the
inner loop reads contiguous 32-byte VNNI operands instead of striding a
row-major matrix.

## Scheduling

- `pan8_pool`: persistent pool, one worker per P-core thread, epoch-dispatched,
  static N-partition. Thread creation happens once, never in the timed loop
  (creating a thread costs ~0.5 ms on Windows, 8x a 64 us decode GEMV).
- `npu_cluster`: one worker per P-core plus a governor on an E-core (the Cell
  PPE/SPE split, and the Kryo/hybrid split).
- `hetero_pool`: P + E + LP-E stealing from one atomic cursor, fine tiles.

## Hardware characterisation

- `core_scan` / `core_classes`: EfficiencyClass-driven P/E/LP-E enumeration.
  Exposes both `pfirst`/`efirst`/`lpefirst` (one logical per physical core, for
  the hetero pools) and `pmask`/`emask` (full hyperthread sibling masks, for the
  legacy `pan8_pool`).
- `large_page_probe`: `MEM_LARGE_PAGES` availability. Result: unavailable, needs
  `SeLockMemoryPrivilege`.
- `cpu_avx2` / `cpu_avxvnni`: CPUID dispatch.
- `clock_ghz`, `bench_read_bw`, `stream_gbs`: measured clock and streaming
  bandwidth, used to build the cache-roof ladder.

## GGUF block shaders (native, no runtime dequant)

`blk_q8_0`, `blk_q4_0`, `blk_q5_0` plus `dot_q8_0_q8_0`, `dot_q8_0_q4_0`,
`dot_q8_0_q5_0`, `f16_to_f32` and an f16 lookup table.

## Tracing

`sherlock.h`, nine traps: DISPATCH, MEMORY, INSTRUCTION, SYNC, ALLOC, TRANSFER,
DEPENDENCY, ACCURACY, COMPUTE. `sherlock_trace_start/end`, `sherlock_dump_trace`,
`sherlock_mute`, `now_ns`. Per-trace bytes read/written, cache misses, ops count,
worst-first hot spots, a B/MAC column, and cache-roof ladder verdicts.
`VNPU_TRACE=1` for the text report, `=2` for a chrome trace too.

## Reference tech, and whether it actually landed

| reference | idea | in this NPU | verdict |
|---|---|---|---|
| 3DNow! | packed ops, no scalar breakouts | `XOR 0x80` sign->unsigned for `dpbusd`, packed `hsum` | in `make_avec` |
| SSE streaming | NT stores, `prefetchnta` | `_mm256_stream_ps`, `_MM_HINT_NTA` | built, measured a wash |
| AltiVec | full-width `vec_perm`, `vec_splat`, `dst` | `permute2x128`+`pshufb`, `vpbroadcastd`, `dst_touch` | in `make_avec` |
| UltraSPARC VIS | `FALIGNDATA`, `FMUL8x16`, `PDIST`, `BMASK` | `dpbusd` (FMUL8x16 descendant); `FALIGNDATA` unused | partial |
| UltraSPARC T | chip multithreading | many runnable threads in the pools | yes |
| Cell SPE | local store + double-buffered DMA | staging tile | **built, lost** (STAGING_TILE.md) |
| z/Architecture | VPERM, VDP, VFENE, huge pages | permute, `dpbusd`, `cmpeq`+`tzcnt`; no huge pages | partial |
| DB2 | buffer pool, pattern prefetch, clustering, partitioning | pre-packing, N partitioning, kernel-per-shape | biggest wins |
| Sun MAJC | VLIW, vertical multithreading, workload-shaped ISA | `pan8` vs `rowblk` split by shape | yes |
| Kryo 2 | per-core DVFS, heterogeneity | P/E/LP-E placement, `hetero_pool` | biggest single lever |
| SGI / Cobalt | ccNUMA, bandwidth arbitration, producer/consumer FIFO | affinity yes; arbitration removed the governor prefetcher | partial |
| POWER10 MMA / z16 NNPA / Intel AMX | real matrix unit | absent on Meteor Lake | no |

Not transferable here: packed-decimal, POWER DSCR.

## Measured verdicts

Everything in this section was measured on the prototype above, single-thread
unless stated, against its own `rawcpu` baseline. It is not the library's
performance and should not be quoted as such.

- `vnni_pan8` is 3.4-4.5x `rawcpu` single-thread, across M = 1..128.
- 8 P-core logicals reach ~7.6x at M=128 (268 GMAC/s against 35 GMAC/s for one).
- E-cores: ~8-17% of a P-core on this kernel, so adding them to the GEMM makes it
  slower; they straggle at the barrier. All 9 E-cores roughly equal 1.4 P-cores.
- Streaming hints: a wash, +-5%. NT stores once cost 0.38x.
- M-blocking: negative while the weights fit L3.
- Staging tile: no win, see STAGING_TILE.md.
- Huge pages: unavailable.
- Decode-shaped block shaders fall behind int8 in order of weight size:
  Q8_0 0.75x, Q4_0 0.56x, Q5_0 0.28x (these are the current merge bounds, not
  the end state).

## Still open

- UltraSPARC VIS `FALIGNDATA` analogue: the byte-shift-and-merge form has not
  been used. Current code uses `permute2x128` + `pshufb`, which covers the cases
  FALIGNDATA would.
- Bandwidth arbitration beyond affinity (SGI/Cobalt). The governor prefetcher is
  the only piece built.
- Issue-side work, which is where the remaining wins are: wider accumulator
  blocking, fewer non-VNNI instructions per MAC, and batching prefill into a real
  GEMM instead of one token per forward pass.
