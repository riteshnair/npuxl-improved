# Performance and measurement methodology

## Reference environment

- 2 physical cores, 1 thread/core (KVM guest), AVX2 + FMA, **no AVX-512**
  (`grep -c avx512 /proc/cpuinfo` = 0; the library contains no EVEX-encoded
  instruction — `objdump -d lib/libvnpu.so | grep -c zmm` = 0)
- default 2 units = physical cores; SMT siblings would share L1/L2
- as this guest reports them: 32 KiB L1D, 512 KiB L2, 32 MiB L3 (`lscpu`);
  a hypervisor may under-report, so prefer the probe below to the spec sheet
- the VM is **shared**. The same binary has produced 19.2 and 31.7 ms for
  1024³ F32 in one session. Absolute numbers are samples; only paired,
  interleaved comparisons survive this noise.
- GCC, C++17, `-O3 -fno-math-errno -ffp-contract=fast -funroll-loops`
- `make benchmark`, 5 iterations for GEMM, 3 for autotune/LLM block
- 2 threads (the issuing thread is one unit, so a 2-thread device runs two
  active threads, not three)

## Where the cycles go

`make peak` (benchmarks/peak_probe.cpp) measures the instruction ceiling with
the same 16-register budget the kernel has, so "efficiency" has a denominator.
On the reference VM (1 thread / 2 threads, GFLOP/s, best of 3):

| loop shape | mem ops per 12 FMA | 1 thread | 2 threads |
|---|---|---:|---:|
| FMA only, no loads | 0 | 136-141 | 260-276 |
| 1 B load + 3 A broadcasts | 4 | 141-155 | 280-304 |
| 4 B loads + 3 A broadcasts (**shipped 3x32 tile**) | 7 | 73-95 | 137-188 |
| 6x16 tile (2 loads + 6 broadcasts) | 8 | 83 | 169 |
| 4x24 tile (3 loads + 4 broadcasts) | 7 | 97 | 147-155 |

Reading it:

- **The kernel is load-bound, not FMA-bound.** Dropping three of the four B
  loads per k (`1 B load` row) recovers the full FMA ceiling; dropping the
  broadcasts changes nothing (`4 loads, no broadcasts` measures the same as
  with them). The gap between 136 and ~90 is the cost of streaming B.
- **BLIS's AVX2 tile is not this machine's optimum.** The textbook 6x16
  register block is *slower* than the shipped 3x32 here, because it trades one
  expensive B load for four cheap broadcasts and still ends up with more memory
  uops per FMA. Copying a tuned shape from another microarchitecture is a
  guess; measuring it is not.
- **A 4x24 tile is the only shape that beat 3x32 in the model** (97 vs 93,
  within noise at 1 thread; 147-155 vs 137-188 at 2 threads, i.e. not
  distinguished). It is the obvious next experiment, not a known win.
- End-to-end, one thread reaches 51-57 GFLOP/s and two threads 80-118, i.e.
  **~40% and ~45% of the measured machine ceiling**. The rest is the load
  stream, panel packing, and the memory hierarchy at 1024³ — not idle CPUs.
- Parallel efficiency on 2 cores is **1.5-1.85x, not 2x** (F32 512³ 51.5 -> 84.0;
  F32 1024³ 54.2 -> 80.8; INT8 1024³ 57.1 -> 105.6), measured 1 vs 2 threads on
  the same binary.

Build flags were checked the same way rather than assumed: rebuilding the
library at `-O2` (no `-funroll-loops`) makes 1024³ INT8 **34-46 ms against
17-20 ms**, 3 of 3 interleaved rounds. The shipped `-O3 -funroll-loops` stays.

## GEMM vs GEMV: two different ceilings

`M x 4096 x 4096`, 2 threads, 3 iterations, `make peak` for the denominators
(64 MiB stream read = 19.8 GB/s; FMA ceiling 260-276 GFLOP/s):

| M | dtype | ms | GFLOP/s | achieved GB/s | vs plain C | bound by |
|---:|---|---:|---:|---:|---:|---|
| 1 | F32 | 2.96 | 11.3 | 22.7 | 1.26x | memory (at ceiling) |
| 1 | INT8 | 1.03 | 32.5 | 16.3 | 3.76x | memory, 82% of roofline |
| 4 | F32 | 6.51 | 20.6 | 10.3 | 2.27x | memory |
| 4 | INT8 | 3.24 | 41.4 | 5.2 | 4.43x | 4.6% over roofline |
| 16 | F32 | 10.65 | 50.4 | 6.3 | 6.03x | compute |
| 16 | INT8 | 8.80 | 61.0 | 1.9 | 7.72x | compute |
| 64 | F32 | 24.5 | 87.8 | 2.7 | 11.37x | compute |
| 64 | INT8 | 19.8 | 108.3 | 0.8 | 13.04x | compute |

Arithmetic intensity predicts the crossover: F32 GEMV is 0.5 flop/byte, so the
roofline caps it at 19.8 x 0.5 = **9.9 GFLOP/s**; INT8 GEMV is 2 flop/byte,
capped at **39.6 GFLOP/s**. F32 at M=1 measures 11.3 GFLOP/s (22.7 GB/s) — at
the memory ceiling, so no vector width or thread count can improve F32 decode.
INT8 at M=1 measures 32.5 GFLOP/s, 82% of its roofline and 16.3 GB/s of a 19.8
GB/s stream. From M=16 up both are compute-bound, which is where the FMA
ceilings in "Where the cycles go" apply.

### What actually limited INT8 decode

Not the multiply. The single-row path used to keep C in registers for a
32-column block, which forces each pass over B to walk all K rows at a stride of
N bytes for 32-64 useful bytes each. Two changes fixed it, both exact (identical
integer arithmetic in the same k order, so every output is unchanged bit for
bit):

1. **Rank-1 streaming for the left-over row.** Instead of 32-column register
   blocks, stage the C row (4 bytes per column) and stream B one row at a time.
   Measured in one binary, 4096x4096 int8, single thread, best of 6, all
   variants bit-identical to each other:

   | variant | ms | GB/s |
   |---|---:|---:|
   | 32-column register blocks (old shape) | 3.69 | 4.5 |
   | 128-column register blocks | 2.21 | 7.6 |
   | 256-column register blocks | 1.46 | 11.5 |
   | rank-1 streaming | **1.19** | **14.1** |

2. **The decode tile width, from N and the cache.** The rank-1 loop only
   streams if the tile is wide: with the old fixed 64-column decode tile each
   unit still walked B at the row stride. `1x4096x4096` int8 by decode tile
   width, 2 threads: 64 cols 2.85 ms, 128 cols 3.13, 256 cols 1.35, 512 cols
   1.20, 1024 cols 1.25, 2048 cols 1.13, 4096 cols 1.21.

Together: **2.47 -> 0.95 ms** at M=1 (2.6x, 13.6 -> 33 GFLOP/s).

#### How the width is chosen

`tile_geometry` gives a single row one contiguous slice of N per unit, then
corrects that split twice:

- **A slice narrower than 256 columns is not worth a second unit.** The run per
  row gets too short to pay for the split: `1x4096x128` measured 0.053 ms
  unsplit against 0.063 split, `1x4096x256` 0.069 against 0.078.
- **A tile may not outgrow L2.** The rank-1 row stages one `int32` per column of
  the tile and re-reads it on every k step, so the staging row (4 bytes per
  column) shares L2 with the B stream and gets half of it: `l2_bytes / 8`
  columns, also capped by the decode scratch (`VNPU_LOCAL_STORE_BYTES / 4`).
  Without this cap the split is right for small N and badly wrong for wide ones,
  because `N / units` grows past what the staging row can hold: at `K=64,
  N=262144` 131072 columns is 512 KiB of staging row, the whole L2, and 65536
  columns is nearly twice as fast. When the cap binds, the tile count is rounded
  up to a multiple of units so the per-unit share stays even - five tiles over
  two units (3+2) cost 1.22 ms against 0.99 ms for four (2+2).

int8 decode, 2 threads, best of 11 interleaved rounds in one process, one row
against a `K x N` matrix. Comparing widths needs the interleaving, so the sweep
ran in a probe against the C API; a single row is reproducible with
`vnpu-cli bench --shape 1x64x262144 --dtype i8 --threads 2 --unit voodoo`, which
accepts wide shapes as long as the f32 working set stays under 1 GiB:

| N | K | old 64-column tile | `N / units` alone | chosen width |
|---:|---:|---:|---:|---:|
| 256 | 4096 | 0.095 ms | 0.079 ms (one tile) | 0.079 ms |
| 1024 | 4096 | 0.227 ms | 0.175 ms | 0.175 ms |
| 4096 | 4096 | 2.250 ms | 0.823 ms | 0.858 ms |
| 65536 | 1024 | 8.333 ms | 2.949 ms | 2.811 ms |
| 262144 | 64 | 2.174 ms | 1.795 ms | 0.954 ms |
| 1048576 | 64 | 9.400 ms | 3.700 ms | 3.700 ms |

The chosen width is within measurement noise (5%, the same spread identical
configurations show against each other) of the best fixed width on every shape,
and 2.4-2.6x faster than the old fixed tile from N=4096 up. The two corrections
are worth 1.9x on their own at N=262144, where the bare `N / units` rule picks
a tile that no longer fits next to the stream.

The wide split is gated on the row count, and on the unit run length from M=2
up. Two or three rows cannot keep a wide column block in registers, and the
packed panel couples rows in pairs rather than covering them, so the int8 kernel
stages one int32 row per row of the tile and
streams B once for both; that doubles the accumulator traffic, so it only pays
while a unit still owns a long run. 2x4096x4096 int8 (2048 columns each) went
1.58 -> 1.07 ms, while 2x4096x1024 (512 columns each) went 0.28 -> 0.53 ms and
keeps the 64-column tile.

Three rows had the same problem in a different place. The packed panel couples
rows in pairs, so an odd row re-reads the whole panel: at M=3 that is 128 bytes
of panel per k step for three outputs, against the same 128 bytes for four at
M=4, which is the entire 0.65 against 0.51 ms/row gap. A rank-3 stream reads B
once for all three rows instead, and it wins at every width tried (int8, 2
threads, median / min of 11-15 alternating rounds):

| shape | before | after | |
|---|---:|---:|---:|
| 3x4096x4096 | 1.98 / 1.79 ms | 1.70 / 1.21 ms | 1.16-1.48x |
| 3x4096x2048 | 0.84 / 0.80 ms | 0.67 / 0.62 ms | 1.26x |
| 3x2048x2048 | 0.62 / 0.47 ms | 0.44 / 0.33 ms | 1.41x |
| 3x512x262144 | 27.0 / 25.8 ms | 14.5 / 14.4 ms | 1.86x |
| 3x64x262144 | 3.06 / 2.63 ms | 2.42 / 2.27 ms | 1.26x |

The wide decode shapes gain most because the old path also copied a `K x 64`
panel per tile, and there are `N / 64` of them; the rank-3 stream reads the
original B once.

Four rows are where the stream stops paying. Four staged `int32` rows thrash L1
badly enough that the panel's register-held accumulators win: `4x4096x4096` int8
went 1.98 -> 2.32 ms with a rank-4 stream, so the wide gate deliberately stops at
M=3 and M=4 keeps the packed panel with its 64-column tile. f32 has no staged
decode stream at all and keeps the narrow tile, where a wide tile at M=2 cost
4.9 -> 6.0 ms.

### Two ideas that measured worse, and why they are not here

- **`vpmaddubsw` + `vpmaddwd` int8 dot product.** On paper 2 operations per 32
  MACs instead of ~13. It saturates: `vpmaddubsw` clamps its int16 pair sum, and
  255 x 127 x 2 = 64770 overflows 32767, so with full-range int8 operands the
  biased form is simply wrong (199 of 200 random full-range trials disagreed
  with the exact product). Interleaving activations with zero bytes keeps one
  product per pair and is exact, but measured 0.90 ms against 1.11 ms for the
  plain widening path - about 11%, for a scheme that also needs a matching byte
  order in the packed weights. The layout work above is 3.9x on the same shape.
- **BLIS-style 6x16 tiles.** Measured slower here; the kernel is load-bound, see
  "Where the cycles go".

## What a speedup means here

`vnpu-cli bench` measures two plain-C references in the same binary and reports
against the **faster** of them:

1. `baseline_scalar_ms` — the naive scalar triple loop, m-n-k order, no
   intrinsics, no blocking;
2. `baseline_compiler_ms` — the same arithmetic in the order the compiler can
   auto-vectorise (K outer, rows walked linearly), still no intrinsics.

Both are timed as the best of three runs after a warm-up. Taking the minimum of
the two means a reported factor can never be inflated by a slow reference loop.

**Withdrawn numbers.** Earlier revisions of this file quoted 50–85× against a
single-shot baseline that stored through a `volatile` inside the measured loop
and walked B with stride N. That measured the volatile pipeline and a
pessimised access pattern, roughly an order of magnitude worse than a fair
compiler loop, and every factor derived from it was inflated by about that
much. Those figures have been removed rather than adjusted.

These are still **not** a claim against MKL, OpenBLAS, BLIS or oneDNN. The VM
is noisy — the same binary has measured 80–101 GFLOP/s at 512³ F32 — so
`benchmarks/*.json` is authoritative for a given run.

## Dense GEMM

Best unit per shape from `benchmarks/gemm.json`:

| Shape | dtype | Time | GFLOP/s | vs plain C |
|---|---:|---:|---:|---:|
| 128×128×128 | F32 | 0.096 ms | 43.8 | 2.15× |
| 128×128×128 | INT8 | 0.075 ms | 55.7 | 2.74× |
| 256×256×256 | F32 | 0.446 ms | 75.2 | 3.70× |
| 256×256×256 | INT8 | 0.358 ms | 93.7 | 4.61× |
| 512×512×512 | F32 | 2.79 ms | 96.1 | 4.66× |
| 512×512×512 | INT8 | 2.61 ms | 102.8 | 4.98× |
| 1024×1024×1024 | F32 | 19.2 ms | 111.9 | 5.56× |
| 1024×1024×1024 | INT8 | 18.2 ms | 117.9 | 5.85× |
| 1×4096×4096 (decode) | F32 | 3.04 ms | 11.1 | 1.28× |
| 1×4096×4096 (decode) | INT8 | 2.20 ms | 15.2 | 1.76× |

### Decode is the honest weak spot

A 1×4096×4096 GEMM reads 64 MiB of weights and does one row of arithmetic: it
is bandwidth-bound, and the compiler loop reads the same bytes. F32 decode
therefore lands at 0.96–1.25× across runs — **parity, not a win**. INT8 decode
wins ~2.3× mainly because it halves the bytes it must stream.

### Panel packing (implemented for F32)

Row-major B costs a fresh cache line per K step, and each 3-row block re-reads
the same K×32 strip: for 512³ that is ~22 MB of strip traffic per N tile,
mostly served from L2/L3. The F32 NN path now packs each tile's columns into a
`[kc][tn]` panel (`kc=128, tn=64` → 32 KiB, exactly L1) once per K block, and
the microkernel walks that contiguous panel.

Tiles also become full-height (`tm = M`) whenever N alone supplies at least
`2 × units` tiles, so a panel is packed once for the whole GEMM rather than
once per M tile.

A/B on the same binary, `VNPU_PACK=0` as the control, 6 interleaved rounds:

| Shape | unpacked (median) | packed (median) | change |
|---|---:|---:|---:|
| 256³ F32 | 0.433 ms | 0.399 ms | +8% |
| 512³ F32 | 3.275 ms | 3.104 ms | +5% |
| 1024³ F32 | 36.4 ms | 21.7 ms | **+68%** |
| 256³ INT8 | 0.431 ms | 0.415 ms | +4% (within noise) |
| 512³ INT8 | 3.221 ms | 2.909 ms | +10% |
| 1024³ INT8 | 25.6 ms | 19.2 ms | **+33%** |

The win grows with the working set, exactly as the strip-traffic model predicts,
and is neutral-to-slightly-positive below that. Decode (M<3) is untouched: it
stays on the streaming GEMV / single-row INT8 path, because a one-row tile reads
each panel entry exactly once and packing it would be a pure copy.

The two dtypes get away with different block strategies because an int8 panel is
four times smaller. F32 packs `kc=128` rows at a time (32 KiB) and accumulates C
across K blocks. INT8 packs the *whole* K of the tile in one go (64 KiB at
`tn=64, K=1024`), so the int32 accumulators stay in registers and the summation
order is bit-for-bit what the row-major path computes. `make conformance`
asserts that equality rather than leaving it as a comment.

The measured INT8 decode delta was noise: an A/B run in alternating order showed
an 8% "win", and the same comparison in reversed order interleaved the two
configurations. The code path is provably identical there (the tile is one row),
which is why it is reported as unchanged.

### Command launch overhead

Small commands pay a fixed cost that has nothing to do with the kernel. Measured
with a probe that submits the *same* op list two ways -- one command per
submit+sync, or all commands in a single submit+sync -- best of 3:

| Op | separate submit+sync | one batch | cost of one round trip |
|---|---:|---:|---:|
| 128×128×128 F32 | 0.088 ms | 0.071 ms | 17 µs (19%) |
| 1×512×1376 F32 (decode) | 0.073 ms | 0.045 ms | 28 µs (38%) |
| 512×512×512 F32 | 2.689 ms | 2.733 ms | none (-2%) |

The delta is unchanged at `num_threads=1`, where there is no pool worker to wake
at all. So the cost is the async round trip -- host -> stream thread -> host --
not the parallel-for. Two consequences:

- **Callers should batch independent commands into one `vnpu_stream_submit`.**
  For decode-shaped ops that is worth a third of the op, and it is a caller-side
  change with no kernel work.
- **Holding the thread pool hot across a batch (`VNPU_LEASE=1`) does not
  recover it**, because it targets the pool hop rather than the round trip.
  Measured on this box it is a small *regression* (prefill 11.9 vs 11.5 ms,
  4 rounds, medians): a spinning worker competes for the 2 vCPUs with the stream
  thread and the host. It is off by default. On a many-core host the spinning
  would land on idle cores and could pay off, but that is untested here and
  should not be assumed.

## Observations

1. **Prefill-shaped work is 1.6–5.3× a compiler-vectorised loop**, typically
   3–5× from 256³ up. That is the layer's actual value: it is not 50×.
2. **The 3-row × 32-column register block is a correctness-and-theory win, not
   a measured one here.** Twelve accumulators (3 rows × 4 vectors) plus three
   broadcasts and one live B vector fill the 16 YMM registers, which is what
   hides a 5-cycle FMA latency across two ports; run-to-run noise on this VM
   (±40%) is larger than the expected 5–10% gain.
3. **CELL is slower than the plain-C baseline everywhere** (0.3–0.6×). It is
   opt-in only and is never selected automatically any more; when it was
   (for `M≤4` and for small `K,N`), it made `AUTO` decode 6× slower than
   `VOODOO` while looking like a default.
4. **Per-command cost matters for decode.** Submit+sync costs ~30 µs when a
   worker has to be woken, versus ~1.5 µs/op when 100 commands are batched.
   The pool now runs single-chunk commands inline on the issuing thread and
   wakes only `units-1` workers.
5. **Sherlock shows decode is memory-bound, prefill compute-bound.** ~96
   FLOP/byte on the dense GEMM benchmark; decode is a fraction of that.

## Synthetic transformer block

Command:

```bash
./bin/vnpu-cli llm-bench --hidden 512 --intermediate 1376 \
  --heads 8 --layers 2 --prefill 32 --decode 1 --dtype both \
  --iterations 3 --json benchmarks/llm-block.json \
  --trace benchmarks/llm-trace.json
```

Latest results:

| Phase | F32 | INT8 |
|---|---:|---:|
| prefill 32 tokens | 7.81 ms, 4,098 tok/s, 52.4 GFLOP/s | 7.44 ms, 4,301 tok/s, 55.0 GFLOP/s |
| decode 1 token | 1.32 ms, 758 tok/s, 9.6 GFLOP/s | 1.38 ms, 722 tok/s, 9.1 GFLOP/s |

`QUANTIZE_I8` remains a visible cost in the INT8 path; pre-quantized hidden
states or a quantized residual stream would remove most of it.

## Cluster and speculation

- Two loopback nodes, 256³ GEMM: correct result, 1.35× effective worker
  parallelism, 0.86× wall-clock vs local. Network overhead dominated.
- Distributed speculative demo: 20 tokens in 13.8 ms, 16/20 accepted,
  theoretical 1.08–1.47× with a toy model. Real gains require a cheap draft
  head and a single batched verifier round trip.

## Reproducing

```bash
make clean
make -j$(nproc)
make benchmark
cat benchmarks/gemm.json
cat benchmarks/autotune-f32.json
cat benchmarks/autotune-i8.json
cat benchmarks/llm-block.json
cat benchmarks/cluster-demo.txt
cat benchmarks/specdemo.txt
```

`make bench-matrix` additionally builds `bin/bench_kernels` and
`bin/bench_scaling`, which print each unit, the scalar baseline and the
compiler-autovectorised baseline side by side.

Chrome traces:

- `benchmarks/sherlock-trace.json` – GEMM units/dtypes;
- `benchmarks/llm-trace.json` – transformer block.

Summarize a trace:

```bash
python3 scripts/trace_report.py benchmarks/sherlock-trace.json 20
```

## Caveats

- These are synthetic random matrices, not real model weights. Quantization
  error and cache behavior will differ.
- No tokenizer, sampling, KV cache or model loader is included.
- Shared VMs are noisy: treat differences under ~40% on large shapes as
  unresolved rather than as regressions or wins.
- Network v1 is plaintext TCP and native-endian.
