# Virtual NPU rewrite — engineering report

## Outcome

The PCem dependency is gone from the buildable runtime. In its place is a
standalone virtual NPU that can be consumed as a shared library, static
library, Windows DLL (via CMake), CLI tool, or network node. The original
experiments are preserved under `legacy/` and `archive/`, but no PCem/Voodoo
renderer, SDL, wxWidgets or emulator source is compiled.

All correctness tests pass:

- `16 tests, 0 failures` in `vnpu-selftest`;
- `120 iterations, 0 failures` in `vnpu-fuzz`;
- `make test` including local cluster and speculative demos;
- dynamic-loader C example verified;
- Python ctypes example verified;
- `make pcem-check` confirms buildable sources are PCem-free.

## What was built

### 1. Stable C ABI and device model

`include/vnpu.h` and `include/vnpu_backend.h` define a versioned C ABI:

- device/stream/event/buffer/packed-weight handles;
- fixed-size `vnpu_command_t` packets with 17 opcodes;
- virtual MMIO registers, doorbell, IRQ mask/callback;
- status counters for commands, FLOPs, bytes, tiles, culls, DMA;
- `vnpu_get_backend_v1()` function table for `dlopen`/`LoadLibrary`.

### 2. Three real compute-unit schedulers

- **Voodoo** – FIFO command-stream tiles;
- **Wildcat** – interleaved work-stealing tiles with optional cull mask;
- **Cell** – SPU-style workers with explicit DMA into 256 KiB local stores.

They share runtime-dispatched kernels but have different tile geometry,
ordering, culling and memory behavior. `AUTO` and `vnpu_autotune_gemm()`
choose among them empirically.

### 3. SIMD and quantized kernels

- AVX2/FMA 3-row × 32-column F32 GEMM: 12 accumulators + 3 broadcasts + one
  live B vector = exactly the 16 architectural YMM registers, which is what
  hides the 5-cycle FMA latency across two ports;
- AVX2 streaming GEMV for M≤8: K outermost, so B is walked linearly and the
  hardware prefetcher sees one run instead of a cache line per column;
- AVX2 INT8 rank-1 decode row: the C row is staged (4 bytes per column) and B is
  streamed one row at a time, instead of holding a 32-column block of C in
  registers; paired with one contiguous slice of N per unit at M=1 it took
  1×4096×4096 INT8 from 2.47 ms to 1.03 ms (13.6 to 32.5 GFLOP/s) with
  bit-identical results;
- AVX2 INT8 rank-R decode stream (R = 2 and 3): R rows stage R int32 rows and
  share a single pass over B, which took 2×4096×4096 INT8 from 1.58 ms to
  1.07 ms and 3×4096×4096 from 1.98 to 1.70 ms (1.21 ms best), all with
  bit-identical results. It needs a long run per unit (≥1024 columns) because
  each staged row adds accumulator traffic, and f32 keeps the old narrow tile.
  R=3 exists because the packed panel pairs rows, so at M=3 the odd row re-read
  the whole panel — the same panel bytes for three outputs as for four, which
  was the entire 0.65 against 0.51 ms/row gap. R=4 was implemented and measured
  worse (4×4096×4096 went 1.98 -> 2.32 ms: four staged rows thrash L1, and the
  panel's register-held accumulators win), so the wide gate stops at M=3;
- AVX2 column-blocked INT8 GEMM: 2 rows × 32 columns per block, eight int32
  accumulators plus the operand vectors;
- AArch64 NEON 4×4 F32 GEMM;
- scalar correctness path;
- packed INT8 weights with per-row/per-column scales (packing itself is still
  the VNNI-shaped `[n/16][k4][16][4]` layout, which the AVX2 kernel does not
  consume);

There is no AVX-512 or VNNI kernel. Both were removed after the audit showed
the VNNI path was unreachable on Meteor Lake (no AVX-512) and unmeasurable on
Zen 3; the build is AVX2+FMA with a scalar fallback.
- RMSNorm, Softmax, RoPE, SwiGLU, GELU, ADD, MUL, streaming attention,
  quantization and MTP head.

A thread-pool lifetime bug found by AddressSanitizer was fixed with a
heap-allocated parallel state and worker-done barrier.

### 4. Sherlock tracing and autotune

Sherlock now records real nanosecond timestamps instead of the original
hard-coded 0.5 ms. It emits text roofline summaries and Chrome Trace JSON with
per-command bytes, FLOPs, opcode, unit and worker. Autotune sweeps tile sizes
and units with tracing disabled, then restores the previous state.

### 5. Cluster and distributed speculation

- `vnpu-node` binary TCP server with token, persistent connections, resident
  weights and GEMM requests;
- `vnpu_cluster_*` API splitting GEMM rows across local devices/remote nodes;
- `vnpu_speculative_step()` DFlash/MTP-style draft/verify coordinator;
- `vnpu-cli cluster-demo` and `specdemo` start two in-process nodes and
  demonstrate both modes.

v1 uses greedy acceptance; exact stochastic acceptance and TLS/RDMA are on the
roadmap.

### 6. Legacy compatibility

`src/compat/compat.cpp` implements the original `virtual_npu_*`, `vgpu_npu_*`
and Sherlock APIs without PCem. Graphics-only calls are deterministic no-ops;
matmul and unit-1 ALU compute are real. Address truncation and cull semantics
were fixed.

### 7. Documentation, bindings and tooling

- root `README.md`;
- `docs/ARCHITECTURE.md`, `INTEGRATION.md`, `CLUSTER.md`, `PERFORMANCE.md`,
  `TUNING.md`, `COMPAT.md`, `SECURITY.md`, `ROADMAP.md`;
- `examples/dynamic_load.c`, `examples/engine_adapter.cpp`;
- `bindings/python/vnpu.py`;
- `scripts/bench_matrix.sh`, `autotune_all.py`, `trace_report.py`;
- `tests/fuzz_shapes.cpp`;
- `Makefile` and `CMakeLists.txt`.

## Measured gains

Reference VM: 2 logical vCPUs with AVX2+FMA, 2 threads. A speedup is against
the faster of two plain-C references in the same binary: the naive scalar
triple loop, and the same arithmetic in compiler-vectorisable order (no
intrinsics, best of 3 after warm-up).

- 256³ F32: **0.446 ms, 75.2 GFLOP/s, 3.70×**; INT8: **0.358 ms, 93.7 GFLOP/s,
  4.61×**;
- 512³ F32: **2.79 ms, 96.1 GFLOP/s, 4.66×**; INT8: **2.61 ms, 102.8 GFLOP/s,
  4.98×**;
- 1024³ F32: **19.2 ms, 111.9 GFLOP/s, 5.56×**; INT8: **18.2 ms, 117.9 GFLOP/s,
  5.85×**;
- decode 1×4096×4096: F32 **2.96 ms, 1.26×**; INT8 **1.03 ms, 3.76×**;

These are from `benchmarks/gemm.json`, which is regenerated by `make bench`;
they were stale here for three revisions after packing landed. Treat a single
run as a sample, not a result: this VM is shared and the same binary has
produced 19.2 and 26.6 ms for 1024³ F32 within one session. Only differences
measured in the same process, interleaved, mean anything.
- synthetic transformer block prefill: **4,098 tok/s F32**, **4,301 tok/s INT8**;
- decode: **820 tok/s F32**, **883 tok/s INT8**.

Earlier revisions of this file quoted 50–85× against a single-shot baseline
that stored through a `volatile` inside the measured loop; that inflated every
factor by roughly an order of magnitude and the figures have been withdrawn.
The honest summary is 1.3–5.9× against a compiler-vectorised plain-C loop.
Decode F32 was parity before panel packing and is now 1.28×, because packing is
correctly skipped there and other work (scheduling, scratch) got cheaper.
Block sizes are measured-out (~1–7%).

The one structural change that did pay is panel packing: tiles copy their
columns into a `[k][tn]` panel sized to L1 and the microkernel walks it
sequentially instead of striding row-major K×N. Against the same binary with
`VNPU_PACK=0`: F32 neutral at 256³, +5% at 512³, **1.68× at 1024³**
(36.4 → 21.7 ms); INT8 +4%/n, +10% at 512³, **1.33× at 1024³** (25.6 → 19.2 ms).
The win grows with the working set, as the strip-traffic model predicts, and
decode is untouched by construction.

Packing also introduced a data race that the suite did not catch: the panel lived
in scratch indexed by the *task* slot a `parallel_for` hands out, and two streams
on one device run two `parallel_for` calls at once, so two tasks with the same
slot index could overwrite each other. Two concurrent streams on a 512³ GEMM
returned wrong results in 10 of 12 runs. Scratch is now `thread_local`, which
makes the hazard structural rather than ordering-dependent, and
`selftest concurrent_streams_same_device` covers it (verified to fail when the
scratch is made shared again).
Clustering over loopback TCP is still slower than local execution for small
kernels, as network overhead predicts.

## Honest limitations

- This is a CPU software device, not a physical NPU.
- Speedups are vs the faster of a naive scalar loop and a compiler-vectorised
  plain-C loop, not optimised BLAS. Decode F32 is at parity with the compiler.
- No full model loader, tokenizer, sampler or KV-cache manager is included.
- Network v1 is plaintext/native-endian; use a VPN or TLS proxy.
- Speculative v1 uses greedy acceptance.
- GPU support is reserved in the ABI but not implemented.

## Reproduce

```bash
make clean
make -j$(nproc)
make test
make benchmark
make release
```

Deliverables are `lib/libvnpu.so`, `lib/libvnpu.a`, `bin/vnpu-cli`,
`bin/vnpu-node`, `bin/vnpu-selftest`, `bin/vnpu-fuzz`, and
`release/vnpu-standalone-1.0.0.tar.gz`.
