# Architecture

## 1. Design goals

The runtime is a **software device**. An LLM engine should be able to treat it
like a small accelerator: allocate buffers, push commands into a stream, wait
on events, read status registers and collect telemetry. The implementation is
free to use CPU threads, SIMD, local stores, culling and network workers.

Non-goals for v1: pretending to be a physical NPU, emulating a PCI device,
implementing a full model/tokenizer, or depending on a vendor SDK.

## 2. Layer diagram

```text
  LLM engine / Python / CLI / legacy test
        | C ABI (vnpu.h) or backend vtable
        v
  Command processor (vnpu_stream_submit)
        | fixed POD command packets
        v
  Scheduler (choose_unit / build_tiles_generic)
        +----------------+----------------+----------------+
        | Voodoo FIFO    | Wildcat cull   | Cell DMA       |
        +----------------+----------------+----------------+
        v
  Kernel dispatcher (runtime CPUID)
        +---------+---------+---------+
        | scalar  | AVX2    | NEON    |
        +---------+---------+---------+
        v
  Persistent thread pool + per-worker local store
        |
        +--> Sherlock trace / status registers / events
        |
        +--> cluster split ------------------> vnpu-node TCP workers
```

## 3. Device object

`vnpu_device_t` owns:

- a `CpuFeatures` snapshot (ISA, SIMD width, cache sizes, memory);
- a persistent `ThreadPool` for compute;
- a vector of 256 KiB local stores, one per worker;
- a `TraceState` with per-opcode aggregates and Chrome trace events;
- 16 virtual 32-bit MMIO registers;
- atomic status counters: commands, kernel/dispatch/memory/sync time, bytes,
  FLOPs, tiles, culls, DMA, interrupts, errors.

`vnpu_mmio_read32()` and `vnpu_mmio_write32()` expose ID, version, control,
status, error, command head/tail, doorbell, IRQ status/mask, active CUs,
feature bits. Writing the doorbell raises IRQ 0; completing a command batch
raises IRQ 1 and invokes the optional interrupt callback.

## 4. Streams and events

Each `vnpu_stream_t` owns one worker thread and a FIFO of `CommandBatch`
objects. Submission is non-blocking; `vnpu_stream_sync()` waits until the queue
is empty and all pending commands have retired. `vnpu_event_t` is a timed
completion object. Commands inside one stream are ordered; different streams
execute concurrently on the shared compute pool.

Convenience functions such as `vnpu_matmul_f32()` submit one command and then
sync. High-performance engines should batch commands and sync once per layer or
per token step.

## 5. Command packet

`vnpu_command_t` is a fixed-size POD structure:

```text
uint32 opcode, flags
uint64 id
union {
  matmul_f32, matmul_i8, rmsnorm, softmax, rope, binary, unary,
  swiglu, attention, quantize_i8, pack_weights_i8, mtp_head,
  copy, memset, raw[192]
}
```

The union is intentionally flat. No command owns memory, so engines can build
batches on the stack or in pinned host memory. Future operators are added as
new opcodes and new union fields while `VNPU_ABI_VERSION` stays 1.

## 6. Compute units

### Voodoo

- static tile order: M outer, N inner;
- persistent row/col bases inside the microkernel;
- tile geometry from the device or autotuner;
- best for large dense prefill GEMMs.

### Wildcat

- interleaved N-major tile order for work stealing;
- optional `tile_mask` culls whole zero tiles;
- smaller M tile (default ≤32) and K window (≤128);
- good for sparse/conditional execution and MoE-style masks.

### Cell

- each worker gets an aligned 256 KiB local store;
- A/B panels are explicitly copied (DMA) into that store in K windows;
- register-blocked accumulation stays inside the local store;
- ideal for remote workers, tiny independent tasks and cache-sensitive shapes.

`VNPU_UNIT_AUTO` chooses Cell for low-latency/decode shapes, Wildcat when a
cull mask is present, otherwise Voodoo.

## 7. GEMM kernels

### F32 NN

`A[M,K] * B[K,N] -> C[M,N]`.

- AVX2: 3 rows × 32 columns in registers — twelve accumulators, three
  broadcasts and one live B vector, which is exactly the 16 architectural YMM
  registers. Eight accumulators (the old 4×8 block) cannot hide a 5-cycle FMA
  latency across two ports.
- AVX2, packed B: each tile copies its columns into a `[kc][tn]` panel
  (`kc=128, tn=64` → 32 KiB, L1-resident) and the microkernel walks the panel
  sequentially instead of striding row-major K×N. Tiles go full-height
  (`tm = M`) when N alone gives the units enough tiles, so B is packed once.
  `VNPU_PACK=0` selects the unpacked path for A/B measurement.
- AVX2, M≤8 (decode): one row at a time with K outermost, so B is walked
  linearly and the prefetcher sees one run rather than a cache line per column.
  Rows left over from the 2-row block use the same path.
- AVX2, M=2 and M=3 int8 over a wide slice: the rows of the tile are staged as
  `int32` (4 bytes per column each) and B is swept once for all of them, so
  every B byte feeds R FMAs. R=4 was measured and is worse than the packed
  panel (four staged rows thrash L1), so M≥4 keeps the panel.
- NEON: 4 rows × 4 columns.
- Scalar: correctness oracle and fallback.
- Transposed A falls back to scalar tiles.

There is no AVX-512 path in this build; `ISA_AVX512` remains in the enum only
because the public ABI and the NEON/LLM operator code share it, and runtime
detection never selects it.

### F32 NT (transposed B)

For decode, weights are often `[N,K]`. A vectorized dot-product kernel streams
both A and B rows contiguously and horizontally reduces one accumulator.

### Kernel scratch ownership

Kernel scratch (the packed panel, the CELL staging buffers) is `thread_local`,
not a per-worker-device buffer. A `parallel_for` worker id is a *task* slot, not
a thread: two streams on one device run two `parallel_for` calls concurrently,
so two tasks carrying the same slot index can land on different threads. Measured
with two concurrent streams on a 512³ F32 GEMM before the fix: 10 of 12 runs
returned wrong results. `selftest concurrent_streams_same_device` covers it, and
fails if the scratch stops being per-thread.

### INT8 (AVX2)

- symmetric per-row activation scales, per-column weight scales;
- 2 rows × 32 columns per block: eight int32 accumulators plus the operand
  vectors fit the 16 YMM registers;
- without VNNI there is no int8 dot-product instruction, so the only exact and
  vectorisable axis on row-major B is the column axis: broadcast one A value
  and multiply eight contiguous B bytes (`cvtepi8_epi32` + `mullo_epi32`);
- rows left over from the 2-row block, including the whole tile when decode
  makes `tm==1`, take the same single-row path rather than the scalar kernel;
- B is **panel-packed** per tile into `[k][tn]` for the whole K at once (64 KiB
  at `tn=64, K=1024`), which keeps the int32 accumulators in registers and makes
  the packed result bit-identical to the row-major one. Tiles with `tm < 3` skip
  packing: a single row reads every panel entry once, so copying it is pure
  overhead. That is the decode shape, which stays row-major;
- weights can still be packed as `[N/16][K/4][16][4]`, but that layout is
  VNNI-shaped: the AVX2 kernel consumes the row-major copy, and the tile-time
  `[k][tn]` panel above is what the kernel actually reads now.

## 8. Transformer operators

- RMSNorm: vectorized sum of squares, one reciprocal sqrt, vectorized scale.
- Softmax: vectorized max, fast `exp2` polynomial, normalize.
- RoPE: half-rotation or interleaved GPT-J rotation, optional positions.
- SwiGLU/GELU: fast sigmoid/tanh approximations with vectorized exp.
- Attention: streaming online softmax over K blocks with causal/additive mask;
  vectorized dot products and V accumulation.

## 9. Tracing

`ScopedTrace` wraps each command. It records start, duration, opcode, unit,
worker, bytes read/written and FLOPs. `TraceState` aggregates the four buckets
(compute/memory/dispatch/sync), per-opcode totals and the slowest events.
`vnpu_trace_dump()` prints a roofline-style report and
`vnpu_trace_write_chrome()` emits Chrome Trace JSON.

Tracing adds roughly 25–45 ns/event on the reference VM. Disable it for final
timing; autotune does this automatically.

## 10. Cluster

`vnpu_cluster_t` holds weighted local workers and remote workers. Local workers
own a stream on an existing device; remote workers own a persistent TCP
connection to a `vnpu-node`. Weights are cached by a stable FNV-1a key; matmul
splits M rows by worker weight and merges results into the caller’s C buffer.

Speculative decoding uses generic logits callbacks, so draft and verifier can
be local, remote, or an MTP head. The cluster object only records job stats.

## 11. GPU extension point

The public ABI already has `VNPU_UNIT_GPU` and
`VNPU_FEATURE_GPU_FUTURE`. A future backend can:

- keep the same `vnpu_command_t`;
- map `vnpu_buffer_t` to device allocations;
- implement `MATMUL_F32/I8` with cuBLAS/MIOpen/oneDNN;
- keep Sherlock events, cluster splitting and speculative orchestration
  unchanged.

No ABI break is required.
