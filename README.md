# Virtual NPU 1.0 — standalone CPU accelerator for LLM workloads

**Virtual NPU** is a dependency-free software device that turns ordinary CPU-only
VMs into accelerator-style inference targets. It replaces the old PCem/Voodoo
emulator coupling entirely: there is no SDL, wxWidgets, Voodoo renderer, PCem
header, BIOS, display or graphics driver anywhere in the build.

The runtime can be consumed in four ways:

1. **`libvnpu.so` / `libvnpu.a` / `vnpu.dll`** – stable C ABI with a dynamic
   backend vtable (`vnpu_get_backend_v1`).
2. **`vnpu-cli`** – info, selftest, benchmark, autotune, trace, node server,
   cluster demo, speculative demo and synthetic LLM block benchmark.
3. **`vnpu-node`** – a standalone network appliance that keeps weights resident
   and answers binary GEMM requests.
4. **Legacy compatibility** – the original `virtual_npu_*` and `vgpu_npu_*`
   APIs still compile, but now execute on the new kernels.

It is deliberately called a *virtual* NPU: it is not pretending to be a physical
NPU, and it does not bypass the CPU. It gives an engine a device-like contract —
streams, buffers, command packets, doorbells, interrupts, status registers,
completion events, DMA and telemetry — while mapping the work onto AVX2/FMA,
AArch64 NEON or scalar CPU kernels.

---

## Why this rewrite exists

The original project had two coupled trees:

- `npu/` – a standalone batched/tensor matmul experiment;
- `pcem/src/video/vgpu_npu.c` – a PCem Voodoo renderer turned into a compute
  unit with a 2 GB flat address space and a tiny unit-1 ALU.

That design was hard to use from a real LLM engine: PCem was a hard dependency,
matmul was expressed incorrectly in one of the PCem tests, Sherlock tracing used
a hard-coded `0.0005` second delay, the API was not `dlopen`-friendly, and there
was no async device model, quantization, transformer operators, cluster, or
remote execution.

This repository keeps the old experiments under `legacy/` and the original zip
under `archive/`, but the buildable product is now a clean runtime under
`include/`, `src/`, `tools/` and `bindings/`.

---

## Quick start

```bash
make -j$(nproc)
make test                      # selftest + shape fuzzer + mini cluster/spec demos
./bin/vnpu-cli info
./bin/vnpu-selftest
./bin/vnpu-fuzz 100
./bin/vnpu-cli bench --shape 512x512x512 --dtype both --iterations 5
./bin/vnpu-cli autotune --shape 1024x1024x1024 --dtype i8
./bin/vnpu-cli trace --shape 512x512x512 --output trace.json
cc -O2 -Iinclude examples/dynamic_load.c -ldl -o /tmp/dynamic_load
VNPU_LIBRARY=$PWD/lib/libvnpu.so /tmp/dynamic_load
```

Artifacts:

```text
lib/libvnpu.so        shared library (C ABI)
lib/libvnpu.so.1      SONAME symlink
lib/libvnpu.a         static library
bin/vnpu-cli          all-in-one command line tool
bin/vnpu-node         network node executable
bin/vnpu-selftest     standalone correctness suite
```

CMake also produces a Windows DLL and static `.lib`:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --parallel
```

No BLAS, oneDNN, OpenMP, protobuf, gRPC, CUDA, ROCm, SDL, wxWidgets or PCem is
required. Only a C++17 compiler and POSIX/WinSock threads.

---

## What “proper NPU” means here

The runtime exposes a small virtual device rather than a bag of functions:

| Device feature | Implementation |
|---|---|
| Command processor | fixed-size `vnpu_command_t` packets on ordered streams |
| Compute units | Voodoo FIFO, Wildcat work-stealing, Cell SPU-style DMA |
| Device memory | 64-byte aligned buffers, host/device copy direction flags |
| Streams | one worker thread per stream, FIFO ordering |
| Events | completion objects with timed waits |
| Doorbell / MMIO | `vnpu_mmio_read32/write32`, IRQ mask and callbacks |
| Status registers | commands, FLOPs, bytes, tiles, culls, DMA, errors |
| DMA | explicit copy/memset commands and Cell local-store staging |
| Telemetry | Sherlock text summaries and Chrome Trace JSON |
| Autotuning | empirical tile/unit sweep driven by measured timings |
| Cluster | local multi-device and remote `vnpu-node` workers |
| Future GPU path | `VNPU_UNIT_GPU` and `VNPU_FEATURE_GPU_FUTURE` reserved |

### The three compute-unit styles are real schedulers

They are not cosmetic names over one loop:

- **VOODOO** – classic command-stream pipeline: static FIFO tile order with
  persistent row/column bases. Best for large dense prefill GEMMs.
- **WILDCAT** – programmable work-stealing tile scheduler. It interleaves tile
  order, supports a per-tile cull mask and skips whole zero tiles. Best for
  sparse or masked workloads.
- **CELL** – SPU-style workers. Each worker DMAs A/B panels into its own 256 KiB
  aligned local store before a register-blocked microkernel runs. Best for
  small independent chunks and remote/distributed workers.

`VNPU_UNIT_AUTO` selects a unit from shape, configuration flags and cull mask.
`vnpu-cli autotune` replaces the heuristic with measurements.

---

## Operator surface

The v1 command stream supports:

- `MATMUL_F32` with optional bias, `alpha/beta`, transposed B, fused ReLU/SiLU/GELU;
- `MATMUL_I8` with packed resident INT8 weights and per-row/per-column scales;
- `RMSNORM_F32`, `SOFTMAX_F32`, `ROPE_F32`, `SWIGLU_F32`, `GELU_F32`;
- `ADD_F32`, `MUL_F32`, `ATTENTION_F32` (streaming online softmax);
- `QUANTIZE_I8`, `PACK_WEIGHTS_I8`, `MTP_HEAD_F32`;
- `COPY`, `MEMSET`, `BARRIER`, `NOP`.

Kernels use runtime dispatch:

| CPU | Path |
|---|---|
| AVX2/FMA (Zen 3 baseline) | 3-row × 32-column F32 microkernel (12 accumulators, 16 YMM) and a column-blocked AVX2 INT8 microkernel |
| AVX2/FMA, M≤8 | streaming GEMV: K outermost so B is walked linearly |
| AArch64 NEON | 4×4 F32 microkernel |
| Generic | scalar correctness path |

All ISA-specific functions are compiled with target attributes, so a single
binary also runs on machines without AVX2. `VNPU_FORCE_ISA=scalar|avx2` can
downgrade dispatch for debugging or sanitizer builds; any other value is
ignored with a log line rather than executed.

There is **no AVX-512 or VNNI kernel in this build**. Those paths were removed:
they could not be verified on the target hardware (a 4×8 hybrid laptop and a Zen
3 desktop, neither with AVX-512) and wide-vector code downclocked the laptop.
INT8 runs on AVX2, where the only exact vectorisation axis is the column axis.

---

## Integrating an LLM engine

A C/C++ engine either links `-lvnpu` or dynamically resolves the vtable:

```c
#include "vnpu_backend.h"

void *lib = dlopen("libvnpu.so", RTLD_NOW);
const vnpu_backend_v1_t *(*get)(void) =
    (const vnpu_backend_v1_t *(*)(void))dlsym(lib, "vnpu_get_backend_v1");
const vnpu_backend_v1_t *npu = get();

vnpu_config_t cfg;
npu->config_init(&cfg);          /* or use vnpu_config_init from vnpu.h */
cfg.num_threads = 0;             /* auto */
vnpu_device_t *dev;
npu->device_create(&cfg, &dev);

vnpu_stream_t *stream;
npu->stream_create(dev, 0, &stream);

vnpu_matmul_f32_args_t mm = {
    .A = activations, .B = weights, .bias = bias, .C = output,
    .M = tokens, .K = hidden, .N = out_dim,
    .alpha = 1.0f, .beta = 0.0f,
    .epilogue = VNPU_EPILOGUE_NONE,
    .unit = VNPU_UNIT_AUTO
};
npu->matmul_f32(stream, &mm);
npu->stream_sync(stream);
```

`vnpu_backend_v1_t` contains every public function as a pointer, so an engine
can ship one binary and discover the NPU at runtime. See `docs/INTEGRATION.md`
for GGML/llama.cpp-style op mapping, Python bindings and out-of-process usage.

---

## Sherlock tracing and iterative tuning

Sherlock is now real instrumentation:

- monotonic nanosecond timestamps, no hard-coded delays;
- per-command duration, bytes, FLOPs, opcode, unit and worker id;
- dispatch / memory / compute / sync buckets;
- per-opcode call counts and hot-spot list;
- roofline-style verdict;
- Chrome Trace JSON output for `chrome://tracing` or Perfetto UI;
- ~25–45 ns/event overhead on the test machine.

```bash
VNPU_TRACE=1 ./bin/vnpu-cli llm-bench --hidden 512 --intermediate 1376 \
  --heads 8 --layers 2 --prefill 32 --decode 1 --dtype both \
  --trace llm-trace.json
./bin/vnpu-cli autotune --shape 2048x4096x11008 --dtype i8 --iterations 3
```

The autotuner sweeps tile geometries and Voodoo/Wildcat/Cell schedulers,
disables tracing during measurement, restores the previous state, and writes
the winning configuration back to the device.

---

## Clustering and distributed speculative decoding

`vnpu_cluster_*` treats local devices and remote `vnpu-node` processes as
weighted workers. Weight matrices are uploaded once with a stable key; each
inference call sends only activations and receives only results.

```bash
# machine A (draft/MTP model or head)
./bin/vnpu-node --port 9449 --token secret --name draft

# machine B (main verifier model)
./bin/vnpu-node --port 9450 --token secret --name verify
```

The client can then:

- split one GEMM by rows across both nodes;
- run a DFlash/MTP-style draft head on machine A;
- run the main verifier on machine B;
- verify a batch of `gamma` draft tokens in one network round trip.

`vnpu-cli specdemo` starts two nodes in-process and demonstrates the pipeline.
`vnpu_speculative_step()` accepts arbitrary logits callbacks, so it can wrap a
real draft model, an MTP head, or a remote inference service. v1 uses greedy
acceptance; temperature affects draft/bonus sampling. Exact stochastic
acceptance needs draft probabilities in a future callback version.

Network v1 is a compact binary TCP protocol with a token. It is intentionally
**not encrypted**; run it over WireGuard, Tailscale, an SSH tunnel or an mTLS
sidecar. See `docs/CLUSTER.md`.

---

## Measured performance

Sandbox: 2 physical cores (1 thread/core), AVX2+FMA, 2 units, GCC, `-O3`,
`make benchmark`. The default unit count is **physical cores**, not logical
CPUs: SMT siblings share the L1/L2 these kernels depend on, so a 14c/18t hybrid
laptop gets 14 units. Override with `cfg.num_threads` or `--threads`.

A speedup here is against the **faster of two plain-C references** measured in
the same binary: the naive scalar triple loop, and the same arithmetic written
in the loop order the compiler can auto-vectorise (no intrinsics, no blocking,
best of 3 after warm-up). Earlier revisions quoted factors up to 85× against a
single-shot baseline with a volatile store inside the loop; those numbers were
measuring the reference, not the kernels, and are gone. These are **not** a
claim against optimised BLAS.

The VM is noisy — the same binary has measured 80–101 GFLOP/s at 512³ F32 — so
`benchmarks/*.json` is the source of truth for a given run. Best unit per row:

| Shape | dtype | Time | Throughput | vs plain C |
|---|---:|---:|---:|---:|
| 128×128×128 | F32 | 0.096 ms | 43.8 GFLOP/s | 2.15× |
| 128×128×128 | INT8 | 0.075 ms | 55.7 GFLOP/s | 2.74× |
| 256×256×256 | F32 | 0.446 ms | 75.2 GFLOP/s | 3.70× |
| 256×256×256 | INT8 | 0.358 ms | 93.7 GFLOP/s | 4.61× |
| 512×512×512 | F32 | 2.79 ms | 96.1 GFLOP/s | 4.66× |
| 512×512×512 | INT8 | 2.61 ms | 102.8 GFLOP/s | 4.98× |
| 1024×1024×1024 | F32 | 19.2 ms | 111.9 GFLOP/s | 5.56× |
| 1024×1024×1024 | INT8 | 18.2 ms | 117.9 GFLOP/s | 5.85× |
| 1×4096×4096 (decode) | F32 | 2.96 ms | 11.3 GFLOP/s | 1.26× |
| 1×4096×4096 (decode) | INT8 | 1.03 ms | 32.5 GFLOP/s | 3.76× |

Read this honestly. Prefill-shaped work is 1.6–6× a compiler-vectorised loop.
**Decode F32 is a parity win at best**: 1.26× against a plain `-O3` loop,
because a single row against a 67 MB matrix is pure memory bandwidth and both
sides read the same bytes. INT8 decode is the strongest ratio in the table: it
halves those bytes *and* streams them, and at 16.3 GB/s it runs at 82% of its
2 flop/byte roofline. It was 1.76× until the decode tile width and the int8
row kernel were fixed; `docs/PERFORMANCE.md` has the measurements.

Both dtypes are **panel-packed**: each tile copies its columns into a `[kc][tn]`
panel sized to L1 and the microkernel walks that contiguous panel instead of
striding a row-major K×N matrix. Measured on the same binary with
`VNPU_PACK=0` as the control (medians, interleaved rounds):

| Shape | F32 | INT8 |
|---|---:|---:|
| 256³ | 0.433 → 0.399 ms (+8%) | 0.431 → 0.415 ms (+4%, within noise) |
| 512³ | 3.275 → 3.104 ms (+5%) | 3.221 → 2.909 ms (+10%) |
| 1024³ | 36.4 → 21.7 ms (**+68%**) | 25.6 → 19.2 ms (**+33%**) |

The win grows with the working set, as the strip-traffic model predicts. F32
uses K blocking (`kc=128`); INT8 panels are 4× smaller, so the whole K fits in
the worker's local store and the int32 accumulators never leave registers --
which makes packed and unpacked INT8 **bit-identical**, and the conformance run
asserts exactly that. Decode (M=1) is untouched by design: a single row reads
each panel entry once, so packing it would be a pure copy.

### Conformance and reference mode

"Same answer everywhere" is tested, not asserted: `make conformance` runs every
unit, thread count, B structure (packed and unpacked) and transposition over
five shapes against an **independent plain-C reference written in double
precision**, and checks two invariance properties a caller can rely on:

- **unit-count invariance** — the same GEMM produces bit-identical bytes on 1
  unit and on 2, because each output element accumulates over the whole K
  inside one tile;
- **INT8 exactness** — integer accumulation makes INT8 results bit-identical
  across every unit and thread count.

Current run: 10 checks, 0 failures. F32 worst relative error against the
double-precision reference is **9.2e-07** (tolerance 1e-4). Packing B changes
the K-block summation order, so packed and unpacked results agree to that
tolerance but are *not* bit-identical — the invariance checks are made within
one structure, and the docs say so rather than pretending otherwise.

`VNPU_MODE=reference` forces the simplest path the library can produce (scalar
kernels, unpacked B, no pool lease). It is the equivalent of running a graphics
app on a software rasterizer: if a bug still reproduces here, it is the
caller's, not a fast-kernel or dispatch bug. The conformance run checks that
this mode agrees with the plain-C reference too (rel err 2–4e-07).

### Synthetic transformer block

`llm-bench --hidden 512 --intermediate 1376 --heads 8 --layers 2`:

| Phase | F32 | INT8 |
|---|---:|---:|
| prefill 32 tokens | 4,098 tok/s, 52.4 GFLOP/s | 4,301 tok/s, 55.0 GFLOP/s |
| decode 1 token | 758 tok/s, 9.6 GFLOP/s | 722 tok/s, 9.1 GFLOP/s |

This is a realistic op mix (RMSNorm, RoPE, attention, QKV/O/gate-up/down GEMMs,
SwiGLU), not a tokenizer or sampling engine.

### Cluster / speculative demo on one 2-vCPU box

- Two loopback nodes, 256³ GEMM: correct result, 1.35× effective worker
  parallelism, but 0.86× wall-clock vs one local device because ~1.8 MB crossed
  TCP for a ~5 ms kernel. On separate machines and larger work this inverts.
- Distributed draft/verify demo: 20 tokens in 13.8 ms, 16/20 draft tokens
  accepted, theoretical 1.08–1.47× for the toy model. Real gains require a
  cheap draft head and one batched verifier round trip.

Full JSON/Chrome outputs are in `benchmarks/`.

## Repository layout

```text
include/                  public C ABI, backend vtable, legacy shims
src/core/                 runtime, scheduler, autotune, backend table
src/kernels/              SIMD GEMM + transformer kernels
src/cluster/              sockets, node server, cluster, speculation
src/compat/               PCem-free virtual_npu/vgpu_npu implementation
tools/                    CLI, node, selftest, demos, LLM benchmark, bench_matrix
                          (tools/laptop/ is a separate Meteor Lake prototype and is
                          not built by make or CMake)
bindings/python/          ctypes wrapper
examples/                 dynamic loader and engine integration examples
tests/                    fuzz/property tests
docs/                     architecture, integration, cluster, tuning, compat
scripts/                  benchmark matrix and trace report helpers
legacy/                   original experiments (not built)
archive/                  original npu_experiments.zip (not built)
benchmarks/               generated measurement artifacts
release/                  release tarball
```

`make pcem-check` verifies that no buildable runtime source includes PCem,
Voodoo renderer, SDL or wxWidgets headers.

---

## Compatibility and migration

Old tests can include `virtual_npu.h` or the original
`includes/private/video/vgpu_npu.h` path; these are shims over
`vnpu_compat.h`. Graphics-only calls (`draw_triangle`, `swapbuffers`, texture
upload/bind) are deterministic no-ops with status accounting. Matmul and unit-1
compute are real.

New code should use `vnpu.h` directly. See `docs/COMPAT.md`.

---

## Roadmap

- GPU backend behind `VNPU_UNIT_GPU` with the same stream/event/buffer ABI.
- INT4 / grouped quantization and FP8/BF16 activation paths.
- Real GGUF/GGML backend plugin and ONNX Runtime execution-provider shim.
- RDMA/libfabric transport and weight sharding instead of row splitting.
- Exact stochastic speculative acceptance with draft probabilities.
- NUMA-aware worker placement and cgroup-aware thread detection.
- Windows DLL and ARM NEON CI matrix.

See `docs/ROADMAP.md` for the full plan.

---

## License

MIT. See `LICENSE`.

**Honest note:** a virtual NPU cannot create compute that the CPU does not have.
The gains come from removing the emulator, keeping data resident, using SIMD,
batching commands, culling empty work, and overlapping local/remote workers.
Measure against your own model and a baseline you trust — the benchmarking and
tracing tools are included specifically for that, and `vnpu-cli bench` now
reports the plain-C reference times next to each result.
