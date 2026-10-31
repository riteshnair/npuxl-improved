# Iterative tuning with Sherlock

## 1. Enable tracing

Compile-time flag or runtime API:

```bash
VNPU_TRACE=1 VNPU_TRACE_FILE=trace.json ./bin/vnpu-cli bench ...
```

```c
vnpu_trace_enable(device, 1);
/* ... submit commands ... */
vnpu_trace_dump(device, "sherlock.txt");
vnpu_trace_write_chrome(device, "trace.json");
vnpu_trace_get_summary(device, &summary);
```

`VNPU_CONFIG_TRACE` enables text aggregation; `VNPU_CONFIG_TRACE_CHROME` makes
the device automatically write Chrome JSON to `config.trace_file` at shutdown.

## 2. Read the text report

Important fields:

- `compute / memory / dispatch / sync` buckets;
- `achieved GFLOP/s` and `achieved GB/s`;
- `arithmetic intensity` = FLOPs / bytes;
- `hot opcode`;
- `roofline verdict`;
- per-opcode call count and us/call;
- slowest 12 individual events;
- trace overhead.

Example verdicts:

| Verdict | Meaning | Next move |
|---|---|---|
| dispatch-bound | too many tiny commands | batch ops, fuse bias/epilogue, enlarge tiles |
| memory-bound | low arithmetic intensity | quantize weights, cache packed weights, larger K tile |
| compute-bound | high arithmetic intensity | more units, larger M/N blocks, INT8 |
| balanced | no single dominant bucket | try autotune |

## 3. Chrome trace

Open `trace.json` in `chrome://tracing` or <https://ui.perfetto.dev/>.
Each event has:

- `name` – opcode + command id;
- `cat` – compute/memory/dispatch/sync;
- `ts`, `dur` – microseconds;
- `tid` – worker 0..7;
- `args` – opcode, unit, bytes, FLOPs.

Look for:

- long gaps between commands (engine-side dispatch);
- one worker doing all the work (bad tile granularity);
- repeated tiny GEMMs (fusion opportunity);
- quantize/copy commands dominating decode.

## 4. Autotune

```bash
./bin/vnpu-cli autotune --shape 2048x4096x11008 --dtype i8 \
  --iterations 3 --json tune.json
```

The autotuner sweeps:

- Voodoo: 32/64/128 M tiles, 64/128 N tiles, 128/256 K tiles;
- Wildcat: 16/32 M tiles, 32/64 N tiles, 64/128 K tiles;
- Cell: 8/16 M tiles, 32/64 N tiles, 32/64/128 K tiles.

It disables tracing, runs warmups, measures the best time, restores the previous
tiles and writes the winning config back to the device. Use different shapes for
prefill and decode; a single tile config cannot be optimal for both.

## 5. Environment variables

| Variable | Effect |
|---|---|
| `VNPU_THREADS` | override worker count |
| `VNPU_TRACE` | enable tracing when non-zero |
| `VNPU_TRACE_FILE` | default Chrome trace path |
| `cfg.num_threads` / `--threads` | unit count; default is physical cores, not logical (SMT siblings share L1/L2) |
| `VNPU_PACK` | `0` disables F32 B panel packing (A/B control); on by default |
| `VNPU_MODE` | `reference` forces scalar kernels, unpacked B and no pool lease |
| `VNPU_LEASE` | `1` keeps the thread pool hot across a batch; off by default (measured negative, see PERFORMANCE) |
| `VNPU_FORCE_ISA` | `scalar` or `avx2` for debug/sanitizers (anything else is ignored) |
| `VNPU_DEBUG` | verbose runtime diagnostics |

## 6. Bottleneck playbook

### Large prefill GEMM

- Use `VNPU_UNIT_VOODOO` or `WILDCAT`.
- Autotune with the exact hidden/intermediate dimensions.
- Pack weights once with `vnpu_pack_weights_i8()`.
- Batch multiple tokens; avoid `M=1` on a 64-thread VM.

### Decode / `M=1`

- Use `trans_b=1` weight layout and `VNPU_UNIT_CELL` or `AUTO`.
- Fuse RMSNorm + QKV where possible.
- Keep KV cache and weights in device buffers.
- Batch requests to increase M.
- Avoid per-token quantize/copy commands by keeping activations quantized.

### Sparse / MoE

- Provide `tile_mask` and use `WILDCAT`.
- Set `VNPU_CMD_FLAG_WILDCAT_CULL`.
- Watch `tiles_culled` in `vnpu_device_status_t`.

### Network cluster

- Increase `num_streams` until bandwidth saturates.
- Use large M splits; avoid many tiny remote calls.
- Cache weights once.
- Compare `local_time_ns` vs `remote_time_ns` and `network_bytes_*`.
- If `speedup_estimate` is high but wall time is worse, network RTT dominates.

### Speculative decoding

- Increase `gamma` until verifier cost dominates draft cost.
- Put the draft model/MTP head on a nearby or smaller node.
- Batch all draft tokens into one verifier call.
- Track `accepted / produced` and `theoretical_speedup`.
- For sampling, add draft probabilities for exact acceptance in v2.

## 7. Regression workflow

1. `./bin/vnpu-selftest`
2. `./bin/vnpu-cli bench --shape <shape> --dtype both --iterations 5`
3. `./bin/vnpu-cli trace --shape <shape> --output trace.json`
4. Inspect Sherlock text and Chrome trace.
5. Change one kernel/tile/scheduler at a time.
6. Re-run steps 1–3 and compare `benchmarks/*.json`.

## 8. Sanitizer note

For sanitizer builds, force the scalar path — target-attribute SIMD and stack
stores are known to confuse some GCC versions under ASan:

```bash
VNPU_FORCE_ISA=scalar ASAN_OPTIONS=detect_leaks=0 ./your-asan-binary
```

`VNPU_FORCE_ISA` accepts `scalar` and `avx2` only; any other value is ignored
with a log line instead of executing instructions the CPU may not implement.
The release build is unaffected.
