# Clustering and distributed speculative decoding

## 1. Topology

```text
                 +----------------------+
                 |  LLM engine / client |
                 +----------+-----------+
                            |
              vnpu_cluster_matmul_f32()
                            |
        +-------------------+-------------------+
        |                                       |
   local device worker                    remote node worker
   (same process)                         (vnpu-node over TCP)
        |                                       |
   AVX2+FMA / NEON / scalar                 AVX2+FMA / NEON / scalar
```

A cluster contains weighted workers. `vnpu_cluster_add_device()` adds local
devices; `vnpu_cluster_add_remote()` adds remote nodes. `num_streams` controls
how many streams/connections each device or node gets.

## 2. Running nodes

```bash
# machine A
./bin/vnpu-node --port 9449 --bind 0.0.0.0 --token secret --name draft

# machine B
./bin/vnpu-node --port 9449 --bind 0.0.0.0 --token secret --name verify
```

`vnpu-cli serve --port 9449 --token secret` is equivalent.

### Client setup

```c
vnpu_cluster_config_t cc;
vnpu_cluster_config_init(&cc);
cc.num_streams = 2;
cc.auth_token = "secret";
cc.request_timeout_ms = 30000;

vnpu_cluster_t *cluster;
vnpu_cluster_create(&cc, &cluster);

vnpu_remote_node_config_t r = {0};
r.host = "10.0.0.11";
r.port = 9449;
r.auth_token = "secret";
r.weight = 1;
vnpu_cluster_add_remote(cluster, &r, NULL);

vnpu_weight_handle_t *W;
vnpu_cluster_cache_weights_f32(cluster, "layer0.qkv", B, K, N, 0, &W);

vnpu_cluster_matmul_f32(cluster, W, A, C, M, K, N, VNPU_UNIT_AUTO);
```

Weights are sent once with `OP_STORE_WEIGHTS` and stored on each node by a
64-bit FNV-1a key. Each inference sends `OP_MATMUL_F32` with only the A tile
and receives only the C tile.

## 3. Wire protocol (v1)

Every message is a 32-byte native-endian header followed by a payload:

```text
magic, version, op, request_id, payload_size, status, flags, reserved
```

Ops:

| Op | Meaning |
|---|---|
| `OP_HELLO` | token + client name, must be first message |
| `OP_CAPS` | backend, threads, features |
| `OP_PING` | liveness |
| `OP_STORE_WEIGHTS` | key + canonical KxN float matrix |
| `OP_MATMUL_F32` | weight id, M/K/N, unit, A floats -> C floats |
| `OP_GET_STATS` | device counters |
| `OP_SHUTDOWN` | graceful node stop |

Payloads are capped at 1 GiB. The token is compared with a constant-time-ish
plain comparison in v1; treat it as a shared secret, not strong authentication.
There is **no TLS** in v1.

## 4. Security

- Use WireGuard, Tailscale, an SSH tunnel, or an mTLS sidecar for any network
  that is not a trusted loopback.
- Use a long random token and rotate it.
- Bind to a private interface, not `0.0.0.0`, when possible.
- Do not expose `vnpu-node` to the public internet.
- Treat `payload_size` and all shapes as untrusted; v1 validates them but a
  production deployment should add rate limits and quotas.
- The protocol is native-endian; run the same architecture family on both ends
  or add explicit little-endian serialization in v2.

See `docs/SECURITY.md`.

## 5. Row splitting

`vnpu_cluster_matmul_f32()` splits M rows proportionally to worker weight.
Each worker computes a contiguous C block. This keeps writes disjoint and avoids
reductions. It is not optimal when K or N is the dominant dimension; future
versions can split K with reductions or N with concatenated outputs.

Weights are duplicated on each node. That is intentional for LLM decode: memory
is cheaper than sending weights per token. For very large models, use pipeline
parallelism (different layers on different nodes) instead.

## 6. DFlash / MTP-style speculative decoding

`vnpu_speculative_step()` implements the orchestration:

1. Call the draft callback `gamma` times to produce draft tokens.
2. Call the verifier once with the context plus all draft tokens to get
   `gamma + 1` logits rows.
3. Accept the longest greedy-matching prefix.
4. Sample/argmax one bonus token from the verifier.
5. Report accepted tokens, latency and theoretical speedup.

Callbacks have this shape:

```c
vnpu_status_t logits_fn(void *user,
                        const int32_t *tokens, uint32_t num_tokens,
                        uint32_t first_position, uint32_t num_positions,
                        float *logits, uint32_t vocab);
```

`logits` is row-major `[num_positions, vocab]`. A callback can:

- run a small draft model locally;
- call `vnpu_mtp_head_f32()` on local hidden states;
- call `vnpu_cluster_matmul_f32()` on a remote draft node;
- wrap an existing HTTP/gRPC inference endpoint.

`vnpu-cli specdemo` starts two loopback nodes: one for the draft head and one
for the verifier. Both use the same toy weight matrix, so acceptance is high.
Replace `remote_logits()` with a real model call to use this in production.

### v1 limitation

Acceptance is greedy/argmax. `temperature > 0` affects draft and bonus sampling,
but exact speculative sampling needs `min(1, p/q)` with draft-model
probabilities. A v2 callback can return both logits and probabilities without
changing the device ABI.

## 7. Performance guidance

- Keep weights resident; never call `STORE_WEIGHTS` per token.
- Use persistent connections (`num_streams > 1` for concurrency).
- Batch draft tokens into one verifier call.
- Use INT8/BF16 activations on the wire when network bandwidth dominates.
- Use larger M (prefill or batched decode) before splitting across machines.
- On one small VM, loopback clustering often loses to a local kernel because
  TCP overhead is larger than the GEMM. Measure with
  `vnpu_cluster_get_stats()` and Sherlock.

Example from a 2-vCPU sandbox with two loopback nodes and a 256³ GEMM:
effective worker parallelism 1.35×, wall-clock 0.86× local because ~1.8 MB
crossed TCP for a ~5 ms kernel. On separate machines with a 100+ ms model step,
the same pattern is beneficial.

The in-process speculative demo produced 20 tokens in 13.8 ms with 16/20 draft
tokens accepted and a theoretical 1.08–1.47× speedup. Loopback TCP still
dominates the toy model; a real cheap draft head on one machine and a batched
verifier on another is the intended deployment.
