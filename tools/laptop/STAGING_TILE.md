# Cell-style staging tile: built, measured, and it does not pay

The last big memory-side technique on the list was the Cell SPE model: a small
software-managed local store plus a double-buffered DMA. It is now implemented in
`vnpu_laptop.cpp` and measured in two modes. The honest result is that it loses to
the plain panel kernel almost everywhere, and the reason is interesting.

## What was built

`gemm_i8_vnni_stage<MODE, PF>(...)` stages one K-tile of a 64-column panel at a
time through a persistent thread-local, 64-byte-aligned buffer sized to fit L1
(48 KB on this P-core). A tile is a run of K-groups, `tile_g` of them, so the
compute loop reads with unit stride out of a known-resident buffer instead of
walking the whole panel.

- `STAGE_COPY` is the literal DMA: `memcpy` tile t+1 into the second buffer while
  computing tile t.
- `STAGE_PREFETCH` skips the copy and only issues T0/NTA touches one whole tile
  ahead. The plain pan8 kernel already prefetches, but only one cache line deep,
  so this changes the lookahead distance rather than the access pattern.

Correctness: max abs diff 0 against the scalar oracle at tile sizes that do not
divide the K-group count (97 and 61), so the double-buffering and the tail are
right.

## Measured

Clock 2.90 GHz, compute peak ~92.9 GMAC/s, bandwidth ladder L1 22 / L2 21 /
L3 17 / DRAM 17 GB/s. `vs pan8` is the number that matters; absolutes move with
the clock.

### B = 2048 x 2048 (4.2 MB), fits L3

| M | variant | ms | GMAC/s | vs pan8 |
|---|---|---|---|---|
| 1 | pan8 (baseline) | 0.100 | 42.1 | 1.00x |
| 1 | stage pf tile64 | 0.168 | 25.0 | **0.59x** |
| 1 | stage pf tile128 | 0.123 | 34.1 | 0.81x |
| 1 | stage pfNTA tile64 | 0.145 | 28.9 | 0.69x |
| 1 | stage copy tile64 | 0.162 | 25.9 | 0.61x |
| 1 | stage copy tile32 | 0.152 | 27.5 | 0.65x |
| 128 | pan8 (baseline) | 13.355 | 40.2 | 1.00x |
| 128 | stage pf tile64 | 16.723 | 32.1 | 0.80x |
| 128 | stage pf tile128 | 15.731 | 34.1 | 0.85x |
| 128 | stage pfNTA tile64 | 19.713 | 27.2 | 0.68x |
| 128 | stage copy tile64 | 22.353 | 24.0 | 0.60x |
| 128 | stage copy tile32 | 19.833 | 27.1 | 0.67x |

### B = 8192 x 4096 (33.6 MB), exceeds L3

This is the regime the technique is for: a decode pass cannot lean on cache.

| M | variant | ms | GMAC/s | GB/s | vs pan8 |
|---|---|---|---|---|---|
| 1 | pan8 (baseline) | 1.567 | 21.4 | 21.4 | 1.00x |
| 1 | stage pf tile64 | 1.618 | 20.7 | 20.7 | 0.97x |
| 1 | stage pf tile128 | 1.705 | 19.7 | 19.7 | 0.92x |
| 1 | stage pfNTA tile64 | 2.632 | 12.8 | 12.8 | 0.60x |
| 1 | stage copy tile64 | 3.375 | 9.9 | 9.9 | 0.46x |
| 1 | stage copy tile32 | 3.134 | 10.7 | 10.7 | 0.50x |
| 32 | pan8 (baseline) | 73.187 | 14.7 | 0.5 | 1.00x |
| 32 | stage pf tile64 | 70.308 | 15.3 | 0.5 | 1.04x |
| 32 | stage pf tile128 | 74.611 | 14.4 | 0.4 | 0.98x |
| 32 | stage pfNTA tile64 | 118.865 | 9.0 | 0.3 | 0.62x |
| 32 | stage copy tile64 | 127.951 | 8.4 | 0.3 | 0.57x |
| 32 | stage copy tile32 | 96.999 | 11.1 | 0.3 | 0.75x |

## Verdict

No win. The only number above 1.00x is 1.04x (deeper prefetch at M=32 on the
large matrix), which is inside run-to-run noise.

Why, concretely:

- On the small shape the kernel is not memory-bound at all. pan8 sustains
  42 GMAC/s against a 92.9 GMAC/s issue peak while the whole weight matrix is
  L3-resident, so it is limited by instruction issue, and the DMA copy spends
  issue slots and load/store bandwidth without relieving anything. That is the
  0.59x.
- On the large shape pan8 already streams at ~21 GB/s, which is the machine's
  own streaming rate. The hardware prefetcher has the pattern figured out, so
  escalating to software prefetch finds nothing left. `pf tile64` lands at 0.97x,
  i.e. neutral.
- Escalating the prefetch depth to a whole 32 KB tile makes things worse (0.81x,
  0.92x), and NTA is worse still (0.60x). The limit is cache capacity, not
  latency: a tile we are about to re-read must not be marked non-temporal.
- The literal Cell DMA loses hardest because it pays an extra read *and* write of
  every weight byte. The technique was profitable on an SPE only because the SPE
  had no cache; here the copy is redundant work on top of a working hierarchy.

This is the same shape of result as the earlier streaming-store and non-temporal
result: the memory-side tricks that mattered on architectures without a cache
hierarchy do not transfer to a core that has one.

## What this implies for the NPU

- Do not put this in the pool. Eight workers would each hold their own tile
  buffer and duplicate the same traffic into a system that is already saturated
  at the DRAM end.
- The remaining wins are on the issue side, not the memory side: wider
  accumulator blocking, fewer non-VNNI instructions per MAC, and batching the
  prompt into a real GEMM instead of feeding prefill one token at a time.
