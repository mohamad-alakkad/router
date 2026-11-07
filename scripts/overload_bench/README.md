# Overload and recovery harness for per-worker admission (RFC #435)

**Everything here is a local mock.** Each "engine" is an aiohttp server with a
fixed number of concurrent batch slots (`--slots`) and an unbounded FIFO wait
queue, the shape of vLLM's running and waiting queues. A request holds a slot
for `--service-ms`, streaming 10 SSE chunks. These numbers show *where queueing
happens and how fast the system recovers* under each router configuration.
They are not GPU, vLLM, or production measurements.

## Run

```bash
cargo build --release --bin vllm-router
python -m venv .venv && .venv/bin/pip install aiohttp==3.10.10
.venv/bin/python scripts/overload_bench/overload_bench.py \
  --router-bin target/release/vllm-router --caps none,8,12 > result.json
python3 scripts/overload_bench/table.py result.json
```

Scenarios used for the published results:

| File | Flags |
|---|---|
| `results/result_overload.json` | defaults (2 engines x 8 slots, 500 ms service, 64 rps for 20 s = 2x capacity, then 8 rps for 10 s) |
| `results/result_failure.json` | `--caps none,8 --kill-at 5` (second engine stops listening 5 s into the overload) |
| `results/result_undercap.json` | `--caps none,8 --overload-rps 24 --overload-secs 15 --recovery-secs 3` (75% of capacity) |

The router runs with `--policy round_robin` and default retry and circuit
breaker settings. "within SLO" counts 200 responses whose first token arrived
within 1 s.

Environment: 4 vCPU Linux cloud container (kernel 6.18), router, engines and
load generator on the same host. The benchmarked binary was built from
`feat/435-overload-admission` at functional state `e3bff1e`; later commits on
that branch change only comments, docs and tests.

## Results

engines=2 slots=8 service=500ms overload=64rps x 20s recovery=8rps x 10s kill_at=None slo_ttft=1.0s
| config | phase | offered | 200 | 429 | other | TTFT p50 s | TTFT p99 s | TTFT max s | 200 within SLO | 429 latency p99 s | peak engine waiting | peak engine running |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap=none | overload | 1280 | 1280 | 0 | 0 | 10.295 | 20.411 | 20.431 | 64 | - | 322 | 8 |
| cap=none | recovery | 80 | 80 | 0 | 0 | 16.938 | 20.691 | 20.691 | 0 | - |  |  |
| cap=8 | overload | 1280 | 624 | 656 | 0 | 0.001 | 0.003 | 0.006 | 624 | 0.001 | 1 | 8 |
| cap=8 | recovery | 80 | 79 | 1 | 0 | 0.001 | 0.004 | 0.004 | 79 | 0.001 |  |  |
| cap=12 | overload | 1280 | 644 | 636 | 0 | 0.126 | 0.383 | 0.384 | 644 | 0.002 | 4 | 8 |
| cap=12 | recovery | 80 | 80 | 0 | 0 | 0.001 | 0.380 | 0.380 | 80 | - |  |  |

engines=2 slots=8 service=500ms overload=64rps x 20s recovery=8rps x 10s kill_at=5.0 slo_ttft=1.0s
| config | phase | offered | 200 | 429 | other | TTFT p50 s | TTFT p99 s | TTFT max s | 200 within SLO | 429 latency p99 s | peak engine waiting | peak engine running |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap=none | overload | 1280 | 1280 | 0 | 0 | 20.360 | 50.774 | 63.740 | 80 | - | 808 | 8 |
| cap=none | recovery | 80 | 80 | 0 | 0 | 48.934 | 51.511 | 51.511 | 0 | - |  |  |
| cap=8 | overload | 1280 | 392 | 888 | 0 | 0.001 | 0.004 | 0.006 | 392 | 0.048 | 1 | 8 |
| cap=8 | recovery | 80 | 77 | 3 | 0 | 0.001 | 0.003 | 0.003 | 77 | 0.001 |  |  |

engines=2 slots=8 service=500ms overload=24.0rps x 15.0s recovery=8.0rps x 3.0s kill_at=None slo_ttft=1.0s
| config | phase | offered | 200 | 429 | other | TTFT p50 s | TTFT p99 s | TTFT max s | 200 within SLO | 429 latency p99 s | peak engine waiting | peak engine running |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap=none | overload | 360 | 360 | 0 | 0 | 0.001 | 0.004 | 0.021 | 360 | - | 1 | 7 |
| cap=none | recovery | 24 | 24 | 0 | 0 | 0.001 | 0.002 | 0.002 | 24 | - |  |  |
| cap=8 | overload | 360 | 360 | 0 | 0 | 0.001 | 0.005 | 0.013 | 360 | - | 1 | 7 |
| cap=8 | recovery | 24 | 24 | 0 | 0 | 0.001 | 0.003 | 0.003 | 24 | - |  |  |

