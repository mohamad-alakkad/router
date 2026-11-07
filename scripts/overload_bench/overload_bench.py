#!/usr/bin/env python3
"""Local overload / recovery benchmark for per-worker admission (RFC #435).

Everything is a MOCK: each "engine" is an aiohttp server with a fixed number
of concurrent batch slots and an unbounded internal FIFO wait queue (the shape
of vLLM's running/waiting queues). It is not a GPU and the numbers are not a
claim about real vLLM throughput; they show how the router's admission
decision changes where queueing happens and how fast the system recovers.

Scenario: open-loop arrivals at `--overload-rps` for `--overload-secs`, then
`--recovery-rps` for `--recovery-secs`. Optionally one engine is killed partway
through the overload phase (`--kill-at`). The router binary is started once
per configuration with and without --max-inflight-requests-per-worker.
"""

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time

import aiohttp
from aiohttp import web

CHUNKS = 10


class Engine:
    def __init__(self, slots, service_ms):
        self.slots = asyncio.Semaphore(slots)
        self.service_s = service_ms / 1000.0
        self.running = self.waiting = 0
        self.peak_running = self.peak_waiting = self.served = 0
        self.runner = None

    async def chat(self, request):
        await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)  # vLLM sends headers before scheduling
        self.waiting += 1
        self.peak_waiting = max(self.peak_waiting, self.waiting)
        scheduled = False
        try:
            async with self.slots:
                self.waiting -= 1
                scheduled = True
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
                try:
                    for _ in range(CHUNKS):
                        await response.write(
                            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                        )
                        await asyncio.sleep(self.service_s / CHUNKS)
                    await response.write(b"data: [DONE]\n\n")
                    self.served += 1
                finally:
                    self.running -= 1
        finally:
            if not scheduled:  # client went away while queued
                self.waiting -= 1
        return response

    async def health(self, _request):
        return web.json_response({"status": "ok"})

    async def start(self, port):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.health)
        self.runner = web.AppRunner(app, handle_signals=False)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", port).start()

    async def kill(self):
        await self.runner.cleanup()


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def one_request(session, url, phase, results, t0):
    sent = time.monotonic()
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    record = {"phase": phase, "t": sent - t0}
    try:
        async with session.post(url, json=body) as response:
            record["status"] = response.status
            if response.status == 200:
                ttft = None
                async for line in response.content:
                    if ttft is None and line.startswith(b"data: {"):
                        ttft = time.monotonic() - sent
                record["ttft"] = ttft
            else:
                await response.read()
                record["retry_after"] = response.headers.get("Retry-After")
    except Exception as error:  # transport failure
        record["status"] = f"error:{type(error).__name__}"
    record["latency"] = time.monotonic() - sent
    results.append(record)


async def load(url, args, engines):
    results, tasks = [], []
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.monotonic()
        killed = False
        samples = []
        for phase, rps, secs in (
            ("overload", args.overload_rps, args.overload_secs),
            ("recovery", args.recovery_rps, args.recovery_secs),
        ):
            start = time.monotonic()
            n = int(rps * secs)
            for i in range(n):
                target = start + i / rps
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                now = time.monotonic() - t0
                if args.kill_at is not None and not killed and now >= args.kill_at:
                    killed = True
                    await engines[-1].kill()
                if i % max(1, int(rps / 4)) == 0:
                    samples.append(
                        (round(now, 2), [(e.running, e.waiting) for e in engines])
                    )
                tasks.append(asyncio.create_task(one_request(session, url, phase, results, t0)))
        await asyncio.gather(*tasks)
    return results, samples


def summarize(results, engines, samples, slo):
    out = {}
    for phase in ("overload", "recovery"):
        rows = [r for r in results if r["phase"] == phase]
        ok = [r for r in rows if r["status"] == 200]
        rejected = [r for r in rows if r["status"] == 429]
        other = [r for r in rows if r["status"] not in (200, 429)]
        ttft = [r["ttft"] for r in ok if r.get("ttft") is not None]
        out[phase] = {
            "offered": len(rows),
            "completed_200": len(ok),
            "rejected_429": len(rejected),
            "other": len(other),
            "other_statuses": sorted({str(r["status"]) for r in other}),
            "ttft_p50_s": pct(ttft, 0.5),
            "ttft_p99_s": pct(ttft, 0.99),
            "ttft_max_s": max(ttft) if ttft else None,
            "within_slo": sum(1 for t in ttft if t <= slo),
            "reject_latency_p99_s": pct([r["latency"] for r in rejected], 0.99),
            "retry_after_values": sorted({r.get("retry_after") for r in rejected} - {None}),
        }
    out["engines"] = [
        {"peak_running": e.peak_running, "peak_waiting": e.peak_waiting, "served": e.served}
        for e in engines
    ]
    out["max_engine_waiting_over_time"] = [
        (t, max(w for _, w in s)) for t, s in samples[:: max(1, len(samples) // 12)]
    ]
    return out


async def run_config(args, cap, port_base):
    engines = [Engine(args.slots, args.service_ms) for _ in range(args.engines)]
    ports = [port_base + 1 + i for i in range(args.engines)]
    for engine, port in zip(engines, ports):
        await engine.start(port)
    router_port = port_base
    cmd = [args.router_bin, "--port", str(router_port), "--policy", "round_robin",
           "--worker-urls", *[f"http://127.0.0.1:{p}" for p in ports],
           "--log-level", "warn"]
    if cap is not None:
        cmd += ["--max-inflight-requests-per-worker", str(cap)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(300):
                try:
                    async with session.get(f"http://127.0.0.1:{router_port}/readiness") as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("router did not become ready")
        results, samples = await load(
            f"http://127.0.0.1:{router_port}/v1/chat/completions", args, engines
        )
        return summarize(results, engines, samples, args.slo_ttft_s)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait()
        for engine in engines:
            try:
                await engine.kill()
            except Exception:
                pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-bin", required=True)
    parser.add_argument("--engines", type=int, default=2)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--service-ms", type=int, default=500)
    parser.add_argument("--overload-rps", type=float, default=64)
    parser.add_argument("--overload-secs", type=float, default=20)
    parser.add_argument("--recovery-rps", type=float, default=8)
    parser.add_argument("--recovery-secs", type=float, default=10)
    parser.add_argument("--kill-at", type=float, default=None)
    parser.add_argument("--slo-ttft-s", type=float, default=1.0)
    parser.add_argument("--caps", default="none,8,12")
    parser.add_argument("--port-base", type=int, default=39000)
    args = parser.parse_args()
    report = {"params": vars(args), "runs": {}}
    for index, cap in enumerate(args.caps.split(",")):
        value = None if cap == "none" else int(cap)
        report["runs"][f"cap={cap}"] = await run_config(args, value, args.port_base + 10 * index)
        print(f"done cap={cap}", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
