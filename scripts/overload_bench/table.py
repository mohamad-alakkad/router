import json, sys
r = json.load(open(sys.argv[1]))
p = r["params"]
print(f"engines={p['engines']} slots={p['slots']} service={p['service_ms']}ms "
      f"overload={p['overload_rps']}rps x {p['overload_secs']}s recovery={p['recovery_rps']}rps x {p['recovery_secs']}s "
      f"kill_at={p['kill_at']} slo_ttft={p['slo_ttft_s']}s")
f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
print("| config | phase | offered | 200 | 429 | other | TTFT p50 s | TTFT p99 s | TTFT max s | 200 within SLO | 429 latency p99 s | peak engine waiting | peak engine running |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, run in r["runs"].items():
    waiting = max(e["peak_waiting"] for e in run["engines"])
    running = max(e["peak_running"] for e in run["engines"])
    for phase in ("overload", "recovery"):
        s = run[phase]
        other = s["other"] if not s["other_statuses"] else f"{s['other']} {s['other_statuses']}"
        print(f"| {name} | {phase} | {s['offered']} | {s['completed_200']} | {s['rejected_429']} | {other} | "
              f"{f(s['ttft_p50_s'])} | {f(s['ttft_p99_s'])} | {f(s['ttft_max_s'])} | {s['within_slo']} | "
              f"{f(s['reject_latency_p99_s'])} | {waiting if phase=='overload' else ''} | {running if phase=='overload' else ''} |")
