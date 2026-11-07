//! Deterministic CPU-only simulation of Program resume ordering.
//!
//! The simulation drives the real scheduler state: `ProgramRuntime`
//! retention and dispatch, `ensure_decision_state`, force-resume deadlines,
//! `order_resume_candidates`, and every fair-share hook. Only two things are
//! modeled. Backend capacity is `slots` concurrent Programs, which is
//! equivalent to the Program-count guard. Service time comes from the
//! reference prefill and decode calibration surfaces. Every arm replays the
//! same seeded trace in virtual time.
//!
//! Not modeled: KV reuse, so MRU's cache-locality benefit is invisible here;
//! token-capacity admission; Progress-TTL retention; and real engine
//! batching. Throughput and end-to-end latency claims need real hardware.
//!
//! Run the full report with:
//! `cargo test --release --features cpu,ready --lib fair_share_sim -- --ignored --nocapture`

#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::fair_share::FairShareConfig;
use super::scheduler::ProgramScheduler;
use super::{
    DecodeThroughputModel, PrefillCostModel, ProgramDispatch, ProgramIdentity,
    ProgramRequestHandle, ProgramResumeOrder, ProgramSchedulerConfig, ProgramState, ProgramStatus,
    ProgramTarget, ProgramUsageObservation,
};
use serde_json::json;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

const TARGET: &str = "rank-0";

/// One workload class: `flows` flows, each with `agents` closed-loop agents.
#[derive(Debug, Clone)]
struct FlowClass {
    name: &'static str,
    flows: usize,
    agents: usize,
    context_tokens: usize,
    output_tokens: usize,
    /// Multiplier applied to the Router's context estimate only.
    estimate_factor: f64,
    /// Number of turns an agent issues; `None` runs for the whole horizon.
    turns: Option<usize>,
}

#[derive(Debug, Clone)]
struct Workload {
    classes: Vec<FlowClass>,
    slots: usize,
    horizon_seconds: f64,
    mean_think_seconds: f64,
    max_tracked_flows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Arm {
    Fcfs,
    Mru,
    FairShare {
        band_tokens: f64,
        tie_break: ProgramResumeOrder,
    },
}

impl Arm {
    fn label(self) -> String {
        match self {
            Self::Fcfs => "fcfs".into(),
            Self::Mru => "mru".into(),
            Self::FairShare {
                band_tokens,
                tie_break,
            } => format!("fair_share(band={band_tokens},tie={tie_break:?})").to_lowercase(),
        }
    }
}

/// SplitMix64: tiny, deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn exponential(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.next_f64()).ln()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EventKind {
    Completion(usize),
    Arrival(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Event {
    at: f64,
    seq: u64,
    kind: EventKind,
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.at
            .total_cmp(&other.at)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

struct Agent {
    class: usize,
    flow: usize,
    program_id: String,
    turns_issued: usize,
    waiting: Option<(ProgramRequestHandle, f64)>,
    in_flight: Option<(ProgramDispatch, f64)>,
}

#[derive(Debug, Clone, Default)]
struct ClassReport {
    name: &'static str,
    completed: usize,
    served_tokens: f64,
    busy_seconds: f64,
    waits: Vec<f64>,
}

impl ClassReport {
    fn percentile(&self, quantile: f64) -> f64 {
        if self.waits.is_empty() {
            return 0.0;
        }
        let mut sorted = self.waits.clone();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
        sorted[index]
    }

    fn max_wait(&self) -> f64 {
        self.waits.iter().copied().fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, Default)]
struct SimReport {
    arm: String,
    classes: Vec<ClassReport>,
    /// Served token work per flow, for Jain's index.
    flow_tokens: Vec<f64>,
    flow_busy_seconds: Vec<f64>,
    completed: usize,
    served_tokens: f64,
    peak_waiting: usize,
    peak_tracked_flows: usize,
    forced_dispatches: usize,
    ordering_calls: usize,
    ordering_nanos: u128,
}

fn jain(values: &[f64]) -> f64 {
    let sum = values.iter().sum::<f64>();
    let squares = values.iter().map(|value| value * value).sum::<f64>();
    if squares == 0.0 {
        return 1.0;
    }
    sum * sum / (values.len() as f64 * squares)
}

fn prefill_seconds(model: &PrefillCostModel, uncached_tokens: usize) -> f64 {
    let x = uncached_tokens as f64 / 1000.0;
    model.intercept_seconds
        + model.linear_seconds_per_1k_tokens * x
        + model.quadratic_seconds_per_1k_tokens_squared * x * x
}

fn decode_step_seconds(model: &DecodeThroughputModel, batch: usize, context: usize) -> f64 {
    model.fixed_step_seconds
        + model.batch_step_seconds_per_request * batch as f64
        + model.context_step_seconds_per_token * context as f64
}

fn identity(program_id: &str, task_id: &str) -> ProgramIdentity {
    ProgramIdentity::from_request(
        None,
        Some(&json!({"vllm_xargs":{"agentic_context":{
            "program_id":program_id,"task_id":task_id,"expected_resume":true
        }}})),
        Some("model"),
    )
    .unwrap()
    .unwrap()
}

fn scheduler_for(arm: Arm, workload: &Workload) -> ProgramScheduler {
    let (resume_order, fair_share) = match arm {
        Arm::Fcfs => (ProgramResumeOrder::Fcfs, None),
        Arm::Mru => (ProgramResumeOrder::Mru, None),
        Arm::FairShare {
            band_tokens,
            tie_break,
        } => (tie_break, Some(band_tokens)),
    };
    let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
        resume_order,
        max_active_programs_per_target: workload.slots,
        admission_waiting_request_threshold: 0,
        queue_timeout: Duration::from_secs(3600 * 24),
        fair_share: FairShareConfig {
            enabled: fair_share.is_some(),
            band_tokens: fair_share.unwrap_or(0.0),
            max_tracked_flows: workload.max_tracked_flows,
            ..FairShareConfig::default()
        },
        ..ProgramSchedulerConfig::default()
    });
    scheduler.sync_targets(
        "model",
        &[ProgramTarget {
            id: TARGET.into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        }],
    );
    scheduler
}

fn simulate(workload: &Workload, arm: Arm, seed: u64) -> SimReport {
    let scheduler = scheduler_for(arm, workload);
    let prefill = PrefillCostModel::default();
    let decode = DecodeThroughputModel::default();
    let base = Instant::now();
    let at = |seconds: f64| base + Duration::from_secs_f64(seconds);
    let mut rng = Rng(seed);
    let mut events = BinaryHeap::new();
    let mut seq = 0_u64;
    let mut push = |events: &mut BinaryHeap<Reverse<Event>>, at: f64, kind: EventKind| {
        seq += 1;
        events.push(Reverse(Event { at, seq, kind }));
    };

    let mut agents = Vec::new();
    let mut flow_count = 0;
    for (class_index, class) in workload.classes.iter().enumerate() {
        for _ in 0..class.flows {
            for agent in 0..class.agents {
                agents.push(Agent {
                    class: class_index,
                    flow: flow_count,
                    program_id: format!("{}-f{flow_count}-a{agent}", class.name),
                    turns_issued: 0,
                    waiting: None,
                    in_flight: None,
                });
            }
            flow_count += 1;
        }
    }
    for index in 0..agents.len() {
        let start = rng.exponential(workload.mean_think_seconds);
        push(&mut events, start, EventKind::Arrival(index));
    }

    let mut report = SimReport {
        arm: arm.label(),
        classes: workload
            .classes
            .iter()
            .map(|class| ClassReport {
                name: class.name,
                ..ClassReport::default()
            })
            .collect(),
        flow_tokens: vec![0.0; flow_count],
        flow_busy_seconds: vec![0.0; flow_count],
        ..SimReport::default()
    };
    let mut by_program: HashMap<String, usize> = HashMap::new();
    let mut active = 0_usize;
    let mut active_context = 0_usize;
    let mut processed = 0_u64;

    while let Some(Reverse(event)) = events.pop() {
        if event.at > workload.horizon_seconds {
            break;
        }
        let now = at(event.at);
        let mut state = scheduler.state.lock();
        match event.kind {
            EventKind::Arrival(index) => {
                let agent = &mut agents[index];
                let class = &workload.classes[agent.class];
                let identity = identity(&agent.program_id, &format!("flow-{}", agent.flow));
                let estimate =
                    (class.context_tokens as f64 * class.estimate_factor).round() as usize;
                let handle = state.runtime.retain_request(&identity, estimate, None, now);
                let reference = handle.program().clone();
                scheduler.ensure_decision_state(
                    &mut state,
                    &identity,
                    reference.clone(),
                    estimate,
                    now,
                    None,
                );
                let queue = state.rank_queues.entry(TARGET.into()).or_default();
                if !queue.contains(&reference) {
                    queue.push_back(reference.clone());
                }
                if let Some(decision) = state.decisions.get_mut(&reference) {
                    decision.queued_at = Some(now);
                }
                scheduler.set_force_resume_deadline(&mut state, TARGET, &reference, now);
                by_program.insert(reference.program_id().to_string(), index);
                agent.turns_issued += 1;
                agent.waiting = Some((handle, event.at));
            }
            EventKind::Completion(index) => {
                let agent = &mut agents[index];
                let class = &workload.classes[agent.class];
                let (dispatch, dispatched_at) = agent.in_flight.take().expect("in flight");
                let usage = ProgramUsageObservation {
                    prompt_tokens: Some(class.context_tokens),
                    completion_tokens: Some(class.output_tokens),
                    cached_prompt_tokens: Some(0),
                };
                scheduler.fair_share_settle(&mut state, &dispatch, &usage, true);
                assert!(state.runtime.complete_request(&dispatch, usage));
                assert!(state.runtime.pause_idle(dispatch.program()));
                if let Some(decision) = state.decisions.get_mut(dispatch.program()) {
                    decision.last_request_finished_at = Some(now);
                    decision.force_resume_deadline = None;
                }
                active -= 1;
                active_context -= class.context_tokens;
                let served = (class.context_tokens + class.output_tokens) as f64;
                let busy = event.at - dispatched_at;
                let class_report = &mut report.classes[agent.class];
                class_report.completed += 1;
                class_report.served_tokens += served;
                class_report.busy_seconds += busy;
                report.flow_tokens[agent.flow] += served;
                report.flow_busy_seconds[agent.flow] += busy;
                report.completed += 1;
                report.served_tokens += served;
                if class.turns.is_none_or(|turns| agent.turns_issued < turns) {
                    let think = rng.exponential(workload.mean_think_seconds);
                    push(&mut events, event.at + think, EventKind::Arrival(index));
                }
            }
        }

        // Dispatch while capacity remains, using the production comparator.
        while active < workload.slots {
            scheduler.refresh_fair_share_floor(&mut state);
            let mut queued = state
                .rank_queues
                .get(TARGET)
                .into_iter()
                .flatten()
                .filter(|program| {
                    state.runtime.view(program).is_some_and(|view| {
                        view.state == ProgramState::Paused
                            && view.status == ProgramStatus::Reasoning
                            && view.waiting_requests > 0
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            report.peak_waiting = report.peak_waiting.max(queued.len());
            if queued.is_empty() {
                break;
            }
            let started = Instant::now();
            scheduler.order_resume_candidates(&state, &mut queued, now);
            report.ordering_nanos += started.elapsed().as_nanos();
            report.ordering_calls += 1;
            let chosen = queued.swap_remove(0);
            if state.decisions[&chosen]
                .force_resume_deadline
                .is_some_and(|deadline| deadline <= now)
            {
                report.forced_dispatches += 1;
            }
            if let Some(queue) = state.rank_queues.get_mut(TARGET) {
                queue.retain(|program| program != &chosen);
            }
            let index = by_program[chosen.program_id()];
            let agent = &mut agents[index];
            let class = &workload.classes[agent.class];
            let (handle, arrived) = agent.waiting.take().expect("waiting");
            let mut dispatch = state
                .runtime
                .admit_front(&handle, TARGET.into(), now)
                .expect("admit");
            scheduler.fair_share_charge_dispatch(&mut state, &mut dispatch);
            active += 1;
            active_context += class.context_tokens;
            let service = prefill_seconds(&prefill, class.context_tokens)
                + class.output_tokens as f64 * decode_step_seconds(&decode, active, active_context);
            report.classes[agent.class].waits.push(event.at - arrived);
            agent.in_flight = Some((dispatch, event.at));
            push(
                &mut events,
                event.at + service,
                EventKind::Completion(index),
            );
        }
        if let Some(ledger) = state.fair_share.as_ref() {
            report.peak_tracked_flows = report
                .peak_tracked_flows
                .max(ledger.snapshot().tracked_flows);
        }
        processed += 1;
        if processed.is_multiple_of(64) {
            scheduler.prune_fair_share(&mut state);
        }
    }
    report
}

fn heavy_vs_light(estimate_factor: f64) -> Workload {
    Workload {
        classes: vec![
            FlowClass {
                name: "heavy",
                flows: 1,
                agents: 8,
                context_tokens: 32_000,
                output_tokens: 1_024,
                estimate_factor,
                turns: None,
            },
            FlowClass {
                name: "light",
                flows: 4,
                agents: 2,
                context_tokens: 2_000,
                output_tokens: 128,
                estimate_factor: 1.0,
                turns: None,
            },
        ],
        slots: 4,
        horizon_seconds: 3_600.0,
        mean_think_seconds: 1.0,
        max_tracked_flows: 4_096,
    }
}

fn uniform() -> Workload {
    Workload {
        classes: vec![FlowClass {
            name: "uniform",
            flows: 5,
            agents: 3,
            context_tokens: 8_000,
            output_tokens: 256,
            estimate_factor: 1.0,
            turns: None,
        }],
        slots: 4,
        horizon_seconds: 3_600.0,
        mean_think_seconds: 1.0,
        max_tracked_flows: 4_096,
    }
}

fn flow_churn() -> Workload {
    Workload {
        classes: vec![
            FlowClass {
                name: "heavy",
                flows: 1,
                agents: 4,
                context_tokens: 32_000,
                output_tokens: 1_024,
                estimate_factor: 1.0,
                turns: None,
            },
            FlowClass {
                name: "short_lived",
                flows: 2_000,
                agents: 1,
                context_tokens: 1_000,
                output_tokens: 64,
                estimate_factor: 1.0,
                turns: Some(2),
            },
        ],
        slots: 4,
        horizon_seconds: 3_600.0,
        mean_think_seconds: 600.0,
        max_tracked_flows: 256,
    }
}

const fn fair(band_tokens: f64, tie_break: ProgramResumeOrder) -> Arm {
    Arm::FairShare {
        band_tokens,
        tie_break,
    }
}

const STRICT: Arm = fair(0.0, ProgramResumeOrder::Mru);
/// Documented recommendation: default band with an FCFS tie-break.
const RECOMMENDED: Arm = fair(8_192.0, ProgramResumeOrder::Fcfs);

const ARMS: [Arm; 6] = [
    Arm::Fcfs,
    Arm::Mru,
    fair(0.0, ProgramResumeOrder::Mru),
    fair(0.0, ProgramResumeOrder::Fcfs),
    fair(8_192.0, ProgramResumeOrder::Mru),
    fair(8_192.0, ProgramResumeOrder::Fcfs),
];

fn class<'a>(report: &'a SimReport, name: &str) -> &'a ClassReport {
    report
        .classes
        .iter()
        .find(|class| class.name == name)
        .unwrap()
}

#[test]
fn simulation_is_deterministic() {
    let workload = heavy_vs_light(1.0);
    let first = simulate(&workload, STRICT, 7);
    let second = simulate(&workload, STRICT, 7);
    assert_eq!(first.completed, second.completed);
    assert_eq!(first.flow_tokens, second.flow_tokens);
    assert_eq!(class(&first, "light").waits, class(&second, "light").waits);
}

#[test]
fn fair_share_cuts_light_flow_wait_under_a_heavy_flow() {
    let workload = heavy_vs_light(1.0);
    let fcfs = simulate(&workload, Arm::Fcfs, 7);
    let mru = simulate(&workload, Arm::Mru, 7);
    for fair in [
        simulate(&workload, STRICT, 7),
        simulate(&workload, RECOMMENDED, 7),
    ] {
        for baseline in [&fcfs, &mru] {
            assert!(
                class(&fair, "light").percentile(0.95)
                    < 0.5 * class(baseline, "light").percentile(0.95),
                "{}: light p95 {} vs {}",
                baseline.arm,
                class(&fair, "light").percentile(0.95),
                class(baseline, "light").percentile(0.95)
            );
            assert!(jain(&fair.flow_tokens) > jain(&baseline.flow_tokens));
        }
    }
    // The heavy flow keeps at least an equal GPU-time share and is bounded by
    // force resume rather than starved.
    let fair = simulate(&workload, RECOMMENDED, 7);
    let heavy = class(&fair, "heavy");
    let busy = fair.flow_busy_seconds.iter().sum::<f64>();
    assert!(heavy.busy_seconds / busy > 0.9 / 5.0);
    assert!(
        heavy.max_wait() <= 330.0,
        "heavy max wait {}",
        heavy.max_wait()
    );
}

#[test]
fn reconciliation_corrects_a_heavy_flow_that_underestimates() {
    let honest = simulate(&heavy_vs_light(1.0), STRICT, 7);
    let biased = simulate(&heavy_vs_light(0.5), STRICT, 7);
    let fcfs = simulate(&heavy_vs_light(0.5), Arm::Fcfs, 7);
    let share = |report: &SimReport| class(report, "heavy").served_tokens / report.served_tokens;
    assert!((share(&biased) - share(&honest)).abs() < 0.05);
    assert!(share(&biased) < share(&fcfs));
}

#[test]
fn uniform_workload_does_not_regress() {
    let workload = uniform();
    let fcfs = simulate(&workload, Arm::Fcfs, 11);
    let fair = simulate(&workload, RECOMMENDED, 11);
    let ratio = fair.served_tokens / fcfs.served_tokens;
    assert!((0.99..=1.01).contains(&ratio), "throughput ratio {ratio}");
    assert!(jain(&fair.flow_tokens) >= jain(&fcfs.flow_tokens) - 0.01);
    let tail = class(&fair, "uniform").percentile(0.99);
    let baseline = class(&fcfs, "uniform").percentile(0.99);
    assert!(tail <= 1.1 * baseline, "p99 {tail} vs fcfs {baseline}");
}

#[test]
fn ledger_stays_bounded_under_flow_churn() {
    let workload = flow_churn();
    let fair = simulate(&workload, RECOMMENDED, 13);
    let fcfs = simulate(&workload, Arm::Fcfs, 13);
    assert!(
        class(&fair, "heavy").percentile(0.95)
            <= 2.0 * class(&fcfs, "heavy").percentile(0.95) + 1.0
    );
    // Live flows can exceed the idle-account cap only by in-flight charges.
    assert!(
        fair.peak_tracked_flows <= workload.max_tracked_flows + workload.slots,
        "peak tracked flows {}",
        fair.peak_tracked_flows
    );
    assert!(fair.completed > 1_000);
}

#[test]
#[ignore = "prints the benchmark report"]
fn report() {
    let seeds = [7_u64, 11, 13];
    let workloads = [
        ("heavy_vs_light", heavy_vs_light(1.0)),
        ("heavy_vs_light_underestimate_50pct", heavy_vs_light(0.5)),
        ("heavy_vs_light_overestimate_50pct", heavy_vs_light(1.5)),
        ("uniform", uniform()),
        ("flow_churn", flow_churn()),
    ];
    for (label, workload) in workloads {
        println!(
            "\n### {label} (slots={}, horizon={}s, seeds={seeds:?})\n",
            workload.slots, workload.horizon_seconds
        );
        println!("| arm | class | completed | token share | GPU-s share | wait p50 s | p95 s | p99 s | max s |");
        println!("|---|---|---:|---:|---:|---:|---:|---:|---:|");
        let mut summary = Vec::new();
        for arm in ARMS {
            let runs = seeds
                .iter()
                .map(|seed| simulate(&workload, arm, *seed))
                .collect::<Vec<_>>();
            for (index, class) in workload.classes.iter().enumerate() {
                let mut merged = ClassReport {
                    name: class.name,
                    ..ClassReport::default()
                };
                let (mut tokens, mut busy, mut total_tokens, mut total_busy) = (0.0, 0.0, 0.0, 0.0);
                for run in &runs {
                    merged.completed += run.classes[index].completed;
                    merged.waits.extend(&run.classes[index].waits);
                    tokens += run.classes[index].served_tokens;
                    busy += run.classes[index].busy_seconds;
                    total_tokens += run.served_tokens;
                    total_busy += run.classes.iter().map(|c| c.busy_seconds).sum::<f64>();
                }
                println!(
                    "| {} | {} | {} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {:.1} |",
                    arm.label(),
                    class.name,
                    merged.completed,
                    tokens / total_tokens,
                    busy / total_busy,
                    merged.percentile(0.5),
                    merged.percentile(0.95),
                    merged.percentile(0.99),
                    merged.max_wait()
                );
            }
            let mean =
                |f: &dyn Fn(&SimReport) -> f64| runs.iter().map(f).sum::<f64>() / runs.len() as f64;
            summary.push(format!(
                "| {} | {:.3} | {:.3} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} |",
                arm.label(),
                mean(&|r| jain(&r.flow_tokens)),
                mean(&|r| jain(&r.flow_busy_seconds)),
                mean(&|r| r.served_tokens / workload.horizon_seconds),
                mean(&|r| r.completed as f64),
                mean(&|r| r.peak_waiting as f64),
                mean(&|r| r.peak_tracked_flows as f64),
                mean(&|r| r.forced_dispatches as f64),
                mean(&|r| r.ordering_nanos as f64 / r.ordering_calls.max(1) as f64),
            ));
        }
        println!("\n| arm | Jain (tokens) | Jain (GPU-s) | tokens/s | completed | peak waiting | peak flows | forced | ns/order |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        for line in summary {
            println!("{line}");
        }
    }
}

#[test]
#[ignore = "prints comparator overhead"]
fn ordering_overhead() {
    println!("\n| waiters | arm | ns per order_resume_candidates |");
    println!("|---:|---|---:|");
    for waiters in [10_usize, 100, 1_000] {
        for arm in [Arm::Mru, STRICT] {
            let workload = Workload {
                classes: vec![FlowClass {
                    name: "flow",
                    flows: waiters,
                    agents: 1,
                    context_tokens: 1_000,
                    output_tokens: 10,
                    estimate_factor: 1.0,
                    turns: None,
                }],
                slots: 1,
                horizon_seconds: 0.0,
                mean_think_seconds: 1.0,
                max_tracked_flows: 4_096,
            };
            let scheduler = scheduler_for(arm, &workload);
            let now = Instant::now();
            let mut state = scheduler.state.lock();
            let mut refs = Vec::new();
            for flow in 0..waiters {
                let identity = identity(&format!("p{flow}"), &format!("t{flow}"));
                let handle = state.runtime.retain_request(&identity, 1_000, None, now);
                let reference = handle.program().clone();
                scheduler.ensure_decision_state(
                    &mut state,
                    &identity,
                    reference.clone(),
                    1_000,
                    now,
                    None,
                );
                let flow_key = state.decisions[&reference].fair_share_flow.clone();
                if let (Some(ledger), Some(flow)) = (state.fair_share.as_mut(), flow_key) {
                    ledger.charge(
                        &flow,
                        super::fair_share::FairShareEstimate {
                            context_tokens: (refs.len() * 7_919) % 50_000,
                            shared_prefix_tokens: 0,
                            output_tokens: 0,
                        },
                    );
                }
                state
                    .decisions
                    .get_mut(&reference)
                    .unwrap()
                    .last_request_finished_at =
                    Some(now - Duration::from_millis((flow * 37 % 1000) as u64));
                refs.push(reference);
            }
            let iterations = 200_000 / waiters;
            let started = Instant::now();
            for _ in 0..iterations {
                let mut queued = refs.clone();
                scheduler.order_resume_candidates(&state, &mut queued, now);
                std::hint::black_box(&queued);
            }
            println!(
                "| {waiters} | {} | {:.0} |",
                arm.label(),
                started.elapsed().as_nanos() as f64 / iterations as f64
            );
        }
    }
}
