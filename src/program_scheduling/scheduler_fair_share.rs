//! ProgramScheduler hooks for opt-in token-cost fair share.
//!
//! Every hook is a no-op when `fair_share.enabled` is false, so the default
//! scheduler commits exactly the decisions it committed before this feature.

use super::fair_share::FairShareEstimate;
use super::scheduler::ProgramScheduler;
use super::scheduler_state::ProgramSchedulerState;
use super::{
    ProgramDispatch, ProgramIdentity, ProgramRef, ProgramState, ProgramStatus,
    ProgramUsageObservation,
};
use crate::metrics::RouterMetrics;
use std::cmp::Ordering;
use std::sync::Arc;

impl ProgramScheduler {
    /// Bind the request's flow to its Program and apply the arrival floor.
    pub(crate) fn fair_share_observe_arrival(
        &self,
        state: &mut ProgramSchedulerState,
        identity: &ProgramIdentity,
        reference: &ProgramRef,
    ) {
        let Some(ledger) = state.fair_share.as_mut() else {
            return;
        };
        let flow = ledger.flow_for(identity);
        ledger.observe_arrival(&flow);
        if let Some(decision) = state.decisions.get_mut(reference) {
            decision.fair_share_flow = Some(flow);
        }
    }

    /// Charge the estimated work of a committed dispatch.
    pub(crate) fn fair_share_charge_dispatch(
        &self,
        state: &mut ProgramSchedulerState,
        dispatch: &mut ProgramDispatch,
    ) {
        let Some(decision) = state.decisions.get(dispatch.program()) else {
            return;
        };
        let Some(flow) = decision.fair_share_flow.clone() else {
            return;
        };
        let estimate = FairShareEstimate {
            context_tokens: dispatch.estimated_context_tokens(),
            shared_prefix_tokens: decision.shared_prefix_tokens,
            output_tokens: dispatch
                .request_hints()
                .expected_output_tokens
                .or(decision.output_token_reservation)
                .unwrap_or(self.config.progress_ttl.decode_buffer_tokens),
        };
        let Some(ledger) = state.fair_share.as_mut() else {
            return;
        };
        RouterMetrics::record_agent_aware_fair_share_resume_lag(
            &dispatch.target_id,
            ledger.lag(&flow),
        );
        let charge = ledger.charge(&flow, estimate);
        dispatch.set_fair_share_charge(Some(charge));
    }

    /// Settle the dispatch's provisional charge against reported usage.
    ///
    /// Runs before any generation or placement check: a stale completion still
    /// consumed backend work and must still release its provisional charge.
    pub(crate) fn fair_share_settle(
        &self,
        state: &mut ProgramSchedulerState,
        dispatch: &ProgramDispatch,
        usage: &ProgramUsageObservation,
    ) {
        let (Some(ledger), Some(charge)) =
            (state.fair_share.as_mut(), dispatch.fair_share_charge())
        else {
            return;
        };
        if let Some(settlement) = ledger.settle(charge, usage) {
            RouterMetrics::record_agent_aware_fair_share_settlement(
                &dispatch.target_id,
                settlement.estimated_tokens,
                settlement.actual_tokens,
            );
        }
    }

    /// Raise the service floor from the flows that currently wait to resume.
    pub(crate) fn refresh_fair_share_floor(&self, state: &mut ProgramSchedulerState) {
        if state.fair_share.is_none() {
            return;
        }
        let backlogged = state
            .rank_queues
            .values()
            .chain(state.global_queues.values())
            .flatten()
            .filter(|program| {
                state.runtime.view(program).is_some_and(|view| {
                    view.state == ProgramState::Paused
                        && view.status == ProgramStatus::Reasoning
                        && view.waiting_requests > 0
                })
            })
            .filter_map(|program| state.decisions.get(program)?.fair_share_flow.clone())
            .collect::<Vec<Arc<str>>>();
        if let Some(ledger) = state.fair_share.as_mut() {
            ledger.advance_floor(backlogged.iter());
        }
    }

    /// Drop idle accounts that the floor makes redundant.
    pub(crate) fn prune_fair_share(&self, state: &mut ProgramSchedulerState) {
        if let Some(ledger) = state.fair_share.as_mut() {
            ledger.prune();
            RouterMetrics::set_agent_aware_fair_share_tracked_flows(
                ledger.snapshot().tracked_flows,
            );
        }
    }

    /// Ordinary-tier comparison by charged flow service; `Equal` when disabled.
    pub(crate) fn fair_share_cmp(
        &self,
        state: &ProgramSchedulerState,
        left: &ProgramRef,
        right: &ProgramRef,
    ) -> Ordering {
        let Some(ledger) = state.fair_share.as_ref() else {
            return Ordering::Equal;
        };
        ledger.compare(
            state.decisions[left].fair_share_flow.as_ref(),
            state.decisions[right].fair_share_flow.as_ref(),
        )
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        FairShareConfig, ProgramResumeOrder, ProgramSchedulerConfig, ProgramTarget,
    };
    use serde_json::json;
    use std::time::{Duration, Instant};

    fn target() -> ProgramTarget {
        ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        }
    }

    fn identity(program: &str, task: &str) -> ProgramIdentity {
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs":{"agentic_context":{
                "program_id":program,"task_id":task,"expected_resume":true
            }}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn usage(prompt: usize, completion: usize) -> ProgramUsageObservation {
        ProgramUsageObservation {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            cached_prompt_tokens: Some(0),
        }
    }

    /// One Program slot, no backend gate: resume order alone decides service.
    fn single_slot(fair_share: bool, resume_order: ProgramResumeOrder) -> Arc<ProgramScheduler> {
        Arc::new(ProgramScheduler::new(ProgramSchedulerConfig {
            max_active_programs_per_target: 1,
            admission_waiting_request_threshold: 0,
            resume_order,
            fair_share: FairShareConfig {
                enabled: fair_share,
                ..FairShareConfig::default()
            },
            ..ProgramSchedulerConfig::default()
        }))
    }

    /// Returns which of a heavy-flow and a light-flow waiter resumes first
    /// after the heavy flow has already consumed `heavy_history` tokens.
    async fn first_resumed(scheduler: Arc<ProgramScheduler>) -> &'static str {
        let history = scheduler
            .acquire(identity("heavy-0", "heavy"), 30_000, &[target()], None)
            .await
            .unwrap();
        scheduler.complete(&history, true, true, usage(30_000, 1_000), None);
        let blocker = scheduler
            .acquire(identity("blocker", "other"), 100, &[target()], None)
            .await
            .unwrap();

        // The heavy waiter enters the RequestPool strictly before the light one.
        let heavy = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .acquire(identity("heavy-1", "heavy"), 30_000, &[target()], None)
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        let light = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .acquire(identity("light-1", "light"), 2_000, &[target()], None)
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(scheduler.state.lock().runtime.retained_request_count(), 2);

        scheduler.complete(&blocker, true, true, usage(100, 10), None);
        let winner = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if heavy.is_finished() {
                    return "heavy";
                }
                if light.is_finished() {
                    return "light";
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("one waiter resumes");
        heavy.abort();
        light.abort();
        winner
    }

    #[tokio::test]
    async fn baseline_orders_cost_blind_and_fair_share_prefers_the_light_flow() {
        assert_eq!(
            first_resumed(single_slot(false, ProgramResumeOrder::Fcfs)).await,
            "heavy"
        );
        assert_eq!(
            first_resumed(single_slot(false, ProgramResumeOrder::Mru)).await,
            "heavy"
        );
        assert_eq!(
            first_resumed(single_slot(true, ProgramResumeOrder::Fcfs)).await,
            "light"
        );
        assert_eq!(
            first_resumed(single_slot(true, ProgramResumeOrder::Mru)).await,
            "light"
        );
    }

    #[tokio::test]
    async fn disabled_fair_share_allocates_no_state_and_no_charges() {
        let scheduler = single_slot(false, ProgramResumeOrder::Mru);
        let dispatch = scheduler
            .acquire(identity("p", "task"), 1_000, &[target()], None)
            .await
            .unwrap();
        assert_eq!(dispatch.fair_share_charge(), None);
        {
            let state = scheduler.state.lock();
            assert!(state.fair_share.is_none());
            assert!(state.decisions[dispatch.program()]
                .fair_share_flow
                .is_none());
        }
        assert!(scheduler.diagnostics().fair_share.is_none());
        let serialized =
            serde_json::to_value(crate::config::types::ProgramSchedulingConfig::default()).unwrap();
        assert!(serialized.get("fair_share").is_none());
    }

    #[tokio::test]
    async fn each_logical_request_settles_exactly_once() {
        let scheduler = single_slot(true, ProgramResumeOrder::Mru);
        let dispatch = scheduler
            .acquire(identity("p", "task"), 1_000, &[target()], None)
            .await
            .unwrap();
        let charge = dispatch.fair_share_charge().expect("dispatch is charged");
        let flow = scheduler.state.lock().decisions[dispatch.program()]
            .fair_share_flow
            .clone()
            .unwrap();
        // Estimate: 1000 uncached prompt + 100 default decode buffer.
        assert_eq!(
            scheduler
                .state
                .lock()
                .fair_share
                .as_ref()
                .unwrap()
                .virtual_service(&flow),
            1_100.0
        );

        // A transport restart continues the same logical request and charge.
        let replacement = {
            let mut state = scheduler.state.lock();
            state
                .runtime
                .restart_attempt(&dispatch, Instant::now())
                .unwrap()
        };
        assert_eq!(replacement.fair_share_charge(), Some(charge));

        scheduler.complete(&replacement, true, false, usage(1_000, 40), None);
        scheduler.complete(&replacement, true, false, usage(9_000, 9_000), None);
        scheduler.complete(&dispatch, false, false, usage(9_000, 9_000), None);
        let state = scheduler.state.lock();
        let ledger = state.fair_share.as_ref().unwrap();
        assert_eq!(ledger.virtual_service(&flow), 1_040.0);
        assert_eq!(ledger.pending_charges(), 0);
    }

    #[tokio::test]
    async fn cancelled_waiter_is_never_charged() {
        let scheduler = single_slot(true, ProgramResumeOrder::Mru);
        let blocker = scheduler
            .acquire(identity("blocker", "other"), 100, &[target()], None)
            .await
            .unwrap();
        let waiter = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .acquire(identity("waiter", "light"), 5_000, &[target()], None)
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        waiter.abort();
        let _ = waiter.await;
        let state = scheduler.state.lock();
        assert_eq!(state.runtime.retained_request_count(), 0);
        let ledger = state.fair_share.as_ref().unwrap();
        // Only the blocker's dispatch is outstanding.
        assert_eq!(ledger.pending_charges(), 1);
        let light = Arc::<str>::from("model\u{0}light");
        assert_eq!(ledger.lag(&light), 0.0);
        drop(state);
        scheduler.complete(&blocker, true, true, usage(100, 1), None);
    }

    #[tokio::test]
    async fn priority_and_force_resume_still_outrank_fair_share() {
        let scheduler = single_slot(true, ProgramResumeOrder::Mru);
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let mut refs = Vec::new();
        for (program, task) in [("heavy", "heavy"), ("light", "light")] {
            let identity = identity(program, task);
            let handle = state.runtime.retain_request(&identity, 100, None, now);
            let reference = handle.program().clone();
            scheduler.ensure_decision_state(
                &mut state,
                &identity,
                reference.clone(),
                100,
                now,
                None,
            );
            refs.push(reference);
        }
        let heavy_flow = state.decisions[&refs[0]].fair_share_flow.clone().unwrap();
        state.fair_share.as_mut().unwrap().charge(
            &heavy_flow,
            super::super::fair_share::FairShareEstimate {
                context_tokens: 100_000,
                shared_prefix_tokens: 0,
                output_tokens: 0,
            },
        );
        let mut ordered = refs.clone();
        scheduler.order_resume_candidates(&state, &mut ordered, now);
        assert_eq!(ordered[0], refs[1]);
        state
            .decisions
            .get_mut(&refs[0])
            .unwrap()
            .force_resume_deadline = Some(now);
        scheduler.order_resume_candidates(&state, &mut ordered, now);
        assert_eq!(ordered[0], refs[0]);
    }
}
