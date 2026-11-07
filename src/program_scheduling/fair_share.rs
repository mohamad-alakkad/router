//! Opt-in token-cost fair-share ordering for retained Program requests.
//!
//! The ledger orders already-eligible resume candidates by the token work their
//! flow has been charged. It never grants capacity, bypasses an admission gate,
//! or outranks force-resume, request priority, or segment yield; it only
//! replaces the ordinary MRU/FCFS order with a cost-aware order whose ties fall
//! back to that configured order.
//!
//! The policy is start-time fair queueing over flows. Each flow owns a virtual
//! service counter in charged tokens. A request is charged its estimated work
//! when it is dispatched and reconciled once against reported usage when it
//! completes. A flow that arrives with less service than the current floor is
//! raised to the floor, so idle time cannot be banked into a later burst.
//! All state is protected by the ProgramScheduler mutex.

use super::{ProgramIdentity, ProgramUsageObservation};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Request identity that defines one fair-share flow.
///
/// Every source is request-body metadata. A flow is therefore a cooperative
/// workflow grouping, not an authenticated tenant: a client can split work
/// across identifiers. Tenant isolation requires trusted identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FairShareFlowKey {
    /// `task_id`, then `root_program_id`, then `program_id`.
    #[default]
    TaskId,
    /// `root_program_id`, then `program_id`.
    RootProgramId,
    /// One flow per Program.
    ProgramId,
}

/// Opt-in token-cost fair-share configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FairShareConfig {
    /// Enables cost-aware ordering. Disabled allocates no ledger.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub flow_key: FairShareFlowKey,
    /// Charge per uncached prompt token.
    #[serde(default = "default_prefill_token_weight")]
    pub prefill_token_weight: f64,
    /// Charge per generated token.
    #[serde(default = "default_decode_token_weight")]
    pub decode_token_weight: f64,
    /// Upper bound of one request's estimated or reconciled charge.
    #[serde(default = "default_max_charge_tokens")]
    pub max_charge_tokens: f64,
    /// Service difference treated as equal; ties use the ordinary order.
    #[serde(default = "default_band_tokens")]
    pub band_tokens: f64,
    /// Upper bound of retained idle flow accounts.
    #[serde(default = "default_max_tracked_flows")]
    pub max_tracked_flows: usize,
}

fn default_prefill_token_weight() -> f64 {
    1.0
}

fn default_decode_token_weight() -> f64 {
    1.0
}

fn default_max_charge_tokens() -> f64 {
    262_144.0
}

fn default_band_tokens() -> f64 {
    8_192.0
}

fn default_max_tracked_flows() -> usize {
    4096
}

impl Default for FairShareConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            flow_key: FairShareFlowKey::default(),
            prefill_token_weight: default_prefill_token_weight(),
            decode_token_weight: default_decode_token_weight(),
            max_charge_tokens: default_max_charge_tokens(),
            band_tokens: default_band_tokens(),
            max_tracked_flows: default_max_tracked_flows(),
        }
    }
}

impl FairShareConfig {
    /// Serialization keeps historical output when the feature is untouched.
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }

    /// Validate operator-supplied values.
    pub fn validate(&self) -> Result<(), String> {
        let weights = [
            ("prefill_token_weight", self.prefill_token_weight),
            ("decode_token_weight", self.decode_token_weight),
            ("band_tokens", self.band_tokens),
        ];
        for (field, value) in weights {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("fair_share.{field} must be finite and >= 0"));
            }
        }
        if self.prefill_token_weight == 0.0 && self.decode_token_weight == 0.0 {
            return Err("fair_share token weights must not both be 0".to_string());
        }
        if !self.max_charge_tokens.is_finite() || self.max_charge_tokens <= 0.0 {
            return Err("fair_share.max_charge_tokens must be finite and > 0".to_string());
        }
        if self.max_tracked_flows == 0 {
            return Err("fair_share.max_tracked_flows must be > 0".to_string());
        }
        Ok(())
    }

    fn flow_name<'a>(&self, identity: &'a ProgramIdentity) -> &'a str {
        match self.flow_key {
            FairShareFlowKey::TaskId => identity
                .task_id()
                .or_else(|| identity.root_program_id())
                .unwrap_or_else(|| identity.program_id()),
            FairShareFlowKey::RootProgramId => identity
                .root_program_id()
                .unwrap_or_else(|| identity.program_id()),
            FairShareFlowKey::ProgramId => identity.program_id(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct FlowAccount {
    virtual_service: f64,
    in_flight_charges: usize,
}

#[derive(Debug, Clone)]
struct PendingCharge {
    flow: Arc<str>,
    prefill_estimate: f64,
    decode_estimate: f64,
    shared_prefix_tokens: usize,
}

/// Estimated work of one dispatched request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FairShareEstimate {
    pub(crate) context_tokens: usize,
    pub(crate) shared_prefix_tokens: usize,
    pub(crate) output_tokens: usize,
}

/// Result of settling one provisional charge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FairShareSettlement {
    pub(crate) estimated_tokens: f64,
    pub(crate) actual_tokens: f64,
}

/// Bounded per-flow service ledger.
#[derive(Debug)]
pub(crate) struct FairShareLedger {
    config: FairShareConfig,
    flows: HashMap<Arc<str>, FlowAccount>,
    /// Non-decreasing service floor of backlogged and in-flight flows.
    floor: f64,
    /// Provisional charges by dispatch order; settled at most once.
    pending: BTreeMap<u64, PendingCharge>,
    next_charge_id: u64,
    evictions: u64,
    abandoned_charges: u64,
}

/// Bound on unsettled charges. Completion guards settle every dispatch, so
/// this only limits damage from a lost completion.
const MAX_PENDING_CHARGES: usize = 65_536;

impl FairShareLedger {
    pub(crate) fn new(config: FairShareConfig) -> Self {
        Self {
            config,
            flows: HashMap::new(),
            floor: 0.0,
            pending: BTreeMap::new(),
            next_charge_id: 0,
            evictions: 0,
            abandoned_charges: 0,
        }
    }

    /// Model-pool-scoped flow key. Pools never share capacity or service.
    pub(crate) fn flow_for(&self, identity: &ProgramIdentity) -> Arc<str> {
        let name = self.config.flow_name(identity);
        Arc::from(format!("{}\u{0}{}", identity.model_pool(), name))
    }

    /// A request arrived for `flow`: raise idle credit to the floor.
    pub(crate) fn observe_arrival(&mut self, flow: &Arc<str>) {
        if !self.flows.contains_key(flow) && self.flows.len() >= self.config.max_tracked_flows {
            // Free an eighth of the cap at once so eviction cost is amortized.
            let headroom = (self.config.max_tracked_flows / 8).max(1);
            self.prune();
            if self.flows.len() >= self.config.max_tracked_flows {
                self.evict_to(self.config.max_tracked_flows.saturating_sub(headroom));
            }
        }
        let floor = self.floor;
        let account = self.flows.entry(flow.clone()).or_insert(FlowAccount {
            virtual_service: floor,
            in_flight_charges: 0,
        });
        if account.virtual_service < floor {
            account.virtual_service = floor;
        }
    }

    /// Current charged service, or the floor for an untracked flow.
    pub(crate) fn virtual_service(&self, flow: &str) -> f64 {
        self.flows
            .get(flow)
            .map_or(self.floor, |account| account.virtual_service)
    }

    /// Total order of two flows; `Equal` defers to the ordinary order.
    ///
    /// A candidate without a flow ranks as an untracked flow on the floor, so
    /// the comparison stays a total preorder for `sort_by`.
    pub(crate) fn compare(&self, left: Option<&Arc<str>>, right: Option<&Arc<str>>) -> Ordering {
        if let (Some(left), Some(right)) = (left, right) {
            if Arc::ptr_eq(left, right) || left == right {
                return Ordering::Equal;
            }
        }
        let service =
            |flow: Option<&Arc<str>>| flow.map_or(self.floor, |flow| self.virtual_service(flow));
        let left = service(left);
        let right = service(right);
        if self.config.band_tokens > 0.0 {
            self.band(left).cmp(&self.band(right))
        } else {
            left.total_cmp(&right)
        }
    }

    fn band(&self, service: f64) -> u64 {
        // Saturating float-to-int casts keep the order total for every input.
        ((service - self.floor).max(0.0) / self.config.band_tokens).floor() as u64
    }

    /// Service lead of `flow` over the floor, in charged tokens.
    pub(crate) fn lag(&self, flow: &str) -> f64 {
        (self.virtual_service(flow) - self.floor).max(0.0)
    }

    /// Charge the estimated work of a dispatched request.
    pub(crate) fn charge(&mut self, flow: &Arc<str>, estimate: FairShareEstimate) -> u64 {
        let uncached = estimate
            .context_tokens
            .saturating_sub(estimate.shared_prefix_tokens) as f64;
        let prefill_estimate = uncached * self.config.prefill_token_weight;
        let decode_estimate = estimate.output_tokens as f64 * self.config.decode_token_weight;
        let charged = self.cap(prefill_estimate + decode_estimate);
        if self.pending.len() >= MAX_PENDING_CHARGES {
            self.abandon_oldest_pending();
        }
        self.observe_arrival(flow);
        let account = self
            .flows
            .get_mut(flow)
            .expect("observe_arrival inserts the flow");
        account.virtual_service += charged;
        account.in_flight_charges += 1;
        self.next_charge_id = self.next_charge_id.wrapping_add(1);
        let id = self.next_charge_id;
        self.pending.insert(
            id,
            PendingCharge {
                flow: flow.clone(),
                prefill_estimate,
                decode_estimate,
                shared_prefix_tokens: estimate.shared_prefix_tokens,
            },
        );
        id
    }

    /// Replace one provisional charge with reported usage, exactly once.
    ///
    /// Missing usage fields keep the corresponding estimate. An attempt that
    /// produced no output is refunded: Router retries are fresh arrivals, so
    /// this keeps a retried request charged once, and a client gains nothing
    /// from work it never received. A second call for the same charge is a
    /// no-op.
    pub(crate) fn settle(
        &mut self,
        id: u64,
        usage: &ProgramUsageObservation,
        produced_output: bool,
    ) -> Option<FairShareSettlement> {
        let pending = self.pending.remove(&id)?;
        let estimated = self.cap(pending.prefill_estimate + pending.decode_estimate);
        let prefill_actual = match usage.prompt_tokens {
            _ if !produced_output => 0.0,
            Some(prompt) => {
                let cached = usage
                    .cached_prompt_tokens
                    .unwrap_or(pending.shared_prefix_tokens)
                    .min(prompt);
                (prompt - cached) as f64 * self.config.prefill_token_weight
            }
            None => pending.prefill_estimate,
        };
        let decode_actual = match usage.completion_tokens {
            _ if !produced_output => 0.0,
            Some(tokens) => tokens as f64 * self.config.decode_token_weight,
            None => pending.decode_estimate,
        };
        let actual = self.cap(prefill_actual + decode_actual);
        let floor = self.floor;
        let account = self.flows.entry(pending.flow).or_insert(FlowAccount {
            virtual_service: floor + estimated,
            in_flight_charges: 1,
        });
        account.virtual_service = (account.virtual_service + actual - estimated).max(0.0);
        account.in_flight_charges = account.in_flight_charges.saturating_sub(1);
        Some(FairShareSettlement {
            estimated_tokens: estimated,
            actual_tokens: actual,
        })
    }

    /// Raise the floor to the least service among flows that have backlog or
    /// unsettled work. `backlogged` lists the flows of eligible waiters.
    pub(crate) fn advance_floor<'a>(&mut self, backlogged: impl IntoIterator<Item = &'a Arc<str>>) {
        let waiting = backlogged
            .into_iter()
            .map(|flow| self.virtual_service(flow));
        let in_flight = self
            .flows
            .values()
            .filter(|account| account.in_flight_charges > 0)
            .map(|account| account.virtual_service);
        if let Some(minimum) = waiting.chain(in_flight).min_by(f64::total_cmp) {
            if minimum > self.floor {
                self.floor = minimum;
            }
        }
    }

    /// Drop accounts that the floor would reset anyway. Lossless.
    pub(crate) fn prune(&mut self) {
        let floor = self.floor;
        self.flows
            .retain(|_, account| account.in_flight_charges > 0 || account.virtual_service > floor);
        self.evict_to(self.config.max_tracked_flows);
    }

    /// Evict idle accounts with the smallest unforgiven lead first.
    fn evict_to(&mut self, limit: usize) {
        if self.flows.len() <= limit {
            return;
        }
        let mut idle = self
            .flows
            .iter()
            .filter(|(_, account)| account.in_flight_charges == 0)
            .map(|(flow, account)| (account.virtual_service, flow.clone()))
            .collect::<Vec<_>>();
        idle.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let excess = self.flows.len() - limit;
        for (_, flow) in idle.into_iter().take(excess) {
            self.flows.remove(&flow);
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn abandon_oldest_pending(&mut self) {
        if let Some((_, pending)) = self.pending.pop_first() {
            if let Some(account) = self.flows.get_mut(&pending.flow) {
                account.in_flight_charges = account.in_flight_charges.saturating_sub(1);
            }
            self.abandoned_charges = self.abandoned_charges.saturating_add(1);
        }
    }

    fn cap(&self, tokens: f64) -> f64 {
        tokens.clamp(0.0, self.config.max_charge_tokens)
    }

    pub(crate) fn snapshot(&self) -> FairShareDiagnostic {
        let max_lag_tokens = self
            .flows
            .values()
            .map(|account| (account.virtual_service - self.floor).max(0.0))
            .fold(0.0, f64::max);
        FairShareDiagnostic {
            flow_key: self.config.flow_key,
            band_tokens: self.config.band_tokens,
            tracked_flows: self.flows.len(),
            pending_charges: self.pending.len(),
            floor_tokens: self.floor,
            max_lag_tokens,
            evictions_total: self.evictions,
            abandoned_charges_total: self.abandoned_charges,
        }
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn tracked_flows(&self) -> usize {
        self.flows.len()
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn pending_charges(&self) -> usize {
        self.pending.len()
    }
}

/// Aggregate fair-share state. Flow identifiers are never exposed.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FairShareDiagnostic {
    pub flow_key: FairShareFlowKey,
    pub band_tokens: f64,
    pub tracked_flows: usize,
    pub pending_charges: usize,
    pub floor_tokens: f64,
    pub max_lag_tokens: f64,
    pub evictions_total: u64,
    pub abandoned_charges_total: u64,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn flow(name: &str) -> Arc<str> {
        Arc::from(name)
    }

    fn estimate(context: usize, output: usize) -> FairShareEstimate {
        FairShareEstimate {
            context_tokens: context,
            shared_prefix_tokens: 0,
            output_tokens: output,
        }
    }

    fn usage(prompt: usize, cached: usize, completion: usize) -> ProgramUsageObservation {
        ProgramUsageObservation {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            cached_prompt_tokens: Some(cached),
        }
    }

    #[test]
    fn default_config_is_disabled_and_valid() {
        let config = FairShareConfig::default();
        assert!(!config.enabled);
        assert!(config.is_default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn invalid_values_are_rejected() {
        for config in [
            FairShareConfig {
                prefill_token_weight: -1.0,
                ..Default::default()
            },
            FairShareConfig {
                decode_token_weight: f64::NAN,
                ..Default::default()
            },
            FairShareConfig {
                prefill_token_weight: 0.0,
                decode_token_weight: 0.0,
                ..Default::default()
            },
            FairShareConfig {
                max_charge_tokens: 0.0,
                ..Default::default()
            },
            FairShareConfig {
                band_tokens: f64::INFINITY,
                ..Default::default()
            },
            FairShareConfig {
                max_tracked_flows: 0,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?}");
        }
        assert!(serde_json::from_str::<FairShareConfig>(r#"{"enabld":true}"#).is_err());
    }

    #[test]
    fn cheaper_flow_orders_first_and_equal_service_defers() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            band_tokens: 0.0,
            ..Default::default()
        });
        let heavy = flow("heavy");
        let light = flow("light");
        ledger.observe_arrival(&heavy);
        ledger.observe_arrival(&light);
        assert_eq!(ledger.compare(Some(&heavy), Some(&light)), Ordering::Equal);
        ledger.charge(&heavy, estimate(32_000, 1_000));
        ledger.charge(&light, estimate(2_000, 128));
        assert_eq!(ledger.compare(Some(&light), Some(&heavy)), Ordering::Less);
        assert_eq!(ledger.compare(Some(&heavy), Some(&heavy)), Ordering::Equal);
        // A flow-less candidate sits on the floor: ahead of served flows.
        assert_eq!(ledger.compare(None, Some(&heavy)), Ordering::Less);
        assert_eq!(ledger.compare(None, None), Ordering::Equal);
    }

    #[test]
    fn comparison_is_a_total_preorder_for_sorting() {
        for band_tokens in [0.0, 1_000.0] {
            let mut ledger = FairShareLedger::new(FairShareConfig {
                band_tokens,
                ..Default::default()
            });
            let flows = (0..64).map(|i| flow(&format!("f{i}"))).collect::<Vec<_>>();
            for (i, f) in flows.iter().enumerate() {
                ledger.charge(f, estimate((i * 7_919) % 5_000, 0));
            }
            ledger.advance_floor(flows.iter().take(8));
            let mut candidates = flows
                .iter()
                .map(Some)
                .chain(std::iter::repeat_n(None, 8))
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| ledger.compare(*left, *right));
            for pair in candidates.windows(2) {
                assert_ne!(ledger.compare(pair[0], pair[1]), Ordering::Greater);
            }
        }
    }

    #[test]
    fn band_treats_small_differences_as_ties() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            band_tokens: 10_000.0,
            ..Default::default()
        });
        let a = flow("a");
        let b = flow("b");
        ledger.charge(&a, estimate(4_000, 0));
        ledger.charge(&b, estimate(9_000, 0));
        assert_eq!(ledger.compare(Some(&a), Some(&b)), Ordering::Equal);
        ledger.charge(&b, estimate(9_000, 0));
        assert_eq!(ledger.compare(Some(&a), Some(&b)), Ordering::Less);
    }

    #[test]
    fn settlement_replaces_estimate_exactly_once() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            decode_token_weight: 2.0,
            ..Default::default()
        });
        let a = flow("a");
        let id = ledger.charge(&a, estimate(1_000, 100));
        assert_eq!(ledger.virtual_service(&a), 1_200.0);
        let settled = ledger.settle(id, &usage(1_000, 600, 50), true).unwrap();
        assert_eq!(settled.estimated_tokens, 1_200.0);
        assert_eq!(settled.actual_tokens, 500.0);
        assert_eq!(ledger.virtual_service(&a), 500.0);
        assert!(ledger.settle(id, &usage(9_999, 0, 9_999), true).is_none());
        assert_eq!(ledger.virtual_service(&a), 500.0);
        assert_eq!(ledger.pending_charges(), 0);
    }

    #[test]
    fn missing_usage_keeps_the_estimate() {
        let mut ledger = FairShareLedger::new(FairShareConfig::default());
        let a = flow("a");
        let id = ledger.charge(
            &a,
            FairShareEstimate {
                context_tokens: 1_000,
                shared_prefix_tokens: 400,
                output_tokens: 100,
            },
        );
        ledger
            .settle(id, &ProgramUsageObservation::default(), true)
            .unwrap();
        assert_eq!(ledger.virtual_service(&a), 700.0);
        // A prompt without cache evidence assumes the estimated shared prefix,
        // and missing completion tokens keep the decode estimate.
        let id = ledger.charge(
            &a,
            FairShareEstimate {
                context_tokens: 0,
                shared_prefix_tokens: 400,
                output_tokens: 100,
            },
        );
        ledger
            .settle(id, &ProgramUsageObservation::from(Some(1_000)), true)
            .unwrap();
        assert_eq!(ledger.virtual_service(&a), 700.0 + 600.0 + 100.0);
    }

    #[test]
    fn attempt_without_output_is_refunded() {
        let mut ledger = FairShareLedger::new(FairShareConfig::default());
        let a = flow("a");
        let id = ledger.charge(&a, estimate(10_000, 100));
        assert_eq!(ledger.virtual_service(&a), 10_100.0);
        let settled = ledger
            .settle(id, &ProgramUsageObservation::default(), false)
            .unwrap();
        assert_eq!(settled.actual_tokens, 0.0);
        assert_eq!(ledger.virtual_service(&a), 0.0);
        assert_eq!(ledger.pending_charges(), 0);
    }

    #[test]
    fn charges_are_capped() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            max_charge_tokens: 10_000.0,
            ..Default::default()
        });
        let a = flow("a");
        let id = ledger.charge(&a, estimate(1_000_000, 0));
        assert_eq!(ledger.virtual_service(&a), 10_000.0);
        ledger.settle(id, &usage(5_000_000, 0, 0), true).unwrap();
        assert_eq!(ledger.virtual_service(&a), 10_000.0);
    }

    #[test]
    fn idle_credit_is_not_banked() {
        let mut ledger = FairShareLedger::new(FairShareConfig::default());
        let busy = flow("busy");
        let idle = flow("idle");
        let id = ledger.charge(&busy, estimate(50_000, 0));
        ledger.settle(id, &usage(50_000, 0, 0), true).unwrap();
        ledger.advance_floor([&busy]);
        assert_eq!(ledger.lag(&busy), 0.0);
        ledger.observe_arrival(&idle);
        assert_eq!(ledger.virtual_service(&idle), 50_000.0);
        assert_eq!(ledger.compare(Some(&idle), Some(&busy)), Ordering::Equal);
    }

    #[test]
    fn floor_never_decreases_and_tracks_in_flight_work() {
        let mut ledger = FairShareLedger::new(FairShareConfig::default());
        let a = flow("a");
        let b = flow("b");
        ledger.charge(&a, estimate(100, 0));
        let id = ledger.charge(&b, estimate(300, 0));
        ledger.advance_floor(std::iter::empty());
        assert_eq!(ledger.snapshot().floor_tokens, 100.0);
        ledger.settle(id, &usage(0, 0, 0), true).unwrap();
        ledger.advance_floor(std::iter::empty());
        assert_eq!(ledger.snapshot().floor_tokens, 100.0);
    }

    #[test]
    fn prune_is_lossless_and_cap_evicts_smallest_lead() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            max_tracked_flows: 2,
            ..Default::default()
        });
        let a = flow("a");
        let b = flow("b");
        let c = flow("c");
        for (f, tokens) in [(&a, 10), (&b, 20)] {
            let id = ledger.charge(f, estimate(tokens, 0));
            ledger.settle(id, &usage(tokens, 0, 0), true).unwrap();
        }
        ledger.advance_floor([&a]);
        ledger.prune();
        // `a` sits on the floor and is dropped; re-arrival restores the same value.
        assert_eq!(ledger.tracked_flows(), 1);
        assert_eq!(ledger.virtual_service(&a), 10.0);
        ledger.observe_arrival(&a);
        // Admitting `c` first prunes `a` losslessly instead of evicting `b`.
        ledger.observe_arrival(&c);
        assert_eq!(ledger.tracked_flows(), 2);
        assert_eq!(ledger.snapshot().evictions_total, 0);
        assert_eq!(ledger.virtual_service(&b), 20.0);

        // With every account ahead of the floor, the smallest lead is forgiven.
        let mut ledger = FairShareLedger::new(FairShareConfig {
            max_tracked_flows: 2,
            ..Default::default()
        });
        for (f, tokens) in [(&a, 10), (&b, 20)] {
            let id = ledger.charge(f, estimate(tokens, 0));
            ledger.settle(id, &usage(tokens, 0, 0), true).unwrap();
        }
        ledger.observe_arrival(&c);
        assert_eq!(ledger.tracked_flows(), 2);
        assert_eq!(ledger.snapshot().evictions_total, 1);
        assert_eq!(ledger.virtual_service(&a), 0.0);
        assert_eq!(ledger.virtual_service(&b), 20.0);
    }

    #[test]
    fn in_flight_accounts_are_never_evicted() {
        let mut ledger = FairShareLedger::new(FairShareConfig {
            max_tracked_flows: 1,
            ..Default::default()
        });
        let a = flow("a");
        let b = flow("b");
        let id = ledger.charge(&a, estimate(10, 0));
        ledger.charge(&b, estimate(10, 0));
        assert_eq!(ledger.tracked_flows(), 2);
        assert!(ledger.settle(id, &usage(10, 0, 0), true).is_some());
        assert_eq!(ledger.virtual_service(&a), 10.0);
    }
}
