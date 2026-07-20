# MORNING BRIEF — 20 July 2026 (overnight autonomous window)

## ⚡ ROUND-2 RESULTS (landed ~06:50, all six arms; read through the
## audit's instrument rules)

- **Replicated and real**: M1 clean 0.88, M3 clean 0.93, M5 deliveries
  MEET again. Compulsion persists (M5 cmd rate 0.429 vs 0.476; M4 now
  fires its honest HARD MISS — through-flight instructed).
- **M1 ordering after 2400 eps: 0.29 (init) → 0.02 (final), last-4
  mean −0.07.** Formally no-verdict (instrument), but the direction is
  now consistent across rounds: training under this recipe ERASES the
  ordering random init provides. That is a finding about the training
  dynamics, not the instrument.
- **The decisive M2 result — the pure-MC arm FAILED its pre-registered
  success criterion**: Q(NOOP) did not converge to the measured
  contamination floor (mean gap 13.4; prices −5..−38 at states whose
  floor is −1..−2). And the single-knob contrast is null: m2b ≈ m2a
  everywhere. With bootstrap tails REMOVED from cf rows and ~7,100
  funded probe states per arm (telemetry matches the audit's cost
  model exactly), NOOP stays doom-priced. **Conclusion: the doom
  equilibrium is maintained by the LIVE side, not the cf targets** —
  the audit's flagged round-3 candidates are now the confirmed
  mechanism set: censored-window bootstraps recycling Q(NOOP) doom,
  stale terminal rows at 37.5% of every batch, and the commanding
  compulsion feeding real LoS into live data. CF-replay's corrective
  signal loses that tug-of-war regardless of its own purity.
- Round-3 shape this implies (design pass first, per your process):
  fix the live-side doom recyclers + the compulsion, with the
  (already-proven-clean) CF machinery as the grounding source, and the
  rebuilt instruments as the eyes. No further training runs before the
  instrument rebuild — nothing rho-shaped is readable until then.

Everything below is measured, audited, and recorded; nothing was
escalated. No full run launched. Decision list for you at the bottom.

## The two findings that change how everything reads

1. **The M2 ordering bar was unreachable in principle.** A perfect
   learner of what we actually train the network on (frozen-policy
   futures, your D1 ruling) can score at most ~0.18 on the bar we
   grade with (do-nothing futures) — because the two "ground truths"
   only agree at 0.18 on these states. The 0.4 bar could never have
   been passed by ANY network under this training basis. This is the
   design audit's A1 circularity warning, now measured
   (audit_measurements/basis_ceiling.json).
2. **Every ordering instrument in the battery is currently unreadable.**
   Freshly initialized random networks score anywhere from −0.6 to
   +0.6/+0.85 on these probe sets (they are 20 timestamps of ONE
   trajectory — effective sample size ~1). The 0.4 bar sits INSIDE the
   random band on both M1 and M3, as does every trained value ever
   recorded. Nothing rho-shaped from round 1 OR round 2 can certify or
   refute anything until the instrument is rebuilt
   (audit_measurements/null_bands.json). Width is exonerated (c-sweep:
   same story at c=0, widths stable).

## The M2 contamination, now fully measured (cf_decomposition_by_ckpt)

The do-nothing lessons were manufactured negative by three mechanisms,
all now quantified per checkpoint on the production code path:
- REAL collisions inside what-if playouts: the network's trigger-happy
  continuation re-levels the safe pair (9/20 states violating at
  ep800; mean recorded lesson −14.8 where truth is 0).
- PHANTOM events past episode end: training playouts are uncapped while
  grading is capped (6/20 states priced on post-episode events).
- Bootstrap-tail doom: γ³⁶ ≈ 0.334 of the network's own −30-ish tail
  re-injected into otherwise-clean lessons (−2..−16).
The m2b (pure-observed-targets) arm's success criterion is
PRE-REGISTERED (puremc_floor.json): valuations converging to the
measured floor (−1..−2 early, −18..−46 at violating states), NOT to
zero. Its miss must not be read as "counterfactual grounding fails."

## Honest round-1 scoreboard (post-audit)

- Real positives: M1 clean 0.88, M3 clean 0.90, M5 deliveries 0.90
  (survived refutation — first delivery pathway success ever).
- No-verdict (instrument-limited): M1 rho, M3 rho — and now, by the
  ceiling result, all M2 rho readings.
- Real misses: all of M2 (mechanism fully understood, above); M4
  re-scored FULL MISS (first-command bar = coin flip at ~0.65 chance;
  policy commanded the protected through-flight 10/60 times, evidence
  in m4_forensics.json); M5 command rate 0.476 vs 0.2 (the compulsion
  is cross-cutting: it drives M2's playout collisions and M4's
  harassment).
- Round 1 licenses NEITHER "counterfactual grounding works" NOR "it
  fails." The mechanism is proven sound end-to-end (RA-CF exact
  identities); its learning effect is unmeasured pending instruments.

## Running / done overnight

- Round-2 arms running: M1 2400, M3 2400, M4 1800, M5 1200
  (pre-registered), M2a floor-0.5, M2b floor-0.5+pure-MC (exploratory,
  restarted with collision-proof output tags after the audit caught
  both arms clobbering each other's files — pid now in every tag).
  NOTE for reading M2a: the k=2 flag is silently clamped to 3
  (CF_K_FLOOR), so it funds ~5.6 states/ep vs ~51 demanded — it tests
  "modest quiet-state coverage," not "floor 0.5."
- M4 arms save no checkpoints (rho_fn=None path) — round-2 M4 will
  also end checkpoint-less; fix queued for round 3.
- All six overnight audit measurements complete in audit_measurements/
  (TONIGHT_REPORT.md is the plain-English summary).

## Your decision list (with my recommendations)

D-i.  Ordering-instrument rebuild (BLOCKS everything rho): probes from
      many independent scenarios/seeds; every instrument ships with its
      random-network null band; dual-basis GT (report against both the
      training basis and the do-nothing basis); drop rho_mid (it is
      algebraically rho at c=0.5). REC: approve; this is round 3's
      first work item, before any new training.
D-ii. M2 target-basis fix (BLOCKS M2 meaning): cap training playouts at
      episode end (bug-level, uncontroversial) + for the do-nothing
      probe row specifically, use a non-self-destructive continuation
      (do-nothing or scripted-safe) so its lesson prices the QUESTION
      being asked. REC: approve both; the frozen-policy basis stays
      for command candidates (your D1 intent preserved).
D-iii. M4 bar redesign: replace the coin-flip first-command bar with
      "any through-flight command in the greedy episode = FAIL" (the
      A15 intent) + multi-episode first-intervention test. REC:
      approve.
D-iv. K-floor clamp: expose real k=2 (cuts per-state cost ~33%) or
      accept clamp in budget math. REC: expose it.
D-v.  Config echo in every summary JSON + persist target_net in
      checkpoints (two bookkeeping gaps that cost us tonight). REC:
      approve.
D-vi. Round-3 structural candidates (flagged, not designed): stale
      terminal rows are 37.5% of every batch; censored-window
      bootstraps recycle Q(NOOP) doom; the commanding compulsion as a
      cross-cutting target. REC: run these through a design pass
      before building anything.
- Standing guardrail affirmed: pure-MC targets stay OFF conflict
  scenarios (would truncate real −50 tails — optimistic exactly where
  dangerous).
