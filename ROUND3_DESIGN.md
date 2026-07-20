# ROUND 3 DESIGN — FOR ADVERSARIAL REVIEW (20 July 2026)

## ⚠ REVIEW OUTCOME (20 July — full text in ROUND3_REVIEW.md)

25 agents; every section AMEND; the amended design REPLACES the drafts
below where they conflict. Headlines: (1) A2's drafted mechanism is
replaced wholesale — recovery becomes a LOGGED WIN EVENT with credit
flowing through (NO segmentation by default; the drafted anchor-0
terminal states a provably false value, 0..+22 measured); W_rec must
sit at or below the audited pump cost (~0.3-1.0/cycle — my drafted
W_rec=5 would have PAID a pumper +4/cycle; recommended 0 or ≤0.1);
guards (i)/(ii) dropped (vacuous / bypassed by clearance persistence /
disarm exactly the dominant endogenous-onset cases) — the binding
guard is the W_rec bound + two-direction pump audit, which gates any
W_rec > 0 and JK reads it. (2) C rebuilt around attribution: arm-1
read at the A1+B-only checkpoint BEFORE A2 exists; W_rec=0 variant as
the clean separator; pre-registered noise floor on "command rate
falls"; a third outcome pre-committed (NOOP losing the argmax lottery
on prediction noise → tie-break tolerance, NOT a lever). (3) B1
becomes EXCLUDE-not-cap (~36% late-state coverage cost at nstep=36,
pre-registered). (4) B2 requires a class-aware buffer schema
(terminal_kind tag) BEFORE any win rows exist, else wins capture the
3× boost built for sparse crashes and dilute violations ~16×.
(5) New M6 recovery instrument mandatory (rate, time-in-band, armed-
fraction telemetry — the round trains a behavior nothing measured).
(6) A1 completion check specified (roster-based, AFTER the violation
branch — the OUT_SECTOR same-step corner). (7) Amended 10-step build
order in ROUND3_REVIEW.md; six JK decision items pending.

## ✅ JK RULINGS COMPLETE (20 July) — governing design, supersedes
## conflicting drafts below AND the review's W_rec=0 recommendation

1. WINS ARE PAID (W_rec > 0, real reward): a label alone influences
   nothing. Pump immunity comes from PATH-DEPENDENCE, not from zeroing
   the reward: a resolution pays ONLY if the danger was PRE-EXISTING.
   Operationalization: every command's issuance-time safety payment
   (the live CBP conflict term, already computed at lag 1) signs
   whether the controller endangered the pair; band entries downstream
   of own negative-signed commands (decaying flag per pair) are
   UNARMED — resolving self-made danger pays nothing, and making it
   already costs. Needless flirting with the band = loss only (conflict
   shaping + delivery decay), never gain. Two-direction pump audit
   still gates W_rec magnitude and must specifically attack the
   pre-existence test (persistent-clearance causation, flag decay
   edges).
2. SUB-EPISODES: env episodes run through recoveries; the REWARD
   ACCOUNTING is segmented — each danger→recovery arc is a sub-episode
   whose learning windows collapse at the recovery boundary with W_rec
   as the arc's ending; cross-boundary expectation uses the review's
   floored bootstrap (realized + gamma^m * max(Q_target, 0)), never a
   bare 0.
3. SAFETY PRICING STAYS (heuristic, not hard rule): charging for
   commands is empirically motivated in ATC — controllers leave flying
   to pilots except where intervention is necessary. The requirement is
   that fees/win-payments must not be the OVERRIDING driver: the system
   must GENUINELY LEARN that restraint pays where it does. The
   attribution instrumentation (C arm-1 at A1+B-only checkpoint,
   W_rec=0 diagnostic arm, belief-vs-payment decomposition) therefore
   stays — as science, not as a compliance gate.
4. TIE-BREAK RULE APPROVED: "do nothing" is the genuine ATC default.
   Build a NOOP-preference tolerance into action selection (select
   NOOP unless the best candidate beats it by more than a threshold at
   fee scale, draft 0.1) as part of round 3 (section C2), not merely
   as a contingency. Re-review must attack: threshold choice,
   interaction with exploration/Hurwicz, and whether it can mask
   learning failures (instrument: log tolerance-invoked selections
   separately).
5. GT cadence as proposed (NOOP-basis full cadence; training-basis at
   4 checkpoints). 6. Pre-registration constants as proposed in
   ROUND3_REVIEW.md.

Status: DRAFT. Nothing here is built. JK rulings incorporated:
recovery-to-baseline is a WIN with a known value (20 July); belief
repair before any new command-penalty lever; adversarial design review
gates the build.

Evidence base: round-1/2 battery + audits (MORNING_BRIEF_20JUL.md).
Confirmed mechanism: the doom equilibrium is maintained by the LIVE
training side — (i) time-limit episode endings bootstrap on the net's
own pessimistic guess (self-feeding); (ii) stale terminal −50 rows are
37.5% of every batch; (iii) the commanding compulsion manufactures
real LoS that tops up the pessimism — while CF-replay's corrective
lessons (proven exactly priced; RA-CF 0.0 error) lose the tug-of-war.
Q(NOOP) on safe states reads −5..−38 vs truth ~0 in BOTH round-2 M2
arms (pure-MC contrast null).

## A. Win anchors (JK ruling — the structural fix)

The asymmetry to remove: crashes anchor on realized −50; successes
anchor on the net's own guess. LL-style repair: give success a true
terminal with KNOWN value.

- A1 (micro): all-aircraft-delivered = TRUE COMPLETION terminal.
  Remaining episode is provably empty; terminal value 0 (delivery
  bonuses already paid as realized events). Windows reaching it use
  realized returns, no bootstrap. Uncontroversial; implement first.
- A2 (full sector): RECOVERY-AS-WIN. Definition (draft, attack it):
  a risk episode begins when any vertically-proximate pair enters the
  conflict band (min lateral sep < 10 nm, |dFL| < 20 — stratum-0
  geometry); it RESOLVES when, for K consecutive steps (draft K=10),
  no pair is inside the band AND every aircraft is re-established on
  course (centreline offset < tolerance, draft 2 nm). On resolution:
  SEGMENT TERMINAL — the training episode ends there as a WIN with
  known anchor (realized returns through the terminal; win bonus
  +W_rec, draft +5, magnitude for review), and a NEW training episode
  continues from the live state (env not reset; segmentation is a
  training-bookkeeping construct, not an env change).
- A3 anti-farming (the obvious exploit — review must attack): an agent
  could CREATE risk then resolve it to harvest +W_rec. Draft guards:
  (i) a risk episode only arms the win if its onset was exogenous —
  no own-command within the preceding L steps (draft L=6) was issued
  to either member of the first in-band pair; (ii) at most one armed
  win per pair per training episode; (iii) W_rec small relative to
  −50 so cause-and-fix is strictly unprofitable even if guards leak
  (worst-case pump audit required, RA-style, before any run).
- A4 time-limit endings that remain (no completion, no recovery):
  still censored; bootstrap stays BUT its weight shrinks (fewer such
  windows once segmentation exists). Option for review: bootstrap
  from the TARGET net's value at the SAFEST candidate rather than
  a_cont, or cap the bootstrap magnitude. Pick one in review.

## B. Live-side pessimism repairs (mechanical)

- B1: cap CF branch windows at the episode end (bug-level; the eval
  side is already capped; training was not).
- B2: terminal-replay rebalance: age-cap stale −50 rows (draft: evict
  or down-weight terminal rows older than 400 episodes, mirroring the
  cf age cap) and re-derive the boost target so terminal rows are
  ≤ 20% of a batch (draft; review the number against the original
  terminal-boost rationale — sparse-terminal learning still needs
  them).
- B3: flush_censored doom recycling: with A1/A2 in place, re-measure;
  only add further mechanism if the residual is material (pre-register
  the measurement, not a fix).

## C. Compulsion — explicitly NO new lever this round

Prior attempts (fee 0.1, tax 0.3, change-penalty) were price levers
against hallucinated −10..−30 NOOP values; all failed structurally.
Round 3 repairs the values instead (A+B) and PRE-REGISTERS the
discriminating prediction: after A+B (+ CF NOOP lessons, unchanged),
Q(NOOP) on safe states rises toward 0 and command rate falls with NO
new penalty. If Q(NOOP) recovers but command rate does not fall,
compulsion is a genuine preference → THEN design a dedicated lever
(that failure mode and threshold to be fixed in review).

## D. Instruments and bookkeeping (build alongside, review the specs)

- D1: ordering-instrument rebuild: probe sets drawn from N ≥ 8
  independent scenario seeds/geometries per test; every instrument
  ships its untrained-null band (30 inits) and is only read against
  it; dual-basis GT (training basis + NOOP basis) reported side by
  side; rho at c ∈ {0, c_train, 1}; rho_mid dropped (algebraic
  duplicate at c=0.5).
- D2: config echo (all cf/pricing/nstep flags) in every summary JSON;
  persist target_net in checkpoints; M4 arms save checkpoints (fix the
  rho_fn=None path).
- D3: M4 bar redesign: hard-fail = ANY through-flight command in the
  greedy episode; first-intervention test over ≥ 10 probe episodes
  (chance ~0.667^10); command-discipline bar unchanged.
- D4: expose real cf_k (remove the silent CF_K_FLOOR clamp via CLI
  override; budget math updated).

## E. Sequencing after review

1. Apply review amendments; JK sees verdicts + any REJECTs.
2. Build order: D (instruments first — nothing is readable without
   them) → B1 → A1 → B2 → A2/A3 (biggest surface, reviewed guards)
   → RA-style pump audit for A2 → battery rerun (corrected bars).
3. Full-run decision returns to JK only after a battery pass on
   readable instruments.
