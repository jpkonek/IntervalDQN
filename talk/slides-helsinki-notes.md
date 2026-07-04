# slides-helsinki — opening notes

Deck is 42 slides. The Helsinki revisions vs. `slides.html`:

- Title slide now reads "Imprecision in control problems / Jason Konek
  / Beyond Basic Bayesianism / 26 May 2026 / University of Helsinki".
- `$D$-propriety` and `Complete class theorem (Wald, SSK)` have been
  swapped: $D$-propriety is now slide 12, Complete class theorem is
  slide 13.
- Removed: Bayes-optimal sets of gambles (proposition + picture
  slides), Coherence-preserving maps (Godzilla), Existence:
  piecewise-linear CPMs, Existence: signed-perspective CPMs.
- Inserted, after the Section III divider ("Role in control problems",
  slide 15): two motivation slides (16, 17) drawing on Douven (2025).
- Subsequent slides shift accordingly to /42 total.

The two new slides are inline SVG with the existing robot, so no asset
dependencies.

---

## Slide 16 — Ecological rationality

**SVG:** the robot walking through a jungle, a tiger emerging from the
left foliage facing the robot, ground vegetation, hanging vines,
canopy at top. Robot retains the wide nervous eyes and squiggle mouth.

**On-slide text:**

> Cognitive strategies should be evaluated by their success in helping
> people (or even nonhuman animals) achieve their goals (including
> epistemic goals) within specific contexts.

**What to say:**

- Open with Douven's statement of the ecological rationality
  perspective.
- Contrast with the universalist tradition (probability axioms, Bayes'
  rule as norms applying regardless of context).
- Standard defenses of precise probabilism — dynamic Dutch books,
  accuracy domination — presuppose agents evaluating arbitrary
  combinations of bets across time, or computing expected inaccuracy
  for all possible future credences. Capacities ordinary agents
  cannot even approximate.
- Pivot: we take the ecological perspective seriously and ask what
  imprecise probability contributes to it.

## Slide 17 — The strategy selection problem

**SVG:** the robot reaching toward an open red toolbox containing
hammer, screwdriver, wrench, saw; a thought bubble above the robot
contains a question mark.

**On-slide text:**

> How does a cognitive system identify which tool is appropriate for
> which task?

**What to say:**

- Once the ecological view licenses different rules in different
  environments, the operational question is which rule to use here.
- Cite Rieskamp & Otto (2006) on strategy selection; Gigerenzer's
  "adaptive toolbox"; Douven's "dependency problem" — the agent often
  has to learn about the environment by using a rule, but choosing the
  right rule is what knowing about the environment would tell her.
- IP loss functions plus a coverage-tracking mechanism are *part* of
  the solution. The credal interval DQN does not yet wire in the
  error-loss / propriety apparatus — flagged later on the "Credal DQN
  Generalized" slide.

---

## Open framing decisions

- Plan slide (slide 3) is unchanged from `slides.html`. Consider
  whether it should reflect the new framing.
- Whether to add a closing "ecological-rationality lessons" slide near
  the take-home remains open.
