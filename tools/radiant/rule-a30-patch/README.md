# Rule A30: an offline prototype of a bend rule

This is an offline experiment and is **not** part of the submitted task: `tasks/radiant-floor-layout` is
unchanged. It asks whether one installability rule could rule out the layouts GPT-6.1 used to solve v2.1
(see `../autopsy/`) while the reference solver still passes every plan.

Everything here ran on the host CPU: no containers, no model usage, no agent trials.

## The rules tested

A *piece* is a straight run between two bends, the same thing the checker's `min_segment` (0.10 m)
limits.

| Rule | Definition |
|---|---|
| A | A piece shorter than 0.20 m must be a U-turn end-cap: both bends turn the same way, both neighbouring pieces are at least 0.40 m, and it is not in the hall. |
| **A30** | Rule A with neighbouring pieces of at least 0.30 m. |
| B | No piece shorter than 0.20 m anywhere (`min_segment` 0.20 m). |

The physical basis is a minimum bend radius. A 16 mm floor-heating pipe bends to roughly 5 x its outside
diameter, about 80 mm. Two bends closer than about 0.16 m are only possible as the return of a parallel
run (a U-turn), not as an S-jog.

## Results on the task's 15 plans (12 held-out + 3 samples)

Reference solver adapted to each rule, judged by the task's checker plus the rule (`gate.py`):

| Reference variant | Patch | Result |
|---|---|---|
| Rule A, smallest change (hall lane shifts of at least 0.20 m) | one line, see `results/history/rule-a-0.40m-minimal.txt` | 2/15: on 8 plans every pitch leaves a finger, so no layout is valid |
| Rule A, plus pruning of one-block fingers | `patches/rule-a-0.40m.patch` | 3/15: bath and kitchen fall short of heat once their fingers go |
| Rule B (all pitches at least 0.20 m, lane shifts of at least 0.20 m) | `patches/rule-b.patch` | 0/15: bath, kitchen and study 13-60 W short of heat |
| **Rule A30** (finger pruning + 0.20 m lane shifts + lane shifts 0.30 m clear of doorways + entries clear of their port) | `patches/rule-a30.patch` | **15/15**, slowest 4.2 s, median 3.5 s (`results/rule-a30.txt`) |

- Rule A30's margins equal the unmodified reference's on 10 of the 12 held-out plans. case-07's tightest
  heat margin drops from 17.4 W to 4.7 W, and case-10's spread margin from 0.30 K to 0.19 K.
- Without load, the Rule A30 version is as fast as the unmodified reference (median 3.7 s vs 4.0 s
  over 14 plans).

Rule A30 violations in layouts regenerated from GPT-6.1's saved v2.1 solver, the copy that passed all 12
held-out plans (`results/layout-rule-counts.txt`):

| Layouts | Rule A30 violations (12 held-out plans) | Plans with none |
|---|---|---|
| GPT-6.1 v2.1 solver | 1,204: 847 in the hall, 273 room S-steps, 84 U-turns with a short side | 0/12 |
| Unmodified v2.1 reference | 26: 23 hall lane shifts, 3 U-turns with a short side | 3/12 |

## What this does not show

- **No container load gate.** On this Windows host, both the unmodified and the Rule A30 reference
  occasionally lose one plan to an over-15 s stall (`results/history/`). Under 12 CPU burners, sample-02
  is slow for both (57 s with the rule, 39 s without, under a profiler). The verifier-style gate in 2-CPU
  containers has not been run.
- **No agent run.** GPT-6.1's saved solver breaks the rule, but a fresh run would be told the rule and
  would adapt. Hall meanders built from 0.20 m pieces stay legal (the reference's own hall-end serpentines
  use 0.25 m rows), so the rule makes the hall heat dump coarser rather than impossible.
- **Only the GPT solver was scored.** The saved Opus solvers have not been scored against the rule.
- **Not adopted.** The task's checker and `instruction.md` do not contain the rule.
- **Built on v2.1.** The patches are against the v2.1 reference (commit 05f9827, the 15 s version), not the
  v2 reference in the task directory.

## Files and commands

- `rules.py`: the rules, and `evaluate()`, the task's checker plus a rule.
- `solver/`: the Rule A30 reference; `solve.py PLAN LAYOUT [RULE]`.
- `gate.py SOLVE_PY RULE TAG`: runs a solver on the 15 plans and judges each layout.
- `score_layouts.py`: rule violations in `../autopsy/layouts/`.
- `patches/`: the three reference variants as diffs against the v2.1 reference.
- `results/`: the gate logs. `rule-a30.txt` is the run above; `history/` holds the earlier passes.

```bash
python tools/radiant/rule-a30-patch/gate.py tools/radiant/rule-a30-patch/solver/solve.py A30 rule-a30
python tools/radiant/rule-a30-patch/score_layouts.py
```
