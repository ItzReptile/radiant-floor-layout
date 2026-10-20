# Autopsy: how GPT-6.1 solved radiant-floor-layout v2.1

GPT-6.1 Sol (xhigh, codex) solved v2.1, the 15 s-per-plan version, in Codex Trial 1. Its solver passed
all 12 held-out plans at the read-only check at 21:31, about 5 h 15 min into the run, and passed 12/12 again
on a re-run of the saved copy. Trial records: `trials/radiant-v2.1/gpt-run1/` and
`tools/radiant/results/v21-trial-gpt-1.txt`.

This folder is a read-only analysis of that solver. The layouts were regenerated on the host CPU from the
saved solver (`trials/radiant-v2.1/gpt-run1/solver-at-12of12/`) and from the v2.1 reference. No containers
and no model usage. The solver is time-budgeted, so exact watts differ slightly from run to run; across six
host runs, case-05's study ranged 155.0-155.6 W and always passed.

## 1. What changed in the last 35 minutes

At 20:56 the solver passed 10/12. case-05's study was 4 W over its cap and case-11's hall 1 W short.
`late-edits/` has the diffs between the solver copies taken at 20:56, 21:21 and 21:31:

- **20:58-21:09:** a penalty for unbalancing the two halves of a room fed by two loops; a 0.12 W margin
  inside room heat limits; more time reserved for the final local-adjustment stage.
- **21:18, the decisive change:** a faster solver for the same heat equations. It eliminates the water
  chains, then solves the two floor layers with BiCGSTAB preconditioned by a DCT, warm-started from the
  last solution, and falls back to the direct solver unless the residual is under 1e-9.
- **21:20-21:31:** pipe moves near a shared wall are ranked by how much of their heat reaches the target
  room, and near-miss rooms are left to the final stage.

**Speed did it, not a new layout idea.** The 20:56 copy, the one that failed case-05 in the verifier,
passes case-05 and case-11 on the host, twice each. It needs 12.6-13.3 s against its own 14.3 s cutoff,
so on the verifier's 2 CPUs it ran out of time with the study still at 160 W. The 21:31 copy's first heat
solve takes 1.51 s instead of 1.86 s, and it reaches a clean case-05 at 7.97 s.

## 2. GPT-6.1's layouts against the reference

`metrics.py` produces these from `layouts/`; `metrics.json` has every number.

| Totals over the 12 held-out plans | GPT-6.1 v2.1 solver | v2.1 reference |
|---|---|---|
| Bends | 4,889 | 4,690 |
| Straight pieces of 0.10 m between two bends (the checker's minimum) | **1,384** | 173 |
| Pipe in the unmeasured comfort-zone border | 1,940 m (49% of room pipe) | 2,014 m (46%) |
| Plans with a room under 1 W from a heat limit | 3 | 0 |

- **Where GPT's 0.10 m pieces are:** across all 15 plans, 967 of 1,558 are in the hall, 524 in room
  borders and 67 in the measured comfort zones. They are zigzags on the leader pipes and S-steps.
- **The reference's 0.10 m pieces** are 182 U-turn end-caps of its 0.10 m perimeter pitch, plus 22 hall
  lane shifts.
- **Border packing is not the trick.** GPT's border share is only slightly higher, and on case-05 it has
  less border pipe than the reference (152 m vs 182 m).

case-05, the last plan to pass:

| Room | GPT-6.1: output, window | GPT-6.1: spread / limit | Reference: output | Reference: spread |
|---|---|---|---|---|
| Study | 155.56 W in [125, 156]: **0.44 W under the cap** | 1.02 / 2.2 K | 136.98 W | 1.66 K |
| Bedroom 2 | 184.48 W in [165, 206] | 2.30 / 3.0 K | 179.34 W | 2.55 K |
| Hall | 646.46 W in [520, 648]: **1.54 W under the cap** | 4.76 / 7.4 K | 574.57 W | 7.06 K |

**On case-05, GPT uses the hall as a heat dump.** 121 of its 150 0.10 m pieces on case-05 are zigzags in
the hall. Its bedroom-2-and-study loop is 71.5 m long, with 16.9 m and 27 zigzags in the hall, and enters
bedroom 2 at 36.92 C. The reference's loop is 89.2 m long, with 22.4 m in the hall, and enters at 35.19 C.

## 3. What could block it

A time limit cannot stop a faster exact solve of the same equations. What the speed buys is fine
tuning with 0.10 m zigzags in the hall and S-steps in the rooms. `../rule-a30-patch/` tests installability
rules that forbid exactly those: Rule A30 keeps an adapted reference at 15/15 on the 15 plans, and
GPT-6.1's layouts break it 1,204 times on the 12 held-out plans (0/12 clean). That folder also lists what
the prototype does not show.

## Files

- `regenerate.py`: regenerates `layouts/gpt` and `layouts/ref` (CPU only).
- `metrics.py`: bends, 0.10 m pieces, border pipe and heat through walls per plan; writes `metrics.json`.
- `layouts/`: the regenerated layouts. `metrics.json`: per plan, per room.
- `late-edits/`: GPT-6.1's solver diffs, 20:56 to 21:21 and 21:21 to 21:31.
