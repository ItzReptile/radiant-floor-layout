# radiant-floor-layout v2.1 gate (offline, no agent trials)

v2.1 = v2 (281d52a) with a 15 s per-plan limit instead of 40 s. Same 3 sample and 12 held-out plans, same checker
and physics. Changes: `tests/test_outputs.py` (TIMEOUT_S 15), `instruction.md` (15 seconds), `task.toml`
(version 1.1.0, verifier timeout 600 s), README, and the reference solver's time handling: at least 6 checker
evaluations before its 5 s search budget applies, a faster sparse ordering (MMD_AT_PLUS_A, same answers within 1e-12)
for its own evaluations, and no redundant final re-check.

## Task checks
| Check | Result |
|---|---|
| TB3 static checks | 28/28 |
| Harbor oracle | reward 1, 13/13 tests; per plan 4.2-5.3 s including checking |
| Harbor nop | reward 0 |
| Reference alone (2 CPUs, one plan at a time) | 15/15, 4.0-5.8 s per plan |
| Reference under heavy load (4 agent solvers + 2 CPU burners running alongside) | 15/15, slowest 10.8 s |

## Saved agent solvers, time-scaled to 15 s
Each solver's own time budget and fixed time thresholds were scaled by 15/40, the change a fresh run would make for
a 15 s limit (GPT deadline 38.8 s -> 14.55 s; Opus 1 budget 29/34 s -> 10.9/12.75 s; Opus 2 30 s -> 11.25 s;
Opus 3 28 s -> 10.5 s). Judged by the verifier's checker with a hard 15 s timeout.

| Solver | At 40 s (v2) hidden | 15 s, alone: hidden | 15 s, alone: all 15 | 15 s, under load: all 15 | Fails alone (hidden) |
|---|---|---|---|---|---|
| GPT-6.1 Trial 1 | 12/12 | **9/12** | 12/15 | 5/15 | case-00 hall heat; case-01 bed1/study heat, bed2/hall spread; case-06 bed1/hall heat |
| Opus Trial 1 | 11/12 | **11/12** | 14/15 | 13/15 | case-00 bed1 heat |
| Opus Trial 2 | 7/12 official (5/12 offline) | **5/12** | 8/15 | 8/15 | 7 plans, mostly hall |
| Opus Trial 3 | 11/12 | **9/12** | 10/15 | 3/15 | case-03 study heat; case-05 bed2/study heat; case-06 hall spread |

Every scaled solver still fails at least one held-out plan, so each would score reward 0 on v2.1.
Caveats: the margins are thin (GPT-6.1 fails 3 plans alone, each a near miss; Opus Trial 1's approach is not slowed
by the limit at all and fails only case-00, as on v2); results move with CPU load; a fresh run designed for 15 s
from the start could do better than a scaled 40 s design. Only real trials settle it.
