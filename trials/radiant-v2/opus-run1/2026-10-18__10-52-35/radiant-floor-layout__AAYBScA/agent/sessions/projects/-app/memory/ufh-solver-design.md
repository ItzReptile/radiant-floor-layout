---
name: ufh-solver-design
description: "Underfloor-heating solver task (/app/solver/solve.py) — key physics findings, chosen design, gotchas, final status"
metadata:
  node_type: memory
  type: project
  originSessionId: [REDACTED]9d2b[REDACTED]cd-e9d7-4f45-be55-934ad[REDACTED]f269[REDACTED][REDACTED]
  modified: 2026-[REDACTED]0-[REDACTED]8T[REDACTED]9:3[REDACTED]:[REDACTED]2.70[REDACTED]Z
---

Task: write /app/solver/solve.py producing pipe layouts passing /app/check_layout.py (40 s limit, stdlib+numpy+scipy only, only /app/solver kept). Worked 2026-[REDACTED]0-[REDACTED]8 [REDACTED]4:52-20:30 UTC.
Scratch in /app/work: viz.py, roomviz.py, waterviz.py, harness.py, genplan.py + genreal.py (realistic random plans -> qplans/), batch.py (`python3 batch.py "qplans/real-0[4-6]*.json" qout [REDACTED]` from /app/work).

Solver modules (/app/solver): planprep.py (canonical frame, exact float door ranges, zone masks), stc.py (cell-tree STC, make_pairs2), layout.py (Designer: hall routing w/ retries, end-wall doors, hall-only loop in a free slot, fill + density prune w/ per-loop bias & cold protection, spirals, split2 two-loop rooms, corridors, trim), heat.py (checker model + local_solve), tune.py (batched parallel Tuner, fork pool sized by cgroup quota, stuck-unit broadening + combined moves, hall densification), solve.py (capacity test -> [REDACTED] vs 2 loops, capacity-gated abort/switch, fallback variants, minimal layout, 29 s budget / 34 s hard cap).

Findings:
- Container has 2-CPU cgroup quota: never batch-test with >2 parallel jobs.
- One loop ~460 W max; 2-loop living min ~575-690 W with OK spread -> output gap for P_ref ~525-575 (main remaining failure mode, with hall capacity).
- Requirements look like round(0.89*P_ref), round([REDACTED].[REDACTED][REDACTED]*P_ref) of a reference design.
- Exact uniform line spacing matters; split-grid per-part spacing and inset knob were harmful.

Final status: samples pass in all 8 orientations (~3-9 s); ~70% (44/60) of synthetic realistic plans pass.
