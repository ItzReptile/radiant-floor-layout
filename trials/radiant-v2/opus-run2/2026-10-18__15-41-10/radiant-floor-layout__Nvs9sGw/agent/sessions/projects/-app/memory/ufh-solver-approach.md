---
name: ufh-solver-approach
description: "Underfloor-heating layout solver in /app/solver — architecture, status and key lessons (task 2026-[REDACTED]0-[REDACTED]8)"
metadata:
  node_type: memory
  type: project
  originSessionId: 503db94e-4557-44de-a5f0-36c2df8ca032
  modified: 2026-[REDACTED]0-[REDACTED]8T2[REDACTED]:32:[REDACTED]2.669Z
---

Solver for /app/check_layout.py lives in /app/solver (solve.py entry; plan.py canonicalises so ports run +x and pipes leave +y; stc.py lane-grid spanning-tree circumnavigation; design.py lanes/adapters; build.py trunks, rooms, hall coverage; sim.py = checker heat model with MMD_AT_PLUS_A ordering ~0.45 s/solve; checker.py = copy of the checker).

Status at end of session: 3/3 samples pass in 2-4 s; 24/24 rotated/mirrored samples pass when run alone; ~[REDACTED]6/30 on a harsher random generator (/tmp/work/gen.py, not kept).

Key lessons:
- Pipes can never cross -> every loop = tour around a tree of grid cells (STC); planarity automatic.
- Rooms need ~0.2 m spacing; big rooms (>~430 W) need two loops. Always use every slot.
- Parent + inner room share one loop; hall heat = trunks + farthest-point EDT coverage.
- Right-going trunks hug the hall's far wall so open hall borders their back-pass (return) side.
- Checker passage bounds are raw floats (5.85/0.05 = [REDACTED][REDACTED]6.999...): derive door spans with the exact crossing rule.
- Lane gaps of ~2s (even-count rounding) create cold bands: pick lane counts by closest spacing.
- Batch tests under parallel load slow the solver ~5x; use iteration budgets, not time slices.

**Why:** remaining failures are near-misses in hall spread, study output, two-loop balance.
**How to apply:** if resuming, improve convergence (per-room line search) before new features.
