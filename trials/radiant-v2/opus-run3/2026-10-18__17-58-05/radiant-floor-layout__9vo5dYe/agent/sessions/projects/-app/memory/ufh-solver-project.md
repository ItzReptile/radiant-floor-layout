---
name: ufh-solver-project
description: "Underfloor-heating layout solver task in /app/solver — architecture decisions, test status and key physics findings"
metadata:
  node_type: memory
  type: project
  originSessionId: 2b06703b-c749-4a5a-9ff7-2b658c833dc2
  modified: 2026-[REDACTED]0-[REDACTED]9T00:26:5[REDACTED].678Z
---

Task (started 2026-[REDACTED]0-[REDACTED]8 22:00 UTC, 6 h budget): write /app/solver/solve.py producing pipe layouts that pass /app/check_layout.py for unseen plans (hall+manifold, 5 hall rooms, [REDACTED] inner room). 40 s limit, stdlib+numpy+scipy only; only /app/solver/ is kept. Container is capped at 2 CPUs (cgroup quota).

Architecture (modules in /app/solver): geom (plan + checker replica), heat (exact heat model, windowed local model, fixed-temperature pipe sources), frame (canonical frame: ports along +x, open side +y), stc (spanning-tree coverage, line fitting, spiral-first Hamiltonian path search), layout (topology, slot/bundle split, door crossings, trunks, room patterns, inner-room splice), fill (hall bumps scored by cold-area coverage, spacing from target flux), optimize (per-room local searches + hall tuning + rounds), solve (entry, never crashes).

Key findings:
- Living room needs 2 loops; inner room (study) is spliced into its parent's loop; hall heated by trunks + bumps.
- Denser spacing + inset gives lower spread than sparse spacing; spread-limited rooms should aim at top of output window.
- Hall fill must avoid stealing length from loops whose rooms need it; supply-side bumps pre-cool water (useful for rooms running hot); freezing supply bumps made things worse.
- Final status (03:35): all 24 rotations/mirrors of the 3 samples pass (6-[REDACTED]5 s); ~84% on harder reference-based generated sets; remaining failures mostly hall spread (cold hall ends when the only adjacent loops have no spare length) and output near-misses. 2 forked workers for room groups (cgroup quota = 2 CPUs), probe-based loop allocation, U-bundle for far-wall doors opposite the manifold, trunk layout chosen by hall score, 28 s budget.
- Not supported: doors on hall end walls (raises LayoutError -> empty layout).

**Why:** context for continuing the solver work.
**How to apply:** check /app/solver modules before redesigning; test scripts live in /tmp/work (not kept).
