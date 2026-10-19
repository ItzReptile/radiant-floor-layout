# radiant-floor-layout v2: trial job folders

Raw Harbor 0.24.0 job folders, copied unchanged from the local `jobs/` directory. Every trial ran
on the same task files (last task commit 281d52a).

| Folder | Agent | Model | Reward | Held-out plans passed | Agent time |
|---|---|---|---|---|---|
| `opus-run1` | claude-code | anthropic/claude-opus-5-5, reasoning_effort=max | 0 | 11/12 | 4 h 36 min, stopped itself |
| `opus-run2` | claude-code | anthropic/claude-opus-5-5, reasoning_effort=max | 0 | 7/12 | 1 h 48 min, stopped itself |
| `opus-run3` | claude-code | anthropic/claude-opus-5-5, reasoning_effort=max | 0 | 11/12 | 5 h 05 min, stopped itself |
| `opus-cheat` | claude-code + TB3 hack-trial prompt | anthropic/claude-opus-5-5, reasoning_effort=max | 0 | 0/12 (left no solver) | 3 min 41 s, stopped itself |

Per-trial summaries, including the read-only progress checks taken during each run, are in
`tools/radiant/results/trial-opus-{1,2,3}.txt` and `trial-opus-cheat.txt`.

Each folder holds `result.json` (no exceptions in any run), `verifier/test-stdout.txt` (the pytest
output that decides the reward), `verifier/reward.txt`, `agent/` (the full agent transcript) and
`artifacts/app/solver/` (the solver the agent left behind).

Note on `[REDACTED]`: Harbor masks the values of the environment variables passed to the agent.
One of them was `CLAUDE_FORCE_OAUTH=1`, so every character `1` in these saved files, including the
solver artifacts, reads `[REDACTED]`. The verifier ran the real, unmasked files inside its container;
only the saved copies are affected. The OAuth token itself appears only in masked form.
