# radiant-floor-layout v2.1: trial job folders

Raw Harbor 0.24.0 job folders, copied unchanged from the local `jobs/` directory. Every trial ran
on the v2.1 task files (branch `radiant-v2.1`, task commit f21aa97: v2 with a 15 s per-plan limit).
Runs go one at a time, so no two trials share the CPU.

| Folder | Agent | Model | Reward | Held-out plans passed | Agent time |
|---|---|---|---|---|---|
| `gpt-cheat-blocked` | codex + TB3 hack-trial prompt | openai/gpt-6.1-sol, reasoning_effort=xhigh | 0 (invalid run) | 0/12 (no solver) | none: blocked by OpenAI's filter on the first request |

`gpt-cheat-blocked`: OpenAI rejected the very first request ("This content was flagged for possible
cybersecurity risk"), so codex exited before running a single command (`AgentSafetyRefusalError` in
`result.json`). The model never saw the task, so this run says nothing about the task's defenses and
does not count as a /cheat trial.

Per-trial summaries are in `tools/radiant/results/v21-trial-*.txt`.
