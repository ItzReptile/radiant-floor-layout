# radiant-floor-layout: trial summary

Local Harbor 0.24.0 with Docker on one Windows PC. Reward 1 only if the agent's solver passes all 12
held-out plans. Raw job folders: `trials/radiant-v2/` and `trials/radiant-v2.1/`. Per-run details:
the `trial-*.txt` and `v21-trial-*.txt` files in this folder.

## v2: the submitted task (task commit 281d52a, 40 s per plan)

Task digest in every run's `lock.json`: `sha256:ca1fbd31dfaf6a8cb44d4022de9e30d5fd31ebd8c1f19788791b7fe89f511985`.

| Run | Agent / model | Reward | Held-out plans passed | Agent time | How it ended | Failing plans |
|---|---|---|---|---|---|---|
| Opus /run 1 | claude-code, claude-opus-5-5, max | 0 | 11/12 | 4 h 36 min | stopped itself | case-00 bed1 heat over max (262 W > 258) |
| Opus /run 2 | claude-code, claude-opus-5-5, max | 0 | 7/12 | 1 h 48 min | stopped itself | case-01, 03, 05, 06, 09 (hall heat/spread, study and bed2 heat caps, kitchen heat) |
| Opus /run 3 | claude-code, claude-opus-5-5, max | 0 | 11/12 | 5 h 05 min | stopped itself | case-05 study heat over max (169 W > 156) and spread |
| Opus /cheat | claude-code + hack-trial prompt, claude-opus-5-5, max | 0 | 0/12 (no solver left) | 3 min 41 s | stopped itself | reported no credible bypass |
| Codex /run 1 | codex, gpt-6.1-sol, xhigh | **solved** (stopped at 12/12) | **12/12** at the 3 h 12 min check | 3 h 12 min | stopped by us at 12/12 | none (re-checked one plan at a time on 2 CPUs, 9-24 s each) |

None of the finished Opus runs had an exception, a timeout, or an auth or rate-limit error. Once Codex
/run 1 solved the held-out set, Codex runs 2 and 3 and the Codex /cheat run were not started on v2.

## v2.1: v2 with a 15 s limit per plan (task commit f21aa97, branch history only)

| Run | Agent / model | Reward | Held-out plans passed | Agent time | How it ended |
|---|---|---|---|---|---|
| Codex /cheat | codex + hack-trial prompt, gpt-6.1-sol, xhigh | invalid | none (no solver) | none | OpenAI rejected the first request: "flagged for possible cybersecurity risk" (it suggests applying for "Daybreak" access); codex never ran a command |
| Codex /run 1 | codex, gpt-6.1-sol, xhigh | **solved** (stopped at 12/12) | **12/12** at the 5 h 15 min check | about 5 h 15 min | stopped by us at 12/12; checks: 6, 6, 5, 7, 7, 10, 9, 10, 10, 11, 12 |

The Codex /cheat refusal was not retried, and the TB3 hack-trial prompt was not reworded to get past
the filter. Opus was not re-run on v2.1.

## Where this leaves the task

- **Opus 5.5 (max):** fails v2 in 3 of 3 runs, and its /cheat run scores 0.
- **GPT-6.1 Sol (xhigh):** solves v2, and solves v2.1 too. The 15 s limit slowed it (3 h 12 min to about
  5 h 15 min) but did not stop it.
- So the task does not meet the "every trial of both models fails" requirement.
- **Why GPT gets through** (`tools/radiant/autopsy/`): it tunes room heat to within a watt of the limits,
  using 0.10 m zigzags on the leader pipes in the hall and S-steps in the rooms. The change that got it
  over the line on v2.1 was a faster solve of the heat equations.
- **A candidate fix, tested offline only** (`tools/radiant/rule-a30-patch/`): Rule A30 allows pieces under
  0.20 m only as U-turn end-caps inside rooms. An adapted reference passes all 15 plans under it, and
  GPT-6.1's saved solver breaks it 1,204 times (0/12 plans clean). No agent has been run against it.
