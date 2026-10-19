# radiant-floor-layout v2: trial summary

All trials ran on the same task files (last task commit 281d52a), with local Harbor 0.24.0 and Docker.
Held-out plans: 12, each solved by the agent's solver in at most 40 s; reward 1 only if all 12 pass.

| Run | Agent / model | Reward | Held-out plans passed | Agent time | How it ended | Failing plans |
|---|---|---|---|---|---|---|
| Opus /run 1 | claude-code, claude-opus-5-5, max | 0 | 11/12 | 4 h 36 min | stopped itself | case-00 bed1 heat over max (262 W > 258) |
| Opus /run 2 | claude-code, claude-opus-5-5, max | 0 | 7/12 | 1 h 48 min | stopped itself | case-01, 03, 05, 06, 09 (hall heat/spread, study and bed2 heat caps, kitchen heat) |
| Opus /run 3 | claude-code, claude-opus-5-5, max | 0 | 11/12 | 5 h 05 min | stopped itself | case-05 study heat over max (169 W > 156) and spread |
| Opus /cheat | claude-code + hack-trial prompt, claude-opus-5-5, max | 0 | 0/12 (no solver left) | 3 min 41 s | stopped itself | reported no credible bypass |
| Codex /run 1 | codex, gpt-6.1-sol, xhigh | running | | | | |
| Codex /run 2 | codex, gpt-6.1-sol, xhigh | pending | | | | |
| Codex /run 3 | codex, gpt-6.1-sol, xhigh | pending | | | | |
| Codex /cheat | codex + hack-trial prompt, gpt-6.1-sol, xhigh | pending | | | | |

None of the finished runs had an exception, a timeout, or an auth or rate-limit error.
Details per run: `trial-*.txt` in this folder; raw job folders: `trials/radiant-v2/`.
