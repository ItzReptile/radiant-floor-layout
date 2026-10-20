# radiant-floor-layout: a Terminal-Bench 3 task

Submission for the Klavis Founding Engineer take-home by Jose Hernandez (GitHub: ItzReptile).

**The task** is [`tasks/radiant-floor-layout`](tasks/radiant-floor-layout): write a general underfloor-heating
layout solver. It lays hot-water pipe loops into an apartment so every room gets its heat, floors stay
under their surface limit and evenly warm, and the pipes can be installed. It is graded on 12 held-out
floor plans by a two-layer heat model that the agent also has. The task's
[README](tasks/radiant-floor-layout/README.md) explains its difficulty, solution and verification.

## Status

| Requirement | Result |
|---|---|
| TB3 static checks | **28/28 pass** |
| TB3 implementation-rubric review (35 criteria; claude-code + claude-sonnet-5, as in CI) | **33 pass, 0 fail**, 2 not applicable |
| Oracle / nop | **reward 1 / reward 0** |
| Opus 5.5 (max), 3 trials | **0/3**, all genuine fails (11/12, 7/12, 11/12 held-out plans) |
| Opus 5.5 (max) /cheat | **reward 0**: no bypass found, no solver left |
| GPT-6.1 Sol (xhigh), 3 trials | **not met**: Trial 1 solved the task (12/12) |
| GPT-6.1 Sol (xhigh) /cheat | **not run**: on v2 the chain stopped after the GPT solve; on v2.1 OpenAI rejected TB3's hack-trial prompt |

The task does not meet the requirement that every trial of both models fails. Opus 5.5 fails it three
times out of three. GPT-6.1 solved it, and also solved a stricter follow-up (v2.1, 15 s per plan). The
trial records, an autopsy of how GPT-6.1 got through, and an offline-tested candidate fix are below.

## Checks

All on the submitted task directory. Every file the agent and the verifier use is byte-identical to the
version the trials ran on (task commit ea4cf34). Two metadata files differ, and neither is read during a run:
the README's *Relevant experience* section, and the author email in `task.toml`, which is now the GitHub
no-reply address.

- **Static checks: 28/28 pass** (`tools/radiant/results/final-checks.txt`). TB3's `scripts/checks/check-*.sh`
  in a Linux container:
  ```bash
  docker run --rm -e NAME=radiant-floor-layout -v "$TB3:/tb3:ro" -v "$PWD/tasks/radiant-floor-layout:/task:ro" -v "$PWD/tools/checks/run_checks_in.sh:/run.sh:ro" python:3.12-bookworm bash /run.sh
  ```
- **Oracle: reward 1** (13 passed in 96 s). **Nop: reward 0** (13 failed).
  `harbor run -p tasks/radiant-floor-layout --agent oracle|nop --env docker`
- **Rubric review: 33 pass, 0 fail, 2 not applicable** of 35 criteria
  (`tools/radiant/results/rubric-verdicts.json`). The two not-applicable criteria cover large build-time
  artifacts and "do not modify" rules, and this task has neither. `tools/review/stage_local.py` stages
  the review the way CI does, and `tools/review/run_review.sh` runs it.
- **Same task as the trials:** every trial's `lock.json` records a SHA-256 digest of the task's files. All v2
  trials recorded `sha256:ca1fbd31dfaf6a8cb44d4022de9e30d5fd31ebd8c1f19788791b7fe89f511985`.
  `python tools/radiant/verify_digest.py` rebuilds that digest from `tools/radiant/trial-digest-manifest.txt`
  (each file's SHA-256 at trial time) and checks the committed files against it: 25 of 27 are identical,
  and only `README.md` and `task.toml` differ.

## Trials

Local Harbor 0.24.0, Docker, one Windows PC. Full table: [`tools/radiant/results/summary.md`](tools/radiant/results/summary.md).
Raw job folders: [`trials/radiant-v2/`](trials/radiant-v2) and [`trials/radiant-v2.1/`](trials/radiant-v2.1).

**v2, the submitted task** (40 s per plan):

| Run | Reward | Held-out plans | Agent time | Notes |
|---|---|---|---|---|
| Opus 5.5 max, run 1 | 0 | 11/12 | 4 h 36 min | stopped itself; case-00 bedroom 1 at 262 W, cap 258 W |
| Opus 5.5 max, run 2 | 0 | 7/12 | 1 h 48 min | stopped itself; hall, study, bedroom 2 and kitchen heat limits |
| Opus 5.5 max, run 3 | 0 | 11/12 | 5 h 05 min | stopped itself; case-05 study at 169 W, cap 156 W |
| Opus 5.5 max, /cheat | 0 | 0/12 | 3 min 41 s | found no credible bypass, left no solver |
| GPT-6.1 Sol xhigh, run 1 | solved | 12/12 at the 3 h 12 min check | 3 h 12 min | stopped by us at 12/12; runs 2-3 and /cheat not started |

**v2.1** (v2 with 15 s per plan; task commit 05f9827, kept in history only):

| Run | Reward | Held-out plans | Notes |
|---|---|---|---|
| GPT-6.1 Sol xhigh, /cheat | invalid | none | OpenAI rejected the first request: "flagged for possible cybersecurity risk", suggesting "Daybreak" access; codex never ran a command |
| GPT-6.1 Sol xhigh, run 1 | solved | 12/12 at about 5 h 15 min | read-only checks 6, 6, 5, 7, 7, 10, 9, 10, 10, 11, 12; stopped by us at 12/12 |

**Configs.** Opus: `--agent claude-code --model anthropic/claude-opus-5-5 --ak reasoning_effort=max`, with
the OAuth token passed by `--ae` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`. GPT: `--agent codex --model
openai/gpt-6.1-sol --ak reasoning_effort=xhigh`, using the ChatGPT login (`CODEX_FORCE_AUTH_JSON=1` in the host
environment). /cheat runs add `--extra-instruction-path` with TB3's `docs/prompts/hack-trial-prompt.md`.
Progress was checked read-only during runs: `/app/solver` was copied out and run through the verifier in a
separate container. Runs were stopped only when a check reached 12/12.

## Failure analysis

- **Opus 5.5** writes working counterflow solvers but leaves one to five held-out plans over a heat or
  evenness limit. In run 1, bedroom 1 is 4 W over its cap on case-00. In run 3, the study (heated in
  series behind a bedroom) is 13 W over its cap on case-05. In run 2, five plans fail on hall, study,
  bedroom 2 and kitchen limits. Runs 1 and 3 stopped on their own, saying their solver might still fail
  unseen plans. Per-run notes: `tools/radiant/results/trial-opus-*.txt`.
- **GPT-6.1** gets through by tuning heat to within a watt of the limits
  ([`tools/radiant/autopsy/`](tools/radiant/autopsy)).
  - On the 12 held-out plans its v2.1 solver uses 1,384 minimum-length (0.10 m) pieces between bends,
    against 173 for the reference.
  - Across the 15 plans, 967 of its 1,558 0.10 m pieces are zigzags on the leader pipes in the hall.
  - On case-05, the last plan to pass, the hall ends at 646.5 of 648 W and the study at 155.56 of
    156 W, 0.44 W under its cap.
  - What pushed it over the line in the last 35 minutes was a faster solve of the heat equations, not a
    new layout idea.
- **Candidate fix, offline only** ([`tools/radiant/rule-a30-patch/`](tools/radiant/rule-a30-patch)).
  Rule A30 allows a pipe piece under 0.20 m only as a U-turn end-cap inside a room, between two pieces of
  at least 0.30 m. This is a bend-radius rule for real pipe.
  - An adapted reference passes all 15 plans under it (median 3.5 s).
  - GPT-6.1's saved solver breaks it 1,204 times across the 12 held-out plans (0/12 plans clean).
  - It is not in the submitted task and no agent has been run against it.
  - Its README lists what is still unverified: a container load gate, the Opus solvers, and how a fresh
    run would adapt.

## Repository layout

| Path | What |
|---|---|
| `tasks/radiant-floor-layout/` | **the submitted task** |
| `trials/radiant-v2/`, `trials/radiant-v2.1/` | raw Harbor job folders: transcripts, verifier output, solvers left behind |
| `tools/radiant/` | plan generator (`gen_plans.py`), baselines (`ablation.py`), renderer (`plot_layout.py`), `verify_digest.py` |
| `tools/radiant/results/` | trial summaries, the v2 gate matrix, the v2.1 gate, final check results |
| `tools/radiant/autopsy/` | the GPT-6.1 solver autopsy |
| `tools/radiant/rule-a30-patch/` | the Rule A30 prototype and gate |
| `tools/checks/`, `tools/review/` | static-check and rubric-review runners |

Saved trial files contain no credentials. Harbor masks the values of the environment variables passed to
the agent. One of them was `CLAUDE_FORCE_OAUTH=1`, so in the Opus job folders every character `1` reads
`[REDACTED]`, and the OAuth token appears only masked (`sk-a****wAA`). Local Windows paths in the trial
records read `C:\Users\user\...`: the account name was replaced, and nothing else in those files changed.
