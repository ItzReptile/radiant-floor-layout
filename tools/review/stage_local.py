"""Stage the TB3 implementation-rubric review for a local task, the way CI does.

Mirrors terminal-bench-3/scripts/review/stage_task.py: the shared reviewer instruction
template (scripts/rubric-regression/templates/instruction.md), the rubric from
docs/prompts/task-implementation.toml, the task under review at
/app/task-under-review/<name>, and verdicts written to /app/verdicts.json. Local
differences, none of which change what the reviewer is asked or given:

- the task is copied from a local directory instead of fetched from GitHub at a commit;
- the rubric is a file at /app/rubric.toml (as in scripts/rubric-regression) instead of
  being inlined into the instruction, because Harbor passes the instruction on the command
  line and Windows caps command lines at 32k characters;
- no network restrictions: CI gives the agent an api.anthropic.com allowlist and the
  environment no network, which Harbor's local docker backend rejects.

Usage: python stage_local.py TB3_CHECKOUT TASK_DIR OUT_DIR
Then:  harbor run --path OUT_DIR --agent claude-code --model anthropic/claude-sonnet-5 ...
"""
import shutil
import sys
from pathlib import Path

DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git nodejs npm procps python3 && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
COPY rubric.toml /app/rubric.toml
COPY task-under-review/ /app/task-under-review/
WORKDIR /app
"""

TASK_TOML = """schema_version = "1.0"
artifacts = ["/app/verdicts.json"]

[agent]
timeout_sec = 1800.0
"""

TEST_SH = """#!/bin/bash
if [ -f /app/verdicts.json ] && [ -s /app/verdicts.json ]; then
  cp /app/verdicts.json /logs/verifier/verdicts.json
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def stage(tb3: Path, task: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    env = out / "environment"
    tests = out / "tests"
    tests.mkdir(parents=True)
    shutil.copytree(task, env / "task-under-review" / task.name)
    shutil.copy2(tb3 / "docs/prompts/task-implementation.toml", env / "rubric.toml")
    shutil.copy2(tb3 / "scripts/rubric-regression/templates/instruction.md", out / "instruction.md")
    write(env / "Dockerfile", DOCKERFILE)
    write(out / "task.toml", TASK_TOML)
    write(tests / "test.sh", TEST_SH)


if __name__ == "__main__":
    stage(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"staged review task in {sys.argv[3]}")
