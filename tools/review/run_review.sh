#!/bin/bash
# TB3 implementation-rubric review, run locally the way CI runs it (claude-code + claude-sonnet-5).
# 1. python tools/review/stage_local.py VENDOR_TB3 tasks/radiant-floor-layout STAGE_DIR
# 2. CLAUDE_CODE_OAUTH_TOKEN=... bash tools/review/run_review.sh STAGE_DIR JOB_DIR
# The token is read from the environment and is never written to this repository.
# Verdicts land in JOB_DIR/<date>/<trial>/verifier/verdicts.json.
: "${CLAUDE_CODE_OAUTH_TOKEN:?set CLAUDE_CODE_OAUTH_TOKEN}"
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 env -u ANTHROPIC_BASE_URL harbor run --path "$1" --agent claude-code \
  --model anthropic/claude-sonnet-5 --ae CLAUDE_FORCE_OAUTH=1 --ae CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
  --env docker --yes -o "$2"
