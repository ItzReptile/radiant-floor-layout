"""Show that the submitted task is the one the v2 trials ran on.

Harbor records a task digest in every trial's lock.json (task.digest): the SHA-256 of a list of
"path NUL sha256(file)" lines, one per file in task.toml, instruction.md, README.md, environment/,
tests/ and solution/. Every v2 trial recorded
sha256:ca1fbd31dfaf6a8cb44d4022de9e30d5fd31ebd8c1f19788791b7fe89f511985.

trial-digest-manifest.txt lists each of those files with its SHA-256 at trial time. This script:
  1. rebuilds the digest from the manifest and checks it equals the trials' digest;
  2. hashes the task directory as committed and compares it with the manifest file by file.
Only two metadata files may differ, and neither is read by the agent or the verifier: README.md
(the "Relevant experience" section was filled in) and task.toml (the author email is the GitHub
no-reply address). Everything the agent and verifier use must be byte-identical.

Usage: python tools/radiant/verify_digest.py   (standard library only)"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1] / "tasks" / "radiant-floor-layout"
TRIALS = "ca1fbd31dfaf6a8cb44d4022de9e30d5fd31ebd8c1f19788791b7fe89f511985"
METADATA = {"README.md", "task.toml"}

manifest = [line.split("  ") for line in (HERE / "trial-digest-manifest.txt").read_text(encoding="utf-8").splitlines() if line]
outer = hashlib.sha256()
for rel, digest in manifest:
    outer.update(f"{rel}\0{digest}\n".encode())
rebuilt = outer.hexdigest()
print(f"digest rebuilt from the manifest:  sha256:{rebuilt}")
print(f"digest recorded by the v2 trials:  sha256:{TRIALS}")

changed, missing = [], []
for rel, digest in manifest:
    f = TASK / rel
    if not f.is_file():
        missing.append(rel)
    elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        changed.append(rel)
listed = {rel for rel, _ in manifest}
extra = sorted(p.relative_to(TASK).as_posix() for d in ("environment", "tests", "solution")
               for p in (TASK / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and p.relative_to(TASK).as_posix() not in listed)
print(f"files in the manifest: {len(manifest)}; identical now: {len(manifest) - len(changed) - len(missing)}; "
      f"changed: {changed or 'none'}; missing: {missing or 'none'}; new: {extra or 'none'}")
ok = rebuilt == TRIALS and not missing and not extra and set(changed) <= METADATA
print("MATCH: every file the agent and verifier use is byte-identical to the trials" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
