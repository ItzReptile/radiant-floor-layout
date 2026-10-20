"""Show that the submitted task is the one the v2 trials ran on.

Harbor records a task digest in every trial's lock.json (task.digest): a SHA-256 over task.toml,
instruction.md, README.md and every file under environment/, tests/ and solution/. The v2 trials recorded
sha256:ca1fbd31... After the trials only README.md changed (the "Relevant experience" section, which
is not shown to the agent or used by the verifier), so this script prints two digests:

  1. the task directory as committed;
  2. the same directory with README.md as it was at 281d52a, which must equal the trials' digest.

Usage: python tools/radiant/verify_digest.py   (needs Harbor installed: harbor.publisher.packager)"""
import shutil, subprocess, sys, tempfile  # noqa: E401
from pathlib import Path

from harbor.publisher.packager import Packager

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "radiant-floor-layout"
TRIALS = "ca1fbd31dfaf6a8cb44d4022de9e30d5fd31ebd8c1f19788791b7fe89f511985"

now, _ = Packager.compute_content_hash(TASK)
print(f"task directory as committed:        sha256:{now}")
with tempfile.TemporaryDirectory() as tmp:
    copy = Path(tmp) / "radiant-floor-layout"
    shutil.copytree(TASK, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    readme = subprocess.run(["git", "-C", str(ROOT), "show", "281d52a:tasks/radiant-floor-layout/README.md"],
                            capture_output=True, check=True).stdout
    (copy / "README.md").write_bytes(readme)
    then, _ = Packager.compute_content_hash(copy)
print(f"with README.md as at 281d52a:       sha256:{then}")
print(f"digest recorded by the v2 trials:   sha256:{TRIALS}")
print("MATCH" if then == TRIALS else "MISMATCH")
sys.exit(0 if then == TRIALS else 1)
