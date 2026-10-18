"""Try one physics / limit setting: generate plans, then run the ablations against them.

Usage: python sweep.py OUT_DIR N_PLANS [gen_plans options...]
Prints how many plans each ablation passes (the reference passes all of them by construction).
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1])
n = int(sys.argv[2])
extra = sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)


def gen(i):
    subprocess.run([sys.executable, str(HERE / "gen_plans.py"), str(out), "--count", "1", "--seed", str(1000 + i),
                    "--prefix", f"s{i:02d}", *extra], capture_output=True, text=True)


with ThreadPoolExecutor(max_workers=min(n, 14)) as ex:
    list(ex.map(gen, range(n)))
plans = sorted(str(p) for p in out.glob("*.json") if not p.name.startswith("abl-"))
print(f"{len(plans)} plans generated with {' '.join(extra)}", flush=True)
for fill, lanes in (("serp", "spread"), ("serp", "north"), ("cf", "north"), ("cf", "south")):
    r = subprocess.run([sys.executable, str(HERE / "ablation.py"), "run", fill, lanes, *plans,
                        "--json", str(out / f"abl-{fill}-{lanes}.json")], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    fails = [ln for ln in lines if " PASS " in ln or ln.endswith("PASS")]
    print(lines[-1] if lines else r.stderr[-800:], flush=True)
    for ln in lines[:-1]:
        print("   ", ln, flush=True)
