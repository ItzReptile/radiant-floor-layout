"""Count each rule's violations in the autopsy layouts (tools/radiant/autopsy/layouts): GPT-6.1's saved
v2.1 solver and the unmodified v2.1 reference, regenerated on the 15 plans. Geometry only, CPU only.
Usage: python score_layouts.py"""
import json, sys  # noqa: E401
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

LAYOUTS = HERE.parent / "autopsy" / "layouts"
PLANS = sorted((rules.TASK / "tests/plans").glob("*.json")) + sorted((rules.TASK / "environment/app/plans").glob("*.json"))
print("violations per rule; 'hidden' = the 12 held-out plans, 'all' = those plus the 3 samples\n")
for name, label in (("gpt", "GPT-6.1 v2.1 solver (12/12 copy)"), ("ref", "reference v2.1, unmodified")):
    tot, hid, where = Counter(), Counter(), Counter()
    for pp in PLANS:
        plan, lay = json.loads(pp.read_text()), json.loads((LAYOUTS / name / f"{pp.stem}.json").read_text())
        for rule in ("A", "A30", "B"):
            n = len(rules.violations(plan, lay, rule))
            tot[rule] += n
            if pp.stem.startswith("case"):
                hid[rule] += n
                hid[f"clean {rule}"] += n == 0
        if pp.stem.startswith("case"):
            for s, L, kind, a, b, room, x, y in rules.violations(plan, lay, "A30"):
                where["hall" if room == "hall" else "room step" if kind == "step" else "room U-turn, short side"] += 1
    print(f"{label}")
    print(f"  hidden: rule A {hid['A']}, rule A30 {hid['A30']}, rule B {hid['B']}; plans with no violation: "
          f"A {hid['clean A']}/12, A30 {hid['clean A30']}/12, B {hid['clean B']}/12")
    print(f"  all 15: rule A {tot['A']}, rule A30 {tot['A30']}, rule B {tot['B']}")
    print(f"  rule A30 violations on the hidden plans by kind: {dict(where)}\n")
