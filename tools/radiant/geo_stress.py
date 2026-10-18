"""Development check: build N random geometries at fixed pitches and count geometry problems."""
import random, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

def one(seed):
    sys.argv = ["x"]
    import gen_plans as gp, layout_solver as ls, check_layout as ck
    rng = random.Random(seed); plan = None
    while plan is None:
        plan = gp.make_geometry(rng)
    for r in plan["rooms"]:
        r["required_w"], r["max_w"] = 0, 10 ** 6
    canon = ls.transform_plan(plan, ls.canonical_transform(plan))
    out = []
    for combo in ((0.15, 0.10), (0.25, 0.20)):
        zones, hi = ls.make_zones(canon)
        for z in zones:
            z.combo = combo
        try:
            rep = ck.evaluate(canon, ls.to_layout(ls.build_layout(canon, zones, hi)))
            out.append(rep["geometry_problems"][:2])
        except Exception as e:
            out.append([f"build error: {e}"])
    return seed, out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    bad = 0
    kinds = Counter()
    with ProcessPoolExecutor(12) as ex:
        for seed, out in ex.map(one, range(1000, 1000 + n)):
            probs = [p for o in out for p in o]
            if probs:
                bad += 1
                print(seed, probs[:2])
                for p in probs:
                    kinds[p.split(" near")[0].split(":")[-1].strip()[:40]] += 1
    print(f"{bad}/{n} geometries with problems", dict(kinds))
