"""Measure the autopsy layouts (layouts/gpt, layouts/ref) with the verifier's checker. CPU only.
Usage: python metrics.py   (prints the tables and writes metrics.json)
Per plan and solver: pass/fail, bends, minimum-length (0.10 m) straight pieces between bends, pipe length,
pipe in the unmeasured comfort-zone border, and each room's net heat imported through its walls
(output + downward loss - heat its own pipe cells inject), from the checker's own solution vector."""
import json, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # keep the task's tests/ free of __pycache__

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[2] / "tasks" / "radiant-floor-layout"
sys.path.insert(0, str(TASK / "tests"))
import check_layout as ck  # noqa: E402

OUT = HERE
PLANS = sorted((TASK / "tests/plans").glob("*.json")) + sorted((TASK / "environment/app/plans").glob("*.json"))
_last = {}
_orig = ck.spsolve
def _keep(a, b):
    t = _orig(a, b); _last["t"] = t; return t
ck.spsolve = _keep


def zone_mask(plan, floor, ri):
    g = plan["grid"]; room = plan["rooms"][ri]
    xs, ys = floor.xs * g, floor.ys * g
    x0, y0, x1, y1 = room["rect"]; zm = room.get("zone_margin", 0.0); E = ck.EPS
    zx = (xs >= x0 + zm - E) & (xs <= x1 - zm + E); zy = (ys >= y0 + zm - E) & (ys <= y1 - zm + E)
    blocked = np.zeros((floor.ny, floor.nx), bool)
    for k in plan.get("keepouts", []):
        blocked |= (((ys >= k[1] - zm - E) & (ys <= k[3] + zm + E))[:, None] & ((xs >= k[0] - zm - E) & (xs <= k[2] + zm + E))[None, :])
    return (floor.room_of == ri) & zy[:, None] & zx[None, :] & ~blocked


def measure(plan, layout):
    rep = ck.evaluate(plan, layout)
    g = plan["grid"]; ph = plan["physics"]; floor = ck.Floor(plan)
    loops, problems = ck.check_geometry(plan, layout, floor)
    t = _last["t"]; n = floor.nx * floor.ny
    bends = jogs = jogs15 = 0; length = 0.0
    hall = next(i for i, r in enumerate(plan["rooms"]) if r["id"] == "hall")
    border = {i: 0.0 for i in range(len(plan["rooms"]))}; inzone = dict(border)
    zones = [zone_mask(plan, floor, i) for i in range(len(plan["rooms"]))]
    inject = np.zeros(len(plan["rooms"]))
    sleeve = int(round(ph["sleeve_length"] / g)); off = 2 * n
    for lp in loops:
        pts = lp["points"]; segs = list(zip(pts, pts[1:]))
        bends += max(0, len(pts) - 2)
        for i, (a, b) in enumerate(segs):
            L = (abs(a[0] - b[0]) + abs(a[1] - b[1])) * g
            if 0 < i < len(segs) - 1:
                jogs += L <= 0.10 + 1e-9; jogs15 += L <= 0.15 + 1e-9
        walk = lp["walk"]; steps = len(walk) - 1; length += steps * g
        gk = np.full(steps, ph["g_pipe"] * g); k = np.arange(steps)
        gk[(k < sleeve) | (k >= steps - sleeve)] *= ph["sleeve_factor"]
        water = t[off:off + steps]; off += steps
        for s, (x, y) in enumerate(walk[1:]):
            cy, cx = y - floor.by0, x - floor.bx0; ri = int(floor.room_of[cy, cx])
            (inzone if zones[ri][cy, cx] else border)[ri] += g
            inject[ri] += gk[s] * (water[s] - t[cy * floor.nx + cx])
    tb = t[:n].reshape(floor.ny, floor.nx); gdown = ph["insulation_u"] * g * g
    rooms = {}
    for ri, r in enumerate(rep["rooms"]):
        sel = floor.room_of == ri
        down = float((gdown * (tb[sel] - ph["below_temp"])).sum())
        rooms[r["id"]] = {"out": r["output_w"], "req": r["required_w"], "max": r["max_w"], "spread": r["zone_spread"],
                          "lim_spread": r["limit_spread"], "peak": r["max_surface"], "lim_peak": r["limit_surface"],
                          "import_w": r["output_w"] + down - float(inject[ri]),
                          "border_m": border[ri], "zone_m": inzone[ri], "ok": all(r["checks"].values())}
    nonhall_border = sum(v for i, v in border.items() if i != hall)
    nonhall_total = nonhall_border + sum(v for i, v in inzone.items() if i != hall)
    return {"passed": rep["passed"], "problems": rep["geometry_problems"][:2], "bends": bends, "jogs10": int(jogs),
            "jogs15": int(jogs15), "length": length, "border_m": nonhall_border, "border_frac": nonhall_border / max(1e-9, nonhall_total),
            "rooms": rooms}


res = {}
for name in ("gpt", "ref"):
    for plan_path in PLANS:
        plan = json.loads(plan_path.read_text())
        f = OUT / "layouts" / name / f"{plan_path.stem}.json"
        res[name, plan_path.stem] = measure(plan, json.loads(f.read_text())) if f.is_file() else None
json.dump({f"{a}|{b}": v for (a, b), v in res.items()}, open(OUT / "metrics.json", "w"), indent=1)

print(f"{'plan':10} | {'pass':9} | {'bends':11} | {'0.10m jogs':11} | {'<=0.15 m':11} | {'length m':13} | {'border m (share)':25}")
tot = {k: np.zeros(6) for k in ("gpt", "ref")}
for plan_path in PLANS:
    s = plan_path.stem; G, R = res["gpt", s], res["ref", s]
    print(f"{s:10} | {('P' if G['passed'] else 'F') + ' / ' + ('P' if R['passed'] else 'F'):9} | {G['bends']:4} / {R['bends']:4} | "
          f"{G['jogs10']:4} / {R['jogs10']:4} | {G['jogs15']:4} / {R['jogs15']:4} | {G['length']:5.0f} / {R['length']:5.0f} | "
          f"{G['border_m']:5.1f} ({G['border_frac']:.0%}) / {R['border_m']:5.1f} ({R['border_frac']:.0%})")
    for k, M in (("gpt", G), ("ref", R)):
        tot[k] += [M["passed"], M["bends"], M["jogs10"], M["jogs15"], M["length"], M["border_m"]]
for k in ("gpt", "ref"):
    print(k, "totals: pass", int(tot[k][0]), "bends", int(tot[k][1]), "jogs10", int(tot[k][2]), "jogs15", int(tot[k][3]),
          "length", round(tot[k][4]), "border", round(tot[k][5], 1))
for s in ("case-05",):
    print(f"\n{s}: room | out W [req, max] | spread/limit | peak/limit | net heat in through walls | border m / zone m   (GPT then ref)")
    for rid in res["gpt", s]["rooms"]:
        for k in ("gpt", "ref"):
            r = res[k, s]["rooms"][rid]
            print(f"  {k} {rid:8} {r['out']:7.2f} [{r['req']}, {r['max']}]  {r['spread']:.3f}/{r['lim_spread']}  {r['peak']:.2f}/{r['lim_peak']}"
                  f"  {r['import_w']:+7.1f} W  {r['border_m']:5.1f}/{r['zone_m']:5.1f} {'' if r['ok'] else 'FAIL'}")
