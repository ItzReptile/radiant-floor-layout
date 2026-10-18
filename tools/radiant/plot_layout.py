"""Render a radiant-floor-layout plan and layout as a PNG heat map.

Usage: python plot_layout.py PLAN.json LAYOUT.json OUT.png [--title TEXT]
       python plot_layout.py PLAN.json --solve OUT.png        (lay out with the reference solver)

Shows the floor surface temperature, room outlines, every room's occupied (comfort) zone as a
dashed outline, keep-outs, doorways, and every pipe coloured by its water temperature. Each
room is labelled with its output against the window, its peak surface temperature and the
spread over its comfort zone; failing checks are marked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1] / "tasks" / "radiant-floor-layout"
sys.path.insert(0, str(TASK / "environment" / "app"))
sys.path.insert(0, str(TASK / "solution" / "solver"))
import check_layout as ck  # noqa: E402


def zone_parts(plan, room):
    x0, y0, x1, y1 = room["rect"]
    zm = room.get("zone_margin", 0.0)
    inner = [x0 + zm, y0 + zm, x1 - zm, y1 - zm]
    holes = []
    for k in plan.get("keepouts", []):
        g = [k[0] - zm, k[1] - zm, k[2] + zm, k[3] + zm]
        if g[0] < inner[2] and g[2] > inner[0] and g[1] < inner[3] and g[3] > inner[1]:
            holes.append([max(g[0], inner[0]), max(g[1], inner[1]), min(g[2], inner[2]), min(g[3], inner[3])])
    return inner, holes


def render(plan, layout, out, title=""):
    rep = ck.evaluate(plan, layout)
    if rep["geometry_problems"]:
        print("geometry problems:", *rep["geometry_problems"][:5], sep="\n  ")
    bx0, by0, bx1, by1 = plan["building"]
    w, h = bx1 - bx0, by1 - by0
    fig, ax = plt.subplots(figsize=(2 + 1.1 * w, 0.6 + 1.1 * h))
    if "_surface" in rep:
        im = ax.imshow(rep["_surface"], origin="lower", extent=[bx0, bx1, by0, by1], cmap="inferno",
                       vmin=18, vmax=31, interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="floor surface (C)")
    g = plan["grid"]
    segs, temps = [], []
    for lp in layout["loops"]:
        walk = ck.lattice_walk([(round(x / g), round(y / g)) for x, y in lp["path"]])
        water = rep.get("_water", {}).get(lp["slot"])
        for i, (a, b) in enumerate(zip(walk, walk[1:])):
            segs.append([(a[0] * g, a[1] * g), (b[0] * g, b[1] * g)])
            temps.append(water[i] if water is not None else 40.0)
    if segs:
        lc = LineCollection(segs, cmap="cool", linewidths=1.1)
        lc.set_array(np.array(temps))
        lc.set_clim(min(temps), plan["physics"]["supply_temp"])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01, label="water (C)")
    for k in plan.get("keepouts", []):
        ax.add_patch(Rectangle((k[0], k[1]), k[2] - k[0], k[3] - k[1], fc="#9a9a9a", ec="white", lw=0.6, alpha=0.85))
    for room in plan["rooms"]:
        x0, y0, x1, y1 = room["rect"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.6))
        inner, holes = zone_parts(plan, room)
        ax.add_patch(Rectangle((inner[0], inner[1]), inner[2] - inner[0], inner[3] - inner[1], fill=False,
                               ec="#7CFC00", lw=0.8, ls="--"))
        for z in holes:
            ax.add_patch(Rectangle((z[0], z[1]), z[2] - z[0], z[3] - z[1], fill=False, ec="#7CFC00", lw=0.6, ls=":"))
        res = next((q for q in rep["rooms"] if q["id"] == room["id"]), None)
        if res:
            c = res["checks"]
            label = (f"{room['id']}\n{res['output_w']:.0f} W [{room['required_w']}-{room['max_w']}]"
                     f"{'' if c['output_min'] and c['output_max'] else ' FAIL'}\n"
                     f"peak {res['max_surface']:.1f}/{room['max_surface']}{'' if c['max_surface'] else ' FAIL'}\n"
                     f"spread {res['zone_spread']:.2f}/{room['max_spread']}{'' if c['max_spread'] else ' FAIL'}")
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, color="white", fontsize=8, ha="center", va="center",
                    bbox=dict(fc="black", alpha=0.55, lw=0))
    for p in plan.get("passages", []):
        ax.plot([p[0], p[2]], [p[1], p[3]], color="lime", lw=5, solid_capstyle="butt")
    m = plan["manifold"]
    ax.plot(*m["origin"], marker="s", color="white", ms=5)
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_aspect("equal")
    verdict = "PASS" if rep.get("passed") else "FAIL"
    ax.set_title(f"{title or plan.get('name', '')}  -  {verdict}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return rep


def main():
    args = sys.argv[1:]
    title = ""
    if "--title" in args:
        i = args.index("--title")
        title = args[i + 1]
        args = args[:i] + args[i + 2:]
    plan = json.loads(Path(args[0]).read_text())
    if args[1] == "--solve":
        import layout_solver as ls
        layout, _ = ls.solve(plan, ck.evaluate)
    else:
        layout = json.loads(Path(args[1]).read_text())
    render(plan, layout, args[2], title)


if __name__ == "__main__":
    main()
