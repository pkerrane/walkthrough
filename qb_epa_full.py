"""
QB EPA per Game vs Success Rate — full pipeline in one script.

Pulls 2026 play-by-play with nflreadpy, builds the QB table (the same
summarise/filter your R script did), then renders the base scatter plus one
chart per QB with that QB's bubble AND name enclosed in a black oval, named by
full player name, with EPA/play as the bubble size and in the subtitle.

SETUP (one time)
    pip install nflreadpy pandas numpy matplotlib

RUN
    python qb_epa_full.py
    -> charts land in ./output/ (base) and ./output/circled/ (one per QB)
"""
import os
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Ellipse
from matplotlib.transforms import IdentityTransform

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# command line: py qb_epa_full.py [--year 2026] [--minplays 10]
_ap = argparse.ArgumentParser()
_ap.add_argument("--year", type=int, default=2026, help="season (default 2026)")
_ap.add_argument("--minplays", type=int, default=10, help="min plays/dropbacks to qualify (default 10)")
_args = _ap.parse_args()
SEASON   = _args.year
MINPLAYS = _args.minplays
DBS      = round(0.8 * MINPLAYS)   # dropback floor = 80% of plays
# output -> Walkthrough 26 / EPA Circled QBs  (absolute, so it doesn't depend on where you run it)
_home    = os.environ.get("USERPROFILE") or os.path.expanduser("~")
OUTPUT   = Path(_home) / "Documents" / "Walkthrough 26" / "EPA Circled QBs"
CREDIT   = "Figure: @PatKerrane  |  Data: @nflfastR"

# Data arrays — filled by load_data() from the live pull.
NAMES, SR, EPAG, EPA_PLAY, CPOE, COL = [], None, None, None, None, []

# ---------------------------------------------------------------------------
# DATA  (your R dplyr pipeline, translated to pandas)
# ---------------------------------------------------------------------------
def load_data():
    global NAMES, SR, EPAG, EPA_PLAY, CPOE, COL
    import nflreadpy as nfl

    pbp = nfl.load_pbp([SEASON]).to_pandas()
    # exclude QB kneels and spikes so EPA/SR match RBSDM
    d = pbp[pbp["epa"].notna() & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]

    qbs = (
        d.groupby(["id", "name"], dropna=True)
         .agg(
             epa=("qb_epa", "mean"),          # EPA per play
             cpoe=("cpoe", "mean"),
             sr=("success", "mean"),
             g=("game_id", "nunique"),
             n_dropbacks=("pass", "sum"),
             n_plays=("epa", "size"),         # n() = rows in the group
             team=("posteam", "last"),
         )
         .reset_index()
    )
    qbs["epa_g"] = qbs["epa"] * qbs["n_plays"] / qbs["g"]   # EPA per game
    qbs = qbs[(qbs["n_dropbacks"] >= DBS) &
              (qbs["n_plays"] >= MINPLAYS)].reset_index(drop=True)

    # official team colors, joined like your left_join(load_teams())
    teams = nfl.load_teams().to_pandas()[["team_abbr", "team_color"]]
    qbs = qbs.merge(teams, left_on="team", right_on="team_abbr", how="left")

    NAMES    = qbs["name"].tolist()
    SR       = qbs["sr"].to_numpy()
    EPAG     = qbs["epa_g"].to_numpy()
    EPA_PLAY = qbs["epa"].to_numpy()
    CPOE     = qbs["cpoe"].to_numpy()
    COL      = qbs["team_color"].fillna("#666666").tolist()
    print(f"{len(NAMES)} QBs qualified (>= {DBS} dropbacks, {MINPLAYS} plays)")
    return qbs

# ---------------------------------------------------------------------------
# FULL NAMES  — files are named from these. Unmapped names fall back to the
# abbreviated form and print a warning so you can add a line.
# ---------------------------------------------------------------------------
FULL_NAMES = {
    "D.Lock": "Drew Lock",
    "P.Rivers": "Philip Rivers",        "A.Rodgers": "Aaron Rodgers",
    "J.Flacco": "Joe Flacco",           "M.Stafford": "Matthew Stafford",
    "T.Taylor": "Tyrod Taylor",         "R.Wilson": "Russell Wilson",
    "K.Cousins": "Kirk Cousins",        "G.Smith": "Geno Smith",
    "M.Mariota": "Marcus Mariota",      "C.Wentz": "Carson Wentz",
    "D.Prescott": "Dak Prescott",       "J.Goff": "Jared Goff",
    "J.Brissett": "Jacoby Brissett",    "P.Mahomes": "Patrick Mahomes",
    "L.Jackson": "Lamar Jackson",       "B.Mayfield": "Baker Mayfield",
    "J.Allen": "Josh Allen",            "S.Darnold": "Sam Darnold",
    "J.Browning": "Jake Browning",      "K.Murray": "Kyler Murray",
    "D.Jones": "Daniel Jones",          "T.Tagovailoa": "Tua Tagovailoa",
    "J.Love": "Jordan Love",            "J.Herbert": "Justin Herbert",
    "J.Hurts": "Jalen Hurts",           "J.Burrow": "Joe Burrow",
    "D.Mills": "Davis Mills",           "J.Fields": "Justin Fields",
    "T.Lawrence": "Trevor Lawrence",    "M.Jones": "Mac Jones",
    "B.Purdy": "Brock Purdy",           "B.Young": "Bryce Young",
    "C.Stroud": "C.J. Stroud",          "S.Rattler": "Spencer Rattler",
    "B.Nix": "Bo Nix",                  "D.Maye": "Drake Maye",
    "J.Daniels": "Jayden Daniels",      "M.Penix": "Michael Penix Jr.",
    "C.Williams": "Caleb Williams",     "J.McCarthy": "J.J. McCarthy",
    "S.Sanders": "Shedeur Sanders",     "C.Ward": "Cam Ward",
    "J.Dart": "Jaxson Dart",            "D.Gabriel": "Dillon Gabriel",
    "T.Shough": "Tyler Shough",
}

def full_name(abbr: str) -> str:
    return FULL_NAMES.get(abbr, abbr)

def slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

# ---------------------------------------------------------------------------
# CHART
# ---------------------------------------------------------------------------
def bubble_sizes():
    """Map EPA per play -> marker area (bigger bubble = higher EPA/play)."""
    lo, hi = EPA_PLAY.min(), EPA_PLAY.max()
    return np.interp(EPA_PLAY, [lo, hi], [140, 1700])


def draw_base(ax):
    sizes = bubble_sizes()
    ax.axhline(0, color="#444444", linewidth=1.1, linestyle="--", zorder=1)
    ax.axvline(SR.mean(), color="#444444", linewidth=1.1, linestyle="--", zorder=1)
    m, b = np.polyfit(SR, EPAG, 1)
    xs = np.array([SR.min() - 0.006, SR.max() + 0.006])
    ax.plot(xs, m * xs + b, color="#7A7A7A", linewidth=1.2, zorder=1)
    ax.scatter(SR, EPAG, s=sizes, c=COL, alpha=0.9,
               edgecolors="white", linewidths=0.8, zorder=3)
    texts = []
    for name, x, y, s in zip(NAMES, SR, EPAG, sizes):
        dx = 0.0016 + (s ** 0.5) / 1000 * 0.006
        t = ax.annotate(name, (x, y), xytext=(x + dx, y), fontsize=12,
                        va="center", ha="left", color="#111111", zorder=4)
        texts.append(t)
    ax.set_xlim(SR.min() - 0.012, SR.max() + 0.02)
    ax.set_ylim(EPAG.min() - 1.6, EPAG.max() + 1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlabel("Success Rate", fontsize=15, fontweight="bold")
    ax.set_ylabel("EPA per Game", fontsize=15, fontweight="bold")
    ax.grid(True, color="#E9E9E9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#CCCCCC")
    ax.tick_params(labelsize=13, length=0)
    return texts, sizes


def new_figure(subtitle=None, note=None):
    fig, ax = plt.subplots(figsize=(20, 13), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    texts, sizes = draw_base(ax)
    fig.text(0.5, 0.965, f"{SEASON} EPA per Game and Success Rate",
             ha="center", va="center", fontsize=20, fontweight="bold")
    note_text = note if note else "bubble size = EPA/play"
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="black")
        fig.text(0.5, 0.910, note_text, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="black")
    else:
        fig.text(0.5, 0.935, note_text, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="black")
    fig.text(0.015, 0.012, f"min {MINPLAYS} plays", ha="left", fontsize=11, color="#555555")
    fig.text(0.985, 0.012, CREDIT, ha="right", fontsize=11, color="#555555")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.06)
    return fig, ax, texts, sizes


def save_base():
    fig, ax, _t, _s = new_figure()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = OUTPUT / "epa_sr_base.png"
    fig.savefig(p, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")
    return p


def circle_one(idx: int, out_dir: Path):
    """One chart with QB #idx's bubble AND name enclosed by a black oval."""
    epa_rank = int(np.sum(EPAG > EPAG[idx])) + 1   # QB1 = highest EPA/game
    sr_rank  = int(np.sum(SR   > SR[idx]))   + 1   # QB1 = highest success rate
    sub = (f"{full_name(NAMES[idx])}: EPA/Game {EPAG[idx]:.1f} (QB{epa_rank}), "
           f"Success Rate {SR[idx]*100:.0f}% (QB{sr_rank})")
    epa_play_rank = int(np.sum(EPA_PLAY > EPA_PLAY[idx])) + 1   # QB1 = highest EPA/play
    note = (f"bubble size = EPA/play.  {full_name(NAMES[idx])}: "
            f"EPA/play {EPA_PLAY[idx]:+.2f} (QB{epa_play_rank})")
    fig, ax, texts, sizes = new_figure(subtitle=sub, note=note)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mx, my = ax.transData.transform((SR[idx], EPAG[idx]))
    r_pix = np.sqrt(sizes[idx] / np.pi) * (fig.dpi / 72.0)
    tb = texts[idx].get_window_extent(renderer)

    x0 = min(mx - r_pix, tb.x0); x1 = max(mx + r_pix, tb.x1)
    y0 = min(my - r_pix, tb.y0); y1 = max(my + r_pix, tb.y1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pad = 8
    width = (x1 - x0) * 1.42 + 2 * pad
    height = (y1 - y0) * 1.42 + 2 * pad

    ax.add_patch(Ellipse((cx, cy), width=width, height=height, fill=False,
                         edgecolor="black", linewidth=3.2, zorder=6,
                         transform=IdentityTransform(), clip_on=False))
    p = out_dir / f"{slugify(full_name(NAMES[idx]))}.png"
    fig.savefig(p, dpi=100, facecolor="white")
    plt.close(fig)
    return p


def circle_all():
    missing = [n for n in NAMES if n not in FULL_NAMES]
    if missing:
        print("  ! no full name mapped for: " + ", ".join(missing))
        print("    (files use the abbreviated name — add them to FULL_NAMES)")
    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [circle_one(i, out_dir) for i in range(len(NAMES))]
    print(f"wrote {len(made)} circled charts -> {out_dir}/")
    return made


if __name__ == "__main__":
    load_data()
    save_base()
    circle_all()
