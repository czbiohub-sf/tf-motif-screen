import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from af3tfm.dataset import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["svg.fonttype"] = "none"

SEQ_LEN = 396

tracks = [
    {
        "db": "PFAM",
        "color": "#87a9d4",
        "domains": [
            {"label": "Paired box", "start": 16, "end": 140},
            {"label": "Pax2_C", "start": 256, "end": 361},
        ],
    },
    {
        "db": "PROSITE",
        "color": "#f4a460",
        "domains": [
            {"label": "Paired box", "start": 16, "end": 142},
        ],
    },
]


# Union = merged overlapping domains from non-CDD tracks; label from longest domain
def merge_union_domains(tracks):
    flat = [
        {**d, "color": t["color"], "db": t["db"]}
        for t in tracks
        if t["db"] != "CDD"
        for d in t["domains"]
    ]
    flat.sort(key=lambda d: d["start"])
    merged = []
    for d in flat:
        if merged and d["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            if (d["end"] - d["start"]) > (prev["end"] - prev["start"]):
                merged[-1] = {**d, "end": max(prev["end"], d["end"])}
            else:
                merged[-1]["end"] = max(prev["end"], d["end"])
        else:
            merged.append(dict(d))
    return merged


union_domains = merge_union_domains(tracks)

# ── Layout ────────────────────────────────────────────────────────────────────
N_TRACKS = len(tracks) + 1  # +1 for union row
TRACK_H = 0.5
TRACK_GAP = 0.3
ROW_STEP = TRACK_H + TRACK_GAP
DOMAIN_H = 0.35
BACKBONE_H = 0.06
PAD = 0.5

fig_h = N_TRACKS * ROW_STEP + 1.2
fig, ax = plt.subplots(figsize=(12, fig_h))
ax.set_xlim(-PAD, SEQ_LEN + PAD)
ax.set_ylim(-ROW_STEP, N_TRACKS * ROW_STEP)
ax.axis("off")


def row_y(i):
    """Bottom y of row i (0 = top row)."""
    return (N_TRACKS - 1 - i) * ROW_STEP


def draw_domain(
    ax, start, end, y_center, height, color, label, text_color="#333", gid=None
):
    w = end - start
    x = start
    y = y_center - height / 2
    fancy = FancyBboxPatch(
        (x, y),
        w,
        height,
        boxstyle=f"round,pad=0.0,rounding_size={height * 0.65}",
        linewidth=0.6,
        edgecolor="grey",
        facecolor=color,
        zorder=3,
        gid=f"{gid}_patch" if gid else None,
    )
    ax.add_patch(fancy)
    if w > 15:
        ax.text(
            x + w / 2,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
            fontweight="medium",
            zorder=4,
            clip_on=True,
            gid=f"{gid}_label" if gid else None,
        )


# ── Draw tracks ───────────────────────────────────────────────────────────────
for i, track in enumerate(tracks):
    yc = row_y(i) + TRACK_H / 2
    db = track["db"]

    # Backbone
    ax.plot(
        [0, SEQ_LEN],
        [yc, yc],
        color="#cccccc",
        linewidth=3,
        zorder=1,
        gid=f"backbone_{db}",
    )

    # DB label
    ax.text(
        -6,
        yc,
        db,
        ha="right",
        va="center",
        fontsize=8,
        color="#555",
        gid=f"track_label_{db}",
    )

    # Domains
    for d in track["domains"]:
        gid = f"domain_{db}_{d['label']}_{d['start']}"
        draw_domain(
            ax, d["start"], d["end"], yc, DOMAIN_H, track["color"], d["label"], gid=gid
        )

# ── Union row ─────────────────────────────────────────────────────────────────
union_i = len(tracks)
yc_union = row_y(union_i) + TRACK_H / 2

# Dashed separator
sep_y = yc_union + TRACK_H * 0.9
ax.plot(
    [0, SEQ_LEN],
    [sep_y, sep_y],
    color="#cccccc",
    linewidth=0.8,
    linestyle="--",
    zorder=1,
    gid="union_separator",
)

# Light purple backdrop
h_backdrop = DOMAIN_H + 0.05
backdrop = FancyBboxPatch(
    (0, yc_union - h_backdrop / 2),
    SEQ_LEN,
    h_backdrop,
    boxstyle=f"round,pad=0.0,rounding_size={h_backdrop * 0.65}",
    linewidth=0.5,
    edgecolor="#cccccc",
    facecolor="#f0e6ff",
    zorder=1,
    gid="union_backdrop",
)
ax.add_patch(backdrop)

ax.text(
    -6,
    yc_union,
    "Union",
    ha="right",
    va="center",
    fontsize=8,
    color="#6a0dad",
    fontweight="semibold",
    gid="track_label_union",
)

for d in union_domains:
    gid = f"domain_union_{d['label']}_{d['start']}"
    draw_domain(
        ax, d["start"], d["end"], yc_union, DOMAIN_H, d["color"], d["label"], gid=gid
    )

# ── Ruler ─────────────────────────────────────────────────────────────────────
ruler_y = N_TRACKS * ROW_STEP - 0.15
tick_step = 50
ax.plot(
    [0, SEQ_LEN],
    [ruler_y, ruler_y],
    color="#aaaaaa",
    linewidth=2,
    zorder=1,
    gid="ruler_line",
)
for v in range(tick_step, SEQ_LEN, tick_step):
    ax.plot(
        [v, v],
        [ruler_y, ruler_y + 0.08],
        color="#777",
        linewidth=0.8,
        gid=f"ruler_tick_{v}",
    )
    ax.text(
        v,
        ruler_y + 0.12,
        str(v),
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555",
        gid=f"ruler_label_{v}",
    )
ax.text(
    1,
    ruler_y + 0.12,
    "1",
    ha="center",
    va="bottom",
    fontsize=7,
    color="#555",
    gid="ruler_label_1",
)
ax.text(
    SEQ_LEN,
    ruler_y + 0.12,
    str(SEQ_LEN),
    ha="center",
    va="bottom",
    fontsize=7,
    color="#555",
    gid=f"ruler_label_{SEQ_LEN}",
)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/domain_architecture.svg", format="svg", bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/domain_architecture.png", dpi=150, bbox_inches="tight")
plt.show()
