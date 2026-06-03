#!/usr/bin/env python
"""Plot performance vs latency (or throughput) from a markdown table.

Usage:
    python plot_latency_performance.py [markdown_file] [--output OUTPUT] [--throughput]
           [--theme THEME] [--context CONTEXT] [--fontsize FS]

If no file is given, defaults to new-latency.md in the same directory.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMALL_THRESHOLD = 150  # Models below this (in M params) are "small"
FONT_FAMILY = "Montserrat"

IBM_COLOR = "#0f62fe"        # IBM Design blue (R2)
IBM_COLOR_R1 = "#a6c8ff"     # Lighter IBM blue for older (R1) models
OTHER_GRAY = "#cccccc"       # Light gray for non-IBM models (fill)
OTHER_ALPHA = 0.60           # Non-IBM fill opacity
R2_RING_SCALE = 2.2          # Outer-circle size multiplier for R2 models

SHADOW_OFFSET = (1.5, -1.5)
SHADOW_COLOR = "#00000022"

DARK_THEMES = {"dark", "darkgrid"}
SEABORN_STYLES = {"whitegrid", "darkgrid", "white", "dark", "ticks"}
SEABORN_CONTEXTS = {"paper", "notebook", "talk", "poster"}


# ---------------------------------------------------------------------------
# Theme-adaptive color palette
# ---------------------------------------------------------------------------
def _make_palette(style: str) -> dict:
    """Return a dict of UI colors adapted to the current theme."""
    dark = style in DARK_THEMES
    return dict(
        # Seaborn rc overrides
        axes_face   = "#2b2b2b" if dark else "#f8f9fa",
        fig_face    = "#1e1e1e" if dark else "white",
        grid        = "#3a3a3a" if dark else "#e4e4e4",
        edge        = "#555555" if dark else "#999999",
        tick        = "#aaaaaa" if dark else "#555555",
        label       = "#cccccc" if dark else "#333333",
        title       = "#eeeeee" if dark else "#1a1a1a",
        text_global = "#cccccc" if dark else "#333333",
        # Connectors, arrows, legend
        connector   = "#666666" if dark else "#bbbbbb",
        arrow       = "#aaaaaa" if dark else "#888888",
        legend_edge = "#444444" if dark else "#dddddd",
        legend_marker = "#aaaaaa" if dark else "#555555",
        # Text halo (matches axes background)
        halo        = "#2b2b2b" if dark else "white",
        # Model label text (lighter / more muted than data colors)
        label_text  = "#bbbbbb" if dark else "#999999",
        # R2 label text (high contrast)
        label_r2    = "white" if dark else "black",
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_markdown_table(filepath: str):
    """Parse a markdown table and return list of row dicts."""
    rows = []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    header_line = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "Model" in line:
            header_line = i
            break
    if header_line is None:
        sys.exit("Could not find table header in markdown file.")

    headers = [h.strip() for h in lines[header_line].split("|")[1:-1]]
    data_start = header_line + 2

    for line in lines[data_start:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _draw_orthogonal_connector(ax, x_text, y_text, x_point, y_point, color):
    """Draw an L-shaped connector from text center to data point."""
    dx = abs(x_text - x_point)
    dy = abs(y_text - y_point)

    if dx > dy:
        ax.plot([x_text, x_point, x_point], [y_text, y_text, y_point],
                color=color, lw=0.7, zorder=3, solid_capstyle="round")
    else:
        ax.plot([x_text, x_text, x_point], [y_text, y_point, y_point],
                color=color, lw=0.7, zorder=3, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot performance vs latency/throughput from a markdown table")
    parser.add_argument("markdown_file", nargs="?",
                        default=str(Path(__file__).parent / "new-latency.md"))
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: save as .png next to input). "
                             "Format is inferred from extension (.svg, .pdf, .png).")
    parser.add_argument("--format", "-f", default=None,
                        choices=["png", "svg", "pdf"],
                        help="Output format when no --output is given (default: png). "
                             "svg and pdf are vector formats editable in PowerPoint/Illustrator.")
    parser.add_argument("--throughput", "-t", action="store_true",
                        help="Plot throughput (samples/s) on x-axis instead of latency")
    parser.add_argument("--fontsize", "-fs", type=float, default=14,
                        help="Base font size in points (default: 14)")
    parser.add_argument("--theme", default="whitegrid",
                        choices=sorted(SEABORN_STYLES),
                        help="Seaborn style (default: whitegrid)")
    parser.add_argument("--context", default="talk",
                        choices=sorted(SEABORN_CONTEXTS),
                        help="Seaborn context / scale (default: talk)")
    args = parser.parse_args()

    rows = parse_markdown_table(args.markdown_file)
    pal = _make_palette(args.theme)

    # -- Seaborn theme ------------------------------------------------------
    fs = args.fontsize
    sns.set_theme(
        style=args.theme,
        context=args.context,
        font=FONT_FAMILY,
        font_scale=fs / 12,
        rc={
            "axes.facecolor": pal["axes_face"],
            "figure.facecolor": pal["fig_face"],
            "grid.color": pal["grid"],
            "grid.linewidth": 0.6,
            "axes.edgecolor": pal["edge"],
            "axes.linewidth": 0.7,
            "axes.labelcolor": pal["label"],
            "text.color": pal["text_global"],
            "xtick.color": pal["tick"],
            "ytick.color": pal["tick"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    # -- Extract data -------------------------------------------------------
    models, latencies, throughputs, performances, sizes = [], [], [], [], []
    for row in rows:
        model_full = re.sub(r'\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*$', '', row["Model"])
        perf_str = row.get("MTEB Multilingual Retrieval", "").strip()
        if perf_str in ("\u2014", "-", ""):
            continue
        models.append(model_full)
        latencies.append(float(row["Latency"]))
        throughputs.append(float(row["Throughput"]))
        performances.append(float(perf_str))
        sizes.append(int(row["Size (M)"]))

    latencies = np.array(latencies)
    throughputs = np.array(throughputs)
    performances = np.array(performances)
    sizes = np.array(sizes)

    if args.throughput:
        x_vals, x_label, x_tag = throughputs, "Throughput (samples / s)", "throughput"
    else:
        x_vals, x_label, x_tag = latencies, "Latency (ms)", "latency"

    # -- Marker sizes (area proportional to params) -------------------------
    min_s, max_s = sizes.min(), sizes.max()
    if max_s > min_s:
        marker_sizes = 120 + (sizes - min_s) / (max_s - min_s) * 500
    else:
        marker_sizes = np.full_like(sizes, 300, dtype=float)

    # -- Figure setup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))

    shadow_effect = [
        pe.withSimplePatchShadow(
            offset=SHADOW_OFFSET, shadow_rgbFace=SHADOW_COLOR,
            alpha=0.25, rho=0.3,
        )
    ]

    # -- Plot each model ----------------------------------------------------
    texts = []
    point_coords = []

    # Tiny rightward nudge so labels start right next to their dot
    x_off = (x_vals.max() - x_vals.min()) * 0.008 or 0.1

    for i, model in enumerate(models):
        is_ibm = "ibm-granite" in model.lower() or "granite" in model.lower().split("/")[0]
        is_r2 = is_ibm and "r2" in model.lower()
        short = model.split("/")[-1]
        is_small = sizes[i] < SMALL_THRESHOLD

        if is_r2:
            color = IBM_COLOR
            alpha = 0.90
        elif is_ibm:
            color = IBM_COLOR_R1
            alpha = 0.90
        else:
            color = OTHER_GRAY
            alpha = OTHER_ALPHA

        # marker = "D" if is_small else "o"   # original: diamond for small
        # hatch = "////" if is_small else None
        marker = "o"
        hatch = None

        sc = ax.scatter(
            x_vals[i], performances[i],
            s=marker_sizes[i],
            facecolors=color,
            edgecolors=color,
            linewidths=2 if is_ibm else 1.5,
            marker=marker, zorder=5,
            hatch=hatch, alpha=alpha,
        )
        sc.set_path_effects(shadow_effect)

        # Extra outer ring for R2 models
        if is_r2:
            ax.scatter(
                x_vals[i], performances[i],
                s=marker_sizes[i] * R2_RING_SCALE,
                facecolors="none",
                edgecolors=IBM_COLOR,
                linewidths=1.5,
                marker="o", zorder=4,
                alpha=0.70,
            )

        label_text = f"{short} ({sizes[i]}M)"
        label_color = pal["label_r2"] if is_r2 else pal["label_text"]
        label_weight = "bold" if is_ibm else "normal"
        txt = ax.annotate(
            label_text,
            xy=(x_vals[i], performances[i]),
            xytext=(x_vals[i] + x_off, performances[i]),
            fontsize=fs - 3, fontfamily=FONT_FAMILY,
            fontweight=label_weight, color=label_color,
            ha="left", va="center", zorder=6,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground=pal["halo"]),
            ],
        )
        texts.append(txt)
        point_coords.append((x_vals[i], performances[i]))

    # Label repulsion — push labels clear of dots and each other
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="none", lw=0, shrinkA=0, shrinkB=0),
        expand=(1.3, 1.6),
        force_text=(0.6, 0.8),
        force_static=(0.3, 0.5),
        force_points=(0.5, 0.7),
        ensure_inside_axes=True,
    )

    # Connectors — only when a label was pushed vertically away from its dot
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for txt, (px, py) in zip(texts, point_coords):
        bbox = txt.get_window_extent(renderer=renderer)
        corners = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        x0d, x1d = min(corners[:, 0]), max(corners[:, 0])
        y0d, y1d = min(corners[:, 1]), max(corners[:, 1])
        label_cy = (y0d + y1d) / 2
        label_h = y1d - y0d

        # Skip if the dot's Y is still within half a label-height of the label center
        if abs(label_cy - py) <= label_h * 0.5:
            continue

        tx = float(np.clip(px, x0d, x1d))
        ty = float(np.clip(py, y0d, y1d))
        _draw_orthogonal_connector(ax, tx, ty, px, py, pal["connector"])

    # -- Axes labels & title ------------------------------------------------
    ax.set_xlabel(x_label, fontsize=fs + 1, fontweight="medium",
                  color=pal["label"], labelpad=10)
    ax.set_ylabel("MTEB Multilingual Retrieval", fontsize=fs + 1, fontweight="medium",
                  color=pal["label"], labelpad=10)
    ax.set_title(
        f"Embedding Model: Performance vs {x_tag.title()}",
        fontsize=fs + 5, fontweight="bold", color=pal["title"], pad=18,
    )

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # Pad axis limits
    x_margin = (x_vals.max() - x_vals.min()) * 0.10
    y_margin = (performances.max() - performances.min()) * 0.10
    ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
    ax.set_ylim(performances.min() - y_margin, performances.max() + y_margin)

    # -- Legend -------------------------------------------------------------
    lm = pal["legend_marker"]
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=IBM_COLOR_R1,
                   markeredgecolor=IBM_COLOR_R1, markersize=10, markeredgewidth=1.5,
                   label="IBM Granite (R1)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=IBM_COLOR,
                   markeredgecolor=IBM_COLOR, markersize=8, markeredgewidth=1.5,
                   label="IBM Granite R2  \u25cb"),  # ○ hints at outer ring
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=OTHER_GRAY,
                   markeredgecolor=OTHER_GRAY, markersize=10, markeredgewidth=1.5,
                   alpha=OTHER_ALPHA, label="Other models"),
        plt.Line2D([0], [0], marker="D", color="w", markeredgecolor=lm,
                   markerfacecolor="none", markersize=8, markeredgewidth=1.5,
                   label="Small (<150M, hatched)"),
    ]
    legend_loc = "lower left" if args.throughput else "lower right"
    leg = ax.legend(
        handles=legend_elements, loc=legend_loc, fontsize=fs - 3,
        frameon=True, framealpha=0.90, edgecolor=pal["legend_edge"],
        fancybox=True, borderpad=1.0, labelspacing=1.0,
        shadow=True,
    )
    leg.get_frame().set_linewidth(0.5)

    # -- Save ---------------------------------------------------------------
    plt.tight_layout()

    # -- Gradient colour bars inside axes (better direction = dark blue) -----
    GRAD_DARK = "#0f62fe"
    GRAD_LIGHT = "#d0e2ff"
    GRAD_ALPHA = 0.45
    BAR_FRAC = 0.025           # bar thickness as fraction of axis range

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    # Horizontal bar (just above the bottom spine, inside the plot)
    # Latency: left = better (dark) → right = worse (light)
    # Throughput: left = worse (light) → right = better (dark)
    h_cmap = mcolors.LinearSegmentedColormap.from_list(
        "h_grad",
        [GRAD_LIGHT, GRAD_DARK] if args.throughput else [GRAD_DARK, GRAD_LIGHT],
    )
    ax.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),
        aspect="auto", cmap=h_cmap, alpha=GRAD_ALPHA,
        extent=[xlim[0], xlim[1], ylim[0], ylim[0] + y_span * BAR_FRAC],
        zorder=1, interpolation="bicubic",
    )

    # Vertical bar (just right of the left spine, inside the plot)
    # Bottom = worse (light), top = better (dark)
    v_cmap = mcolors.LinearSegmentedColormap.from_list(
        "v_grad", [GRAD_LIGHT, GRAD_DARK],
    )
    ax.imshow(
        np.linspace(0, 1, 256).reshape(-1, 1),
        aspect="auto", cmap=v_cmap, alpha=GRAD_ALPHA, origin="lower",
        extent=[xlim[0], xlim[0] + x_span * BAR_FRAC, ylim[0], ylim[1]],
        zorder=1, interpolation="bicubic",
    )

    if args.output:
        out_path = Path(args.output)
    else:
        fmt = args.format or "png"
        stem = Path(args.markdown_file).stem
        out_path = Path(args.markdown_file).parent / f"{stem}_{x_tag}.{fmt}"

    fmt = out_path.suffix.lstrip(".").lower() or "png"
    save_kw = dict(bbox_inches="tight", facecolor=fig.get_facecolor())
    if fmt == "png":
        save_kw["dpi"] = 180
    elif fmt == "svg":
        # Emit real <text> elements so PowerPoint preserves individual characters
        plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_path, format=fmt, **save_kw)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
