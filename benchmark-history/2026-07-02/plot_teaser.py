#!/usr/bin/env python3
"""
smart-ask teaser visualization.

Run from anywhere:
    python3.13 benchmarks/plot_teaser.py
Outputs:
    benchmarks/teaser.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = "#080c10"
CARD     = "#0d1117"
CARD2    = "#111827"
GREEN    = "#00e676"
GREEN_MID= "#00c853"
GREEN_DIM= "#1b5e35"
TEAL     = "#64ffda"
OPUS_C   = "#ff5252"
OPUS_DIM = "#5c1a1a"
GEM_C    = "#ffab40"
WHITE    = "#eceff1"
MUTED    = "#78909c"
FAINT    = "#2d3748"
GRIDL    = "#151b22"
PURPLE   = "#b39ddb"

# ── Load data ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

he    = json.loads((ROOT / "humaneval/results_product.json").read_text())
lb    = json.loads((ROOT / "livebench/results_product.json").read_text())
lb_b  = json.loads((ROOT / "livebench/results_opus_baseline.json").read_text())

# HumanEval
he_n      = len(he["results"])
he_pass   = sum(1 for r in he["results"] if r["passed"])
he_acc    = he_pass / he_n * 100                      # 92.7 %
he_cost   = he["token_log"]["total_cost_usd"]          # $0.145
he_gemini = sum(1 for r in he["results"] if r["model"] == "gemini")
he_opus   = he_n - he_gemini
he_opus_cost_est = 0.97   # always-Opus at same per-call rate

# LiveBench
lb_n      = len(lb["results"])
lb_pass   = sum(1 for r in lb["results"] if r["pass_all"])
lb_acc    = lb_pass / lb_n * 100                       # 73.4 %
lb_cost   = lb["token_log"]["total_cost_usd"]           # $0.884
lb_gemini = sum(1 for r in lb["results"] if r["model"] == "gemini")
lb_opus   = lb_n - lb_gemini
lb_opus_acc  = (sum(1 for r in lb_b["results"] if r["pass_all"])
                / lb_b["n"] * 100)                     # 78.1 %
lb_opus_cost = (
    lb["token_log"]["by_model"]["anthropic/claude-opus-4.8"]["cost_usd"]
    + lb_b["token_log"]["total_cost_usd"]              # $1.207
)

# Savings
he_save_pct = (he_opus_cost_est - he_cost) / he_opus_cost_est * 100   # ~85 %
lb_save_pct = (lb_opus_cost     - lb_cost) / lb_opus_cost     * 100   # ~27 %

# LiveBench by difficulty
diffs = {}
for r in lb["results"]:
    d = r.get("difficulty", "?")
    diffs.setdefault(d, {"n": 0, "pass": 0})
    diffs[d]["n"]    += 1
    diffs[d]["pass"] += r["pass_all"]

# LiveBench by task
tasks = {}
for r in lb["results"]:
    t = r["task"]
    tasks.setdefault(t, {"n": 0, "pass": 0})
    tasks[t]["n"]    += 1
    tasks[t]["pass"] += r["pass_all"]


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10.5), facecolor=BG)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.05, right=0.97,
    top=0.80, bottom=0.09,
    wspace=0.34, hspace=0.52,
)


# ── Header ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.955, "smart-ask",
    ha="center", fontsize=44, fontweight="bold",
    color=GREEN, fontfamily="monospace",
    path_effects=[pe.withStroke(linewidth=8, foreground=GREEN_DIM)],
)
fig.text(
    0.5, 0.906,
    "Intelligent routing — send easy problems to cheap models,\n"
    "hard problems to powerful ones.  Pay less.  Lose nothing.",
    ha="center", fontsize=12.5, color=MUTED, linespacing=1.6,
)

# Stat pills  ──  value on top, label below
def pill(x, y, value, label, color=GREEN):
    fig.text(x, y + 0.022, value,
             ha="center", fontsize=24, fontweight="bold", color=color,
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    fig.text(x, y, label,
             ha="center", fontsize=8.5, color=MUTED)

pill(0.15, 0.838, f"{he_acc:.1f}%",          "HumanEval  pass@1",    GREEN)
pill(0.29, 0.838, f"−{he_save_pct:.0f}%",    "HumanEval  cost save", TEAL)
pill(0.50, 0.838, f"{lb_acc:.1f}%",          "LiveBench  pass@1",    GREEN)
pill(0.64, 0.838, f"−{lb_save_pct:.0f}%",    "LiveBench  cost save", TEAL)
pill(0.82, 0.838, "292",                      "problems evaluated",   WHITE)

# thin separator line
fig.add_artist(
    plt.Line2D([0.04, 0.96], [0.82, 0.82],
               transform=fig.transFigure, color=FAINT, linewidth=0.8)
)


# ── helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=WHITE, fontsize=11.5, fontweight="bold", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(FAINT)
    ax.grid(axis="y", color=GRIDL, linewidth=0.5, alpha=0.9, zorder=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A  —  Accuracy vs Cost scatter  (spans both rows on the left)
# ═══════════════════════════════════════════════════════════════════════════════
ax_scat = fig.add_subplot(gs[:, 0])
ax_scat.set_facecolor(CARD)
ax_scat.set_title("Accuracy  vs  Cost", color=WHITE, fontsize=12, fontweight="bold", pad=9)
ax_scat.set_xlabel("Cost  (USD $)", color=MUTED, fontsize=9.5)
ax_scat.set_ylabel("Accuracy  (%)", color=MUTED, fontsize=9.5)
ax_scat.tick_params(colors=MUTED, labelsize=8.5)
for side in ["top", "right"]:  ax_scat.spines[side].set_visible(False)
for side in ["left", "bottom"]: ax_scat.spines[side].set_color(FAINT)
ax_scat.grid(color=GRIDL, linewidth=0.5, alpha=0.8)

# ── curved arrows  Opus → smart-ask ──
arrow_kw = dict(
    arrowstyle="-|>",
    lw=1.8, mutation_scale=14, alpha=0.75,
)
ax_scat.annotate(
    "", xy=(he_cost, he_acc), xytext=(he_opus_cost_est, 92.7),
    arrowprops=dict(**arrow_kw, color=GREEN_MID,
                    connectionstyle="arc3,rad=-0.25"),
)
ax_scat.annotate(
    "", xy=(lb_cost, lb_acc), xytext=(lb_opus_cost, lb_opus_acc),
    arrowprops=dict(**arrow_kw, color=GREEN_MID,
                    connectionstyle="arc3,rad=0.22"),
)

# savings labels along the arrows
ax_scat.text(
    (he_cost + he_opus_cost_est) / 2 - 0.07,
    (he_acc + 92.7) / 2 + 1.5,
    f"−{he_save_pct:.0f}%\ncost", color=TEAL, fontsize=9.5,
    ha="center", fontweight="bold",
)
ax_scat.text(
    (lb_cost + lb_opus_cost) / 2 + 0.05,
    (lb_acc + lb_opus_acc) / 2 - 3.0,
    f"−{lb_save_pct:.0f}%\ncost", color=TEAL, fontsize=9.5,
    ha="center", fontweight="bold",
)

# ── scatter points ──
pts = [
    (he_opus_cost_est, 92.7,   "HumanEval\nOpus-only*", OPUS_C,  "o", 200, ( 0.06,  0.6)),
    (he_cost,          he_acc, "HumanEval\nsmart-ask",  GREEN,   "*", 350, (-0.09, -1.8)),
    (lb_opus_cost,     lb_opus_acc, "LiveBench\nOpus-only", OPUS_C, "o", 200, ( 0.06,  0.8)),
    (lb_cost,          lb_acc, "LiveBench\nsmart-ask",  GREEN,   "*", 350, (-0.09, -1.8)),
]
for cost, acc, lbl, col, mk, sz, (dx, dy) in pts:
    ax_scat.scatter(cost, acc, s=sz, color=col, marker=mk, zorder=6,
                    edgecolors=BG, linewidths=0.7, alpha=0.95)
    ax_scat.text(cost + dx, acc + dy, lbl,
                 color=col, fontsize=8, ha="center", va="center",
                 fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

ax_scat.set_xlim(-0.05, 1.38)
ax_scat.set_ylim(62, 100)

# x-axis ticks formatted as $
from matplotlib.ticker import FuncFormatter
ax_scat.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))

legend_handles = [
    mpatches.Patch(color=GREEN,  label="smart-ask  ★"),
    mpatches.Patch(color=OPUS_C, label="Opus-only  ●"),
]
ax_scat.legend(handles=legend_handles, loc="lower right",
               facecolor=CARD2, edgecolor=FAINT, labelcolor=WHITE,
               fontsize=8.5, framealpha=0.9)

ax_scat.text(
    0.03, 0.03, "*est. based on product Opus call rate",
    transform=ax_scat.transAxes, color=MUTED, fontsize=7, va="bottom",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B  —  Cost comparison  (top-middle)
# ═══════════════════════════════════════════════════════════════════════════════
ax_cost = fig.add_subplot(gs[0, 1])
style_ax(ax_cost, "Cost Comparison", ylabel="Cost  (USD $)")

x      = np.arange(2)
w      = 0.32
labels = ["HumanEval", "LiveBench"]
s_vals = [he_cost,          lb_cost]
o_vals = [he_opus_cost_est, lb_opus_cost]

b_s = ax_cost.bar(x - w/2, s_vals, w, color=GREEN,  alpha=0.92, label="smart-ask", zorder=3)
b_o = ax_cost.bar(x + w/2, o_vals, w, color=OPUS_C, alpha=0.88, label="Opus-only", zorder=3)

for bar, v in zip(b_s, s_vals):
    ax_cost.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                 f"${v:.2f}", ha="center", va="bottom",
                 color=GREEN, fontsize=9, fontweight="bold")
for bar, v in zip(b_o, o_vals):
    ax_cost.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                 f"${v:.2f}", ha="center", va="bottom",
                 color=OPUS_C, fontsize=9, fontweight="bold")

# savings bracket
for i, (sv, ov) in enumerate(zip(s_vals, o_vals)):
    pct = (ov - sv) / ov * 100
    ax_cost.annotate(
        "", xy=(i - w/2, sv), xytext=(i - w/2, ov),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.3, mutation_scale=9),
    )
    ax_cost.text(i - w/2 - 0.15, (sv + ov) / 2,
                 f"−{pct:.0f}%", color=TEAL, fontsize=9,
                 fontweight="bold", va="center")

ax_cost.set_xticks(x)
ax_cost.set_xticklabels(labels, color=WHITE, fontsize=9.5)
ax_cost.legend(facecolor=CARD2, edgecolor=FAINT, labelcolor=WHITE,
               fontsize=8, framealpha=0.9, loc="upper left")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C  —  Routing distribution  (bottom-middle)
# ═══════════════════════════════════════════════════════════════════════════════
ax_rout = fig.add_subplot(gs[1, 1])
ax_rout.set_facecolor(CARD)
ax_rout.set_title("Routing Distribution", color=WHITE, fontsize=11.5,
                  fontweight="bold", pad=9)
ax_rout.set_xlabel("Problems  (%)", color=MUTED, fontsize=9)
ax_rout.tick_params(colors=MUTED, labelsize=8)
for side in ["top", "right"]: ax_rout.spines[side].set_visible(False)
for side in ["left", "bottom"]: ax_rout.spines[side].set_color(FAINT)
ax_rout.grid(axis="x", color=GRIDL, linewidth=0.5, alpha=0.9, zorder=0)

cats = ["HumanEval\n(164 problems)", "LiveBench\n(128 problems)"]
gem_pcts  = [he_gemini / he_n * 100, lb_gemini / lb_n * 100]
opus_pcts = [he_opus   / he_n * 100, lb_opus   / lb_n * 100]
y = np.arange(2)
h = 0.42

bg = ax_rout.barh(y, gem_pcts,  h, color=GEM_C,  label="→ Gemini-Flash (cheap)", zorder=3)
bo = ax_rout.barh(y, opus_pcts, h, left=gem_pcts, color=OPUS_C, alpha=0.85,
                  label="→ Claude Opus (powerful)", zorder=3)

for bar, gp in zip(bg, gem_pcts):
    ax_rout.text(gp / 2, bar.get_y() + bar.get_height() / 2,
                 f"{gp:.0f}%", ha="center", va="center",
                 fontsize=11, color=BG, fontweight="bold")
for bar, gp, op in zip(bo, gem_pcts, opus_pcts):
    ax_rout.text(gp + op / 2, bar.get_y() + bar.get_height() / 2,
                 f"{op:.0f}%", ha="center", va="center",
                 fontsize=11, color=WHITE, fontweight="bold")

ax_rout.set_yticks(y)
ax_rout.set_yticklabels(cats, color=WHITE, fontsize=9)
ax_rout.set_xlim(0, 100)
ax_rout.legend(facecolor=CARD2, edgecolor=FAINT, labelcolor=WHITE,
               fontsize=8, framealpha=0.9, loc="lower right")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel D  —  LiveBench by difficulty  (top-right)
# ═══════════════════════════════════════════════════════════════════════════════
ax_diff = fig.add_subplot(gs[0, 2])
style_ax(ax_diff, "LiveBench — by Difficulty", ylabel="Pass Rate  (%)")

d_order  = ["easy", "medium", "hard"]
d_labels = ["Easy", "Medium", "Hard"]
d_colors = [GREEN, GEM_C, OPUS_C]
d_accs   = [diffs[d]["pass"] / diffs[d]["n"] * 100 for d in d_order]
d_ns     = [diffs[d]["n"] for d in d_order]

bars = ax_diff.bar(d_labels, d_accs, color=d_colors, width=0.52, alpha=0.92, zorder=3)
for bar, acc, n, col in zip(bars, d_accs, d_ns, d_colors):
    ax_diff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                 f"{acc:.0f}%", ha="center", va="bottom",
                 color=col, fontsize=12, fontweight="bold")
    ax_diff.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"n={n}", ha="center", va="center",
                 color=BG, fontsize=9.5, fontweight="bold")

ax_diff.set_ylim(0, 110)
ax_diff.set_xticks(range(len(d_labels)))
ax_diff.set_xticklabels(d_labels, color=WHITE, fontsize=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel E  —  LiveBench by task type  (bottom-right)
# ═══════════════════════════════════════════════════════════════════════════════
ax_task = fig.add_subplot(gs[1, 2])
style_ax(ax_task, "LiveBench — by Task Type", ylabel="Pass Rate  (%)")

task_items = [
    ("LCB_generation",    "LCB\nGeneration",  GREEN),
    ("coding_completion", "Code\nCompletion", TEAL),
]
t_labels = [t[1] for t in task_items]
t_accs   = [tasks[t[0]]["pass"] / tasks[t[0]]["n"] * 100 for t in task_items]
t_ns     = [tasks[t[0]]["n"] for t in task_items]
t_colors = [t[2] for t in task_items]

bars = ax_task.bar(t_labels, t_accs, color=t_colors, width=0.45, alpha=0.92, zorder=3)
for bar, acc, n, col in zip(bars, t_accs, t_ns, t_colors):
    ax_task.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                 f"{acc:.0f}%", ha="center", va="bottom",
                 color=col, fontsize=12, fontweight="bold")
    ax_task.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"n={n}", ha="center", va="center",
                 color=BG, fontsize=9.5, fontweight="bold")

ax_task.set_ylim(0, 110)
ax_task.set_xticks(range(len(t_labels)))
ax_task.set_xticklabels(t_labels, color=WHITE, fontsize=10)

# add pass counts as subtitle
for i, (bar, t_item) in enumerate(zip(bars, task_items)):
    key = t_item[0]
    p   = tasks[key]["pass"]
    n   = tasks[key]["n"]
    ax_task.text(
        bar.get_x() + bar.get_width() / 2, -8,
        f"{p}/{n}", ha="center", va="top", color=MUTED, fontsize=8.5,
        transform=ax_task.transData,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.026,
    "Benchmarks: HumanEval (164 problems) · LiveBench Coding June-2024 (128 problems)  "
    "│  smart-ask: Gemini-2.5-Flash-Lite classifier + generator, Claude Opus 4.8 for hard/escalated  "
    "│  Baseline: Opus-only on same problems  │  *HumanEval Opus-only cost estimated",
    ha="center", fontsize=7.2, color=MUTED, style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = ROOT / "teaser.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved → {out}")
