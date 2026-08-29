#!/usr/bin/env python
"""
Evaluation Dataset Diagrams
============================
Generates publication-ready figures that describe the *structure* of the LawBot
evaluation dataset (evalData_populated.json).  Intended for the project report.

Figures produced (saved to charts/):
  ds1_overview_pie_grid.png     – 3-panel pie chart: category / domain / difficulty
  ds2_domain_difficulty_bar.png – Stacked bar: questions per domain, split by difficulty
  ds3_category_domain_heatmap.png – Heatmap: category × domain question counts
  ds4_category_difficulty_bar.png – Grouped bar: category × difficulty

Usage:
    python generate_dataset_diagrams.py
    python generate_dataset_diagrams.py --dataset evalData_populated.json --out charts/
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ── Colour palettes ─────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "definition":        "#5BA4CF",
    "simple_fact":       "#57A773",
    "conditional_logic": "#E07B54",
    "procedural":        "#A37CC4",
    "special_provision": "#F2C14E",
    "section_lookup":    "#E06C9F",
    "out_of_scope":      "#9E9E9E",
}

DOMAIN_COLORS = {
    "marriage_law":      "#5BA4CF",
    "divorce_law":       "#E07B54",
    "tax_law":           "#A37CC4",
    "current_affairs":   "#F2C14E",
    "cryptocurrency":    "#9E9E9E",
}

DIFFICULTY_COLORS = {
    "easy":   "#57A773",
    "medium": "#F2C14E",
    "hard":   "#E07B54",
}

CATEGORY_LABELS = {
    "definition":        "Definition",
    "simple_fact":       "Simple Fact",
    "conditional_logic": "Conditional Logic",
    "procedural":        "Procedural",
    "special_provision": "Special Provision",
    "section_lookup":    "Section Lookup",
    "out_of_scope":      "Out-of-Scope",
}

DOMAIN_LABELS = {
    "marriage_law":      "Marriage Law",
    "divorce_law":       "Divorce Law",
    "tax_law":           "Tax Law",
    "current_affairs":   "Current Affairs",
    "cryptocurrency":    "Cryptocurrency",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _autopct(values):
    """Closure that shows percentage + count on wedges ≥ 5 %."""
    total = sum(values)
    def inner(pct):
        count = int(round(pct * total / 100))
        return f"{pct:.1f}%\n(n={count})" if pct >= 5 else ""
    return inner


# ── Figure 1: 3-panel pie grid ───────────────────────────────────────────────

def fig_overview_pie_grid(data: list[dict], out_dir: Path) -> Path:
    cats   = Counter(d.get("category", "unknown") for d in data)
    domains = Counter(d.get("legal_domain", "unknown") for d in data)
    diffs  = Counter(d.get("difficulty", "unknown") for d in data)

    cat_order   = ["definition", "simple_fact", "conditional_logic",
                    "procedural", "special_provision", "section_lookup", "out_of_scope"]
    dom_order   = ["marriage_law", "divorce_law",
                    "tax_law", "current_affairs", "cryptocurrency"]
    diff_order  = ["easy", "medium", "hard"]

    cat_order   = [c for c in cat_order if c in cats]
    dom_order   = [d for d in dom_order if d in domains]
    diff_order  = [d for d in diff_order if d in diffs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"LawBot Evaluation Dataset Overview  (N = {len(data)} questions)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    panels = [
        (axes[0], cats,    cat_order,  CATEGORY_COLORS,  CATEGORY_LABELS,  "Question Category"),
        (axes[1], domains, dom_order,  DOMAIN_COLORS,    DOMAIN_LABELS,    "Legal Domain"),
    ]

    for ax, counter, order, color_map, label_map, title in panels:
        values = [counter[k] for k in order]
        colors = [color_map.get(k, "#CCCCCC") for k in order]
        labels = [
            (label_map.get(k, k.replace("_", " ").title()) if label_map else k.title())
            for k in order
        ]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=_autopct(values),
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(11)

        ax.set_title(title, fontweight="bold", pad=14, fontsize=14)

        # Legend below
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(order))]
        ax.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=2,
            fontsize=11,
            frameon=False,
        )

    plt.tight_layout(w_pad=-4)
    out_path = out_dir / "ds1_overview_pie_grid.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Figure 2: Domain × Difficulty stacked bar ────────────────────────────────

def fig_domain_difficulty_bar(data: list[dict], out_dir: Path) -> Path:
    domains    = ["marriage_law", "divorce_law",
                   "tax_law", "current_affairs", "cryptocurrency"]
    diffs      = ["easy", "medium", "hard"]
    dom_labels = [DOMAIN_LABELS.get(d, d.replace("_", " ").title()) for d in domains]

    # Build matrix [domain × difficulty]
    matrix: dict[str, dict[str, int]] = {
        dom: Counter(
            d.get("difficulty") for d in data if d.get("legal_domain") == dom
        )
        for dom in domains
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bottom = np.zeros(len(domains))
    bars_per_diff = {}
    for diff in diffs:
        values = np.array([matrix[dom].get(diff, 0) for dom in domains], dtype=float)
        bars = ax.bar(
            dom_labels, values,
            bottom=bottom,
            color=DIFFICULTY_COLORS[diff],
            label=diff.title(),
            edgecolor="white", linewidth=0.8,
        )
        bars_per_diff[diff] = (bars, values, bottom.copy())
        # Add count labels inside segments that are tall enough
        for bar, val in zip(bars, values):
            if val >= 1.5:
                rect = bar
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    str(int(val)),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
        bottom += values

    # Total annotations on top
    totals = np.array([sum(matrix[dom].values()) for dom in domains])
    for i, (x_label, total) in enumerate(zip(dom_labels, totals)):
        ax.text(i, total + 0.3, str(int(total)),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#333333")

    ax.set_title("Questions per Legal Domain by Difficulty", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Questions")
    ax.set_ylim(0, max(totals) + 4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Difficulty", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = out_dir / "ds2_domain_difficulty_bar.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Figure 3: Category × Domain heatmap ──────────────────────────────────────

def fig_category_domain_heatmap(data: list[dict], out_dir: Path) -> Path:
    categories = ["definition", "simple_fact", "conditional_logic",
                   "procedural", "special_provision", "section_lookup", "out_of_scope"]
    domains    = ["marriage_law", "divorce_law",
                   "tax_law", "current_affairs", "cryptocurrency"]

    cat_labels = [CATEGORY_LABELS.get(c, c) for c in categories]
    dom_labels = [DOMAIN_LABELS.get(d, d.replace("_", " ").title()) for d in domains]

    # Build count matrix
    matrix = np.zeros((len(categories), len(domains)), dtype=int)
    for item in data:
        cat = item.get("category", "")
        dom = item.get("legal_domain", "")
        if cat in categories and dom in domains:
            matrix[categories.index(cat), domains.index(dom)] += 1

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("lawbot", ["#ffffff", "#1565C0"], N=256)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(dom_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(cat_labels, fontsize=9)

    # Cell annotations
    for r in range(len(categories)):
        for c in range(len(domains)):
            val = matrix[r, c]
            text_color = "white" if val > matrix.max() * 0.6 else "#333333"
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Number of Questions", fontsize=9)

    ax.set_title("Dataset Distribution: Question Category × Legal Domain",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Legal Domain", fontsize=10)
    ax.set_ylabel("Question Category", fontsize=10)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(domains)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(categories)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    out_path = out_dir / "ds3_category_domain_heatmap.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Figure 4: Category × Difficulty grouped bar ──────────────────────────────

def fig_category_difficulty_bar(data: list[dict], out_dir: Path) -> Path:
    categories = ["definition", "simple_fact", "conditional_logic",
                   "procedural", "special_provision", "section_lookup", "out_of_scope"]
    diffs      = ["easy", "medium", "hard"]
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in categories]

    matrix: dict[str, dict[str, int]] = {
        cat: Counter(d.get("difficulty") for d in data if d.get("category") == cat)
        for cat in categories
    }

    x      = np.arange(len(categories))
    width  = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, diff in enumerate(diffs):
        values = [matrix[cat].get(diff, 0) for cat in categories]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, values,
            width=width,
            color=DIFFICULTY_COLORS[diff],
            label=diff.title(),
            edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    str(val),
                    ha="center", va="bottom", fontsize=8.5,
                )

    ax.set_title("Question Category Distribution by Difficulty",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Questions")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=18, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(title="Difficulty", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = out_dir / "ds4_category_difficulty_bar.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Figure 5: Required Dafa coverage summary ─────────────────────────────────

def fig_dafa_coverage(data: list[dict], out_dir: Path) -> Path:
    """Histogram of how many dafas each question requires."""
    dafa_counts = [len(d.get("required_dafas", [])) for d in data]

    counter = Counter(dafa_counts)
    max_dafas = max(counter)
    xs = list(range(0, max_dafas + 1))
    ys = [counter.get(v, 0) for v in xs]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(
        [str(x) for x in xs], ys,
        color="#5BA4CF", edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars, ys):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=9.5,
            )

    ax.set_title("Distribution of Required Legal Provisions (Dafa) per Question",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Required Dafas")
    ax.set_ylabel("Number of Questions")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    mean_dafas = np.mean(dafa_counts)
    ax.axvline(
        mean_dafas, color="#E07B54", linestyle="--", linewidth=1.8,
        label=f"Mean = {mean_dafas:.2f}",
    )
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = out_dir / "ds5_dafa_coverage.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dataset diagrams.")
    parser.add_argument(
        "--dataset",
        default="evalData_populated.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--out",
        default="charts",
        help="Output directory for chart images.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = script_dir / dataset_path

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    data = load_dataset(dataset_path)
    print(f"  Total questions: {len(data)}")
    print(f"Output directory: {out_dir}")
    print()

    print("Generating diagrams...")
    fig_overview_pie_grid(data, out_dir)
    fig_domain_difficulty_bar(data, out_dir)
    fig_category_domain_heatmap(data, out_dir)
    fig_category_difficulty_bar(data, out_dir)
    fig_dafa_coverage(data, out_dir)

    print("\nDone. All dataset diagrams saved.")


if __name__ == "__main__":
    main()
