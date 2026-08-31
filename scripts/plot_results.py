"""Generate publication-ready figures from results/experiment_results.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = ["#2B6CB0", "#C05621", "#2F855A", "#805AD5"]


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_bar_labels(axis, bars, decimals=1, suffix=""):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_negative_construction(data, output_dir):
    methods = data["methods"]
    names = [item["name"] for item in methods]
    panels = [
        ("contradiction_pct", "NLI Contradiction (%)", True, 1, "%"),
        ("ted", "Tree Edit Distance (lower is better)", False, 2, ""),
        ("dependency_jaccard", "Dependency-edge Jaccard", True, 3, ""),
        ("combined_score", "Combined Quality Score", True, 3, ""),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for axis, (key, title, higher_is_better, decimals, suffix) in zip(axes.flat, panels):
        values = [item[key] for item in methods]
        bars = axis.bar(names, values, color=COLORS, width=0.68)
        axis.set_title(title, pad=10)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(values) * 1.2)
        add_bar_labels(axis, bars, decimals=decimals, suffix=suffix)
        best_index = values.index(max(values) if higher_is_better else min(values))
        bars[best_index].set_edgecolor("#1A202C")
        bars[best_index].set_linewidth(2.0)

    figure.suptitle("Hard-negative Construction Quality", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "Top-K triple guidance preserves syntax while increasing semantic contradiction.",
        ha="center",
        color="#4A5568",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_dir / "negative_construction_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_heldout(data, output_dir):
    models = data["models"]
    names = [item["name"] for item in models]
    metrics = [
        ("answer_accuracy_pct", "Answer Acc."),
        ("format_ok_pct", "Format OK"),
        ("strict_key_hit_pct", "Strict Key Hit"),
        ("triple_score", "Triple Score"),
    ]
    x_positions = list(range(len(metrics)))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    for model_index, model in enumerate(models):
        values = [model[key] * (100 if key == "triple_score" else 1) for key, _ in metrics]
        offsets = [x + (model_index - 0.5) * width for x in x_positions]
        bars = axis.bar(
            offsets,
            values,
            width=width,
            label=names[model_index],
            color=COLORS[model_index],
        )
        add_bar_labels(axis, bars, decimals=1)

    axis.set_xticks(x_positions, [label for _, label in metrics])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 112)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    axis.set_title(
        f"Syntax-target Held-out Evaluation (n={data['n']})",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    figure.text(
        0.5,
        0.015,
        "Syntax RL substantially improves relation grounding, with a small answer-accuracy trade-off.",
        ha="center",
        color="#4A5568",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_dir / "heldout_model_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def draw_box(axis, x, y, width, height, title, subtitle, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", weight="bold")
    axis.text(x + width / 2, y + height * 0.31, subtitle, ha="center", va="center", fontsize=8, color="#4A5568")


def plot_pipeline(output_dir):
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = [
        (0.03, "Parse", "Dependency triples", COLORS[0]),
        (0.23, "Construct", "Controlled negatives", COLORS[2]),
        (0.43, "Verify", "Syntax + NLI", COLORS[1]),
        (0.63, "Fine-tune", "CoT SFT -> DPO -> RL", COLORS[3]),
        (0.83, "Evaluate", "Answer + relation", "#B83280"),
    ]
    width, height, y = 0.14, 0.25, 0.48
    for index, (x, title, subtitle, color) in enumerate(stages):
        draw_box(axis, x, y, width, height, title, subtitle, color)
        if index < len(stages) - 1:
            next_x = stages[index + 1][0]
            axis.add_patch(FancyArrowPatch(
                (x + width + 0.008, y + height / 2),
                (next_x - 0.008, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#4A5568",
            ))

    draw_box(axis, 0.23, 0.11, 0.14, 0.18, "Image Route", "Optional ranking", "#718096")
    axis.add_patch(FancyArrowPatch(
        (0.30, 0.29),
        (0.30, 0.47),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        linestyle="--",
        color="#718096",
    ))
    axis.text(0.5, 0.91, "Syntax-Aware MLLM Pipeline", ha="center", fontsize=16, fontweight="bold")
    axis.text(
        0.5,
        0.84,
        "Dependency triples are the primary signal; Image Route remains an optional ablation.",
        ha="center",
        color="#4A5568",
    )
    figure.savefig(output_dir / "pipeline_overview.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment_results.json")
    parser.add_argument("--output_dir", default="assets/figures")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text(encoding="utf-8"))

    configure_style()
    plot_negative_construction(data["negative_construction"], output_dir)
    plot_heldout(data["syntax_target_heldout"], output_dir)
    plot_pipeline(output_dir)
    print(f"Generated figures in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
