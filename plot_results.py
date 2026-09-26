"""根据汇总结果绘制结构对照和 MLP 容量对照。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize import tables


ROOT = Path(__file__).resolve().parent


def make_figures():
    result = tables()
    import json
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / "report" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42,
        "svg.fonttype": "none", "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), constrained_layout=True)
    structure = result["structure"].copy()
    structure["order"] = structure.apply(lambda r: {"body": 0, "joined": 1, "separate": 2}[r["view"]]
                                         + (r.title_weight / 10 if r["view"] == "separate" else 0), axis=1)
    structure = structure.sort_values("order")
    labels = [{"body": "Body", "joined": "Title + body", "separate": f"Separate ({r.title_weight:g})"}[r["view"]]
              for _, r in structure.iterrows()]
    scores = structure.macro_f1.to_numpy() * 100
    axes[0].plot(np.arange(len(scores)), scores, marker="o", color="#236B8E", lw=1.4, ms=4)
    for i, value in enumerate(scores):
        axes[0].annotate(f"{value:.2f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set_xticks(np.arange(len(scores)), labels, rotation=22, ha="right")
    axes[0].set_ylim(scores.min() - 0.7, scores.max() + 1.0)
    axes[0].set_ylabel("Validation macro-F1 (%)")
    axes[0].set_title("a  Title and body", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.18)
    capacity = result["capacity"]
    for alpha, color in [(0.0001, "#236B8E"), (1.0, "#B66B35")]:
        subset = capacity[capacity.params.map(lambda p: json.loads(p)["alpha"] == alpha)]
        ordered = [subset[subset.params.map(lambda p: json.loads(p)["hidden"] == h)].iloc[0]
                   for h in [[64], [128], [64, 32]]]
        axes[1].plot(range(3), [r.macro_f1 * 100 for r in ordered], marker="o", color=color,
                     label=f"Validation, alpha={alpha:g}", lw=1.4, ms=4)
        axes[1].plot(range(3), [r.train_macro_f1 * 100 for r in ordered], marker=".",
                     color=color, linestyle="--", alpha=0.7, label=f"Train, alpha={alpha:g}", lw=1)
    axes[1].set_xticks(range(3), ["64", "128", "64 + 32"])
    axes[1].set_xlabel("Hidden layer width")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_title("b  MLP capacity", loc="left", fontweight="bold")
    axes[1].legend(fontsize=6.3, loc="best")
    axes[1].grid(axis="y", alpha=0.18)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"comparisons.{extension}", dpi=220)
    plt.close(fig)

    frame = result["all_runs"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), constrained_layout=True)
    for ax, model, parameter in zip(axes, ["NB", "SVM", "LR"], ["alpha", "C", "C"]):
        for representation, color in [("bow", "#B66B35"), ("tfidf", "#236B8E")]:
            sub = frame[(frame.seed == config["seed"]) & (frame.stage == "comparison") & (frame.model == model)
                        & (frame.representation == representation)].copy()
            sub["parameter"] = sub.params.map(lambda p: json.loads(p)[parameter])
            sub = sub.sort_values("parameter")
            ax.semilogx(sub.parameter, sub.macro_f1 * 100, "o-", color=color, label=representation.upper(), ms=4)
        ax.set_title(model)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Validation macro-F1 (%)")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7)
    fig.savefig(output / "parameter_search.png", dpi=180)
    fig.savefig(output / "parameter_search.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figures()
