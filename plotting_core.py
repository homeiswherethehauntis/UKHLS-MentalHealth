from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from importlib import import_module

config = import_module("00_config")

COLORS = {
    "Total cohesion": "#D7CEE4",
    "Psychological sense of community": "#B6DFB3",
    "Neighbouring": "#FAD9B8",
}
TEXT_COLORS = {
    "Total cohesion": "#8B78A8",
    "Psychological sense of community": "#4E915A",
    "Neighbouring": "#C1813E",
}
LABELS = {
    "Total cohesion": "Total cohesion",
    "Psychological sense of community": "Psychological sense\nof community",
    "Neighbouring": "Neighbouring",
}
ORDER = ["Total cohesion", "Psychological sense of community", "Neighbouring"]
WIDTH_INCHES = 17.76 / 2.54


def journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    config.make_output_directories()
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            config.FIGURE_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def clean_axis(ax: plt.Axes, zero: bool = True) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero:
        ax.axvline(0, color="#8F8F8F", linestyle=(0, (4, 3)), linewidth=0.9, zorder=0)
    ax.tick_params(axis="y", length=0, pad=3)


def ci_label(row: pd.Series, digits: int = 2) -> str:
    return f"{row['estimate']:.{digits}f} ({row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f})"


def forest_three(
    frame: pd.DataFrame,
    *,
    xlabel: str,
    stem: str,
    xlim: tuple[float, float],
) -> None:
    journal_style()
    data = frame.set_index("cohesion").loc[ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(WIDTH_INCHES, 3.15))
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.23, top=0.96)
    y = np.arange(len(data))[::-1]
    for position, (_, row) in zip(y, data.iterrows()):
        ax.errorbar(
            row["estimate"],
            position,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=COLORS[row["cohesion"]],
            ecolor=COLORS[row["cohesion"]],
            markersize=6,
            capsize=3,
            linewidth=1.8,
        )
        ax.annotate(
            ci_label(row),
            (row["ci_high"], position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_COLORS[row["cohesion"]],
        )
    ax.set_yticks(y, [LABELS[name] for name in data["cohesion"]])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    clean_axis(ax)
    save_figure(fig, stem)


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def draw_grouped_estimates(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    groups: list[str],
    labels: dict[str, str],
    xlim: tuple[float, float],
    annotation_ax: plt.Axes | None = None,
) -> None:
    offsets = {ORDER[0]: 0.22, ORDER[1]: 0.0, ORDER[2]: -0.22}
    base = np.arange(len(groups))[::-1]
    for measure in ORDER:
        subset = frame[frame["cohesion"].eq(measure)].set_index("group")
        for group, centre in zip(groups, base):
            row = subset.loc[group]
            y = centre + offsets[measure]
            ax.errorbar(
                row["slope_estimate"],
                y,
                xerr=[
                    [row["slope_estimate"] - row["slope_ci_low"]],
                    [row["slope_ci_high"] - row["slope_estimate"]],
                ],
                fmt="o",
                color=COLORS[measure],
                markersize=4.5,
                capsize=2,
                linewidth=1.3,
            )
            text = (
                f"{row['slope_estimate']:.2f} "
                f"({row['slope_ci_low']:.2f}, {row['slope_ci_high']:.2f})"
            )
            if pd.notna(row["difference_p_value"]):
                text += (
                    f"   Δ {row['difference_vs_reference']:+.2f}"
                    f"{significance_stars(row['difference_p_value'])}"
                )
            if annotation_ax is None:
                ax.annotate(
                    text,
                    (row["slope_ci_high"], y),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7.2,
                    color=TEXT_COLORS[measure],
                )
            else:
                annotation_ax.text(
                    0.0,
                    y,
                    text,
                    transform=annotation_ax.get_yaxis_transform(),
                    va="center",
                    fontsize=7.0,
                    color=TEXT_COLORS[measure],
                )
    ax.set_yticks(base, [labels[group] for group in groups])
    ax.set_xlim(*xlim)
    clean_axis(ax)
    if annotation_ax is not None:
        annotation_ax.set_ylim(ax.get_ylim())
        annotation_ax.set_axis_off()


def legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], marker="o", color=COLORS[name], linewidth=0, label=name)
        for name in ORDER
    ]


def draw_sensitivity_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    models: list[tuple[str, str]],
    xlim: tuple[float, float],
    title: str,
    annotation_ax: plt.Axes | None = None,
) -> None:
    offsets = {ORDER[0]: 0.22, ORDER[1]: 0.0, ORDER[2]: -0.22}
    base = np.arange(len(models))[::-1]
    for measure in ORDER:
        subset = frame[frame["cohesion"].eq(measure)].set_index("model")
        for (model, _), centre in zip(models, base):
            row = subset.loc[model]
            y = centre + offsets[measure]
            ax.errorbar(
                row["estimate"],
                y,
                xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
                fmt="o",
                color=COLORS[measure],
                markersize=4,
                capsize=2,
                linewidth=1.2,
            )
            if annotation_ax is not None:
                annotation_ax.text(
                    0.0,
                    y,
                    ci_label(row),
                    transform=annotation_ax.get_yaxis_transform(),
                    va="center",
                    fontsize=6.8,
                    color=TEXT_COLORS[measure],
                )
    ax.set_yticks(base, [label for _, label in models])
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.2, pad=7)
    clean_axis(ax)
    if annotation_ax is not None:
        annotation_ax.set_ylim(ax.get_ylim())
        annotation_ax.set_axis_off()


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
