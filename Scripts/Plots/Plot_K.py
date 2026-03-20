#!/usr/bin/env python3

import math
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

BENCHMARK_COMPONENT_NAMES = [
    "dataOrder",
    "neighCon",
    "neighQue",
    "dens",
    "viscAcc + extAcc",
    "srcTerm + diagElem",
    "jacSolv",
    "presAcc",
    "velUpd + posUpd",
]

# 192 threads, averaged over the first 500 steps, in milliseconds
# (breaking dam with 200 x 300 x 200)
NEIGHBOR_LIST_BENCHMARK_TIMES_MS = {
    "Explicit": {
        "dataOrder":            0.0,
        "neighCon":             0.0,
        "neighQue":             0.0,
        "dens":                 0.0,
        "viscAcc + extAcc":     0.0,
        "srcTerm + diagElem":   0.0,
        "jacSolv":              0.0,
        "presAcc":              0.0,
        "velUpd + posUpd":      0.0,
    },
    "Compressed": {
        "dataOrder":            0.0,
        "neighCon":             0.0,
        "neighQue":             0.0,
        "dens":                 0.0,
        "viscAcc + extAcc":     0.0,
        "srcTerm + diagElem":   0.0,
        "jacSolv":              0.0,
        "presAcc":              0.0,
        "velUpd + posUpd":      0.0,
    },
    "On-Demand": {
        "dataOrder":            0.0,
        "neighCon":             0.0,
        "neighQue":             0.0,
        "dens":                 0.0,
        "viscAcc + extAcc":     0.0,
        "srcTerm + diagElem":   0.0,
        "jacSolv":              0.0,
        "presAcc":              0.0,
        "velUpd + posUpd":      0.0,
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Simulation operation"
Y_LABEL = "Time per simulation step [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (24.0, 10.0)
BAR_WIDTH = 0.25

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
Y_AXIS_TICK_INTERVAL = 2.0

SHOW_LEGEND = True
LEGEND_LOC = "upper left"
ROTATE_X_TICK_LABELS = False
X_TICK_LABEL_ROTATION_DEG = 0

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


NEIGHBOR_LIST_STYLES = {
    "Explicit": {
        "label": "Explicit",
        "color": RGB(40, 80, 180),
    },
    "Compressed": {
        "label": "Compressed",
        "color": RGB(250, 0, 50),
    },
    "On-Demand": {
        "label": "On-Demand",
        "color": RGB(40, 180, 80),
    },
}

OPERATION_GROUPS = {
    "Construction": [
        "neighCon",
    ],
    "Query": [
        "neighQue",
    ],
    "Non-Pressure\nAcceleration": [
        "dens",
        "viscAcc + extAcc",
    ],
    "Pressure\nSolving": [
        "srcTerm + diagElem",
        "jacSolv",
        "presAcc",
    ],
    "Misc": [
        "dataOrder",
        "velUpd + posUpd",
    ],
}


def validate_plot_data() -> None:
    if len(NEIGHBOR_LIST_BENCHMARK_TIMES_MS) != 3:
        raise ValueError(
            f"Expected exactly 3 neighbor-list categories, got {len(NEIGHBOR_LIST_BENCHMARK_TIMES_MS)}."
        )

    for category_name, benchmark_times_ms in NEIGHBOR_LIST_BENCHMARK_TIMES_MS.items():
        if category_name not in NEIGHBOR_LIST_STYLES:
            raise ValueError(
                f"Missing style entry for category '{category_name}'."
            )

        for component_name in BENCHMARK_COMPONENT_NAMES:
            if component_name not in benchmark_times_ms:
                raise ValueError(
                    f"Category '{category_name}' is missing component '{component_name}'."
                )


def compute_operation_group_times_ms() -> dict[str, dict[str, float]]:
    grouped_times_ms: dict[str, dict[str, float]] = {}

    for category_name, benchmark_times_ms in NEIGHBOR_LIST_BENCHMARK_TIMES_MS.items():
        grouped_times_ms[category_name] = {}

        for group_name, component_names in OPERATION_GROUPS.items():
            grouped_times_ms[category_name][group_name] = sum(
                benchmark_times_ms[component_name]
                for component_name in component_names
            )

    return grouped_times_ms


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    grouped_times_ms = compute_operation_group_times_ms()
    category_names = list(grouped_times_ms.keys())
    operation_group_names = list(OPERATION_GROUPS.keys())
    x_positions = list(range(len(operation_group_names)))
    max_group_time_ms = 0.0

    for category_idx, category_name in enumerate(category_names):
        style = NEIGHBOR_LIST_STYLES[category_name]
        category_group_times_ms = [
            grouped_times_ms[category_name][operation_group_name]
            for operation_group_name in operation_group_names
        ]
        max_group_time_ms = max(max_group_time_ms, max(category_group_times_ms))

        x_offset = (category_idx - (len(category_names) - 1) / 2.0) * BAR_WIDTH
        bar_positions = [x + x_offset for x in x_positions]

        ax.bar(
            bar_positions,
            category_group_times_ms,
            width=BAR_WIDTH,
            color=style["color"],
            label=style["label"],
        )

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(operation_group_names)

    if ROTATE_X_TICK_LABELS:
        plt.setp(
            ax.get_xticklabels(),
            rotation=X_TICK_LABEL_ROTATION_DEG,
            ha="right",
            rotation_mode="anchor",
        )

    y_max = math.ceil((max_group_time_ms * 1.08) / Y_AXIS_TICK_INTERVAL) * Y_AXIS_TICK_INTERVAL
    ax.set_ylim(0.0, y_max)
    ax.set_yticks([
        i * Y_AXIS_TICK_INTERVAL
        for i in range(int(y_max / Y_AXIS_TICK_INTERVAL) + 1)
    ])

    if GRID_ENABLED:
        ax.grid(
            True,
            axis="y",
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax.legend(
            fontsize=AXES_FONT_SIZE,
            loc=LEGEND_LOC,
        )

    fig.tight_layout()

    if OUTPUT_PATH is not None:
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved plot to '{output_path}'.")
    else:
        plt.show()


def main() -> None:
    validate_plot_data()
    plot_scaling()


if __name__ == "__main__":
    main()
