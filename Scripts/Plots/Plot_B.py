#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# 1 thread, fluid pillar, averaged over 1000 steps, in milliseconds, for 20k particles
METHOD_RUNTIME_BREAKDOWN_PCT = {
    "Brute Force": {
        "Construction": 0.0,
        "Query": 95.7,
        "Other": 4.3,
    },
    "Spatial Hashing": {
        "Construction": 3.5,
        "Query": 33.2,
        "Other": 63.3,
    },
    "Cell-Linked List": {
        "Construction": 3.0,
        "Query": 30.9,
        "Other": 66.1,
    },
    "Linear BVH": {
        "Construction": 2.1,
        "Query": 59.6,
        "Other": 38.3,
    },
}

METHOD_ORDER = [
    "Brute Force",
    "Spatial Hashing",
    "Cell-Linked List",
    "Linear BVH",
]

COMPONENT_ORDER = [
    "Construction",
    "Query",
    "Other",
]


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Neighbor search method"
Y_LABEL = "Share of computation time [%]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (8.0, 16.0)
BAR_WIDTH = 0.67

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0

SHOW_LEGEND = True
SHOW_BAR_EDGES = False
BAR_EDGE_COLOR = (0.0, 0.0, 0.0)
BAR_EDGE_WIDTH = 0.0

ROTATE_X_TICK_LABELS = True
X_TICK_LABEL_ROTATION_DEG = 20

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


COMPONENT_STYLES = {
    "Construction": {
        "label": "Construction",
        "color": RGB(20, 40, 60),
    },
    "Query": {
        "label": "Query",
        "color": RGB(40, 80, 180)
    },
    "Other": {
        "label": "Other",
        "color": RGB(90, 180, 250),
    },
}


def validate_plot_data() -> None:
    if len(METHOD_ORDER) != 4:
        raise ValueError(
            f"Expected exactly 4 methods, got {len(METHOD_ORDER)}."
        )

    if len(COMPONENT_ORDER) != 3:
        raise ValueError(
            f"Expected exactly 3 components, got {len(COMPONENT_ORDER)}."
        )

    for method_name in METHOD_ORDER:
        if method_name not in METHOD_RUNTIME_BREAKDOWN_PCT:
            raise ValueError(
                f"Missing runtime breakdown for method '{method_name}'."
            )

        breakdown = METHOD_RUNTIME_BREAKDOWN_PCT[method_name]

        total_pct = 0.0
        for component_name in COMPONENT_ORDER:
            if component_name not in breakdown:
                raise ValueError(
                    f"Method '{method_name}' is missing component '{component_name}'."
                )

            if component_name not in COMPONENT_STYLES:
                raise ValueError(
                    f"Missing style entry for component '{component_name}'."
                )

            total_pct += breakdown[component_name]

        if abs(total_pct - 100.0) > 1e-6:
            raise ValueError(
                f"Method '{method_name}' sums to {total_pct:.6f}% instead of 100%."
            )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x_positions = list(range(len(METHOD_ORDER)))
    cumulative_bottom = [0.0 for _ in METHOD_ORDER]

    for component_name in COMPONENT_ORDER:
        style = COMPONENT_STYLES[component_name]
        values = [
            METHOD_RUNTIME_BREAKDOWN_PCT[method_name][component_name]
            for method_name in METHOD_ORDER
        ]

        ax.bar(
            x_positions,
            values,
            width=BAR_WIDTH,
            bottom=cumulative_bottom,
            color=style["color"],
            label=style["label"],
            edgecolor=BAR_EDGE_COLOR if SHOW_BAR_EDGES else None,
            linewidth=BAR_EDGE_WIDTH,
        )

        for i, value in enumerate(values):
            cumulative_bottom[i] += value

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        METHOD_ORDER,
        rotation=X_TICK_LABEL_ROTATION_DEG if ROTATE_X_TICK_LABELS else 0,
        ha="right" if ROTATE_X_TICK_LABELS else "center",
    )
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    if GRID_ENABLED:
        ax.grid(
            True,
            axis="y",
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            labels[::-1],
            fontsize=AXES_FONT_SIZE,
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
