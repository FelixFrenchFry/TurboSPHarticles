#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

PARTICLE_COUNTS = [
    100_000_000,
    200_000_000,
    300_000_000,
    400_000_000,
    500_000_000
]

# 192 threads, averaged over the first 25 steps, in milliseconds
# (breaking dam with 1000 x <adjustable> x 1000)
METHOD_COMPONENT_TIMES_MS = {
    "Spatial Hashing": {
        "Construction": [6468,  12198,  19115,  26349,  32753],
        "Query":        [2077,  4020,   6109,   8235,   9954],
        "Fluid Data":   [0,     0,      0,      0,      0], # MB
        "Boundary Data":[0,     0,      0,      0,      0] # MB
    },
    "Cell-Linked List": {
        "Construction": [516,   1065,   1460,   1960,   2217],
        "Query":        [1451,  2850,   4245,   5659,   7053],
        "Fluid Data":   [0,     2699,   4050,   5399,   6750], # MB
        "Boundary Data":[0,     260,    281,    303,    325] # MB
    },
    "Linear BVH": {
        "Construction": [844,   1818,   2890,   4079,   5936],
        "Query":        [2185,  3958,   5781,   7386,   8900],
        "Fluid Data":   [10400, 20800,  31199,  41599,  52000], # MB
        "Boundary Data":[777,   848,    919,    989,    1059] # MB
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Fluid particle count [millions]"
Y_LABEL = "Neighbor search time [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (12.0, 10.0)
BAR_WIDTH = 0.25

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0

SHOW_LEGEND = True
FORMAT_X_AXIS_IN_MILLIONS = True
SHOW_COMPONENT_SEPARATOR = True
COMPONENT_SEPARATOR_COLOR = (1.0, 1.0, 1.0)
COMPONENT_SEPARATOR_LINE_WIDTH = 2.0
COMPONENT_SEPARATOR_LINE_STYLE = "-"

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


METHOD_STYLES = {
    "Spatial Hashing": {
        "label": "Spatial Hashing",
        "construction_color": RGB(25, 50, 80),
        "query_color": RGB(40, 80, 180),
    },
    "Cell-Linked List": {
        "label": "Cell-Linked List",
        "construction_color": RGB(150, 0, 30),
        "query_color": RGB(250, 0, 50),
    },
    "Linear BVH": {
        "label": "Linear BVH",
        "construction_color": RGB(25, 100, 50),
        "query_color": RGB(40, 180, 80),
    },
}


def validate_plot_data() -> None:
    if not PARTICLE_COUNTS:
        raise ValueError("PARTICLE_COUNTS must not be empty.")

    if len(METHOD_COMPONENT_TIMES_MS) != 3:
        raise ValueError(
            f"Expected exactly 3 methods, got {len(METHOD_COMPONENT_TIMES_MS)}."
        )

    num_points = len(PARTICLE_COUNTS)

    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        if "Construction" not in component_times_ms:
            raise ValueError(
                f"Method '{method_name}' is missing 'Construction' values."
            )

        if "Query" not in component_times_ms:
            raise ValueError(
                f"Method '{method_name}' is missing 'Query' values."
            )

        construction_times_ms = component_times_ms["Construction"]
        query_times_ms = component_times_ms["Query"]

        if len(construction_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(construction_times_ms)} "
                f"construction measurements, but PARTICLE_COUNTS has {num_points} entries."
            )

        if len(query_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(query_times_ms)} "
                f"query measurements, but PARTICLE_COUNTS has {num_points} entries."
            )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    method_names = list(METHOD_COMPONENT_TIMES_MS.keys())
    num_methods = len(method_names)
    x_positions = list(range(len(PARTICLE_COUNTS)))
    legend_handles = []
    legend_labels = []
    max_total_time_ms = 0.0

    for method_idx, method_name in enumerate(method_names):
        style = METHOD_STYLES[method_name]
        construction_times_ms = METHOD_COMPONENT_TIMES_MS[method_name]["Construction"]
        query_times_ms = METHOD_COMPONENT_TIMES_MS[method_name]["Query"]
        total_times_ms = [
            construction_time_ms + query_time_ms
            for construction_time_ms, query_time_ms
            in zip(construction_times_ms, query_times_ms)
        ]
        max_total_time_ms = max(max_total_time_ms, max(total_times_ms))

        x_offset = (method_idx - (num_methods - 1) / 2.0) * BAR_WIDTH
        bar_positions = [x + x_offset for x in x_positions]

        ax.bar(
            bar_positions,
            construction_times_ms,
            width=BAR_WIDTH,
            color=style["construction_color"],
        )
        ax.bar(
            bar_positions,
            query_times_ms,
            width=BAR_WIDTH,
            bottom=construction_times_ms,
            color=style["query_color"],
        )

        if SHOW_COMPONENT_SEPARATOR:
            for bar_x, construction_time_ms in zip(bar_positions, construction_times_ms):
                ax.hlines(
                    construction_time_ms,
                    bar_x - 0.5 * BAR_WIDTH,
                    bar_x + 0.5 * BAR_WIDTH,
                    colors=COMPONENT_SEPARATOR_COLOR,
                    linewidth=COMPONENT_SEPARATOR_LINE_WIDTH,
                    linestyles=COMPONENT_SEPARATOR_LINE_STYLE,
                )

        legend_handles.append(
            (
                Patch(facecolor=style["construction_color"]),
                Patch(facecolor=style["query_color"]),
            )
        )
        legend_labels.append(f"{style['label']}")

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)

    x_tick_labels = []
    for particle_count in PARTICLE_COUNTS:
        if FORMAT_X_AXIS_IN_MILLIONS:
            x_tick_labels.append(str(int(particle_count / 1_000_000)))
        else:
            x_tick_labels.append(str(particle_count))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(0.0, max_total_time_ms * 1.05)

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
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
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
