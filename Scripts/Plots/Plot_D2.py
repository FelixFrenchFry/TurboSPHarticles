#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

KERNEL_SUPPORT_RATIOS = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0
]

# 192 threads, averaged over the first 50 steps, in milliseconds
# (breaking dam with 300 x 300 x 300)
METHOD_COMPONENT_TIMES_MS = {
    "Spatial Hashing": {
        "Construction": [1450.56, 1074.84, 706.65, 554.25, 462.71, 445.23, 350.23, 440.72],
        "Query":        [572.55, 879.94, 1270.77, 1797.45, 2464.21, 3402.05, 4336.84, 5669.26],
    },
    "Cell-Linked List": {
        "Construction": [164.61, 153.78, 165.49, 176.77, 158.77, 158.38, 159.15, 149.75, 162.27],
        "Query":        [403.94, 756.40, 1182.18, 1769.29, 2523.96, 3598.47, 4670.36, 6220.07, 7882.24],
    },
    "Linear BVH": {
        "Construction": [228.89, 200.22, 232.59, 228.35, 263.16, 235.61, 268.44, 323.90, 408.09],
        "Query":        [515.77, 883.58, 1259.20, 1639.01, 2188.40, 3171.82, 3738.98, 4902.84, 5924.82],
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Kernel support radius / particle size h"
Y_LABEL = "Neighbor search time [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (24.0, 10.0)
LINE_WIDTH = 4.0
MARKER_SIZE = 16.0

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
Y_AXIS_TICK_INTERVAL = 500.0

SHOW_LEGEND = True
LEGEND_LOC = "upper left"

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


METHOD_STYLES = {
    "Spatial Hashing": {
        "label": "Spatial Hashing",
        "construction_color": RGB(25, 50, 80),
        "query_color": RGB(40, 80, 180),
        "construction_marker": "v",
        "query_marker": "o",
        "line_style": "-",
    },
    "Cell-Linked List": {
        "label": "Cell-Linked List",
        "construction_color": RGB(150, 0, 30),
        "query_color": RGB(250, 0, 50),
        "construction_marker": "v",
        "query_marker": "o",
        "line_style": "-",
    },
    "Linear BVH": {
        "label": "Linear BVH",
        "construction_color": RGB(25, 100, 50),
        "query_color": RGB(40, 180, 80),
        "construction_marker": "v",
        "query_marker": "o",
        "line_style": "-",
    },
}

LEGEND_ORDER = [
    "Spatial Hashing Query",
    "Cell-Linked List Query",
    "Linear BVH Query",
    "Spatial Hashing Construction",
    "Cell-Linked List Construction",
    "Linear BVH Construction",
]


def validate_plot_data() -> None:
    if not KERNEL_SUPPORT_RATIOS:
        raise ValueError("KERNEL_SUPPORT_RATIOS must not be empty.")

    if len(METHOD_COMPONENT_TIMES_MS) != 3:
        raise ValueError(
            f"Expected exactly 3 methods, got {len(METHOD_COMPONENT_TIMES_MS)}."
        )

    num_points = len(KERNEL_SUPPORT_RATIOS)

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
                f"Method '{method_name}' has {len(construction_times_ms)} construction measurements, "
                f"but KERNEL_SUPPORT_RATIOS has {num_points} entries."
            )

        if len(query_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(query_times_ms)} query measurements, "
                f"but KERNEL_SUPPORT_RATIOS has {num_points} entries."
            )

def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    legend_entries: dict[str, plt.Line2D] = {}
    max_time_ms = 0.0

    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        style = METHOD_STYLES[method_name]
        construction_times_ms = component_times_ms["Construction"]
        query_times_ms = component_times_ms["Query"]
        max_time_ms = max(
            max_time_ms,
            max(construction_times_ms),
            max(query_times_ms),
        )

        construction_handle, = ax.plot(
            KERNEL_SUPPORT_RATIOS,
            construction_times_ms,
            color=style["construction_color"],
            marker=style["construction_marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )
        query_handle, = ax.plot(
            KERNEL_SUPPORT_RATIOS,
            query_times_ms,
            color=style["query_color"],
            marker=style["query_marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

        legend_entries[f"{style['label']} Construction"] = construction_handle
        legend_entries[f"{style['label']} Query"] = query_handle

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_xticks(KERNEL_SUPPORT_RATIOS)
    ax.set_xticklabels([f"{ratio:.1f}" for ratio in KERNEL_SUPPORT_RATIOS])
    y_max = (
        int(((max_time_ms * 1.08) + Y_AXIS_TICK_INTERVAL - 1.0) / Y_AXIS_TICK_INTERVAL)
        * Y_AXIS_TICK_INTERVAL
    )
    ax.set_ylim(0.0, y_max)
    ax.set_yticks([
        i * Y_AXIS_TICK_INTERVAL
        for i in range(int(y_max / Y_AXIS_TICK_INTERVAL) + 1)
    ])

    if GRID_ENABLED:
        ax.grid(
            True,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax.legend(
            [legend_entries[label] for label in LEGEND_ORDER if label in legend_entries],
            [label for label in LEGEND_ORDER if label in legend_entries],
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
