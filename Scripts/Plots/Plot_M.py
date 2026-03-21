#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

IISPH_ITERATION_COUNTS = [
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50
]

# 192 threads, averaged over the first 100 steps, in milliseconds
# (breaking dam with 300 x 300 x 300)
METHOD_COMPONENT_TIMES_MS = {
    "Mulitple Individual Lists": {
        "Construction": [153.10, 149.82, 156.17, 152.62, 148.68, 155.92, 148.81, 152.74, 151.64, 152.78, 147.23],
        "Query":        [411.43, 409.88, 410.89, 406.47, 406.44, 407.00, 405.74, 406.75, 402.87, 403.97, 414.92],
        "IISPH":        [1485.68-153.10-411.43,
                         2136.84-149.82-409.88,
                         3232.61-156.17-410.89,
                         4298.45-152.62-406.47,
                         5393.71-148.68-406.44,
                         6449.11-155.92-407.00,
                         7571.16-148.81-405.74,
                         8628.62-152.74-406.75,
                         9690.49-151.64-402.87,
                         10820.76-152.78-403.97,
                         11855.59-147.23-414.92],
    },
    "Single Contiguous List": {
        "Construction": [161.38, 160.88, 167.00, 159.33, 164.30, 164.72, 162.32],
        "Query":        [535.09, 537.14, 535.11, 533.70, 535.88, 533.26, 533.51],
        "IISPH":        [1466.50-161.38-535.09,
                         2030.50-160.88-537.14,
                         2963.06-167.00-535.11,
                         3872.68-159.33-533.70,
                         4824.24-164.30-535.88,
                         5729.37-164.72-533.26,
                         6644.69-162.32-533.51,],
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Number of IISPH iterations"
Y_LABEL = "Time per simulation step [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (24.0, 10.0)
BAR_WIDTH = 0.35

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0

SHOW_LEGEND = True
LEGEND_LOC = "upper left"

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


METHOD_STYLES = {
    "Mulitple Individual Lists": {
        "label": "Mulitple Individual Lists",
        "construction_color": RGB(170, 60, 10),
        "query_color": RGB(240, 110, 40),
        "iisph_color": RGB(250, 170, 130),
    },
    "Single Contiguous List": {
        "label": "Single Contiguous List",
        "construction_color": RGB(15, 25, 40),
        "query_color": RGB(60, 70, 100),
        "iisph_color": RGB(130, 140, 160),
    },
}


def validate_plot_data() -> None:
    if not IISPH_ITERATION_COUNTS:
        raise ValueError("IISPH_ITERATION_COUNTS must not be empty.")

    if len(METHOD_COMPONENT_TIMES_MS) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_COMPONENT_TIMES_MS)}."
        )

    num_points = len(IISPH_ITERATION_COUNTS)
    required_components = [
        "Construction",
        "Query",
        "IISPH",
    ]

    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        for component_name in required_components:
            if component_name not in component_times_ms:
                raise ValueError(
                    f"Method '{method_name}' is missing component '{component_name}'."
                )

            if len(component_times_ms[component_name]) != num_points:
                raise ValueError(
                    f"Method '{method_name}' component '{component_name}' has "
                    f"{len(component_times_ms[component_name])} measurements, "
                    f"but IISPH_ITERATION_COUNTS has {num_points} entries."
                )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    method_names = list(METHOD_COMPONENT_TIMES_MS.keys())
    num_methods = len(method_names)
    x_positions = list(range(len(IISPH_ITERATION_COUNTS)))
    legend_handles = []
    legend_labels = []
    max_total_time_ms = 0.0

    for method_idx, method_name in enumerate(method_names):
        style = METHOD_STYLES[method_name]
        construction_times_ms = METHOD_COMPONENT_TIMES_MS[method_name]["Construction"]
        query_times_ms = METHOD_COMPONENT_TIMES_MS[method_name]["Query"]
        iisph_times_ms = METHOD_COMPONENT_TIMES_MS[method_name]["IISPH"]
        total_times_ms = [
            construction_time_ms + query_time_ms + iisph_time_ms
            for construction_time_ms, query_time_ms, iisph_time_ms
            in zip(construction_times_ms, query_times_ms, iisph_times_ms)
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
        ax.bar(
            bar_positions,
            iisph_times_ms,
            width=BAR_WIDTH,
            bottom=[
                construction_time_ms + query_time_ms
                for construction_time_ms, query_time_ms
                in zip(construction_times_ms, query_times_ms)
            ],
            color=style["iisph_color"],
        )

        legend_handles.append(
            (
                Patch(facecolor=style["construction_color"]),
                Patch(facecolor=style["query_color"]),
                Patch(facecolor=style["iisph_color"]),
            )
        )
        legend_labels.append(style["label"])

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(count) for count in IISPH_ITERATION_COUNTS])
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
