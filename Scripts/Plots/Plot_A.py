#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

PARTICLE_COUNTS = [
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
    19000,
    20000
]

# 1 thread, averaged over the first 1000 steps, in milliseconds
# (fluid pillar with 100 x <adjustable> x 100)
METHOD_STEP_TIMES_MS = {
    "Brute Force":      [4.75, 13.20, 24.93, 40.26, 59.69, 82.37, 108.39, 139.39, 173.03, 210.51,
                         256.34, 301.46, 351.31, 398.69, 464.63, 517.80, 567.04, 642.13, 706.25, 789.44],
    "Spatial Hashing":  [2.71, 5.25, 7.71, 10.27, 12.70, 15.22, 17.74, 20.42, 23.08, 25.59,
                         27.99, 30.31, 33.10, 35.93, 37.92, 40.90, 43.27, 46.78, 49.57, 52.71],
    "Cell-Linked List": [2.83, 5.23, 7.80, 10.21, 12.72, 15.17, 17.70, 20.16, 22.69, 25.28,
                         27.88, 30.18, 32.78, 35.47, 37.91, 40.60, 42.93, 45.44, 48.04, 50.85],
    "Linear BVH":       [3.76, 7.35, 10.86, 14.82, 18.81, 22.55, 26.81, 31.30, 35.94, 39.95,
                         43.87, 48.69, 52.89, 57.55, 62.36, 67.75, 72.88, 77.04, 81.65, 86.18],
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Fluid particle count [thousands]"
Y_LABEL = "Total time per simulation step [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (16.0, 10.0)
LINE_WIDTH = 5.0
MARKER_SIZE = 20.0

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0

SHOW_LEGEND = True
FORMAT_X_AXIS_IN_THOUSANDS = True

OUTPUT_PATH = None
OUTPUT_DPI = 300

def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)

METHOD_STYLES = {
    "Brute Force": {
        "label": "Brute Force",
        "color": RGB(130, 60, 10),
        "marker": "o",
        "line_style": "-",
    },
    "Spatial Hashing": {
        "label": "Spatial Hashing",
        "color": RGB(40, 80, 180),
        "marker": "o",
        "line_style": "-",
    },
    "Cell-Linked List": {
        "label": "Cell-Linked List",
        "color": RGB(250, 0, 50),
        "marker": "o",
        "line_style": "-",
    },
    "Linear BVH": {
        "label": "Linear BVH",
        "color": RGB(40, 180, 80),
        "marker": "o",
        "line_style": "-",
    },
}


def validate_plot_data() -> None:
    if not PARTICLE_COUNTS:
        raise ValueError("PARTICLE_COUNTS must not be empty.")

    if len(METHOD_STEP_TIMES_MS) != 4:
        raise ValueError(
            f"Expected exactly 4 methods, got {len(METHOD_STEP_TIMES_MS)}."
        )

    num_points = len(PARTICLE_COUNTS)

    for method_name, step_times_ms in METHOD_STEP_TIMES_MS.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        if len(step_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(step_times_ms)} measurements, "
                f"but PARTICLE_COUNTS has {num_points} entries."
            )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for method_name, step_times_ms in METHOD_STEP_TIMES_MS.items():
        style = METHOD_STYLES[method_name]

        ax.plot(
            PARTICLE_COUNTS,
            step_times_ms,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=style["label"],
        )

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)

    if GRID_ENABLED:
        ax.grid(
            True,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax.legend(fontsize=AXES_FONT_SIZE)

    if FORMAT_X_AXIS_IN_THOUSANDS:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000.0:g}"))

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
