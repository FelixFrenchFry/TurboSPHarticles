#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

CPU_CORE_COUNTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]

# 1-20 threads, averaged over the first 100 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
METHOD_STEP_TIMES_MS = {
    "Spatial Hashing":  [71+684, 71+339, 72+221, 72+171, 72+138, 72+114, 70+97, 70+87, 71+80, 70+73,
                         70+68, 70+64, 70+59, 70+56, 70+53, 70+50, 70+48, 70+46, 70+44, 70+43],
    "Cell-Linked List": [49+568, 27+292, 18+191, 15+143, 13+115, 11+97, 10+82, 9+73, 8+66, 8+61,
                         8+55, 7+51, 7+47, 7+44, 7+41, 7+39, 6+37, 6+36, 6+34, 6+33],
    "Linear BVH":       [72+1100, 43+577, 29+368, 24+274, 21+221, 19+185, 17+160, 16+141, 16+127, 15+116,
                         15+106, 14+98, 14+91, 13+86, 13+81, 13+76, 13+73, 13+69, 13+66, 12+64],
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "CPU cores used"
Y_LABEL = "Speedup relative to 1 core"
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
SHOW_ALL_CORE_TICKS = True
FORMAT_Y_AXIS_AS_SPEEDUP = True
SHOW_IDEAL_SPEEDUP_LINE = True
IDEAL_SPEEDUP_LINE_COLOR = (0.0, 0.0, 0.0)
IDEAL_SPEEDUP_LINE_STYLE = ":"
IDEAL_SPEEDUP_LINE_WIDTH = 3.0

OUTPUT_PATH = None
OUTPUT_DPI = 300

def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)

METHOD_STYLES = {
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
    if not CPU_CORE_COUNTS:
        raise ValueError("CPU_CORE_COUNTS must not be empty.")

    if len(METHOD_STEP_TIMES_MS) != 3:
        raise ValueError(
            f"Expected exactly 3 methods, got {len(METHOD_STEP_TIMES_MS)}."
        )

    num_points = len(CPU_CORE_COUNTS)

    for method_name, step_times_ms in METHOD_STEP_TIMES_MS.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        if step_times_ms[0] <= 0.0:
            raise ValueError(
                f"Method '{method_name}' must have a positive 1-core timing."
            )

        if len(step_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(step_times_ms)} measurements, "
                f"but CPU_CORE_COUNTS has {num_points} entries."
            )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    max_speedup = 0.0

    for method_name, step_times_ms in METHOD_STEP_TIMES_MS.items():
        style = METHOD_STYLES[method_name]
        baseline_time_ms = step_times_ms[0]
        speedup_values = [
            baseline_time_ms / step_time_ms
            for step_time_ms in step_times_ms
        ]
        max_speedup = max(max_speedup, max(speedup_values))

        ax.plot(
            CPU_CORE_COUNTS,
            speedup_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=style["label"],
        )

    if SHOW_IDEAL_SPEEDUP_LINE:
        ax.plot(
            CPU_CORE_COUNTS,
            CPU_CORE_COUNTS,
            color=IDEAL_SPEEDUP_LINE_COLOR,
            linestyle=IDEAL_SPEEDUP_LINE_STYLE,
            linewidth=IDEAL_SPEEDUP_LINE_WIDTH,
            label="Ideal Speedup",
        )
        max_speedup = max(max_speedup, max(CPU_CORE_COUNTS))

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

    if SHOW_ALL_CORE_TICKS:
        ax.set_xticks(CPU_CORE_COUNTS)

    if FORMAT_Y_AXIS_AS_SPEEDUP:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    ax.set_ylim(0.0, max_speedup * 1.05)

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
