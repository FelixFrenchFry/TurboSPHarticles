#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

IISPH_ITERATION_COUNTS = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
]

# 32 threads, averaged over the first 500 steps, in milliseconds
# (breaking dam with fixed IISPH iteration counts)
METHOD_COMPONENT_TIMES_MS = {
    "IndividualVectors": {
        "Neighbor Build": [70.0, 70.1, 70.0, 70.4, 70.2, 70.3, 70.5, 70.6],
        "Pre-Solver SPH": [165.0, 166.2, 168.0, 170.4, 172.8, 175.1, 179.8, 184.4],
        "Jacobi Solver":  [75.0, 145.0, 285.0, 565.0, 845.0, 1125.0, 1685.0, 2245.0],
        "Post-Solver":    [26.0, 26.1, 26.3, 26.6, 26.9, 27.2, 27.8, 28.4],
    },
    "CSRVector_DoubleQuery": {
        "Neighbor Build": [118.0, 118.3, 118.1, 118.5, 118.7, 118.9, 119.4, 120.0],
        "Pre-Solver SPH": [148.0, 149.0, 150.4, 152.6, 154.8, 156.7, 160.9, 164.7],
        "Jacobi Solver":  [68.0, 131.0, 257.0, 509.0, 761.0, 1013.0, 1517.0, 2021.0],
        "Post-Solver":    [24.0, 24.1, 24.2, 24.5, 24.7, 25.0, 25.5, 26.1],
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Fixed IISPH iterations [-]"
TOTAL_Y_LABEL = "Total step time [ms]"
BUILD_Y_LABEL = "Neighbor build [ms]"
JACOBI_Y_LABEL = "Jacobi solver [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 24
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (16.0, 13.0)
LINE_WIDTH = 4.0
MARKER_SIZE = 14.0

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
    "IndividualVectors": {
        "label": "Individual Vectors",
        "color": RGB(40, 80, 180),
        "marker": "o",
        "line_style": "-",
    },
    "CSRVector_DoubleQuery": {
        "label": "CSR Vector (Double Query)",
        "color": RGB(250, 0, 50),
        "marker": "o",
        "line_style": "-",
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
        "Neighbor Build",
        "Pre-Solver SPH",
        "Jacobi Solver",
        "Post-Solver",
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


def compute_total_step_times_ms() -> dict[str, list[float]]:
    total_step_times_ms: dict[str, list[float]] = {}

    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        total_step_times_ms[method_name] = [
            component_times_ms["Neighbor Build"][i]
            + component_times_ms["Pre-Solver SPH"][i]
            + component_times_ms["Jacobi Solver"][i]
            + component_times_ms["Post-Solver"][i]
            for i in range(len(IISPH_ITERATION_COUNTS))
        ]

    return total_step_times_ms


def apply_axis_styling(ax: plt.Axes, y_label: str) -> None:
    ax.set_ylabel(y_label, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)

    if GRID_ENABLED:
        ax.grid(
            True,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )


def plot_scaling() -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=FIG_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]},
    )

    ax_total, ax_build, ax_jacobi = axes
    total_step_times_ms = compute_total_step_times_ms()

    for method_name, style in METHOD_STYLES.items():
        ax_total.plot(
            IISPH_ITERATION_COUNTS,
            total_step_times_ms[method_name],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=style["label"],
        )

        ax_build.plot(
            IISPH_ITERATION_COUNTS,
            METHOD_COMPONENT_TIMES_MS[method_name]["Neighbor Build"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

        ax_jacobi.plot(
            IISPH_ITERATION_COUNTS,
            METHOD_COMPONENT_TIMES_MS[method_name]["Jacobi Solver"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

    if SHOW_PLOT_TITLE:
        ax_total.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    apply_axis_styling(ax_total, TOTAL_Y_LABEL)
    apply_axis_styling(ax_build, BUILD_Y_LABEL)
    apply_axis_styling(ax_jacobi, JACOBI_Y_LABEL)

    if SHOW_LEGEND:
        ax_total.legend(
            fontsize=AXES_FONT_SIZE,
            loc=LEGEND_LOC,
        )

    ax_jacobi.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax_jacobi.set_xticks(IISPH_ITERATION_COUNTS)

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
