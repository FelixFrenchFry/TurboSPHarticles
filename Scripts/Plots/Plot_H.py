#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

BVH_LEAF_RADIUS_RATIOS = [
    0.1,
    1.0
]

# 20 threads or 5070 Ti, averaged over the first 200 steps, in milliseconds
# (breaking dam with 200 x 200 x 200)
METHOD_COMPONENT_TIMES_MS = {
    "LBVH CPU": {
        "Construction": [107.21,    112.42], # 0.1 vs 1.0
        "Query":        [1446.07,   5328.34], # 0.1 vs 1.0
    },
    "LBVH GPU": {
        "Construction": [6.22,      6.34], # 0.1 vs 1.0
        "Query":        [163.50,    886.97], # 0.1 vs 1.0
    },
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "BVH leaf radius ratio"
Y_LABEL = "Time per simulation step [ms]"
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

OUTPUT_PATH = None
OUTPUT_DPI = 300

def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)

METHOD_STYLES = {
    "LBVH CPU": {
        "label": "LBVH CPU",
        "color": RGB(40, 180, 80),
        "marker": "o",
        "line_style": "-",
    },
    "LBVH GPU": {
        "label": "LBVH GPU",
        "color": RGB(25, 100, 50),
        "marker": "o",
        "line_style": "-",
    },
}


def validate_plot_data() -> None:
    if not BVH_LEAF_RADIUS_RATIOS:
        raise ValueError("BVH_LEAF_RADIUS_RATIOS must not be empty.")

    if len(METHOD_COMPONENT_TIMES_MS) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_COMPONENT_TIMES_MS)}."
        )

    num_points = len(BVH_LEAF_RADIUS_RATIOS)

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
                f"but BVH_LEAF_RADIUS_RATIOS has {num_points} entries."
            )

        if len(query_times_ms) != num_points:
            raise ValueError(
                f"Method '{method_name}' has {len(query_times_ms)} query measurements, "
                f"but BVH_LEAF_RADIUS_RATIOS has {num_points} entries."
            )


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        style = METHOD_STYLES[method_name]
        total_times_ms = [
            construction_time_ms + query_time_ms
            for construction_time_ms, query_time_ms
            in zip(component_times_ms["Construction"], component_times_ms["Query"])
        ]

        ax.plot(
            BVH_LEAF_RADIUS_RATIOS,
            total_times_ms,
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
