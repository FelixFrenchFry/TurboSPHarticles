#!/usr/bin/env python3

import math
from pathlib import Path

import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

BVH_LEAF_RADIUS_RATIOS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]

# 20 threads or 5070 Ti, averaged over the first 200 steps, in milliseconds
# (breaking dam with 200 x 200 x 200)
METHOD_COMPONENT_TIMES_MS = {
    "LBVH CPU": {
        "Construction": [114.82,    115.58,     116.13,     117.78,     118.09,     117.86,     116.22],
        "Query":        [1584.08,   1992.89,    2999.33,    3996.59,    5570.30,    6772.08,    8406.33],
    },
    "LBVH GPU": {
        "Construction": [6.39,      6.45,       6.50,       6.41,       6.51,       6.48,       6.60],
        "Query":        [105.36,    217.61,     375.20,     596.63,     916.41,     1354.37,    1755.92],
    },
}

NEGATIVE_CANDIDATE_CHECKS_PCT = [
    31.04,
    71.61,
    83.86,
    90.73,
    93.68,
    95.77,
    96.82
]


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "BVH leaf radius ratio"
LEFT_Y_LABEL = "Time relative to radius 0.0"
RIGHT_Y_LABEL = "Negative candidate checks [%]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30
QUERY_TIME_LABEL_FONT_SIZE = 24
QUERY_TIME_LABEL_OFFSET_X_PTS = 8
QUERY_TIME_LABEL_OFFSET_Y_PTS = -8
QUERY_TIME_LABEL_DECIMALS = 0

FIG_SIZE = (16.0, 10.0)
LINE_WIDTH = 5.0
MARKER_SIZE = 20.0

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
X_AXIS_TICK_INTERVAL = 0.25
RIGHT_Y_TICK_INTERVAL = 10.0

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
    "Negative Candidates": {
        "label": "Negative Candidates",
        "color": RGB(0, 0, 0),
        "marker": "v",
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

    if len(NEGATIVE_CANDIDATE_CHECKS_PCT) != num_points:
        raise ValueError(
            f"NEGATIVE_CANDIDATE_CHECKS_PCT has {len(NEGATIVE_CANDIDATE_CHECKS_PCT)} measurements, "
            f"but BVH_LEAF_RADIUS_RATIOS has {num_points} entries."
        )

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


def compute_relative_time_factors(query_times_ms: list[float]) -> list[float]:
    baseline_time_ms = query_times_ms[0]

    if baseline_time_ms <= 0.0:
        raise ValueError("The baseline total time must be positive.")

    return [
        query_time_ms / baseline_time_ms
        for query_time_ms in query_times_ms
    ]


def plot_scaling() -> None:
    fig, ax_left = plt.subplots(figsize=FIG_SIZE)
    ax_right = ax_left.twinx()
    ax_right.patch.set_alpha(0.0)

    timing_handles = []
    for method_name, component_times_ms in METHOD_COMPONENT_TIMES_MS.items():
        style = METHOD_STYLES[method_name]
        query_times_ms = component_times_ms["Query"]
        relative_time_factor = compute_relative_time_factors(query_times_ms)

        handle, = ax_left.plot(
            BVH_LEAF_RADIUS_RATIOS,
            relative_time_factor,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line_style"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=style["label"],
        )
        timing_handles.append(handle)

        for leaf_radius_ratio, relative_increase, query_time_ms in zip(
            BVH_LEAF_RADIUS_RATIOS,
            relative_time_factor,
            query_times_ms,
        ):
            label_offset_x_pts = QUERY_TIME_LABEL_OFFSET_X_PTS
            label_offset_y_pts = QUERY_TIME_LABEL_OFFSET_Y_PTS
            label_ha = "left"
            label_va = "top"

            if method_name == "LBVH GPU":
                label_offset_x_pts = -8
                label_offset_y_pts = 8
                label_ha = "right"
                label_va = "bottom"
            ax_left.annotate(
                f"{query_time_ms:.{QUERY_TIME_LABEL_DECIMALS}f}",
                xy=(leaf_radius_ratio, relative_increase),
                xytext=(label_offset_x_pts, label_offset_y_pts),
                textcoords="offset points",
                ha=label_ha,
                va=label_va,
                fontsize=QUERY_TIME_LABEL_FONT_SIZE,
                color=style["color"],
            )

    negative_style = METHOD_STYLES["Negative Candidates"]
    negative_handle, = ax_right.plot(
        BVH_LEAF_RADIUS_RATIOS,
        NEGATIVE_CANDIDATE_CHECKS_PCT,
        color=negative_style["color"],
        marker=negative_style["marker"],
        linestyle=negative_style["line_style"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=negative_style["label"],
    )

    if SHOW_PLOT_TITLE:
        ax_left.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax_left.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax_left.set_ylabel(LEFT_Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax_right.set_ylabel(RIGHT_Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax_left.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax_right.tick_params(axis="y", labelsize=AXES_FONT_SIZE)
    ax_left.set_xticks([
        i * X_AXIS_TICK_INTERVAL
        for i in range(int(max(BVH_LEAF_RADIUS_RATIOS) / X_AXIS_TICK_INTERVAL) + 1)
    ])

    if GRID_ENABLED:
        ax_left.grid(
            True,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax_left.legend(
            [*timing_handles, negative_handle],
            ["LBVH Query CPU", "LBVH Query GPU", "Negative Candidates"],
            fontsize=AXES_FONT_SIZE,
        )

    right_y_max = math.ceil((max(NEGATIVE_CANDIDATE_CHECKS_PCT) * 1.05) / RIGHT_Y_TICK_INTERVAL) * RIGHT_Y_TICK_INTERVAL
    ax_right.set_ylim(0.0, right_y_max)
    ax_right.set_yticks([
        i * RIGHT_Y_TICK_INTERVAL
        for i in range(int(right_y_max / RIGHT_Y_TICK_INTERVAL) + 1)
    ])

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
