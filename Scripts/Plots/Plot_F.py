#!/usr/bin/env python3

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

ACTIVE_SCENARIO = 1

SCENARIO_DATA = {
    1: {
        # 20 threads, averaged over the first 1000 steps, in milliseconds
        # (fluid pillar with 200 x 350 x 200)
        "label": "Scenario 1",
        "particle_counts": [
            14_000_000,
        ],
        "method_component_times_ms": {
            "XYZ-order": {
                "Construction": [104],
                "Query":        [616],
            },
            "Z-order": {
                "Construction": [91],
                "Query":        [553],
            },
        },
    },
    2: {
        # 20 threads, averaged over the first 1000 steps, in milliseconds
        # (converging dam with 400 x 200 x 400 with a 300 x 200 x 300 hole)
        "label": "Scenario 2",
        "particle_counts": [
            14_000_000,
        ],
        "method_component_times_ms": {
            "XYZ-order": {
                "Construction": [116],
                "Query":        [627],
            },
            "Z-order": {
                "Construction": [97],
                "Query":        [558],
            },
        },
    },
    3: {
        # 20 threads, averaged over the first 1000 steps, in milliseconds
        # (big droplet with 1.5 radius droplet and no fluid bed)
        "label": "Scenario 3",
        "particle_counts": [
            14_000_000,
        ],
        "method_component_times_ms": {
            "XYZ-order": {
                "Construction": [115],
                "Query":        [606],
            },
            "Z-order": {
                "Construction": [98],
                "Query":        [537],
            },
        },
    },
}

SELECTED_SCENARIO = SCENARIO_DATA.get(ACTIVE_SCENARIO, {})
PARTICLE_COUNTS = SELECTED_SCENARIO.get("particle_counts", [])
METHOD_COMPONENT_TIMES_MS = SELECTED_SCENARIO.get("method_component_times_ms", {})


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Fluid particles [millions]"
Y_LABEL = "Neighbor search time [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (7.0, 8.0)
BAR_WIDTH = 0.25

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
Y_AXIS_TICK_INTERVAL = 100.0

SHOW_LEGEND = True
LEGEND_LOC = "center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.5)
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
    "XYZ-order": {
        "label": "XYZ-order",
        "construction_color": RGB(25, 50, 80),
        "query_color": RGB(40, 80, 180),
    },
    "Z-order": {
        "label": "Z-order",
        "construction_color": RGB(150, 0, 30),
        "query_color": RGB(250, 0, 50),
    },
}


def validate_plot_data() -> None:
    if ACTIVE_SCENARIO not in SCENARIO_DATA:
        raise ValueError(
            f"ACTIVE_SCENARIO must be one of {sorted(SCENARIO_DATA.keys())}."
        )

    if not PARTICLE_COUNTS:
        raise ValueError("PARTICLE_COUNTS must not be empty.")

    if len(METHOD_COMPONENT_TIMES_MS) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_COMPONENT_TIMES_MS)}."
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

    y_max = math.ceil((max_total_time_ms * 1.05) / Y_AXIS_TICK_INTERVAL) * Y_AXIS_TICK_INTERVAL
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
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=AXES_FONT_SIZE,
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
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
