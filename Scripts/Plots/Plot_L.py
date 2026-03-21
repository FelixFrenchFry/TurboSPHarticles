#!/usr/bin/env python3

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

KERNEL_SUPPORT_RATIOS = [
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
    4.50,
    5.00,
]

FLUID_PARTICLE_COUNT = 1000000

AVERAGE_NEIGHBORS_PER_PARTICLE = [
    30.38,
    62.24,
    101.96,
    161.89,
    237.07,
    337.00,
    453.55
]

# 20 threads, averaged over the first 200 steps
# (breaking dam with 100 x 100 x 100)
METHOD_COMPONENT_RAW_BYTES = {
    "Explicit": {
        "Metadata": [8000000,   8000000,    8000000,    8000000,    8000000,    8000000,    8000000],
        "Storage":  [121520000, 248960000,  407840000,  647560000,  948280000,  1348000000, 1814200000],
    },
    "Compressed": {
        "Metadata": [12000000,  12000000,   12000000,   12000000,   12000000,   12000000,   12000000],
        "Storage":  [26497515,  43358617,   62170071,   88185241,   118468116,  157129113,  199722270],
    },
    #"Compressed_OLD_OUTDATED_WRONG": {
    #    "Metadata": [12000000,  12000000,   12000000,   12000000,   12000000,   12000000,   12000000],
    #    "Storage":  [24945623,  39027998,   57115934,   79817605,   107489525,  140372869,  178845122],
    #},
}

# not plotted at the moment
COMPRESSED_DELTA_BUCKET_PERCENTAGES = {
    "Explicit":    [3.29,   1.61,   0.98,   0.62,   0.42,   0.30,   0.22],
    "0":           [47.18,  56.94,  62.66,  67.69,  71.45,  74.46,  76.94],
    "1":           [13.26,  12.08,  11.18,  10.11,  9.24,   8.48,   7.81],
    "[2, 2^8)":    [32.09,  26.47,  22.92,  19.80,  17.43,  15.54,  13.99],
    "[2^8, 2^32)": [4.17,   2.90,   2.25,   1.78,   1.45,   1.22,   1.03],
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
Y_LABEL = "Average bytes per neighbor [B]"
X_LABEL = "Kernel support radius ratio"
RIGHT_Y_LABEL = "Average neighbors per particle"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30
BAR_VALUE_FONT_SIZE = 26
AVERAGE_NEIGHBOR_VALUE_FONT_SIZE = 26

FIG_SIZE = (24.0, 10.0)
BAR_WIDTH = 0.15
LINE_WIDTH = 5.0
MARKER_SIZE = 20.0
LINE_HALO_COLOR = (1.0, 1.0, 1.0, 0.45)
LINE_HALO_EXTRA_WIDTH = 2.0
AVERAGE_NEIGHBOR_LABEL_ABOVE_INDICES = {0, 2}
AVERAGE_NEIGHBOR_LABEL_OFFSET_ABOVE_PTS = 12
AVERAGE_NEIGHBOR_LABEL_OFFSET_BELOW_PTS = -16

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
RIGHT_Y_TICK_INTERVAL = 50.0

SHOW_LEGEND = True
LEGEND_LOC = "center left"

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


METHOD_STYLES = {
    "Explicit": {
        "label": "Explicit",
        "metadata_color": RGB(150, 180, 235),
        "storage_color": RGB(40, 80, 180),
    },
    "Compressed": {
        "label": "Compressed",
        "metadata_color": RGB(255, 170, 185),
        "storage_color": RGB(250, 0, 50),
    },
}


def validate_plot_data() -> None:
    if not KERNEL_SUPPORT_RATIOS:
        raise ValueError("KERNEL_SUPPORT_RATIOS must not be empty.")

    if FLUID_PARTICLE_COUNT <= 0:
        raise ValueError("FLUID_PARTICLE_COUNT must be positive.")

    if len(METHOD_COMPONENT_RAW_BYTES) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_COMPONENT_RAW_BYTES)}."
        )

    num_points = len(KERNEL_SUPPORT_RATIOS)

    if len(AVERAGE_NEIGHBORS_PER_PARTICLE) != num_points:
        raise ValueError(
            f"AVERAGE_NEIGHBORS_PER_PARTICLE has {len(AVERAGE_NEIGHBORS_PER_PARTICLE)} entries, "
            f"but KERNEL_SUPPORT_RATIOS has {num_points} entries."
        )

    if any(avg_neighbors <= 0.0 for avg_neighbors in AVERAGE_NEIGHBORS_PER_PARTICLE):
        raise ValueError("AVERAGE_NEIGHBORS_PER_PARTICLE entries must be positive.")

    for method_name, component_bytes in METHOD_COMPONENT_RAW_BYTES.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        if set(component_bytes.keys()) != {"Metadata", "Storage"}:
            raise ValueError(
                f"Method '{method_name}' must define exactly 'Metadata' and 'Storage'."
            )

        for component_name, component_values in component_bytes.items():
            if len(component_values) != num_points:
                raise ValueError(
                    f"Method '{method_name}' component '{component_name}' has "
                    f"{len(component_values)} measurements, but KERNEL_SUPPORT_RATIOS has "
                    f"{num_points} entries."
                )


def compute_component_bytes_per_neighbor(raw_component_bytes: list[float]) -> list[float]:
    return [
        raw_bytes / (FLUID_PARTICLE_COUNT * avg_neighbors)
        for raw_bytes, avg_neighbors in zip(raw_component_bytes, AVERAGE_NEIGHBORS_PER_PARTICLE)
    ]


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax_right = ax.twinx()

    method_names = list(METHOD_COMPONENT_RAW_BYTES.keys())
    legend_handles = []
    legend_labels = []

    for method_idx, method_name in enumerate(method_names):
        style = METHOD_STYLES[method_name]
        storage_bytes = compute_component_bytes_per_neighbor(
            METHOD_COMPONENT_RAW_BYTES[method_name]["Storage"]
        )
        x_offset = (method_idx - (len(method_names) - 1) / 2.0) * BAR_WIDTH

        storage_handle = ax.bar(
            [x + x_offset for x in KERNEL_SUPPORT_RATIOS],
            storage_bytes,
            width=BAR_WIDTH,
            color=style["storage_color"],
            label=f"{style['label']} Storage",
        )

        legend_handles.append(storage_handle[0])
        legend_labels.append(f"{style['label']} Storage")

        for x_pos, visible_bar_top in zip(
            [x + x_offset for x in KERNEL_SUPPORT_RATIOS],
            storage_bytes,
        ):
            ax.annotate(
                f"{visible_bar_top:.2f}".replace(".", ","),
                xy=(x_pos, visible_bar_top),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_FONT_SIZE,
                color=style["storage_color"],
            )

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    average_neighbors_handle, = ax_right.plot(
        KERNEL_SUPPORT_RATIOS,
        AVERAGE_NEIGHBORS_PER_PARTICLE,
        color="black",
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Average Neighbors",
    )
    average_neighbors_handle.set_path_effects([
        path_effects.Stroke(
            linewidth=LINE_WIDTH + LINE_HALO_EXTRA_WIDTH,
            foreground=LINE_HALO_COLOR,
        ),
        path_effects.Normal(),
    ])

    for point_idx, (x_pos, avg_neighbors) in enumerate(
        zip(KERNEL_SUPPORT_RATIOS, AVERAGE_NEIGHBORS_PER_PARTICLE)
    ):
        label_is_above = point_idx in AVERAGE_NEIGHBOR_LABEL_ABOVE_INDICES
        average_neighbor_text = ax_right.annotate(
            f"{avg_neighbors:.2f}".replace(".", ","),
            xy=(x_pos, avg_neighbors),
            xytext=(0, AVERAGE_NEIGHBOR_LABEL_OFFSET_ABOVE_PTS if label_is_above else AVERAGE_NEIGHBOR_LABEL_OFFSET_BELOW_PTS),
            textcoords="offset points",
            ha="center",
            va="bottom" if label_is_above else "top",
            fontsize=AVERAGE_NEIGHBOR_VALUE_FONT_SIZE,
            color="black",
        )
        average_neighbor_text.set_path_effects([
            path_effects.Stroke(linewidth=4.0, foreground=(1.0, 1.0, 1.0, 0.80)),
            path_effects.Normal(),
        ])

    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_xticks(KERNEL_SUPPORT_RATIOS)
    ax.set_xticklabels(
        [f"{kernel_support_ratio:.2f}" for kernel_support_ratio in KERNEL_SUPPORT_RATIOS]
    )
    ax_right.set_ylabel(RIGHT_Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax_right.tick_params(axis="y", labelsize=AXES_FONT_SIZE)
    right_y_max = max(AVERAGE_NEIGHBORS_PER_PARTICLE)
    right_y_tick_max = (
        math.ceil(right_y_max / RIGHT_Y_TICK_INTERVAL) * RIGHT_Y_TICK_INTERVAL
    )
    ax_right.set_yticks(
        [
            tick_value
            for tick_value in range(
                0,
                int(right_y_tick_max + RIGHT_Y_TICK_INTERVAL),
                int(RIGHT_Y_TICK_INTERVAL),
            )
        ]
    )

    if GRID_ENABLED:
        ax.grid(
            True,
            axis="y",
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    legend_handles.append(average_neighbors_handle)
    legend_labels.append("Average Neighbors")

    if SHOW_LEGEND:
        ax.legend(
            legend_handles,
            legend_labels,
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
