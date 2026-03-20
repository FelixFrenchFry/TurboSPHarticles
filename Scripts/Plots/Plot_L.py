#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt


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

# 20 threads, averaged over the first 200 steps
# (breaking dam with 100 x 100 x 100)
METHOD_COMPONENT_BYTES_PER_NEIGHBOR = {
    "Explicit": {
        "Metadata": [],
        "Storage":  [],
    },
    "Compressed": {
        "Metadata": [],
        "Storage":  [],
    },
}

PLOT_TITLE = "TODO"
Y_LABEL = "Average bytes per neighbor [B]"
X_LABEL = "Kernel support radius / particle size h"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (24.0, 10.0)
BAR_WIDTH = 0.15

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

    if len(METHOD_COMPONENT_BYTES_PER_NEIGHBOR) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_COMPONENT_BYTES_PER_NEIGHBOR)}."
        )

    num_points = len(KERNEL_SUPPORT_RATIOS)

    for method_name, component_bytes in METHOD_COMPONENT_BYTES_PER_NEIGHBOR.items():
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


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    method_names = list(METHOD_COMPONENT_BYTES_PER_NEIGHBOR.keys())
    legend_handles = []
    legend_labels = []

    for method_idx, method_name in enumerate(method_names):
        style = METHOD_STYLES[method_name]
        metadata_bytes = METHOD_COMPONENT_BYTES_PER_NEIGHBOR[method_name]["Metadata"]
        storage_bytes = METHOD_COMPONENT_BYTES_PER_NEIGHBOR[method_name]["Storage"]
        x_offset = (method_idx - (len(method_names) - 1) / 2.0) * BAR_WIDTH

        storage_handle = ax.bar(
            [x + x_offset for x in KERNEL_SUPPORT_RATIOS],
            storage_bytes,
            width=BAR_WIDTH,
            color=style["storage_color"],
            label=f"{style['label']} Storage",
        )
        metadata_handle = ax.bar(
            [x + x_offset for x in KERNEL_SUPPORT_RATIOS],
            metadata_bytes,
            width=BAR_WIDTH,
            bottom=storage_bytes,
            color=style["metadata_color"],
            label=f"{style['label']} Metadata",
        )

        legend_handles.append(storage_handle[0])
        legend_labels.append(f"{style['label']} Storage")
        legend_handles.append(metadata_handle[0])
        legend_labels.append(f"{style['label']} Metadata")

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_xticks(KERNEL_SUPPORT_RATIOS)

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
