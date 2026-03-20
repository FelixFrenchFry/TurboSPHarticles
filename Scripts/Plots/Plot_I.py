#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

DELTA_BUCKET_LABELS = [
    "Uncompressed",
    "0",
    "1",
    r"$[2, 2^8)$",
    r"$[2^8, 2^{32})$",
]

# 20 threads, averaged over the first 1000 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
METHOD_BUCKET_PERCENTAGES = {
    "XYZ":    [3.29, 47.18, 13.26, 32.09, 4.17],
    "Morton": [3.29, 47.18, 13.26, 32.09, 4.17],
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Share of stored neighbor delta encodings [%]"
Y_LABEL = "Delta encoding category"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30
BAR_LABEL_FONT_SIZE = 24

FIG_SIZE = (16.0, 10.0)
BAR_HEIGHT = 0.4
BAR_EDGE_WIDTH = 1.5
BAR_LABEL_OFFSET = 0.8

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
GRID_AXIS = "x"

SHOW_LEGEND = True
SHOW_VALUE_LABELS = True
VALUE_LABEL_DECIMALS = 2

SHOW_X_AXIS_PERCENT = True
X_AXIS_MIN = 0.0
X_AXIS_MAX = None
X_AXIS_TICK_INTERVAL = 10.0

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


METHOD_STYLES = {
    "Morton": {
        "label": "Z-order",
        "facecolor": RGB(120, 180, 230),
        "edgecolor": RGB(40, 80, 120),
        "textcolor": RGB(40, 80, 120),
        "hatch": "///",
    },
    "XYZ": {
        "label": "XYZ-order",
        "facecolor": RGB(220, 200, 140),
        "edgecolor": RGB(110, 100, 70),
        "textcolor": RGB(110, 100, 70),
        "hatch": "\\\\\\",
    },
}


def validate_plot_data() -> None:
    if not DELTA_BUCKET_LABELS:
        raise ValueError("DELTA_BUCKET_LABELS must not be empty.")

    if len(METHOD_BUCKET_PERCENTAGES) != 2:
        raise ValueError(
            f"Expected exactly 2 methods, got {len(METHOD_BUCKET_PERCENTAGES)}."
        )

    num_buckets = len(DELTA_BUCKET_LABELS)

    for method_name, bucket_percentages in METHOD_BUCKET_PERCENTAGES.items():
        if method_name not in METHOD_STYLES:
            raise ValueError(
                f"Missing style entry for method '{method_name}'."
            )

        if len(bucket_percentages) != num_buckets:
            raise ValueError(
                f"Method '{method_name}' has {len(bucket_percentages)} measurements, "
                f"but DELTA_BUCKET_LABELS has {num_buckets} entries."
            )

        for bucket_percentage in bucket_percentages:
            if bucket_percentage < 0.0:
                raise ValueError(
                    f"Method '{method_name}' contains a negative percentage."
                )


def plot_distribution() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    method_names = list(METHOD_BUCKET_PERCENTAGES.keys())
    bucket_positions = list(range(len(DELTA_BUCKET_LABELS)))
    method_index_center = 0.5 * (len(method_names) - 1)

    max_bucket_percentage = 0.0

    for method_idx, method_name in enumerate(method_names):
        style = METHOD_STYLES[method_name]
        bucket_percentages = METHOD_BUCKET_PERCENTAGES[method_name]

        y_positions = [
            bucket_pos + (method_idx - method_index_center) * BAR_HEIGHT
            for bucket_pos in bucket_positions
        ]

        ax.barh(
            y_positions,
            bucket_percentages,
            height=BAR_HEIGHT * 0.82,
            label=style["label"],
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=BAR_EDGE_WIDTH,
        )

        if SHOW_VALUE_LABELS:
            for y_pos, bucket_percentage in zip(y_positions, bucket_percentages):
                ax.text(
                    bucket_percentage + BAR_LABEL_OFFSET,
                    y_pos,
                    f"{bucket_percentage:.{VALUE_LABEL_DECIMALS}f}%",
                    va="center",
                    ha="left",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    color=style["textcolor"],
                )

        max_bucket_percentage = max(max_bucket_percentage, max(bucket_percentages))

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_yticks(bucket_positions)
    ax.set_yticklabels(DELTA_BUCKET_LABELS)
    ax.invert_yaxis()

    if GRID_ENABLED:
        ax.grid(
            True,
            axis=GRID_AXIS,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax.legend(fontsize=AXES_FONT_SIZE)

    if SHOW_X_AXIS_PERCENT:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))

    if X_AXIS_MAX is None:
        computed_x_axis_max = max_bucket_percentage + BAR_LABEL_OFFSET + 6.0
        ax.set_xlim(X_AXIS_MIN, computed_x_axis_max)
    else:
        ax.set_xlim(X_AXIS_MIN, X_AXIS_MAX)

    if X_AXIS_TICK_INTERVAL is not None:
        x_axis_max = ax.get_xlim()[1]
        x_ticks = []
        x_tick_value = X_AXIS_MIN

        while x_tick_value <= x_axis_max + 1e-9:
            x_ticks.append(x_tick_value)
            x_tick_value += X_AXIS_TICK_INTERVAL

        ax.set_xticks(x_ticks)

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
    plot_distribution()


if __name__ == "__main__":
    main()
