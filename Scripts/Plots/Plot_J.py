#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

DELTA_BUCKET_LABELS = [
    "Explicit",
    "0",
    "1",
    r"$[2, 2^8)$",
    r"$[2^8, 2^{32})$",
]

# 20 threads, averaged over the first 1000 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
INTERVAL_BUCKET_PERCENTAGES = {
    "None": [3.29, 25.87, 11.93, 45.27, 13.64],
    "100":  [3.29, 42.19, 13.84, 35.04, 4.85],
    "10":   [3.29, 46.96, 13.33, 32.23, 4.19],
    "1":    [3.29, 47.18, 13.26, 32.09, 4.17],
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

FIG_SIZE = (16.0, 11.0)
BAR_HEIGHT = 0.2
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


INTERVAL_STYLES = {
    "1": {
        "label": "1",
        "facecolor": RGB(250, 160, 0),
        "edgecolor": RGB(180, 110, 0),
        "textcolor": RGB(180, 110, 0),
        "hatch": "///",
    },
    "10": {
        "label": "10",
        "facecolor": RGB(40, 180, 80),
        "edgecolor": RGB(20, 120, 50),
        "textcolor": RGB(20, 120, 50),
        "hatch": "\\\\\\",
    },
    "100": {
        "label": "100",
        "facecolor": RGB(250, 0, 50),
        "edgecolor": RGB(150, 0, 30),
        "textcolor": RGB(150, 0, 30),
        "hatch": "///",
    },
    "None": {
        "label": "None",
        "facecolor": RGB(60, 60, 60),
        "edgecolor": RGB(0, 0, 0),
        "textcolor": RGB(0, 0, 0),
        "hatch": "\\\\\\",
    },
}


def validate_plot_data() -> None:
    if not DELTA_BUCKET_LABELS:
        raise ValueError("DELTA_BUCKET_LABELS must not be empty.")

    if len(INTERVAL_BUCKET_PERCENTAGES) != 4:
        raise ValueError(
            f"Expected exactly 4 intervals, got {len(INTERVAL_BUCKET_PERCENTAGES)}."
        )

    num_buckets = len(DELTA_BUCKET_LABELS)

    for interval_name, bucket_percentages in INTERVAL_BUCKET_PERCENTAGES.items():
        if interval_name not in INTERVAL_STYLES:
            raise ValueError(
                f"Missing style entry for interval '{interval_name}'."
            )

        if len(bucket_percentages) != num_buckets:
            raise ValueError(
                f"Interval '{interval_name}' has {len(bucket_percentages)} measurements, "
                f"but DELTA_BUCKET_LABELS has {num_buckets} entries."
            )

        for bucket_percentage in bucket_percentages:
            if bucket_percentage < 0.0:
                raise ValueError(
                    f"Interval '{interval_name}' contains a negative percentage."
                )


def plot_distribution() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    interval_names = list(INTERVAL_BUCKET_PERCENTAGES.keys())
    bucket_positions = list(range(len(DELTA_BUCKET_LABELS)))
    interval_index_center = 0.5 * (len(interval_names) - 1)

    max_bucket_percentage = 0.0

    for interval_idx, interval_name in enumerate(interval_names):
        style = INTERVAL_STYLES[interval_name]
        bucket_percentages = INTERVAL_BUCKET_PERCENTAGES[interval_name]

        y_positions = [
            bucket_pos + (interval_idx - interval_index_center) * BAR_HEIGHT
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
