#!/usr/bin/env python3

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# 20 threads, averaged over the first 100 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
SPATIAL_HASHING_TIME_MS = [
    (0.1, 361.63+22451.47),
    (0.15, 326.33+5402.97),
    (0.2, 321.77+1528.06),
    (0.25, 325.78+690.49),
    (0.3, 320.00+634.44),
    (0.35, 317.99+302.99),
    (0.4, 318.24+278.90),
    (0.45, 317.85+264.10),
    (0.5, 295.09+113.57),
    (0.55, 247.84+114.05),
    (0.6, 204.57+115.65),
    (0.65, 177.32+114.96),
    (0.7, 149.23+124.18),
    (0.75, 122.38+129.86),
    (0.8, 106.94+140.87),
    (0.85, 98.07+145.33),
    (0.9, 88.29+154.00),
    (0.95, 82.79+167.40),
    (1.0, 75.70+48.52),
    (1.05, 68.14+52.23),
    (1.1, 64.12+54.53),
    (1.15, 55.26+59.09),
    (1.2, 57.58+61.58),
    (1.25, 52.87+60.96),
    (1.3, 50.51+65.99),
    (1.35, 46.90+69.17),
    (1.4, 44.51+72.32),
    (1.45, 41.87+74.44),
    (1.5, 40.23+77.03),
    (1.55, 38.39+81.45),
    (1.6, 38.09+90.51),
    (1.65, 36.63+93.39),
    (1.7, 35.57+99.27),
    (1.75, 34.54+102.96),
    (1.8, 33.58+110.32),
    (1.85, 31.50+115.13),
    (1.9, 31.86+123.27),
    (1.95, 30.55+126.18),
    (2.0, 28.31+130.12),
]

# 20 threads, averaged over the first 100 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
CELL_LINKED_LIST_TIME_MS = [
    (0.1, 10.52+11104.03),
    (0.15, 9.10+4288.71),
    (0.2, 9.07+1909.50),
    (0.25, 9.47+1053.91),
    (0.3, 9.17+1167.83),
    (0.35, 9.30+639.03),
    (0.4, 9.22+649.82),
    (0.45, 9.32+686.18),
    (0.5, 9.35+133.15),
    (0.55, 9.29+120.80),
    (0.6, 8.76+109.46),
    (0.65, 8.45+101.32),
    (0.7, 7.85+96.81),
    (0.75, 7.19+92.16),
    (0.8, 7.42+93.45),
    (0.85, 7.27+93.64),
    (0.9, 7.11+95.94),
    (0.95, 6.99+99.70),
    (1.0, 6.92+37.08),
    (1.05, 6.45+40.73),
    (1.1, 6.94+44.79),
    (1.15, 6.62+46.98),
    (1.2, 6.72+50.31),
    (1.25, 6.30+51.90),
    (1.3, 6.47+56.18),
    (1.35, 6.28+58.46),
    (1.4, 6.32+63.49),
    (1.45, 6.06+65.22),
    (1.5, 6.00+70.18),
    (1.55, 5.97+75.05),
    (1.6, 5.97+82.03),
    (1.65, 5.93+85.32),
    (1.7, 5.80+91.80),
    (1.75, 5.75+94.35),
    (1.8, 5.75+102.59),
    (1.85, 6.02+109.37),
    (1.9, 5.99+115.84),
    (1.95, 6.02+125.38),
    (2.0, 5.60+128.79),
]

# 20 threads, averaged over the first 100 steps, in milliseconds
# (breaking dam with 100 x 100 x 100)
NEGATIVE_CANDIDATE_CHECKS_PCT = [
    (0.1, 74.14),
    (0.15, 75.52),
    (0.2, 74.90),
    (0.25, 74.88),
    (0.3, 76.56),
    (0.35, 75.76),
    (0.4, 81.77),
    (0.45, 87.93),
    (0.5, 75.97),
    (0.55, 82.30),
    (0.6, 85.96),
    (0.65, 88.32),
    (0.7, 91.07),
    (0.75, 93.06),
    (0.8, 93.96),
    (0.85, 94.82),
    (0.9, 95.70),
    (0.95, 96.38),
    (1.0, 86.00),
    (1.05, 88.06),
    (1.1, 89.59),
    (1.15, 90.79),
    (1.2, 91.69),
    (1.25, 92.44),
    (1.3, 93.39),
    (1.35, 94.19),
    (1.4, 94.81),
    (1.45, 95.33),
    (1.5, 95.75),
    (1.55, 96.15),
    (1.6, 96.50),
    (1.65, 96.80),
    (1.7, 97.04),
    (1.75, 97.24),
    (1.8, 97.47),
    (1.85, 97.69),
    (1.9, 97.87),
    (1.95, 98.02),
    (2.0, 98.17),
]


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Cell size ratio (cell size / kernel support radius)"
LEFT_Y_LABEL = "Neighbor search time [ms]"
RIGHT_Y_LABEL = "Negative candidate checks [%]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (24.0, 10.0)
LINE_WIDTH = 5.0
MARKER_SIZE = 20.0
CELL_COUNT_LABEL_FONT_SIZE = 22

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
RIGHT_Y_TICK_INTERVAL = 10.0

SHOW_LEGEND = True
LEGEND_LOC = "lower left"
LEGEND_BBOX_TO_ANCHOR = None
SHOW_GUIDE_RATIO_TICKS = True
FORMAT_X_AXIS_AS_RATIO = True
SHOW_QUERY_CELL_BACKGROUND = True
SHOW_QUERY_CELL_LABELS = True
QUERY_CELL_LABEL_Y_POS_AXES = 1.05
QUERY_CELL_LABEL_MIN_SPAN_WIDTH = 0.05
QUERY_CELL_LAST_LABEL_X = 1.6
PLOT_X_MIN = 0.01
PLOT_X_MAX = 2.10
# toggle for the left y-axis (log or non-log scale)
USE_LOG_LEFT_Y_AXIS = True
LEFT_Y_LOG_SCALE = "symlog"
LEFT_Y_SCALE_OPTIONS = {"linear", "log", "symlog"}
LEFT_Y_LOG_BASE = 10.0
LEFT_Y_SYMLOG_LINTHRESH = 100.0
LEFT_Y_SYMLOG_LINSCALE = 1.0
LEFT_Y_MAX_PADDING_FACTOR = 1.05

X_TICK_RATIOS = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


def RGBA(r: int, g: int, b: int, a: float) -> tuple[float, float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0, a)


TIMING_LINE_HALO_COLOR = RGBA(255, 255, 255, 0.45)
TIMING_LINE_HALO_EXTRA_WIDTH = 2.0
TIMING_MARKER_EDGE_COLOR = RGBA(255, 255, 255, 0.80)
TIMING_MARKER_EDGE_WIDTH = 1.4


SERIES_STYLES = {
    "Spatial Hashing": {
        "color": RGB(40, 80, 180),
        "marker": "o",
        "line_style": "-",
    },
    "Cell-Linked List": {
        "color": RGB(250, 0, 50),
        "marker": "o",
        "line_style": "-",
    },
    "Negative Candidates": {
        "color": RGB(0, 0, 0),
        "marker": "v",
        "line_style": "-",
    },
}

QUERY_CELL_BACKGROUND_COLORS = {
    21 ** 3: RGBA(0, 0, 0, 0.95),
    19 ** 3: RGBA(30, 0, 0, 0.95),
    17 ** 3: RGBA(60, 0, 0, 0.95),
    15 ** 3: RGBA(90, 0, 0, 0.95),
    13 ** 3: RGBA(120, 0, 0, 0.95),
    11 ** 3: RGBA(150, 40, 40, 0.95),
    9 ** 3: RGBA(180, 80, 80, 0.95),
    7 ** 3: RGBA(210, 120, 120, 0.95),
    5 ** 3: RGBA(240, 160, 160, 0.95),
    3 ** 3: RGBA(255, 200, 200, 0.95),
}
FALLBACK_QUERY_CELL_BACKGROUND_COLOR = QUERY_CELL_BACKGROUND_COLORS[21 ** 3]


def derive_candidate_cells_considered(cell_size_ratio: float) -> int:
    # the support radius spans 1 / ratio cells per axis
    cells_per_axis = 2 * math.ceil(1.0 / cell_size_ratio) + 1
    return cells_per_axis ** 3


def unpack_series_points(
    series_points: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    x_values = [cell_size_ratio for cell_size_ratio, _ in series_points]
    y_values = [measurement for _, measurement in series_points]
    return x_values, y_values


def derive_query_cell_spans(x_min: float, x_max: float) -> list[tuple[float, float, int]]:
    spans = []
    max_cell_radius = math.ceil(1.0 / x_min)
    min_cell_radius = math.ceil(1.0 / x_max)

    for cell_radius in range(max_cell_radius, min_cell_radius - 1, -1):
        span_min = max(x_min, 1.0 / cell_radius)
        if cell_radius == 1:
            span_max = x_max
        else:
            span_max = min(x_max, 1.0 / (cell_radius - 1))

        if span_min >= span_max:
            continue

        cells_per_axis = 2 * cell_radius + 1
        spans.append((span_min, span_max, cells_per_axis ** 3))

    return spans


def derive_positive_measurements(*series_list: list[float]) -> list[float]:
    return [
        measurement
        for series in series_list
        for measurement in series
        if measurement > 0.0
    ]


def configure_left_y_axis(
    ax_left: Axes,
    spatial_hashing_times: list[float],
    cll_times: list[float],
) -> None:
    left_y_scale = LEFT_Y_LOG_SCALE if USE_LOG_LEFT_Y_AXIS else "linear"
    positive_measurements = derive_positive_measurements(spatial_hashing_times, cll_times)
    max_measurement = max(max(spatial_hashing_times), max(cll_times)) * LEFT_Y_MAX_PADDING_FACTOR

    if left_y_scale == "linear":
        ax_left.set_ylim(0.0, max_measurement)
        return

    if left_y_scale == "log":
        min_positive_measurement = min(positive_measurements)
        ax_left.set_yscale("log", base=LEFT_Y_LOG_BASE)
        ax_left.set_ylim(min_positive_measurement, max_measurement)
        ax_left.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        return

    ax_left.set_yscale(
        "symlog",
        base=LEFT_Y_LOG_BASE,
        linthresh=LEFT_Y_SYMLOG_LINTHRESH,
        linscale=LEFT_Y_SYMLOG_LINSCALE,
    )
    ax_left.set_ylim(0.0, max_measurement)
    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


def validate_plot_data() -> None:
    if not SPATIAL_HASHING_TIME_MS:
        raise ValueError("SPATIAL_HASHING_TIME_MS must not be empty.")

    if not CELL_LINKED_LIST_TIME_MS:
        raise ValueError("CELL_LINKED_LIST_TIME_MS must not be empty.")

    if not NEGATIVE_CANDIDATE_CHECKS_PCT:
        raise ValueError("NEGATIVE_CANDIDATE_CHECKS_PCT must not be empty.")

    if LEFT_Y_LOG_SCALE not in LEFT_Y_SCALE_OPTIONS - {"linear"}:
        raise ValueError(
            "LEFT_Y_LOG_SCALE must be either 'log' or 'symlog'."
        )

    if LEFT_Y_LOG_BASE <= 1.0:
        raise ValueError("LEFT_Y_LOG_BASE must be greater than 1.0.")

    if LEFT_Y_SYMLOG_LINTHRESH <= 0.0:
        raise ValueError("LEFT_Y_SYMLOG_LINTHRESH must be positive.")

    if USE_LOG_LEFT_Y_AXIS and LEFT_Y_LOG_SCALE == "log":
        positive_measurements = derive_positive_measurements(
            [measurement for _, measurement in SPATIAL_HASHING_TIME_MS],
            [measurement for _, measurement in CELL_LINKED_LIST_TIME_MS],
        )
        total_measurement_count = len(SPATIAL_HASHING_TIME_MS) + len(CELL_LINKED_LIST_TIME_MS)
        if len(positive_measurements) != total_measurement_count:
            raise ValueError(
                "LEFT_Y_SCALE='log' requires all left-axis timing measurements to be strictly positive."
            )

    for series_name, series_points in (
        ("SPATIAL_HASHING_TIME_MS", SPATIAL_HASHING_TIME_MS),
        ("CELL_LINKED_LIST_TIME_MS", CELL_LINKED_LIST_TIME_MS),
        ("NEGATIVE_CANDIDATE_CHECKS_PCT", NEGATIVE_CANDIDATE_CHECKS_PCT),
    ):
        for entry in series_points:
            if len(entry) != 2:
                raise ValueError(f"{series_name} entries must be (cell_size_ratio, measurement) tuples.")

            cell_size_ratio, _ = entry
            if cell_size_ratio <= 0.0:
                raise ValueError(f"{series_name} must contain only positive cell size ratios.")


def plot_scaling() -> None:
    fig, ax_left = plt.subplots(figsize=FIG_SIZE)
    ax_right = ax_left.twinx()
    ax_right.patch.set_alpha(0.0)

    spatial_hashing_ratios, spatial_hashing_times = unpack_series_points(SPATIAL_HASHING_TIME_MS)
    cll_ratios, cll_times = unpack_series_points(CELL_LINKED_LIST_TIME_MS)
    negative_check_ratios, negative_check_pcts = unpack_series_points(NEGATIVE_CANDIDATE_CHECKS_PCT)

    query_cell_spans = derive_query_cell_spans(PLOT_X_MIN, PLOT_X_MAX)

    if SHOW_QUERY_CELL_BACKGROUND:
        for span_idx, (span_min, span_max, queried_cells) in enumerate(query_cell_spans):
            ax_left.axvspan(
                span_min,
                span_max,
                facecolor=QUERY_CELL_BACKGROUND_COLORS.get(
                    queried_cells,
                    FALLBACK_QUERY_CELL_BACKGROUND_COLOR,
                ),
                zorder=0,
            )

            if SHOW_QUERY_CELL_LABELS and (
                (span_max - span_min) >= QUERY_CELL_LABEL_MIN_SPAN_WIDTH
                or span_idx == len(query_cell_spans) - 1
            ):
                cells_per_axis = round(queried_cells ** (1.0 / 3.0))
                if span_idx == len(query_cell_spans) - 1:
                    cell_count_label = f"{cells_per_axis}$^3$ cells queried"
                    cell_count_label_x = QUERY_CELL_LAST_LABEL_X
                else:
                    cell_count_label = f"{cells_per_axis}$^3$"
                    cell_count_label_x = 0.5 * (span_min + span_max)
                ax_left.text(
                    cell_count_label_x,
                    QUERY_CELL_LABEL_Y_POS_AXES,
                    cell_count_label,
                    ha="center",
                    va="center",
                    fontsize=CELL_COUNT_LABEL_FONT_SIZE,
                    transform=ax_left.get_xaxis_transform(),
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                    zorder=1,
                )

    spatial_hashing_style = SERIES_STYLES["Spatial Hashing"]
    cll_style = SERIES_STYLES["Cell-Linked List"]
    negative_checks_style = SERIES_STYLES["Negative Candidates"]

    spatial_hashing_handle, = ax_left.plot(
        spatial_hashing_ratios,
        spatial_hashing_times,
        color=spatial_hashing_style["color"],
        marker=spatial_hashing_style["marker"],
        linestyle=spatial_hashing_style["line_style"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgecolor=TIMING_MARKER_EDGE_COLOR,
        markeredgewidth=TIMING_MARKER_EDGE_WIDTH,
        label="Spatial Hashing",
        zorder=3,
    )
    spatial_hashing_handle.set_path_effects([
        path_effects.Stroke(
            linewidth=LINE_WIDTH + TIMING_LINE_HALO_EXTRA_WIDTH,
            foreground=TIMING_LINE_HALO_COLOR,
        ),
        path_effects.Normal(),
    ])

    cll_handle, = ax_left.plot(
        cll_ratios,
        cll_times,
        color=cll_style["color"],
        marker=cll_style["marker"],
        linestyle=cll_style["line_style"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgecolor=TIMING_MARKER_EDGE_COLOR,
        markeredgewidth=TIMING_MARKER_EDGE_WIDTH,
        label="Cell-Linked List",
        zorder=3,
    )
    cll_handle.set_path_effects([
        path_effects.Stroke(
            linewidth=LINE_WIDTH + TIMING_LINE_HALO_EXTRA_WIDTH,
            foreground=TIMING_LINE_HALO_COLOR,
        ),
        path_effects.Normal(),
    ])

    negative_checks_handle, = ax_right.plot(
        negative_check_ratios,
        negative_check_pcts,
        color=negative_checks_style["color"],
        marker=negative_checks_style["marker"],
        linestyle=negative_checks_style["line_style"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgecolor=TIMING_MARKER_EDGE_COLOR,
        markeredgewidth=TIMING_MARKER_EDGE_WIDTH,
        label="Negative Candidates",
        zorder=3,
    )
    negative_checks_handle.set_path_effects([
        path_effects.Stroke(
            linewidth=LINE_WIDTH + TIMING_LINE_HALO_EXTRA_WIDTH,
            foreground=TIMING_LINE_HALO_COLOR,
        ),
        path_effects.Normal(),
    ])

    if SHOW_PLOT_TITLE:
        ax_left.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax_left.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax_left.set_ylabel(LEFT_Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax_right.set_ylabel(RIGHT_Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax_left.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax_right.tick_params(axis="y", labelsize=AXES_FONT_SIZE)

    if GRID_ENABLED:
        ax_left.grid(
            True,
            linestyle=GRID_LINE_STYLE,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )

    if SHOW_LEGEND:
        ax_left.legend(
            [spatial_hashing_handle, cll_handle, negative_checks_handle],
            ["Spatial Hashing", "Cell-Linked List", "Negative Candidates"],
            fontsize=AXES_FONT_SIZE,
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        )

    if SHOW_GUIDE_RATIO_TICKS:
        ax_left.set_xticks(X_TICK_RATIOS)

    if FORMAT_X_AXIS_AS_RATIO:
        ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    ax_left.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    configure_left_y_axis(ax_left, spatial_hashing_times, cll_times)
    right_y_max = math.ceil((max(negative_check_pcts) * 1.05) / RIGHT_Y_TICK_INTERVAL) * RIGHT_Y_TICK_INTERVAL
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
