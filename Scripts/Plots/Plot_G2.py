#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# --- [ plot data ] --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

USE_BENCHMARK_CSV_INPUT = True
BENCHMARK_CSV_TIME_COLUMN = "total_sim_ms"

# 20 threads, total simulation time per step, in milliseconds
# (breaking dam with 100 x 100 x 100)
INTERVAL_BENCHMARK_CSV_PATHS = {
    "None": Path("Logs/602/benchmark_metrics.csv"),
    "2000": Path("Logs/601/benchmark_metrics.csv"),
    "200": Path("Logs/600/benchmark_metrics.csv"),
    "20": Path("Logs/599/benchmark_metrics.csv"),
    "2": Path("Logs/598/benchmark_metrics.csv"),
}

INTERVAL_STEP_TIMES_MS = {
    "None": [],
    "2000": [],
    "200": [],
    "20": [],
    "2": [],
}


# --------------------------------------------------------------------------------------
# --- [ plot styling ] -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLOT_TITLE = "TODO"
X_LABEL = "Simulation step"
Y_LABEL = "Total time per simulation step [ms]"
SHOW_PLOT_TITLE = False

AXES_FONT_SIZE = 30
FIGURE_TITLE_FONT_SIZE = 30

FIG_SIZE = (18.0, 9.0)
RAW_LINE_WIDTH = 1.5
RAW_LINE_ALPHA = 0.0
MEAN_LINE_WIDTH = 4.5
BAND_ALPHA = 0.25
SHOW_RAW_SERIES = True
USE_ROLLING_MEAN_BAND = True
ROLLING_WINDOW_SIZE = 30
ROLLING_BAND_STD_SCALE = 1.0
ROLLING_WINDOW_ALIGNMENT = "trailing"

GRID_ENABLED = True
GRID_LINE_STYLE = "--"
GRID_LINE_WIDTH = 1.0
GRID_ALPHA = 1.0
Y_AXIS_MIN = 10.0
Y_AXIS_MAX = 110.0
Y_AXIS_TICK_INTERVAL = 10.0

SHOW_LEGEND = True

OUTPUT_PATH = None
OUTPUT_DPI = 300


def RGB(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


INTERVAL_STYLES = {
    "None": {
        "label": "None",
        "color": RGB(0, 0, 0),
        "marker": "o",
        "line_style": "-",
    },
    "2000": {
        "label": "2000",
        "color": RGB(40, 80, 180),
        "marker": "o",
        "line_style": "-",
    },
    "200": {
        "label": "200",
        "color": RGB(250, 0, 50),
        "marker": "o",
        "line_style": "-",
    },
    "20": {
        "label": "20",
        "color": RGB(40, 180, 80),
        "marker": "o",
        "line_style": "-",
    },
    "2": {
        "label": "2",
        "color": RGB(250, 160, 0),
        "marker": "o",
        "line_style": "-",
    },
}


def validate_plot_data() -> None:
    interval_series = load_interval_step_times_ms()

    if len(interval_series) != len(INTERVAL_STYLES):
        raise ValueError(
            f"Expected exactly {len(INTERVAL_STYLES)} interval series, got {len(interval_series)}."
        )

    if ROLLING_WINDOW_SIZE <= 0:
        raise ValueError("ROLLING_WINDOW_SIZE must be positive.")
    if ROLLING_WINDOW_ALIGNMENT not in {"centered", "trailing"}:
        raise ValueError("ROLLING_WINDOW_ALIGNMENT must be 'centered' or 'trailing'.")

    for interval_name, step_time_samples in interval_series.items():
        if interval_name not in INTERVAL_STYLES:
            raise ValueError(
                f"Missing style entry for interval '{interval_name}'."
            )

        if not step_time_samples:
            raise ValueError(
                f"Interval '{interval_name}' must have at least one measurement."
            )

        for sample in step_time_samples:
            if len(sample) != 2:
                raise ValueError(
                    f"Interval '{interval_name}' samples must be (step, time_ms) tuples."
                )


def load_interval_step_times_ms() -> dict[str, list[tuple[int, float]]]:
    if not USE_BENCHMARK_CSV_INPUT:
        return INTERVAL_STEP_TIMES_MS

    interval_series: dict[str, list[tuple[int, float]]] = {}

    for interval_name, csv_path in INTERVAL_BENCHMARK_CSV_PATHS.items():
        samples: list[tuple[int, float]] = []

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(f"CSV '{csv_path}' has no header.")

            if BENCHMARK_CSV_TIME_COLUMN not in reader.fieldnames:
                raise ValueError(
                    f"CSV '{csv_path}' is missing column '{BENCHMARK_CSV_TIME_COLUMN}'."
                )

            for row in reader:
                samples.append((
                    int(row["simStep"]),
                    float(row[BENCHMARK_CSV_TIME_COLUMN]),
                ))

        interval_series[interval_name] = samples

    return interval_series


def compute_rolling_mean_std(
    step_times_ms: list[float],
) -> tuple[list[float], list[float], list[float]]:
    if not step_times_ms:
        return [], [], []

    if ROLLING_WINDOW_SIZE <= 1:
        return step_times_ms, step_times_ms, step_times_ms

    n = len(step_times_ms)
    prefix_sum = [0.0]
    prefix_sum_squ = [0.0]

    for time_ms in step_times_ms:
        prefix_sum.append(prefix_sum[-1] + time_ms)
        prefix_sum_squ.append(prefix_sum_squ[-1] + time_ms * time_ms)

    mean_values: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []

    for i in range(n):
        if ROLLING_WINDOW_ALIGNMENT == "trailing":
            i0 = max(0, i - ROLLING_WINDOW_SIZE + 1)
            i1 = i + 1
        else:
            half_window = ROLLING_WINDOW_SIZE // 2
            i0 = max(0, i - half_window)
            i1 = min(n, i + half_window + 1)

        window_count = i1 - i0

        window_sum = prefix_sum[i1] - prefix_sum[i0]
        window_sum_squ = prefix_sum_squ[i1] - prefix_sum_squ[i0]
        mean = window_sum / window_count
        variance = max(0.0, window_sum_squ / window_count - mean * mean)
        std_dev = math.sqrt(variance)
        band_half_width = ROLLING_BAND_STD_SCALE * std_dev

        mean_values.append(mean)
        lower_values.append(mean - band_half_width)
        upper_values.append(mean + band_half_width)

    return mean_values, lower_values, upper_values


def plot_scaling() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    interval_series = load_interval_step_times_ms()

    for interval_name, step_time_samples in interval_series.items():
        style = INTERVAL_STYLES[interval_name]
        step_numbers = [step for step, _ in step_time_samples]
        step_times_ms = [time_ms for _, time_ms in step_time_samples]
        mean_values, lower_values, upper_values = compute_rolling_mean_std(step_times_ms)

        if SHOW_RAW_SERIES:
            ax.plot(
                step_numbers,
                step_times_ms,
                color=style["color"],
                linestyle=style["line_style"],
                linewidth=RAW_LINE_WIDTH,
                alpha=RAW_LINE_ALPHA,
            )

        if USE_ROLLING_MEAN_BAND:
            ax.fill_between(
                step_numbers,
                lower_values,
                upper_values,
                color=style["color"],
                alpha=BAND_ALPHA,
                linewidth=0.0,
            )

        ax.plot(
            step_numbers,
            mean_values,
            color=style["color"],
            linestyle=style["line_style"],
            linewidth=MEAN_LINE_WIDTH,
            label=style["label"],
        )

    if SHOW_PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)

    ax.set_xlabel(X_LABEL, fontsize=AXES_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXES_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXES_FONT_SIZE)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.set_yticks([
        Y_AXIS_MIN + i * Y_AXIS_TICK_INTERVAL
        for i in range(int((Y_AXIS_MAX - Y_AXIS_MIN) / Y_AXIS_TICK_INTERVAL) + 1)
    ])

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
