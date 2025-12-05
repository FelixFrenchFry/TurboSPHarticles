#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pyvista as pv



# --- [ config ] ---------------------------------------------------------------

DATA_DIR        = Path("../../Export/C")
INPUT_TEMPLATE  = "pos_vel_step_{step:09d}.xdmf"
OUTPUT_DIR      = Path("../../Export/C")

STEP_START      = 0
STEP_END        = 1000
STEP_STRIDE     = 10

TARGET_POINTS   = 320_000_000
USE_SPHERES     = False
POINT_SIZE      = 0.1
CMAP            = "Blues"
BACKGROUND      = "black"

SCALAR_CANDIDATES = ["velMagF", "velMag", "velocity", "p", "pressure", "density"]
SCALAR_RANGE      = None # e.g. (0.0, 5.0) or None for auto

CAMERA_POSITION = [
    (25.0, 18.0, 25.0),  # eye
    (7.5,  7.5,  7.5),   # focal
    (0.0,  1.0,  0.0),   # up
]

USE_MULTIPROCESSING = False
NUM_PROCESSES       = 8

# --- [ logic ] ----------------------------------------------------------------

def step_to_filename(step: int) -> Path:
    name = INPUT_TEMPLATE.format(step=step)
    return DATA_DIR / name


def step_to_output(step: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"frame_{step:09d}.png"


def choose_scalar_name(mesh: pv.DataSet) -> str | None:
    names = set(mesh.array_names)
    for cand in SCALAR_CANDIDATES:
        if cand in names:
            return cand
    return None


def render_step(step: int) -> None:
    in_path = step_to_filename(step)
    out_path = step_to_output(step)

    try:
        if not in_path.exists():
            print(f"[step {step:9d}] missing file: {in_path}")
            return

        print(f"[step {step:9d}] reading {in_path}")
        mesh = pv.read(in_path)

        n_points = mesh.n_points
        if n_points == 0:
            print(f"[step {step:9d}] mesh has no points, skipping")
            return

        n_sample = min(TARGET_POINTS, n_points)
        if n_sample < n_points:
            idx = np.random.choice(n_points, size=n_sample, replace=False)
            submesh = mesh.extract_points(idx, include_cells=False)
            print(f"[step {step:9d}] subsampled {n_sample}/{n_points} points")
        else:
            submesh = mesh
            print(f"[step {step:9d}] using all {n_points} points")

        scalars_name = choose_scalar_name(submesh)
        if scalars_name is None:
            print(f"[step {step:9d}] no scalar from {SCALAR_CANDIDATES}, rendering uncolored")

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(BACKGROUND)

        kwargs = dict(
            scalars=scalars_name,
            cmap=CMAP,
            render_points_as_spheres=USE_SPHERES,
            point_size=POINT_SIZE,
        )
        if SCALAR_RANGE is not None and scalars_name is not None:
            kwargs["clim"] = SCALAR_RANGE

        plotter.add_points(submesh, **kwargs)
        plotter.camera_position = CAMERA_POSITION

        out_path = out_path.resolve()
        plotter.show(screenshot=str(out_path), auto_close=True)
        print(f"[step {step:9d}] saved {out_path}")

    except Exception as e:
        print(f"[step {step:9d}] ERROR: {e}")


def main():
    try:
        pv.start_xvfb()
    except Exception:
        pass

    if len(sys.argv) == 4:
        start = int(sys.argv[1])
        end   = int(sys.argv[2])
        stride = int(sys.argv[3])
    else:
        start = STEP_START
        end   = STEP_END
        stride = STEP_STRIDE

    steps = list(range(start, end + 1, stride))
    print(f"Rendering steps: {steps[0]} .. {steps[-1]} (stride {stride}), total {len(steps)}")

    if USE_MULTIPROCESSING:
        with mp.Pool(NUM_PROCESSES) as pool:
            pool.map(render_step, steps)
    else:
        for s in steps:
            render_step(s)


if __name__ == "__main__":
    main()
