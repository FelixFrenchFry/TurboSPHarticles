#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pyvista as pv


# --- [ config ] ---------------------------------------------------------------

DATA_DIR        = Path("../../Export/B")
INPUT_TEMPLATE  = "pos_vel_step_{step:09d}.xdmf"
OUTPUT_DIR      = Path("../../Export/B")

STEP_START      = 0
STEP_END        = 1000
STEP_STRIDE     = 10

# maximum number of particles used to build the volume
TARGET_POINTS   = 100_000_000

# volume grid resolution (points, not cells)
GRID_RES        = (384, 384, 384)

# render resolution
IMAGE_SIZE      = (3840, 2160)

CMAP            = "Blues"
BACKGROUND      = "black"

SCALAR_CANDIDATES = ["velMagF", "velMag", "velocity", "p", "pressure", "density"]
SCALAR_RANGE      = None  # e.g. (0.0, 5.0) or None for auto
VOLUME_OPACITY    = "sigmoid"  # or e.g. "linear", or a custom array

CAMERA_POSITION = [
    (25.0, 18.0, 25.0),  # eye
    (7.5,  7.5,  7.5),   # focal
    (0.0,  1.0,  0.0),   # up
]

USE_MULTIPROCESSING = False
NUM_PROCESSES       = 8

# --- [ helpers ] --------------------------------------------------------------

def step_to_filename(step: int) -> Path:
    name = INPUT_TEMPLATE.format(step=step)
    return DATA_DIR / name


def step_to_output(step: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"frame_vol_{step:09d}.png"


def choose_scalar_name(dataset: pv.DataSet) -> str | None:
    names = set(dataset.array_names)
    for cand in SCALAR_CANDIDATES:
        if cand in names:
            return cand
    return None


def subsample_points(mesh: pv.DataSet) -> pv.DataSet:
    n_points = mesh.n_points
    if n_points == 0:
        return mesh

    n_sample = min(TARGET_POINTS, n_points)
    if n_sample < n_points:
        idx = np.random.choice(n_points, size=n_sample, replace=False)
        submesh = mesh.extract_points(idx, include_cells=False)
        print(f"  subsampled {n_sample}/{n_points} points")
        return submesh

    print(f"  using all {n_points} points")
    return mesh


def build_uniform_grid_from_bounds(bounds, res) -> pv.UniformGrid:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = res

    # avoid zero-length axes
    x_len = max(xmax - xmin, 1e-6)
    y_len = max(ymax - ymin, 1e-6)
    z_len = max(zmax - zmin, 1e-6)

    dx = x_len / (nx - 1)
    dy = y_len / (ny - 1)
    dz = z_len / (nz - 1)

    grid = pv.UniformGrid()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (dx, dy, dz)
    return grid


# --- [ core rendering ] -------------------------------------------------------

def render_step(step: int) -> None:
    in_path = step_to_filename(step)
    out_path = step_to_output(step)

    try:
        if not in_path.exists():
            print(f"[step {step:9d}] missing file: {in_path}")
            return

        print(f"[step {step:9d}] reading {in_path}")
        mesh = pv.read(in_path)

        if mesh.n_points == 0:
            print(f"[step {step:9d}] mesh has no points, skipping")
            return

        # optional subsampling
        mesh = subsample_points(mesh)

        # prepare a uniform grid over the particle bounds
        bounds = mesh.bounds
        print(f"  bounds: x[{bounds[0]:.3f}, {bounds[1]:.3f}], "
              f"y[{bounds[2]:.3f}, {bounds[3]:.3f}], "
              f"z[{bounds[4]:.3f}, {bounds[5]:.3f}]")

        grid = build_uniform_grid_from_bounds(bounds, GRID_RES)

        # resample point data onto the grid
        print(f"  resampling to grid {GRID_RES} (this may take a bit)...")
        volume = grid.sample(mesh)

        # choose scalar for volume rendering
        scalars_name = choose_scalar_name(volume)
        if scalars_name is None:
            print(f"[step {step:9d}] no scalar from {SCALAR_CANDIDATES}, aborting")
            return

        print(f"  using scalar '{scalars_name}' for volume rendering")

        # build plotter & render
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(BACKGROUND)
        plotter.window_size = IMAGE_SIZE

        kwargs = dict(
            scalars=scalars_name,
            cmap=CMAP,
            opacity=VOLUME_OPACITY,
            shade=True,
        )
        if SCALAR_RANGE is not None:
            kwargs["clim"] = SCALAR_RANGE

        plotter.add_volume(volume, **kwargs)
        plotter.camera_position = CAMERA_POSITION

        out_path = out_path.resolve()
        plotter.show(screenshot=str(out_path), auto_close=True)
        print(f"[step {step:9d}] saved {out_path}")

    except Exception as e:
        print(f"[step {step:9d}] ERROR: {e}")

# --- [ main ] -----------------------------------------------------------------

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
    print(f"Rendering volume steps: {steps[0]} .. {steps[-1]} "
          f"(stride {stride}), total {len(steps)}")

    if USE_MULTIPROCESSING:
        with mp.Pool(NUM_PROCESSES) as pool:
            pool.map(render_step, steps)
    else:
        for s in steps:
            render_step(s)


if __name__ == "__main__":
    main()
