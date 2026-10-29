# TurboSPHarticles
HPC-oriented 3D SPH fluid solver with an interactive GUI mode for local runs and headless MPI for distributed runs (TBB CPU and CUDA GPU backends).

## Layout

- `apps/`, `source/` contain execution entry points and implementation
- `assets/`, `configs/`, `external/` contain models, simulation settings, and dependencies
- `scripts/`, `tests/`, `docs/` contain tools, tests, and documentation
- `logs/` contains run metrics and Slurm logs (.out/.err)
- `exports/` contains output exported from the simulation
- `out/` contains local build output

TODO: add proper documentation/examples/benchmarks
