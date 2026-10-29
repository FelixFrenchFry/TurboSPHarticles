# TurboSPHarticles
HPC-oriented 3D SPH fluid solver with an interactive GUI mode for local runs and headless MPI for distributed runs (TBB CPU and CUDA GPU backends).

## Layout

- [`apps/`](apps/), [`source/`](source/) contain execution entry points and implementation
- [`assets/`](assets/), [`configs/`](configs/), [`external/`](external/) contain models, simulation settings, and dependencies
- [`scripts/`](scripts/), [`tests/`](tests/), [`docs/`](docs/) contain tools, tests, and documentation
- [`logs/`](logs/) contains run metrics and Slurm logs (.out/.err)
- [`exports/`](exports/) contains output exported from the simulation
- `out/` contains local build output

TODO: add proper documentation/examples/benchmarks
