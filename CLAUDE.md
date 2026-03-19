# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TurboSPHarticles is a high-performance SPH (Smoothed Particle Hydrodynamics) fluid simulation framework written in C++20. It supports CPU (TBB-parallelized) and GPU (CUDA) backends, with an optional OpenGL visualization frontend.

## Build Commands

Uses CMake with vcpkg (as a git submodule in `External/vcpkg/`). Presets are defined in `CMakePresets.json`.

```bash
# Configure (Windows example; Linux presets also available)
cmake --preset Windows-x64-RelWithDeb

# Build
cmake --build out/build/Windows-x64-RelWithDeb

# Available presets: Windows-x64-{Debug,RelWithDeb,Release}, Linux-x64-{Debug,RelWithDeb,Release}
```

Two executables are produced in `out/build/<preset>/bin/`:
- **TSP_Headless** — CLI-only simulation (always built)
- **TSP_Visual** — simulation with OpenGL visualization (requires `ENABLE_VIS=ON`, the default)

```bash
# Run with a config profile (A-Z or Default)
TSP_Headless -p A
TSP_Visual -p Default
```

CMake options: `ENABLE_CUDA` (OFF), `ENABLE_VIS` (ON), `ENABLE_NATIVE_OPT` (OFF).

## Architecture

```
Apps/                    Entry points: main_headless.cpp, main_visual.cpp
Source/
  Base/                  Shared enums (enum_hub.h) and math helpers
  Eval/                  Evaluation framework
    Bench/               Per-operation benchmarking
    Mem/                 Memory usage tracking
    Serial/              HDF5/XDMF data export
    Stats/               Simulation statistics (density, velocity, CFL)
  Sim/                   Simulation engine
    Core/                SimulationMgr, SimParams, I_SimBackend interface, DiagnosticsMgr
    CPU/CPU_Backend_01/  CPU backend (TBB): solver, neighbor search, neighbor lists
    GPU/GPU_Backend_01/  GPU/CUDA backend
    Scene/               Scenario definitions (BreakingDam, ConvergingDam, BigDroplet, SmallDroplets)
    Sync/                Lock-free mailboxes and snapshot types for inter-thread communication
  Vis/                   Visualization (OpenGL + ImGui + ImPlot)
Configs/                 Config profiles (Default, A-F), each with params.json and scenarios.json
Export/                  Output directory for serialized simulation data
```

**Key abstractions:**
- `I_SimBackend` (backend_sim.h) — abstract interface for CPU/GPU backends
- `I_NeighborSearchStrat` — 4 strategies: BruteForce, SpatialHashing_Basic, CellLinkedList_Basic, CellLinkedList_Compact
- `I_NeighborLists` — 7 storage variants (individual vectors, CSR, compressed)
- `SimulationMgr` — orchestrator; runs blocking (headless) or async (visual) via `std::jthread`
- Lock-free `Mailbox` types in Sync/ for cross-thread snapshot exchange

**Threading model:** TBB for computation parallelism (`tbb::parallel_for`, `tbb::parallel_sort`). Visualization runs on a separate thread communicating via atomic mailboxes.

## Code Conventions

- **C++20** required, no extensions
- **Tabs** for indentation
- **Comment separator lines** at ~100 characters
- **Naming:** PascalCase for classes and methods, snake_case with trailing underscore for member variables (e.g., `cellSize_`), interface classes prefixed with `I_` (e.g., `I_SimBackend`)
- **Enum style:** `enum class` in `Base::` namespace with companion `constexpr ToCString()` functions
- **Include organization:** `// external` section first, then `// internal` section, with project-root-relative paths (e.g., `#include "Source/Sim/Core/parameters_sim.h"`)
- **Namespaces:** `Base::`, `Sim::`, `Vis::`, `Bench::`, `Mem::`, `Stats::`, `Export::`, `Scene::`, `Sync::`
- **Header guards:** `#pragma once`
- **Platform defines:** `WIN32_LEAN_AND_MEAN`, `NOMINMAX`, `_UNICODE` on Windows

## Configuration

Simulation parameters live in `Configs/<profile>/params.json`. Key settings include backend selection, scenario type, neighbor search method, neighbor list storage, solver parameters, and evaluation/export toggles. Profiles are selected at runtime via `-p <profile>`.
