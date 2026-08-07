// operations.h
//
// TODO: comment

#pragma once

// external
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>



namespace Bench
{
	// "atomic" operations for benchmarking
	enum class BenchOp : uint8_t
	{
		// data structures
		ParticleData_Reordering,            // re-ordering of the particle attribute data structures

		// neighbor search
		Neighbors_Construction,             // construction of the data structure used for the query
		Neighbors_Query,                    // neighborhood tests between particles

		// density
		Density_Computation,

		// pressure
		Pressure_SourceAndDiagonal,         // setup of pressure source terms, diagonal elements, and initial pressure
		Pressure_JacobiSolver,              // computation of pressure values via the Jacobi solver
		Pressure_Acceleration,              // computation of the pressure accelerations

		// non-pressure acceleration
		NonPressure_Acceleration,           // computation of non-pressure accelerations

		// particle integration
		Particle_Integration,               // integration of fluid particle velocities and positions

		// rigid bodies
		Rigid_StateUpdate,                  // update of rigid body states and attached boundary particles
		Rigid_ForceAccumulation,            // accumulation of rigid-body forces and torques

		// statistics
		Statistics_Computation,             // processing of simulation statistics about particle counts, density errors, velocity metrics, etc...

		// output
		Serialization,                      // packing and serialization of simulation data to disk
		Visualization,                      // preparation of simulation data for visualization

		// distributed simulation
		Dist_ParticleRedistribution,        // particle migration, halo exchange, and rank-local rebuild
		Dist_StateSync,                     // rank-local state handoff around backend data structures
		Dist_SolverSync,                    // fluid field halo exchanges during the rank-local solver step (subset of the state sync)
		Dist_StateHandoff,                  // sync plan rebuild, particle-order handoff, and rigid-body sync (subset of the state sync)
		Dist_RankBackendStep,               // rank-local backend step time used only for load balancing (overlaps with the others)

		// totals
		Total,                              // total time for one simulation step

		// test
		Test,                               // for timing new operations

		Count
	};

	struct BenchOpDef
	{
		BenchOp op;
		std::string_view name;  // display name
		std::string_view alias; // abbreviated name
		std::string_view unit;  // unit hint
	};

	constexpr inline std::array<BenchOpDef, static_cast<uint16_t>(BenchOp::Count)> Definitions =
	{
		{
			{
				.op = BenchOp::ParticleData_Reordering,
				.name = "particle data reordering",
				.alias = "particleDataReorder",
				.unit = "ns"
			},
			{
				.op = BenchOp::Neighbors_Construction,
				.name = "neighbor construction",
				.alias = "neighborConstruction",
				.unit = "ns"
			},
			{
				.op = BenchOp::Neighbors_Query,
				.name = "neighbor query",
				.alias = "neighborQuery",
				.unit = "ns"
			},
			{
				.op = BenchOp::Density_Computation,
				.name = "density computation",
				.alias = "densityComp",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_SourceAndDiagonal,
				.name = "pressure source and diagonal",
				.alias = "pressSourceAndDiag",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_JacobiSolver,
				.name = "pressure jacobi solver",
				.alias = "pressJacobiSolver",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_Acceleration,
				.name = "pressure acceleration",
				.alias = "pressAccel",
				.unit = "ns"
			},
			{
				.op = BenchOp::NonPressure_Acceleration,
				.name = "non-pressure acceleration",
				.alias = "nonPressureAcceleration",
				.unit = "ns"
			},
			{
				.op = BenchOp::Particle_Integration,
				.name = "particle integration",
				.alias = "particleIntegration",
				.unit = "ns"
			},
			{
				.op = BenchOp::Rigid_StateUpdate,
				.name = "rigid state update",
				.alias = "rigidStateUpdate",
				.unit = "ns"
			},
			{
				.op = BenchOp::Rigid_ForceAccumulation,
				.name = "rigid force accumulation",
				.alias = "rigidForceAccumulation",
				.unit = "ns"
			},
			{
				.op = BenchOp::Statistics_Computation,
				.name = "statistics computation",
				.alias = "statisticsComp",
				.unit = "ns"
			},
			{
				.op = BenchOp::Serialization,
				.name = "serialization",
				.alias = "serialization",
				.unit = "ns"
			},
			{
				.op = BenchOp::Visualization,
				.name = "visualization",
				.alias = "visualization",
				.unit = "ns"
			},
			{
				.op = BenchOp::Dist_ParticleRedistribution,
				.name = "dist particle redistribution",
				.alias = "distParticleRedistribution",
				.unit = "ns"
			},
			{
				.op = BenchOp::Dist_StateSync,
				.name = "dist state sync",
				.alias = "distStateSync",
				.unit = "ns"
			},
			{
				.op = BenchOp::Dist_SolverSync,
				.name = "dist solver sync",
				.alias = "distSolverSync",
				.unit = "ns"
			},
			{
				.op = BenchOp::Dist_StateHandoff,
				.name = "dist state handoff",
				.alias = "distStateHandoff",
				.unit = "ns"
			},
			{
				.op = BenchOp::Dist_RankBackendStep,
				.name = "dist rank backend step",
				.alias = "distRankBackendStep",
				.unit = "ns"
			},
			{
				.op = BenchOp::Total,
				.name = "total",
				.alias = "total",
				.unit = "ns"
			},
			{
				.op = BenchOp::Test,
				.name = "test",
				.alias = "testOperation",
				.unit = "ns"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint8_t>(BenchOp::Count),
		"Bench::Definitions must list all BenchOp entries");

	constexpr inline uint8_t GetIdx(BenchOp op) { return static_cast<uint8_t>(op); }

	using BenchOpList = std::vector<BenchOp>;

	inline BenchOpList MakeDefaultSimBenchOps()
	{
		return
		{
			BenchOp::ParticleData_Reordering,
			BenchOp::Neighbors_Construction,
			BenchOp::Neighbors_Query,
			BenchOp::Density_Computation,
			BenchOp::NonPressure_Acceleration,
			BenchOp::Pressure_SourceAndDiagonal,
			BenchOp::Pressure_JacobiSolver,
			BenchOp::Pressure_Acceleration,
			BenchOp::Rigid_ForceAccumulation,
			BenchOp::Particle_Integration,
			BenchOp::Rigid_StateUpdate,
			BenchOp::Statistics_Computation,
			BenchOp::Serialization,
			BenchOp::Visualization,
			BenchOp::Total
		};
	}

	inline bool ValidateOpList(const BenchOpList& ops)
	{
		std::array<uint8_t, GetIdx(BenchOp::Count)> opIncluded{};

		for (const BenchOp op : ops)
		{
			const uint8_t i = GetIdx(op);

			if (opIncluded[i]) // duplicate
			{
				return false;
			}

			opIncluded[i] = 1;
		}

		return true; // no duplicates
	}

	// record of all per-operation timings for one particular sim step
	struct BenchRecordStep
	{
		// metadata
		uint64_t simStep = 0;
		double simTime = 0.0;

		// operation timings
		std::array<uint64_t, GetIdx(BenchOp::Count)> time{};
	};

} // namespace Bench
