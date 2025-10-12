// operations.h
//
// TODO: comment

#pragma once

// external
#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>



namespace Bench
{
	// "atomic" operations for benchmarking
	enum class BenchOp : uint16_t
	{
		// data structures
		Data_Sorting,						// re-sorting of the particle attribute data structures

		// neighbor search
		Neighbors_Construction,				// construction of the data structure used for the query
		Neighbors_Query,					// neighborhood tests between particles
		Neighbors_ListCompression,			// construction and (de-)compression of neighbor lists

		// kernel
		Kernel_Weights,
		Kernel_Gradients,

		// density
		Density_Solving,

		// pressure
		Pressure_Solving,					// computation of pressure values
		Pressure_Acceleration,				// computation of the pressure accelerations
		Pressure_TermCondition,				// evaluation of the pressure solver termination condition

		// viscosity
		Viscosity_Solving,					// computation of viscosity values
		Viscosity_Acceleration,				// computation of the viscosity accelerations

		// external
		External_Acceleration,				// computation of the viscosity accelerations

		// advection
		AccelerationUpdate,
		VelocityUpdate,
		PositionUpdate,

		// statistics
		Statistics_Particles,				// processing of particle/neighbor count statistics
		Statistics_Density,					// processing of density statistics
		Statistics_Velocity,				// processing of velocity statistics

		// snapshots
		Snapshot_Serial,					// packing the sim data into a data structure for serialization
		Snapshot_Export,					// packing the sim data into a data structure for export
		Snapshot_Visual,					// packing the sim data into a data structure for visualization

		// visualization
		Rendering,							// rendering of particles and structures
		UserInterface,						// updating and processing of the user interface
		UserInput,							// user input handling

		// totals
		Total_Sim,							// total time for simulation operations
		Total_Vis,							// total time for visualization operations

		// misc
		Misc_Sim,							// time for misc simulation operations
		Misc_Vis,							// time for misc visualization operations

		// test
		Test,								// for timing new operations

		Count
	};

	struct BenchOpDef
	{
		BenchOp op;
		std::string_view name; // display name
		std::string_view unit; // unit hint
	};

	struct BenchOpGroupDef
	{
		std::vector<BenchOp> ops; // group members
	};

	constexpr inline std::array<BenchOpDef, static_cast<uint16_t>(BenchOp::Count)> Definitions =
	{
		{
			{
				.op = BenchOp::Data_Sorting,
				.name = "data structure sorting",
				.unit = "ns"
			},
			{
				.op = BenchOp::Neighbors_Construction,
				.name = "neighbor construction",
				.unit = "ns"
			},
			{
				.op = BenchOp::Neighbors_Query,
				.name = "neighbor query",
				.unit = "ns"
			},
			{
				.op = BenchOp::Neighbors_ListCompression,
				.name = "neighbor list compress",
				.unit = "ns"
			},
			{
				.op = BenchOp::Kernel_Weights,
				.name = "kernel weights",
				.unit = "ns"
			},
			{
				.op = BenchOp::Kernel_Gradients,
				.name = "kernel gradients",
				.unit = "ns"
			},
			{
				.op = BenchOp::Density_Solving,
				.name = "density solving",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_Solving,
				.name = "pressure solving",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_Acceleration,
				.name = "pressure acc",
				.unit = "ns"
			},
			{
				.op = BenchOp::Pressure_TermCondition,
				.name = "pressure term cond",
				.unit = "ns"
			},
			{
				.op = BenchOp::Viscosity_Solving,
				.name = "viscosity solving",
				.unit = "ns"
			},
			{
				.op = BenchOp::Viscosity_Acceleration,
				.name = "viscosity acc",
				.unit = "ns"
			},
			{
				.op = BenchOp::External_Acceleration,
				.name = "external acc",
				.unit = "ns"
			},
			{
				.op = BenchOp::AccelerationUpdate,
				.name = "acceleration update",
				.unit = "ns"
			},
			{
				.op = BenchOp::VelocityUpdate,
				.name = "velocity update",
				.unit = "ns"
			},
			{
				.op = BenchOp::PositionUpdate,
				.name = "position update",
				.unit = "ns"
			},
			{
				.op = BenchOp::Statistics_Particles,
				.name = "stat particles",
				.unit = "ns"
			},
			{
				.op = BenchOp::Statistics_Density,
				.name = "stat density",
				.unit = "ns"
			},
			{
				.op = BenchOp::Statistics_Velocity,
				.name = "stat velocity",
				.unit = "ns"
			},
			{
				.op = BenchOp::Snapshot_Serial,
				.name = "serial snapshot",
				.unit = "ns"
			},
			{
				.op = BenchOp::Snapshot_Export,
				.name = "export snapshot",
				.unit = "ns"
			},
			{
				.op = BenchOp::Snapshot_Visual,
				.name = "visual snapshot",
				.unit = "ns"
			},
			{
				.op = BenchOp::Rendering,
				.name = "rendering",
				.unit = "ns"
			},
			{
				.op = BenchOp::UserInterface,
				.name = "user interface",
				.unit = "ns"
			},
			{
				.op = BenchOp::UserInput,
				.name = "user input",
				.unit = "ns"
			},
			{
				.op = BenchOp::Total_Sim,
				.name = "sim operations total",
				.unit = "ns"
			},
			{
				.op = BenchOp::Total_Vis,
				.name = "vis operations total",
				.unit = "ns"
			},
			{
				.op = BenchOp::Misc_Sim,
				.name = "sim misc",
				.unit = "ns"
			},
			{
				.op = BenchOp::Misc_Vis,
				.name = "vis misc",
				.unit = "ns"
			},
			{
				.op = BenchOp::Test,
				.name = "test",
				.unit = "ns"
			}
		}
	};
	static_assert(Definitions.size() == static_cast<uint16_t>(BenchOp::Count),
		"Bench::Definitions must list all BenchOp entries");

	constexpr inline uint16_t GetIdx(BenchOp op) { return static_cast<uint16_t>(op); }

	inline bool ValidateOpGroups(const std::vector<BenchOpGroupDef>& opGroups)
	{
		std::array<uint8_t, GetIdx(BenchOp::Count)> opIncluded{};

		for (const auto& group : opGroups)
		{
			for (const auto& op : group.ops)
			{
				const uint16_t i = GetIdx(op);

				if (opIncluded[i]) // duplicate
				{
					return false;
				}

				opIncluded[i] = 1;
			}
		}

		return true; // no duplicates
	}

	// record of all per-operation timings for one particular sim step
	struct BenchRecordStep
	{
		// metadata
		int64_t simStep = 0;
		double simTime = 0.0;

		// operation timings
		std::array<uint64_t, GetIdx(BenchOp::Count)> time{};
	};

} // namespace Bench
