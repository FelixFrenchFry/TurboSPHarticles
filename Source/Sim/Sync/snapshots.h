// snapshots.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <vector>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Eval/Bench/operations.h"
#include "Source/Eval/Mem/metrics_mem.h"
#include "Source/Eval/Stats/metrics.h"
#include "Source/Sim/Scene/builder.h"



namespace Sync
{
	// data structure for serializing the simulation state,
	// (including a minimal subset of the data required to reconstruct the full state, enabling lossless save & load)
	struct SerialSnapshot
	{
		// positions
		std::vector<float> xPos;
		std::vector<float> yPos;
		std::vector<float> zPos;

		// velocities
		std::vector<float> xVel;
		std::vector<float> yVel;
		std::vector<float> zVel;

		// accelerations (total)
		std::vector<float> xAcc;
		std::vector<float> yAcc;
		std::vector<float> zAcc;

		// pressure accelerations
		std::vector<float> xAccPress;
		std::vector<float> yAccPress;
		std::vector<float> zAccPress;

		// viscosity accelerations
		std::vector<float> xAccVisc;
		std::vector<float> yAccVisc;
		std::vector<float> zAccVisc;

		// external accelerations
		// (gravity, surface tensions, etc...)
		std::vector<float> xAccExt;
		std::vector<float> yAccExt;
		std::vector<float> zAccExt;

		// densities and pressures
		std::vector<float> density;
		std::vector<float> pressure;
	};

	// data structure for exporting parts of the simulation state
	// (including a minimal subset of the data requested to analyze/model the simulation offline)
	// TODO: have one export snapshot per field (e.g. positions, velocities, etc...) and
	// export the components of an export request individually, because they are in separate files anyway?
	struct ExportSnapshot
	{
		// metadata
		uint64_t simStep;
		double simTime;

		// data for fluid particle export
		uint32_t N_F;
		std::vector<uint32_t> idF;
		std::vector<float> xPosF;
		std::vector<float> yPosF;
		std::vector<float> zPosF;
		std::vector<float> velMagF;

		// data for boundary particle export
		uint32_t N_B;
		std::vector<uint32_t> idB;
		std::vector<float> xPosB;
		std::vector<float> yPosB;
		std::vector<float> zPosB;
		std::vector<float> volB;

		// data for rigid particle export
		uint32_t N_R;
		std::vector<uint32_t> idR;
		std::vector<uint32_t> bodyIdR;
		std::vector<float> xPosR;
		std::vector<float> yPosR;
		std::vector<float> zPosR;
		std::vector<float> volR;
	};

	// data structure for visualizing parts of the simulation state interactively
	// (including a minimal subset of the data required for the interactive visualization)
	struct VisualSnapshot
	{
		// metadata
		uint64_t simStep;                   // simulation step at this snapshot

		// data for fluid particle vis
		uint32_t N_F;                       // number of fluid particles
		std::vector<glm::vec3> posF;        // positions of all fluid particles
		std::vector<float> scalarDataF;     // data according to color coding mode
		Base::ColorCodingMode scalarModeF;  // color coding mode

		// data for boundary structures vis
		uint32_t N_B;                       // number of boundary particles
		std::vector<glm::vec3> posB;        // positions of all boundary particles
		std::vector<float> volB;            // contribution volumes of all boundary particles
		std::vector<Scene::OBB> boxesB;     // OBBs of boundary structures

		// data for rigid particle vis
		uint32_t N_R;                       // number of rigid particles
		std::vector<glm::vec3> posR;        // positions of all rigid particles
		std::vector<float> volR;            // contribution volumes of all rigid particles

		// data for rigid body vis
		uint32_t N_RB;                      // number of rigid bodies
		std::vector<glm::vec3> centerOfMassRB;
		std::vector<glm::mat3> orientationRB;
	};

	struct BenchSnapshot
	{
		// metadata
		uint64_t simStep;
		double simTime;

		// per-operation timings
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> latest{};
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> avgAll{};
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> avgUsed{};
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> min{};
		std::array<uint64_t, Bench::GetIdx(Bench::BenchOp::Count)> max{};
	};

	struct MemSnapshot
	{
		// metadata
		uint64_t simStep;
		double simTime;

		// per-space / per-bucket memory usage
		Mem::MemSpaceRecord latest{};
		Mem::MemSpaceRecord max{};
	};

	struct StatSnapshot
	{
		// metadata
		uint64_t simStep;
		double simTime;

		// per-metric stat data
		std::array<float, Stats::GetIdx(Stats::StatMetric::Count)> latest{};
		std::array<float, Stats::GetIdx(Stats::StatMetric::Count)> avg{};
		std::array<float, Stats::GetIdx(Stats::StatMetric::Count)> min{};
		std::array<float, Stats::GetIdx(Stats::StatMetric::Count)> max{};
	};

} // namespace Sync
