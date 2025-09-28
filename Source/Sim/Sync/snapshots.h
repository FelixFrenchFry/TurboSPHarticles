// snapshots.h
//
// TODO: comment

#pragma once

// external
#include <glm/vec3.hpp>
#include <vector>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Scene/builder.h"



namespace Sync
{
	// data structure for serializing the simulation state,
	// (including a minimal subset of the data required to reconstruct
	// the full state, enabling lossless save & load)
	// TODO: how to deal with simulation method-specific data?
	// TODO: how to deal with backend-specific data?
	struct SerialSnapshot
	{
		// TODO: implement properly

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

		// masses and sizes
		std::vector<float> mass;
		std::vector<float> size;
	};

	// data structure for exporting parts of the simulation state
	// (including a minimal subset of the data requested to
	// analyze/model the simulation offline)
	// TODO: have one export snapshot per field (e.g. positions, velocities, etc...) and
	// export the components of an export request individually, because they are in separate files anyway?
	struct ExportSnapshot
	{
		// TODO: implement properly
	};

	// data structure for visualizing parts of the simulation state interactively
	// (including a minimal subset of the data required for the
	// interactive visualization)
	struct VisualSnapshot
	{
		// metadata
		int64_t simStep;					// simulation step at this snapshot

		// data for fluid particle vis
		uint32_t N_F;						// number of fluid particles
		std::vector<glm::vec3> posF;		// positions of all fluid particles
		std::vector<float> scalarDataF;		// data according to color coding mode
		Base::ColorCodingMode scalarModeF;	// color coding mode

		// data for boundary particles and boundary structures vis
		uint32_t N_B;						// number of boundary particles
		std::vector<glm::vec3> posB;		// positions of all boundary particles
		std::vector<Scene::OBB> boxesB;		// OBBs of boundary structures
	};

} // namespace Sim
