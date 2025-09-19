// snapshots.h
//
// TODO: comment

#pragma once

// external
#include <glm/vec3.hpp>
#include <vector>

// internal
#include "../../Base/enum_hub.h"



namespace Sync
{
	// data structure for serializing the full simulation state
	// (including all data making up the full state, enabling lossless save & load)
	struct SerialSnapshot
	{
		// TODO: implement properly
	};

	// data structure for exporting parts of the simulation state
	// (including a minimal subset of the data requested to analyze/model the simulation offline)
	struct ExportSnapshot
	{
		// TODO: implement properly
	};

	// data structure for visualizing parts of the simulation state interactively
	// (including a minimal subset of the data required for the interactive visualization)
	struct VisualSnapshot
	{
		// TODO: implement properly

		std::vector<glm::vec3> position;	// positions of all particles
		std::vector<float> scalarData;		// data according to color coding mode
		Base::ColorCodingMode scalarMode;	// color coding mode
		std::uint64_t N_particles;			// number of particles
		std::uint64_t step;					// simulation step at this snapshot
	};

} // namespace Sim
