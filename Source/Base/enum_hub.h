// enum_hub.h
//
// TODO: comment

#pragma once



namespace Base
{
	enum class ParticleType : uint8_t
	{
		Fluid,				// 0
		Boundary			// 1
	};

	enum class BackendVersion : uint8_t
	{
		CPU01,				// 0
		CPU02,				// 1

		GPU01				// 2
	};

	enum class ScenarioType : uint8_t
	{
		BreakingDam,		// 0
	};

	enum class NeighborSearchMethod : uint8_t
	{
		BruteForce,

		ExplicitGrid1D,

		ExplicitGrid3D,

		SpatialHashing_Basic,

		// derived from paper:
		// "Optimization strategies for CPU and GPU implementations of a smoothed particle hydrodynamics method" (2013)
		CellLinkedList_Basic,

		// derived from paper:
		// "Compressed Neighbor Lists for SPH" (2019)
		CellLinkedList_Compact
	};

	enum class ColorCodingMode : uint8_t
	{
		None,				// 0
		VelocityMag,		// 1
		Density,			// 2
		Pressure			// 3
	};

} // namespace Base
