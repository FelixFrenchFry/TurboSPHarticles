// enum_hub.h
//
// TODO: comment

#pragma once



namespace Base
{
	// TODO: use 'enum class ... : uint16_t' instead?
	// TODO: put them into separate, local files instead?

	enum ParticleType
	{
		Fluid,				// 0
		Boundary			// 1
	};

	enum BackendVersion
	{
		CPU01,				// 0
		GPU01				// 1
	};

	enum ScenarioType
	{
		BreakingDam,		// 0
	};

	enum NeighborSearchMethod
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

	enum ColorCodingMode
	{
		None,				// 0
		VelocityMag,		// 1
		Density,			// 2
		Pressure			// 3
	};

} // namespace Base
