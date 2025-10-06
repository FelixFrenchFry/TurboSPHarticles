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
		CPU_01,				// 0
		GPU_01				// 1
	};

	enum ScenarioType
	{
		BreakingDam,		// 0
	};

	enum NeighborSearchMethod
	{
		BruteForce,			// 0
		ExplicitGrid1D,		// 1
		ExplicitGrid3D,		// 2
		IndexSortBasic		// 3
	};

	enum ColorCodingMode
	{
		None,				// 0
		VelocityMag,		// 1
		Density,			// 2
		Pressure			// 3
	};

} // namespace Base
