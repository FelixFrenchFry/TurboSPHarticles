// enum_hub.h
//
// TODO: comment

#pragma once



namespace Base
{
	enum class BackendVersion
	{
		CPU_01,				// 0
		GPU_01				// 1
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
