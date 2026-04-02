// enum_hub.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>



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
		GPU01				// 1
	};
	constexpr const char* ToCString(BackendVersion v) noexcept
	{
		switch (v)
		{
		case BackendVersion::CPU01:
			return "CPU01";
		case BackendVersion::GPU01:
			return "GPU01";
		}
		return "BackendVersion::<unknown>";
	}

	enum class ScenarioType : uint8_t
	{
		BreakingDam,		// 0
		ConvergingDam,		// 1
		BigDroplet,			// 2
	};
	constexpr const char* ToCString(ScenarioType v) noexcept
	{
		switch (v)
		{
		case ScenarioType::BreakingDam:
			return "BreakingDam";
		case ScenarioType::ConvergingDam:
			return "ConvergingDam";
		case ScenarioType::BigDroplet:
			return "BigDroplet";
		}
		return "ScenarioType::<unknown>";
	}

	enum class NeighborSearchMethod : uint8_t
	{
		BruteForce,					// 0
		CellLinkedList_Compact,		// 1 ("Compressed neighbour lists for SPH" Band, Gissler, Teschner, 2020)
		BoundVolHier_Linear,		// 2 TODO: implement ("Maximizing parallelism in the construction of BVHs, octrees, and k-d trees" Karras, 2012)
	};
	constexpr const char* ToCString(const NeighborSearchMethod v) noexcept
	{
		switch (v)
		{
		case NeighborSearchMethod::BruteForce: return
			"BruteForce";
		case NeighborSearchMethod::CellLinkedList_Compact: return
			"CellLinkedList_Compact";
		case NeighborSearchMethod::BoundVolHier_Linear: return
			"BoundVolHier_Linear";
		}
		return "NeighborSearchMethod::<unknown>";
	}

	enum class NeighborListMethod : uint8_t
	{
		IndividualVectors_Explicit,		// 0
		SingleCSRVector_Explicit,		// 1
		SingleCSRVector_AdaptCompress,	// 2 TODO: implement
		SingleCSRVector_VByteCompress	// 3 ("Compressed neighbour lists for SPH" Band, Gissler, Teschner, 2020)
	};
	constexpr const char* ToCString(const NeighborListMethod v) noexcept
	{
		switch (v)
		{
		case NeighborListMethod::IndividualVectors_Explicit:
			return "IndividualVectors_Explicit";
		case NeighborListMethod::SingleCSRVector_Explicit:
			return "SingleCSRVector_Explicit";
		case NeighborListMethod::SingleCSRVector_AdaptCompress:
			return "SingleCSRVector_AdaptCompress";
		case NeighborListMethod::SingleCSRVector_VByteCompress:
			return "SingleCSRVector_VByteCompress";
		}
		return "NeighborListStorage::<unknown>";
	}

	enum class GridCellIndexing : uint8_t
	{
		Morton,				// 0
		XYZ					// 1
	};
	constexpr const char* ToCString(GridCellIndexing v) noexcept
	{
		switch (v)
		{
		case GridCellIndexing::Morton:
			return "Morton";
		case GridCellIndexing::XYZ:
			return "XYZ";
		}
		return "GridCellIndexing::<unknown>";
	}

	enum class SolverTerminationCond : uint8_t
	{
		FixedIters,				// 0
		RelDensityError_Raw,	// 1
		RelDensityError_Pos,	// 2
		RelDensityError_Abs		// 3
	};
	constexpr const char* ToCString(SolverTerminationCond v) noexcept
	{
		switch (v)
		{
		case SolverTerminationCond::FixedIters:
			return "FixedIters";
		case SolverTerminationCond::RelDensityError_Raw:
			return "RelDensityError_Raw";
		case SolverTerminationCond::RelDensityError_Pos:
			return "RelDensityError_Pos";
		case SolverTerminationCond::RelDensityError_Abs:
			return "RelDensityError_Abs";
		}
		return "SolverTerminationCond::<unknown>";
	}

	enum class ColorCodingMode : uint8_t
	{
		None,				// 0
		VelocityMag,		// 1
		Density,			// 2
		Pressure			// 3
	};

} // namespace Base
