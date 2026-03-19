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
		SmallDroplets		// 3
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
		case ScenarioType::SmallDroplets:
			return "SmallDroplets";
		}
		return "ScenarioType::<unknown>";
	}

	enum class NeighborSearchMethod : uint8_t
	{
		BruteForce,					// 0
		SpatialHashing_Basic,		// 1 (using std::unordered_map)
		SpatialHashing_Compact,		// 2 (using ankerl::unordered_dense)
		SpatialHashing_Explicit,	// 3 (using std::vector<std::vector<...>>)
		CellLinkedList_Basic,		// 4 (derived from paper: "Optimization strategies for CPU and GPU implementations of a smoothed particle hydrodynamics method" (2013))
		CellLinkedList_Compact,		// 5 (derived from paper: "Compressed Neighbor Lists for SPH" (2019))
		BoundVolHier_Linear_LBVH,	// 6 (derived from blog: "Using HIP RT ray tracing library to accelerate fluid simulation" (2023))
		BoundVolHier_Hybrid_HLBVH,	// 7
		BoundVolHier_Binned_SAH_BVH	// 8
	};
	constexpr const char* ToCString(NeighborSearchMethod v) noexcept
	{
		switch (v)
		{
		case NeighborSearchMethod::BruteForce: return
			"BruteForce";
		case NeighborSearchMethod::SpatialHashing_Basic: return
			"SpatialHashing_Basic";
		case NeighborSearchMethod::SpatialHashing_Compact: return
			"SpatialHashing_Compact";
		case NeighborSearchMethod::SpatialHashing_Explicit: return
			"SpatialHashing_Explicit";
		case NeighborSearchMethod::CellLinkedList_Basic: return
			"CellLinkedList_Basic";
		case NeighborSearchMethod::CellLinkedList_Compact: return
			"CellLinkedList_Compact";
		case NeighborSearchMethod::BoundVolHier_Linear_LBVH: return
			"BoundVolHier_Linear_LBVH";
		case NeighborSearchMethod::BoundVolHier_Hybrid_HLBVH: return
			"BoundVolHier_Hybrid_HLBVH";
		case NeighborSearchMethod::BoundVolHier_Binned_SAH_BVH: return
			"BoundVolHier_Binned_SAH_BVH";
		}
		return "NeighborSearchMethod::<unknown>";
	}

	enum class NeighborListStorage : uint8_t
	{
		OnDemand,									// 0
		IndividualVectors,							// 1
		CSRVector_FinalConcat,						// 2
		CSRVector_DoubleQuery,						// 3
		IndividualVectors_AdaptCompress,			// 4
		IndividualVectors_VByteDerivedCompress,		// 5
		CSRVector_DoubleQuery_VByteDerivedCompress	// 6
	};
	constexpr const char* ToCString(NeighborListStorage v) noexcept
	{
		switch (v)
		{
		case NeighborListStorage::OnDemand:
			return "OnDemand";
		case NeighborListStorage::IndividualVectors:
			return "IndividualVectors";
		case NeighborListStorage::CSRVector_FinalConcat:
			return "CSRVector_FinalConcat";
		case NeighborListStorage::CSRVector_DoubleQuery:
			return "CSRVector_DoubleQuery";
		case NeighborListStorage::IndividualVectors_AdaptCompress:
			return "IndividualVectors_AdaptCompress";
		case NeighborListStorage::IndividualVectors_VByteDerivedCompress:
			return "IndividualVectors_VByteDerivedCompress";
		case NeighborListStorage::CSRVector_DoubleQuery_VByteDerivedCompress:
			return "CSRVector_DoubleQuery_VByteDerivedCompress";
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
