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
		BruteForce,				// 0
		SpatialHashing_Basic,	// 1
		CellLinkedList_Basic,	// 2 (derived from paper: "Optimization strategies for CPU and GPU implementations of a smoothed particle hydrodynamics method" (2013))
		CellLinkedList_Compact	// 3 (derived from paper: "Compressed Neighbor Lists for SPH" (2019))
	};
	constexpr const char* ToCString(NeighborSearchMethod v) noexcept
	{
		switch (v)
		{
		case NeighborSearchMethod::BruteForce: return
			"BruteForce";
		case NeighborSearchMethod::SpatialHashing_Basic: return
			"SpatialHashing_Basic";
		case NeighborSearchMethod::CellLinkedList_Basic: return
			"CellLinkedList_Basic";
		case NeighborSearchMethod::CellLinkedList_Compact: return
			"CellLinkedList_Compact";
		}
		return "NeighborSearchMethod::<unknown>";
	}

	enum class NeighborListStorage : uint8_t
	{
		IndividualVectors,							// 0
		CSRVector_FinalConcat,						// 1
		CSRVector_DoubleQuery,						// 2
		IndividualVectors_AdaptCompress,			// 3
		IndividualVectors_VByteDerivedCompress,		// 4
		CSRVector_FinalConcat_ULEB128Compress,		// 5
		CSRVector_FinalConcat_VByteDerivedCompress	// 6 (derived from paper: "Compressed Neighbor Lists for SPH" (2019))
	};
	constexpr const char* ToCString(NeighborListStorage v) noexcept
	{
		switch (v)
		{
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
		case NeighborListStorage::CSRVector_FinalConcat_ULEB128Compress:
			return "CSRVector_FinalConcat_ULEB128Compress";
		case NeighborListStorage::CSRVector_FinalConcat_VByteDerivedCompress:
			return "CSRVector_FinalConcat_VByteDerivedCompress";
		}
		return "NeighborListStorage::<unknown>";
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
