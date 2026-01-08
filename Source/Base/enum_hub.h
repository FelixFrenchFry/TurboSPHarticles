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
		case BackendVersion::CPU01: return "CPU01";
		case BackendVersion::GPU01: return "GPU01";
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
		case ScenarioType::BreakingDam: return "BreakingDam";
		case ScenarioType::ConvergingDam: return "ConvergingDam";
		case ScenarioType::BigDroplet: return "BigDroplet";
		case ScenarioType::SmallDroplets: return "SmallDroplets";
		}
		return "ScenarioType::<unknown>";
	}

	enum class NeighborSearchMethod : uint8_t
	{
		BruteForce,				// 0
		SpatialHashing_Basic,	// 1
		// derived from paper:
		// "Optimization strategies for CPU and GPU implementations of a smoothed particle hydrodynamics method" (2013)
		CellLinkedList_Basic,	// 2
		// derived from paper:
		// "Compressed Neighbor Lists for SPH" (2019)
		CellLinkedList_Compact	// 3
	};
	constexpr const char* ToCString(NeighborSearchMethod v) noexcept
	{
		switch (v)
		{
		case NeighborSearchMethod::BruteForce: return "BruteForce";
		case NeighborSearchMethod::SpatialHashing_Basic: return "SpatialHashing_Basic";
		case NeighborSearchMethod::CellLinkedList_Basic: return "CellLinkedList_Basic";
		case NeighborSearchMethod::CellLinkedList_Compact: return "CellLinkedList_Compact";
		}
		return "NeighborSearchMethod::<unknown>";
	}

	enum class NeighborListStorage : uint8_t
	{
		IndividualVectors,							// 0
		SmallBuffers,								// 1
		FixedArrays,								// 2
		CompactVectors,								// 3
		CompactVectors_Compressed_ULEB128,			// 4
		// derived from paper:
		// "Compressed Neighbor Lists for SPH" (2019)
		CompactVectors_Compressed_StreamVByteDelta	// 5
	};
	constexpr const char* ToCString(NeighborListStorage v) noexcept
	{
		switch (v)
		{
		case NeighborListStorage::IndividualVectors: return "IndividualVectors";
		case NeighborListStorage::SmallBuffers: return "SmallBuffers";
		case NeighborListStorage::FixedArrays: return "FixedArrays";
		case NeighborListStorage::CompactVectors: return "CompactVectors";
		case NeighborListStorage::CompactVectors_Compressed_ULEB128: return "CompactVectors_Compressed_ULEB128";
		case NeighborListStorage::CompactVectors_Compressed_StreamVByteDelta: return "CompactVectors_Compressed_StreamVByteDelta";
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
		case SolverTerminationCond::FixedIters: return "FixedIters";
		case SolverTerminationCond::RelDensityError_Raw: return "RelDensityError_Raw";
		case SolverTerminationCond::RelDensityError_Pos: return "RelDensityError_Pos";
		case SolverTerminationCond::RelDensityError_Abs: return "RelDensityError_Abs";
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
