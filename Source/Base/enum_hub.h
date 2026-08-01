// enum_hub.h

#pragma once

// external
#include <cstdint>



namespace Base
{
	enum class ParticleType : uint8_t
	{
		Fluid,              // 0 fluid particles
		Boundary,           // 1 static boundary particles
		Rigid               // 2 rigid boundary particles
	};

	enum class BackendVersion : uint8_t
	{
		CPU01,              // 0
		GPU01               // 1
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
		BreakingDam,        // 0
		ConvergingDam,      // 1
		BigDroplet,         // 2
		RigidCube,          // 3
		SampledMesh,        // 4
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
		case ScenarioType::RigidCube:
			return "RigidCube";
		case ScenarioType::SampledMesh:
			return "SampledMesh";
		}
		return "ScenarioType::<unknown>";
	}

	enum class NeighborSearchMethod : uint8_t
	{
		BruteForce,                 // 0
		CellLinkedList_Compact,     // 1 ("Compressed neighbour lists for SPH" Band, Gissler, Teschner, 2020)
	};
	constexpr const char* ToCString(const NeighborSearchMethod v) noexcept
	{
		switch (v)
		{
		case NeighborSearchMethod::BruteForce: return
			"BruteForce";
		case NeighborSearchMethod::CellLinkedList_Compact: return
			"CellLinkedList_Compact";
		}
		return "NeighborSearchMethod::<unknown>";
	}

	enum class NeighborListMethod : uint8_t
	{
		ChunkedSELLX_Explicit,          // 0
		ChunkedSELLX_AdaptCompress      // 1
	};
	constexpr const char* ToCString(const NeighborListMethod v) noexcept
	{
		switch (v)
		{
		case NeighborListMethod::ChunkedSELLX_Explicit:
			return "ChunkedSELLX_Explicit";
		case NeighborListMethod::ChunkedSELLX_AdaptCompress:
			return "ChunkedSELLX_AdaptCompress";
		}
		return "NeighborListStorage::<unknown>";
	}
	constexpr const char* ToEffectiveNeighborListStorageCString(
		const BackendVersion backend,
		const NeighborListMethod v) noexcept
	{
		switch (v)
		{
		case NeighborListMethod::ChunkedSELLX_Explicit:
			return (backend == BackendVersion::GPU01)
				? "ChunkedSELL32_Explicit"
			#if defined(TSP_CPU_SIMD_WIDTH) && TSP_CPU_SIMD_WIDTH == 16
				: "ChunkedSELL16_Explicit";
			#else
				: "ChunkedSELL8_Explicit";
			#endif
		case NeighborListMethod::ChunkedSELLX_AdaptCompress:
			return (backend == BackendVersion::GPU01)
				? "ChunkedSELL32_AdaptCompress"
			#if defined(TSP_CPU_SIMD_WIDTH) && TSP_CPU_SIMD_WIDTH == 16
				: "ChunkedSELL16_AdaptCompress";
			#else
				: "ChunkedSELL8_AdaptCompress";
			#endif
		}
		return "ChunkedSELLX::<unknown>";
	}

	enum class DomainDecompMethod : uint8_t
	{
		AABB_Uniform        // 0 uniform AABB slabs, split along the global X axis
	};
	constexpr const char* ToCString(const DomainDecompMethod v) noexcept
	{
		switch (v)
		{
		case DomainDecompMethod::AABB_Uniform:
			return "AABB_Uniform";
		}
		return "DomainDecompMethod::<unknown>";
	}

	enum class SolverTerminationCond : uint8_t
	{
		FixedIters,             // 0
		RelDensityError_Raw,    // 1
		RelDensityError_Pos,    // 2
		RelDensityError_Abs     // 3
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
		None,               // 0
		VelocityMag,        // 1
		Density,            // 2
		Pressure            // 3
	};

	enum class ColorMap : uint8_t
	{
		Turbo,              // 0
		Viridis,            // 1
		Inferno,            // 2
		Fast,               // 3
		Coolwarm,           // 4
		Blues,              // 5
		Aurora,             // 6
		Earth               // 7
	};

} // namespace Base
