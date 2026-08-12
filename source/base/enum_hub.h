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
		SampledMeshCollection, // 5
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
		case ScenarioType::SampledMeshCollection:
			return "SampledMeshCollection";
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
		AABB_Slabs,   // 0 dynamically sized AABB slabs, split along the global X axis
		AABB_Columns  // 1 dynamically sized AABB columns, split along the global X and Z axes
	};
	constexpr const char* ToCString(const DomainDecompMethod v) noexcept
	{
		switch (v)
		{
		case DomainDecompMethod::AABB_Slabs:
			return "AABB_Slabs";
		case DomainDecompMethod::AABB_Columns:
			return "AABB_Columns";
		}
		return "DomainDecompMethod::<unknown>";
	}

	enum class DistHaloMode : uint8_t
	{
		SingleHalo, // 0 remote particles within 2h
		DoubleHalo  // 1 remote particles within 4h (allows local computation of pressure acceleration for the first halo layer)
	};
	constexpr const char* ToCString(const DistHaloMode v) noexcept
	{
		switch (v)
		{
		case DistHaloMode::SingleHalo:
			return "SingleHalo";
		case DistHaloMode::DoubleHalo:
			return "DoubleHalo";
		}
		return "DistHaloMode::<unknown>";
	}

	enum class LoadBalanceMetric : uint8_t
	{
		Disabled,               // 0
		RankOwnedParticleCount, // 1
		RankBackendStepTime     // 2
		// TODO: add metric for owned + halo particle count?
	};
	constexpr const char* ToCString(const LoadBalanceMetric v) noexcept
	{
		switch (v)
		{
		case LoadBalanceMetric::Disabled:
			return "Disabled";
		case LoadBalanceMetric::RankOwnedParticleCount:
			return "RankOwnedParticleCount";
		case LoadBalanceMetric::RankBackendStepTime:
			return "RankBackendStepTime";
		}
		return "LoadBalanceMetric::<unknown>";
	}

	enum class LoadBalanceTrigger : uint8_t
	{
		Periodic,               // 0
		SustainedImbalance      // 1
		// TODO: separate between periodic with and without threshold checked at each period?
	};
	constexpr const char* ToCString(const LoadBalanceTrigger v) noexcept
	{
		switch (v)
		{
		case LoadBalanceTrigger::Periodic:
			return "Periodic";
		case LoadBalanceTrigger::SustainedImbalance:
			return "SustainedImbalance";
		}
		return "LoadBalanceTrigger::<unknown>";
	}

	enum class LoadBalanceWeightModel : uint8_t
	{
		OwnedParticleCount       // 0 owned fluid particle count
	};
	constexpr const char* ToCString(const LoadBalanceWeightModel v) noexcept
	{
		switch (v)
		{
		case LoadBalanceWeightModel::OwnedParticleCount:
			return "OwnedParticleCount";
		}
		return "LoadBalanceWeightModel::<unknown>";
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
