// dist_particle_state.h

#pragma once

// external
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

// internal
#include "source/eval/memory/metrics_mem.h"
#include "source/simulation/sync/state_sim.h"
#include "domain_decomp.h"



namespace Dist
{
	// sentinel representing no global particle, valid indices are in [0, N_global)
	inline constexpr uint32_t NO_GLOBAL_PARTICLE_IDX = std::numeric_limits<uint32_t>::max();

	// TODO: fine-tune chunk size by trying out 1024, 2048, 4096, 8192, 16384, 32768?
	inline constexpr uint32_t PARTICLE_CLASSIFICATION_CHUNK_SIZE = 16384;
	inline constexpr uint32_t N_PARTICLE_CLASSIFICATION_CLASSES = 3;
	inline constexpr uint32_t PARTICLE_CLASSIFICATION_CLASS_OWNED = 0;
	inline constexpr uint32_t PARTICLE_CLASSIFICATION_CLASS_INNER = 1;
	inline constexpr uint32_t PARTICLE_CLASSIFICATION_CLASS_OUTER = 2;

	inline constexpr uint32_t PARTICLE_BUILD_CHUNK_SIZE = 16384;
	inline constexpr uint32_t N_PARTICLE_BUILD_CLASSES = 2;
	inline constexpr uint32_t PARTICLE_BUILD_CLASS_OWNED = 0;
	inline constexpr uint32_t PARTICLE_BUILD_CLASS_HALO = 1;

	enum class DistParticleClass : uint8_t
	{
		Inner,  // particle owned by this rank, NOT exported to other ranks
		Outer,  // particle owned by this rank, exported to other ranks
		Halo,   // particle NOT owned by this rank, received from neighbor domains within the selected halo radius
		None    // particle is not active in this rank (e.g. outside owned AND halo bounds)
	};

	// reusable scratch for parallelized classification process of owned particles
	struct ParticleClassificationWorkBuffers
	{
		std::vector<uint32_t> classCounts_ByChunkClass; // how many owned/inner/outer particles each chunk contains
		std::vector<uint32_t> classWriteOffsets_ByChunkClass; // where each chunk writes into the final list
		std::vector<uint32_t> classTotals; // total owned/inner/outer counts across all chunks
		std::vector<uint32_t> prefixBlockSumsScratch; // scratch for parallel chunk-prefix construction
	};

	// reusable scratch for parallelized local particle re-ordering
	struct ParticleReOrderWorkBuffers
	{
		std::vector<DistParticleClass> particleClasses;
		std::vector<uint32_t> particleLocalIdx_Owned;
		std::vector<uint32_t> particleLocalIdx_Inner;
		std::vector<uint32_t> particleLocalIdx_Outer;
		std::vector<uint32_t> particleLocalIdx_Halo;
		ParticleClassificationWorkBuffers classification;
	};

	// stores all classification of particles (F/B) based on the domain decomp,
	// backend re-ordering preserves [ owned particles ][ halo particles ] SoA layouts
	struct DistParticleState
	{
		// compact local SoA ranges
		uint32_t N_owned = 0;
		uint32_t N_halo = 0;

		// per-local-particle class flag, indexed by localIdx
		std::vector<DistParticleClass> particleClasses;

		// cached local particle indices for each class
		std::vector<uint32_t> particleLocalIdx_Owned;
		std::vector<uint32_t> particleLocalIdx_Inner;
		std::vector<uint32_t> particleLocalIdx_Outer;
		std::vector<uint32_t> particleLocalIdx_Halo;

		// clears all particle classification metadata and compact range counts
		void Clear();

		// validation helpers
		bool HasMetadataAtLocalIdx(uint32_t localIdx) const;
		bool IsValidLocalParticleLayout(uint32_t N_local, std::string* failMsg = nullptr) const;
		bool IsValidLocalParticleState(
			std::span<const uint32_t> particleIds,
			uint32_t N_global,
			std::string* failMsg = nullptr) const;
	};

	// TODO: move these functions into the DistParticleState class?

	// memory accounting shared by distributed particle storage and metadata owners
	Mem::MemUsage GetParticleMemoryUsage(const Sync::FluidParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::BoundaryParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::RigidParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::UniversalSimState& state);
	Mem::MemUsage GetParticleMemoryUsage(const DistParticleState& particleState);
	Mem::MemUsage GetParticleMemoryUsage(const ParticleReOrderWorkBuffers& workBuffers);

	// rebuilds particle classes and class index lists
	// after the backend changes local particle order
	bool ApplyLocalParticleReOrder(
		DistParticleState& particleState,
		std::span<const uint32_t> oldLocalIdx_FromNewLocalIdx,
		ParticleReOrderWorkBuffers* workBuffers = nullptr);

	// rebuilds fluid/boundary particle class lists by keeping particles owned by this
	// rank and splitting them into inner / outer by selected halo extent
	void ClassifyOwnedFluidParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::FluidParticles& particlesF,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);
	void ClassifyOwnedBoundaryParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::BoundaryParticles& particlesB,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);

	// extracts the fluid/boundary particles needed by this rank and
	// replicates rigid particles and bodies into the rank-local sim state
	void BuildLocalParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& particleStateF,
		DistParticleState& particleStateB,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		int rank, float haloRadius);

} // namespace Dist
