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

	enum class DistParticleClass
	{
		Inner,  // particle owned by this rank, NOT exported to other ranks
		Outer,  // particle owned by this rank, exported to other ranks
		Halo,   // particle NOT owned by this rank, received from neighbor domains within the selected halo radius
		None    // particle is not active in this rank (e.g. outside owned AND halo bounds)
	};

	// reusable scratch for parallelized classification process of owned particles
	struct ParticleClassificationWorkBuffers
	{
		std::vector<uint32_t> classCounts_ByChunkClass;
		std::vector<uint32_t> classWriteOffsets_ByChunkClass;
		std::vector<uint32_t> classTotals;
	};

	// reusable scratch for parallelized local particle re-ordering
	struct ParticleReOrderWorkBuffers
	{
		std::vector<DistParticleClass> particleClasses;
		std::vector<uint32_t> particleLocalIdx_Owned;
		std::vector<uint32_t> particleLocalIdx_Inner;
		std::vector<uint32_t> particleLocalIdx_Outer;
		std::vector<uint32_t> particleLocalIdx_Halo;
		std::vector<uint32_t> particleGlobalIdx_FromLocalIdx;
		ParticleClassificationWorkBuffers classification;
	};

	// stores all classification of particles (F/B) based on the domain decomp,
	// local SoA layout is [ owned particles ][ halo particles ]
	struct DistParticleState
	{
		// compact local SoA ranges (before backend data re-ordering)
		uint32_t N_owned = 0;
		uint32_t N_halo = 0;

		// per-local-particle class flag, indexed by localIdx
		std::vector<DistParticleClass> particleClasses;

		// cached local indices for each class
		std::vector<uint32_t> particleLocalIdx_Owned;
		std::vector<uint32_t> particleLocalIdx_Inner;
		std::vector<uint32_t> particleLocalIdx_Outer;
		std::vector<uint32_t> particleLocalIdx_Halo;

		// index map from the compact rank-local state to the full scenario state
		std::vector<uint32_t> particleGlobalIdx_FromLocalIdx;

		// per-global-particle active flag, set while that particle is held by this rank
		std::vector<uint8_t> activeGlobalIdxFlags;

		// clears all particle classification metadata, compact range counts, and local/global index maps
		void Clear();

		// clears only particle class lists and compact range counts while preserving local/global index maps
		void ClearParticleClasses();

		// checks that all per-local-particle metadata is present at the given index
		bool HasMetadataAtLocalIdx(uint32_t localIdx) const;

		// validates metadata for compact pre-backend-reorder layout [ owned particles ][ halo particles ]
		bool IsValidLocalLayout(uint32_t N_local, std::string* failMsg = nullptr) const;

		// validates local metadata after backend data re-ordering, where owned
		// and halo particles may no longer be stored in compact contiguous ranges
		bool IsValidLocalParticleState(
			uint32_t N_local,
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

	// rebuilds particle classes, class index lists, and local/global maps after
	// the backend changes local particle order
	bool ApplyLocalParticleReOrder(
		DistParticleState& particleState,
		std::span<const uint32_t> oldLocalIdx_FromNewLocalIdx,
		uint32_t N_global,
		ParticleReOrderWorkBuffers* workBuffers = nullptr);

	// rebuilds fluid particle class lists by keeping particles owned by this
	// rank and splitting them into inner / outer by selected halo extent
	void ClassifyOwnedFluidParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::FluidParticles& fluidParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);

	// rebuilds boundary particle class lists by keeping particles owned by this
	// rank and splitting them into inner / outer by selected halo extent
	void ClassifyOwnedBoundaryParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::BoundaryParticles& boundaryParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);

	// extracts the fluid/boundary particles needed by this rank and
	// replicates rigid particles and bodies into the rank-local sim state
	void BuildLocalParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		DistParticleState& boundaryParticleState,
		const Sync::UniversalSimState& globalState,
		const IDomainDecomp& domainDecomp,
		int rank, float haloRadius,
		uint32_t N_globalF, uint32_t N_globalB);

} // namespace Dist
