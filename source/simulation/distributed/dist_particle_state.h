// dist_particle_state.h
#pragma once

// external
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// internal
#include "source/eval/memory/categories.h"
#include "source/simulation/sync/state_sim.h"
#include "domain_decomp.h"



namespace Dist
{
	class MpiContext;

	inline constexpr uint32_t PARTICLE_BUILD_CHUNK_SIZE = 16384;
	inline constexpr uint32_t N_PARTICLE_BUILD_CLASSES = 2;
	inline constexpr uint32_t PARTICLE_BUILD_CLASS_OWNED = 0;
	inline constexpr uint32_t PARTICLE_BUILD_CLASS_HALO = 1;

	// reusable scratch for parallelized classification process of owned particles
	struct ParticleClassificationWorkBuffers
	{
		std::vector<uint32_t> classCounts_ByChunkClass; // how many outer particles each chunk contains
		std::vector<uint32_t> classWriteOffsets_ByChunkClass; // where each chunk writes into the final list
		std::vector<uint32_t> classTotals; // total outer count across all chunks
	};

	// counts of the compact [ owned particles ][ halo particles ] SoA layout of one particle species (F/B/R),
	// which backend re-ordering preserves, so owned particles occupy the local indices [ 0, N_owned )
	struct DistParticleState
	{
		// compact local SoA ranges
		uint32_t N_owned = 0;
		uint32_t N_halo = 0;

		// owned particles that at least one neighbor rank needs as halos
		uint32_t N_outer = 0;

		// validation helpers
		bool IsValidLocalParticleLayout(uint32_t N_local, std::string* failMsg = nullptr) const;
		bool IsValidLocalParticleState(
			std::span<const uint64_t> particleIds,
			uint64_t N_global,
			std::string* failMsg = nullptr) const;
	};

	// validates rigid field sizes and body attachments independently of particle identity
	bool ValidateLocalRigidParticleAttachments(
		const Sync::RigidConstView& particlesR,
		const Sync::RigidBodies& rigidBodies,
		std::string* failMsg = nullptr);

	struct OwnedParticleConflict
	{
		uint64_t id = Sim::INVALID_PARTICLE_ID;
		uint32_t ownerCount = 0;
	};

	bool FindFirstOwnedParticleConflict(
		std::span<const uint64_t> ownedParticleIds,
		uint64_t N_global,
		const MpiContext& mpiContext,
		std::optional<OwnedParticleConflict>& conflict,
		std::string* failMsg = nullptr);

	// memory accounting shared by distributed particle storage owners
	Mem::MemUsage GetParticleMemoryUsage(const Sync::FluidParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::BoundaryParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::RigidParticles& particles);
	Mem::MemUsage GetParticleMemoryUsage(const Sync::UniversalSimState& state);

	// counts the owned F/B/R particles in the compact prefix that are not inner owned particles
	// for the selected halo extent, and collects their ascending local indices if a list is given
	void ClassifyOwnedFluidParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::FluidConstView& particlesF,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		std::vector<uint32_t>* outerLocalIdxs = nullptr,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);
	void ClassifyOwnedBoundaryParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::BoundaryConstView& particlesB,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		std::vector<uint32_t>* outerLocalIdxs = nullptr,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);
	void ClassifyOwnedRigidParticlesIntoInnerOuter(
		DistParticleState& particleState,
		const Sync::RigidConstView& particlesR,
		const Sync::RigidBodies& rigidBodies,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		std::vector<uint32_t>* outerLocalIdxs = nullptr,
		ParticleClassificationWorkBuffers* workBuffers = nullptr);

	// classifies prepared owned particles before applying redistribution
	// (retained indices map the new prefix to old storage, followed by appended owners)
	uint32_t CollectPreparedOuterFluidLocalIdxs(
		uint32_t N_owned,
		std::span<const uint32_t> retainedOldLocalIdxs,
		std::span<const uint8_t> isOuterOwned_ByLocalIdx,
		const Sync::FluidConstView& appendedParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius,
		ParticleClassificationWorkBuffers& workBuffers,
		std::vector<uint32_t>& outerLocalIdxs,
		float extraAllocCapacityPct);
	uint32_t CountPreparedOuterBoundaryParticles(
		uint32_t N_owned,
		std::span<const uint32_t> retainedOldLocalIdxs,
		std::span<const uint8_t> isOuterOwned_ByLocalIdx,
		const Sync::BoundaryConstView& appendedParticles,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius);
	uint32_t CountPreparedOuterRigidParticles(
		uint32_t N_owned,
		std::span<const uint32_t> retainedOldLocalIdxs,
		const Sync::RigidConstView& oldParticles,
		const Sync::RigidConstView& appendedParticles,
		const Sync::RigidBodies& rigidBodies,
		const IDomainDecomp& domainDecomp,
		int rank,
		float haloRadius);

	// extracts F/B/R particles needed by this rank and replicates rigid body state
	// consumes source particle storage, requiring distinct source and destination states
	void BuildLocalParticlesFromGlobalState(
		Sync::UniversalSimState& localState,
		DistParticleState& particleStateF,
		DistParticleState& particleStateB,
		DistParticleState& particleStateR,
		Sync::UniversalSimState&& globalState,
		const IDomainDecomp& domainDecomp,
		int rank);

} // namespace Dist
