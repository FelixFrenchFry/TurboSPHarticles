// dist_snapshots.h
#pragma once

// external
#include <cstdint>
#include <string>
#include <vector>

// internal
#include "source/simulation/sync/snapshots.h"
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"



namespace Export
{
	class SerializationMgr;
}

namespace Dist
{
	class MpiContext;

	// collective snapshot output and per-rank F/B/R counts for XDMF
	bool SaveDistributedExportSnapshot(
		const Sync::ExportSnapshot& exportSnap,
		const Export::SerializationMgr& serializationMgr,
		const std::string& dir,
		bool includeB,
		const MpiContext& mpiContext,
		std::vector<uint64_t>& rankCounts);

	// compacts the sorted selection to owned particles and packs local output
	bool PackOwnedExportSnapshot(
		Sync::ExportSnapshot& exportSnap,
		const Sync::FluidConstView& particlesF,
		const Sync::BoundaryConstView& particlesB,
		const Sync::RigidConstView& particlesR,
		const Sync::RigidBodies& rigidBodies,
		const DistParticleState& particleStateF,
		const DistParticleState& particleStateB,
		const DistParticleState& particleStateR,
		std::vector<uint32_t>& exportFluidParticleLocalIdxs,
		int rank,
		int N_ranks,
		bool includeB,
		uint64_t simStep,
		double simTime);

} // namespace Dist
