// dist_snapshots.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "source/simulation/sync/snapshots.h"
#include "source/simulation/sync/state_sim.h"
#include "dist_particle_state.h"



namespace Dist
{
	// MPI tags for distributed export-snapshot gathering
	inline constexpr int DIST_SNAPSHOT_TAG_EXPORT_FLUID_COUNT = 7101;
	inline constexpr int DIST_SNAPSHOT_TAG_EXPORT_FLUID_PAYLOAD = 7102;
	inline constexpr int DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_COUNT = 7103;
	inline constexpr int DIST_SNAPSHOT_TAG_EXPORT_BOUNDARY_PAYLOAD = 7104;

	class MpiContext;

	// fluid particle payload used to gather distributed export snapshots
	struct DistExportFluidParticlePacket
	{
		uint32_t globalIdx = NO_GLOBAL_PARTICLE_IDX;
		uint32_t id = 0;
		uint32_t rank = 0;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
		float velMag = 0.0f;
	};

	// boundary particle payload used to gather distributed export snapshots
	struct DistExportBoundaryParticlePacket
	{
		uint32_t globalIdx = NO_GLOBAL_PARTICLE_IDX;
		uint32_t id = 0;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
		float volB = 0.0f;
	};

	struct DistExportSnapshotBuffer
	{
		std::vector<DistExportFluidParticlePacket> particlesF;
		std::vector<DistExportBoundaryParticlePacket> particlesB;

		void Clear();
	};

	struct DistSnapshotGatherStats
	{
		uint32_t N_gatheredF = 0; // gathered fluid particles count
		uint32_t N_gatheredB = 0; // gathered boundary particles count
		uint32_t N_gatheredR = 0; // gathered rigid particle count
	};

	bool PackOwnedExportSnapshotBuffer(
		DistExportSnapshotBuffer& gatherBuffer,
		const Sync::FluidConstView& particlesF,
		const Sync::BoundaryConstView& particlesB,
		const DistParticleState& particleStateF,
		const DistParticleState& particleStateB,
		const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
		int rank);

	// sorts the gathered packets in place, where the gather storage doubles as the sort buffer
	bool RebuildExportSnapshotFromGatheredParticles(
		Sync::ExportSnapshot& exportSnap,
		DistExportSnapshotBuffer& gatherBuffer,
		uint64_t simStep,
		double simTime);

	void PackReplicatedRigidExportSnapshot(
		Sync::ExportSnapshot& exportSnap,
		const Sync::RigidConstView& particlesR,
		const Sync::RigidBodies& rigidBodies);

	// reusable caller-owned gather storage, holding the locally packed packets on every rank
	// and additionally the payloads received from all other ranks on the root rank
	bool GatherExportSnapshotToRoot(
		Sync::ExportSnapshot& exportSnap,
		DistExportSnapshotBuffer& gatherBuffer,
		const Sync::FluidConstView& particlesF,
		const Sync::BoundaryConstView& particlesB,
		const Sync::RigidConstView& particlesR,
		const Sync::RigidBodies& rigidBodies,
		const DistParticleState& particleStateF,
		const DistParticleState& particleStateB,
		const std::vector<uint32_t>& exportFluidParticleLocalIdxs,
		const MpiContext& mpiContext,
		uint64_t simStep,
		double simTime,
		DistSnapshotGatherStats* stats = nullptr);

} // namespace Dist
