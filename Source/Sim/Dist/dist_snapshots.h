// dist_snapshots.h

#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "Source/Sim/Sync/snapshots.h"
#include "Source/Sim/Sync/state_sim.h"
#include "dist_particle_state.h"



namespace Dist
{
	class MpiContext;

	// fluid particle payload used to gather distributed export snapshots
	struct DistExportFluidParticlePacket
	{
		int globalIdx = -1;
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
		int globalIdx = -1;
		uint32_t id = 0;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
		float volB = 0.0f;
	};

	struct DistExportSnapshotBuffer
	{
		std::vector<DistExportFluidParticlePacket> fluidParticles;
		std::vector<DistExportBoundaryParticlePacket> boundaryParticles;

		void Clear();
	};

	struct DistSnapshotGatherStats
	{
		uint32_t gatheredFluidParticleCount = 0;
		uint32_t gatheredBoundaryParticleCount = 0;
		uint32_t gatheredRigidParticleCount = 0;
	};

	bool PackOwnedExportSnapshotBuffer(
		DistExportSnapshotBuffer& gatherBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const std::vector<int>& exportFluidParticleLocalIdxs,
		int rank);

	bool RebuildExportSnapshotFromGatheredParticles(
		Sync::ExportSnapshot& exportSnap,
		const DistExportSnapshotBuffer& gatherBuffer,
		uint64_t simStep,
		double simTime);

	void PackReplicatedRigidExportSnapshot(
		Sync::ExportSnapshot& exportSnap,
		const Sync::UniversalSimState& localState);

	bool GatherExportSnapshotToRoot(
		Sync::ExportSnapshot& exportSnap,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const std::vector<int>& exportFluidParticleLocalIdxs,
		const MpiContext& mpiContext,
		uint64_t simStep,
		double simTime,
		DistSnapshotGatherStats* stats = nullptr);

} // namespace Dist
