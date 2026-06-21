// halo_exchange.h

#pragma once

// external
#include <cstdint>
#include <string>
#include <vector>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "dist_particle_state.h"
#include "domain_decomp.h"



namespace Dist
{
	class MpiContext;

	// fluid particle payload used for halo exchange
	struct HaloFluidParticlePacket
	{
		int globalIdx = -1;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;

		float xVel = 0.0f;
		float yVel = 0.0f;
		float zVel = 0.0f;

		float xVelPred = 0.0f;
		float yVelPred = 0.0f;
		float zVelPred = 0.0f;

		float xAccPress = 0.0f;
		float yAccPress = 0.0f;
		float zAccPress = 0.0f;

		float sourceTerm = 0.0f;
		float diagElem = 0.0f;
		float pressure = 0.0f;
		float density = 0.0f;
	};

	// boundary particle payload used for halo exchange
	struct HaloBoundaryParticlePacket
	{
		int globalIdx = -1;

		float xPos = 0.0f;
		float yPos = 0.0f;
		float zPos = 0.0f;
	};

	struct HaloExchangeBuffer
	{
		std::vector<HaloFluidParticlePacket> fluidParticles;
		std::vector<HaloBoundaryParticlePacket> boundaryParticles;

		// clears packed fluid and boundary halo packets
		void Clear();
	};

	// reusable destination storage for rebuilding [ owned ][ halo ] rank-local state
	struct HaloRebuildScratch
	{
		Sync::UniversalSimState state;
		DistParticleState fluidParticleState;
		DistParticleState boundaryParticleState;
	};

	// packs this rank's outer fluid/boundary particles that overlap the halo extent of destination rank
	// into a sendBuffer; returns false + msg if local particle metadata is inconsistent
	bool PackHaloExchangeBufferForRank(
		HaloExchangeBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		int dstRank,
		std::string* failMsg = nullptr);

	// exchanges halo packet counts and payloads with all neighbor ranks and appends received packets
	// into a recvBuffer; returns false + msg if packing fails on any rank or received counts are invalid
	bool ExchangeHaloParticlesWithNeighborRanks(
		HaloExchangeBuffer& recvBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext,
		std::string* failMsg = nullptr);

	// rebuilds local fluid/boundary SoA arrays as [ owned ][ received halos ] from halo package
	// in the receive buffer and refreshes matching particle metadata
	bool RebuildLocalStateWithReceivedHalos(
		const HaloExchangeBuffer& recvBuffer,
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		DistParticleState& boundaryParticleState,
		uint32_t globalFluidParticleCount,
		uint32_t globalBoundaryParticleCount,
		HaloRebuildScratch& scratch);

} // namespace Dist
