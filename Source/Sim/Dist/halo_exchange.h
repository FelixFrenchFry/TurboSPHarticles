// halo_exchange.h

#pragma once

// external
#include <cstdint>
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

		void Clear();
	};

	void PackHaloExchangeBufferForRank(
		HaloExchangeBuffer& sendBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		int dstRank);

	bool ExchangeHaloParticlesWithNeighborRanks(
		HaloExchangeBuffer& recvBuffer,
		const Sync::UniversalSimState& localState,
		const DistParticleState& fluidParticleState,
		const DistParticleState& boundaryParticleState,
		const IDomainDecomp& domainDecomp,
		const MpiContext& mpiContext);

	bool RebuildLocalStateWithReceivedHalos(
		Sync::UniversalSimState& localState,
		DistParticleState& fluidParticleState,
		DistParticleState& boundaryParticleState,
		const HaloExchangeBuffer& recvBuffer,
		uint32_t globalFluidParticleCount,
		uint32_t globalBoundaryParticleCount);

} // namespace Dist
