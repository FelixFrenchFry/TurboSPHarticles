// state_format.h
#pragma once

// external
#include <cstdint>



namespace Export
{
	#pragma pack(push, 1)
	struct BinHeaderSim
	{
		char magic[8];
		uint32_t endianTag;
		uint32_t headerBytes;
		uint64_t step;
		double time;
		uint64_t N_F;
		uint64_t N_B;
		uint64_t N_R;
		uint64_t N_RB;
	};
	#pragma pack(pop)

	static_assert(sizeof(BinHeaderSim) == 64);

	// byte offsets of particle and body arrays in the TSPSIM file
	struct SimStateOffsets
	{
		uint64_t fluid = 0;
		uint64_t boundary = 0;
		uint64_t rigid = 0;
		uint64_t rigidBodies = 0;
		uint64_t totalBytes = 0;
	};

	BinHeaderSim MakeSimStateHeader(
		uint64_t step,
		double time,
		uint64_t N_F,
		uint64_t N_B,
		uint64_t N_R,
		uint64_t N_RB);
	SimStateOffsets ComputeSimStateOffsets(const BinHeaderSim& header);

	// fixed on-disk field order, shared by full, selected and collective state I/O
	// multiple states visit corresponding fields together when selecting particles
	template <class Visit, class... States>
	void VisitFluidStateFields(Visit visit, States&... states)
	{
		visit(states.id...);
		visit(states.xPos...);
		visit(states.yPos...);
		visit(states.zPos...);
		visit(states.xVel...);
		visit(states.yVel...);
		visit(states.zVel...);
		visit(states.xVelPred...);
		visit(states.yVelPred...);
		visit(states.zVelPred...);
		visit(states.xAccPress...);
		visit(states.yAccPress...);
		visit(states.zAccPress...);
		visit(states.sourceTerm...);
		visit(states.diagElem...);
		visit(states.pressure...);
		visit(states.density...);
	}

	template <class Visit, class... States>
	void VisitBoundaryStateFields(Visit visit, States&... states)
	{
		visit(states.id...);
		visit(states.xPos...);
		visit(states.yPos...);
		visit(states.zPos...);
		visit(states.volB...);
		visit(states.psiB...);
	}

	template <class Visit, class... States>
	void VisitRigidStateFields(Visit visit, States&... states)
	{
		visit(states.id...);
		visit(states.bodyId...);
		visit(states.xPosRel...);
		visit(states.yPosRel...);
		visit(states.zPosRel...);
		visit(states.volR...);
		visit(states.psiR...);
	}

	template <class Visit, class... States>
	void VisitRigidBodyStateFields(Visit visit, States&... states)
	{
		visit(states.id...);
		visit(states.rigidParticleCount...);
		visit(states.xCM...);
		visit(states.yCM...);
		visit(states.zCM...);
		visit(states.xVel...);
		visit(states.yVel...);
		visit(states.zVel...);
		visit(states.xOmega...);
		visit(states.yOmega...);
		visit(states.zOmega...);
		visit(states.xAngMom...);
		visit(states.yAngMom...);
		visit(states.zAngMom...);
		visit(states.a00...);
		visit(states.a01...);
		visit(states.a02...);
		visit(states.a10...);
		visit(states.a11...);
		visit(states.a12...);
		visit(states.a20...);
		visit(states.a21...);
		visit(states.a22...);
		visit(states.mass...);
		visit(states.massInv...);
		visit(states.iBodyInv00...);
		visit(states.iBodyInv01...);
		visit(states.iBodyInv02...);
		visit(states.iBodyInv10...);
		visit(states.iBodyInv11...);
		visit(states.iBodyInv12...);
		visit(states.iBodyInv20...);
		visit(states.iBodyInv21...);
		visit(states.iBodyInv22...);
	}

} // namespace Export
