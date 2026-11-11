// rigid_solver.h
#pragma once

// external
#include <cstdint>
#include <vector>

// internal
#include "neighbor_lists.h"

#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



namespace CPU
{
	constexpr float RIGID_CONTACT_VOLUME_FACTOR = 0.7f;

	struct RigidContactReport
	{
		uint32_t N_solverIters = 0;
		float avgDensityError = 0.0f;
		bool converged = true;
	};

	struct RigidBoundaryContactNeighbors
	{
		std::vector<uint32_t> localIdxsB; // boundary particle local index for each contact index
		std::vector<uint64_t> offsetsRB;
		std::vector<uint32_t> indicesRB; // boundary contact indices
		std::vector<uint64_t> offsetsBR;
		std::vector<uint32_t> indicesBR; // rigid particle local indices

		void Clear()
		{
			localIdxsB.clear();
			offsetsRB.clear();
			indicesRB.clear();
			offsetsBR.clear();
			indicesBR.clear();
		}
	};

	// returns whether any rigid particle has a rigid or boundary neighbor
	bool BuildRigidBoundaryNeighbors(
		const Sim::I_NeighborListSELL& neighborList,
		const uint32_t N_R,
		RigidBoundaryContactNeighbors& contactNeighbors);

	void RigidContactGeometry(
		const float* RESTRICT xPosRelR,
		const float* RESTRICT yPosRelR,
		const float* RESTRICT zPosRelR,
		const uint32_t* RESTRICT bodyIdR,
		const float* RESTRICT a00RB,
		const float* RESTRICT a01RB,
		const float* RESTRICT a02RB,
		const float* RESTRICT a10RB,
		const float* RESTRICT a11RB,
		const float* RESTRICT a12RB,
		const float* RESTRICT a20RB,
		const float* RESTRICT a21RB,
		const float* RESTRICT a22RB,
		const float* RESTRICT iBodyInv00RB,
		const float* RESTRICT iBodyInv01RB,
		const float* RESTRICT iBodyInv02RB,
		const float* RESTRICT iBodyInv10RB,
		const float* RESTRICT iBodyInv11RB,
		const float* RESTRICT iBodyInv12RB,
		const float* RESTRICT iBodyInv20RB,
		const float* RESTRICT iBodyInv21RB,
		const float* RESTRICT iBodyInv22RB,
		const uint32_t N_R,
		const uint32_t N_RB,
		// destinations
		float* RESTRICT xPosRelWorldR,
		float* RESTRICT yPosRelWorldR,
		float* RESTRICT zPosRelWorldR,
		float* RESTRICT iWorldInv00RB,
		float* RESTRICT iWorldInv01RB,
		float* RESTRICT iWorldInv02RB,
		float* RESTRICT iWorldInv10RB,
		float* RESTRICT iWorldInv11RB,
		float* RESTRICT iWorldInv12RB,
		float* RESTRICT iWorldInv20RB,
		float* RESTRICT iWorldInv21RB,
		float* RESTRICT iWorldInv22RB);

	void RigidContactDensityAndDiagonal(
		const float* RESTRICT xPosR,
		const float* RESTRICT yPosR,
		const float* RESTRICT zPosR,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const float* RESTRICT volR,
		const float* RESTRICT volB,
		const float* RESTRICT xPosRelWorldR,
		const float* RESTRICT yPosRelWorldR,
		const float* RESTRICT zPosRelWorldR,
		const uint32_t* RESTRICT bodyIdR,
		const float* RESTRICT massInvRB,
		const float* RESTRICT iWorldInv00RB,
		const float* RESTRICT iWorldInv01RB,
		const float* RESTRICT iWorldInv02RB,
		const float* RESTRICT iWorldInv10RB,
		const float* RESTRICT iWorldInv11RB,
		const float* RESTRICT iWorldInv12RB,
		const float* RESTRICT iWorldInv20RB,
		const float* RESTRICT iWorldInv21RB,
		const float* RESTRICT iWorldInv22RB,
		const Sim::I_NeighborListSELL& neighborList,
		const RigidBoundaryContactNeighbors& contactNeighbors,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float timeStep,
		const uint32_t N_R,
		// destinations
		float* RESTRICT densityContactR,
		float* RESTRICT densityContactB,
		float* RESTRICT diagElemContactR,
		float* RESTRICT diagElemContactB,
		float* RESTRICT pressureContactR,
		float* RESTRICT pressureContactB,
		float* RESTRICT sourceTermContactR,
		float* RESTRICT sourceTermContactB,
		float* RESTRICT jacobiOmegaContactR,
		float* RESTRICT jacobiOmegaContactB);

	uint64_t RigidContactRelaxation(
		const uint32_t* RESTRICT particleOffsetsR_ByBody,
		const uint32_t* RESTRICT particleLocalIdxsR_ByBody,
		const uint32_t* RESTRICT bodyIdR,
		const float* RESTRICT massInvRB,
		const uint64_t* RESTRICT offsetsBR,
		const uint32_t* RESTRICT indicesBR,
		const float* RESTRICT diagElemContactB,
		const uint32_t N_RB,
		const uint32_t N_B,
		// destinations
		float* RESTRICT jacobiOmegaContactR,
		float* RESTRICT jacobiOmegaContactB);

	void RigidContactPressureForces(
		const float* RESTRICT xPosR,
		const float* RESTRICT yPosR,
		const float* RESTRICT zPosR,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const float* RESTRICT volR,
		const float* RESTRICT volB,
		const float* RESTRICT densityContactR,
		const float* RESTRICT densityContactB,
		const float* RESTRICT pressureContactR,
		const float* RESTRICT pressureContactB,
		const Sim::I_NeighborListSELL& neighborList,
		const RigidBoundaryContactNeighbors& contactNeighbors,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const uint32_t N_R,
		// destinations
		float* RESTRICT xForceContactR,
		float* RESTRICT yForceContactR,
		float* RESTRICT zForceContactR);

	void RigidContactParticleVelocities(
		const float* RESTRICT xPosRelWorldR,
		const float* RESTRICT yPosRelWorldR,
		const float* RESTRICT zPosRelWorldR,
		const uint32_t* RESTRICT bodyIdR,
		const float* RESTRICT massInvRB,
		const float* RESTRICT iWorldInv00RB,
		const float* RESTRICT iWorldInv01RB,
		const float* RESTRICT iWorldInv02RB,
		const float* RESTRICT iWorldInv10RB,
		const float* RESTRICT iWorldInv11RB,
		const float* RESTRICT iWorldInv12RB,
		const float* RESTRICT iWorldInv20RB,
		const float* RESTRICT iWorldInv21RB,
		const float* RESTRICT iWorldInv22RB,
		const float* RESTRICT xForceContactRB,
		const float* RESTRICT yForceContactRB,
		const float* RESTRICT zForceContactRB,
		const float* RESTRICT xTorqueContactRB,
		const float* RESTRICT yTorqueContactRB,
		const float* RESTRICT zTorqueContactRB,
		const float timeStep,
		const uint32_t N_R,
		// destinations
		float* RESTRICT xVelContactR,
		float* RESTRICT yVelContactR,
		float* RESTRICT zVelContactR);

	double RigidContactDensityError(
		const float* RESTRICT xPosR,
		const float* RESTRICT yPosR,
		const float* RESTRICT zPosR,
		const float* RESTRICT xPosB,
		const float* RESTRICT yPosB,
		const float* RESTRICT zPosB,
		const float* RESTRICT volR,
		const float* RESTRICT volB,
		const float* RESTRICT xVelR,
		const float* RESTRICT yVelR,
		const float* RESTRICT zVelR,
		const float* RESTRICT sourceTermContactR,
		const float* RESTRICT sourceTermContactB,
		const float* RESTRICT pressureContactR,
		const float* RESTRICT pressureContactB,
		const Sim::I_NeighborListSELL& neighborList,
		const RigidBoundaryContactNeighbors& contactNeighbors,
		const float particleSizeInvF,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		const float timeStep,
		const uint32_t N_R,
		// destinations
		float* RESTRICT densityErrorContactR,
		float* RESTRICT densityErrorContactB);

	void RigidContactPressureRelaxedJacobi(
		const float* RESTRICT densityErrorContact,
		const float* RESTRICT diagElemContact,
		const float* RESTRICT jacobiOmegaContact,
		const uint32_t N,
		// destinations
		float* RESTRICT pressureContact);

} // namespace CPU
