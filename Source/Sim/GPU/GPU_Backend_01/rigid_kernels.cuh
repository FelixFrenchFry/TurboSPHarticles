// rigid_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>



namespace GPU01
{
	void Launch_UpdateRigidBoundaryParticles(
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const uint32_t* rigidBoundaryParticleBeginR,
		const uint32_t* rigidBoundaryParticleCountR,
		const float* xCMR, const float* yCMR, const float* zCMR,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* xOmegaR, const float* yOmegaR, const float* zOmegaR,
		const float* a00R, const float* a01R, const float* a02R,
		const float* a10R, const float* a11R, const float* a12R,
		const float* a20R, const float* a21R, const float* a22R,
		const float* xLocalRigidBoundaryParticle,
		const float* yLocalRigidBoundaryParticle,
		const float* zLocalRigidBoundaryParticle,
		uint32_t N_B, uint32_t N_rigidBodies, uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xPosB, float* yPosB, float* zPosB,
		float* xVelB, float* yVelB, float* zVelB);

	void Launch_ClearRigidForcesAndAddGravity(
		const float* massR,
		const float* massInvR,
		float xGravity, float yGravity, float zGravity,
		uint32_t N_rigidBodies,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR,
		float* xTorqueR, float* yTorqueR, float* zTorqueR);

	void Launch_ClearRigidBoundaryParticleForces(
		uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle);

	void Launch_AccumulateRigidBodyNonPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const float* volB,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelB, const float* yVelB, const float* zVelB,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FB,
		float particleSizeInvF,
		float particleSizeSquF,
		float kernelRadiusSqu,
		float cubicSplineAlpha,
		float massF,
		float viscosityB,
		uint32_t N_F, uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle);

	void Launch_AccumulateRigidBodyPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosB, const float* yPosB, const float* zPosB,
		const uint32_t* bodyIdB,
		const uint32_t* rigidBoundaryParticleIdxB,
		const float* psiB,
		const float* pressureF,
		const uint16_t* nbrCountFB,
		const uint64_t* nbrChunkOffsetsFB,
		const uint32_t* nbrChunkBaseIdxFB,
		const uint8_t* nbrChunkFormatFB,
		const uint8_t* nbrListDelta8FB,
		const uint16_t* nbrListDelta16FB,
		const uint32_t* nbrListAbs32FB,
		float particleSizeInvF,
		float kernelRadiusSqu,
		float cubicSplineAlpha,
		float restDensity,
		float massF,
		float pressGamma,
		uint32_t N_F, uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceRigidBoundaryParticle,
		float* yForceRigidBoundaryParticle,
		float* zForceRigidBoundaryParticle);

	void Launch_ReduceRigidBoundaryParticleForces(
		const uint32_t* rigidBoundaryParticleBeginR,
		const uint32_t* rigidBoundaryParticleCountR,
		const float* xLocalRigidBoundaryParticle,
		const float* yLocalRigidBoundaryParticle,
		const float* zLocalRigidBoundaryParticle,
		const float* a00R, const float* a01R, const float* a02R,
		const float* a10R, const float* a11R, const float* a12R,
		const float* a20R, const float* a21R, const float* a22R,
		const float* xForceRigidBoundaryParticle,
		const float* yForceRigidBoundaryParticle,
		const float* zForceRigidBoundaryParticle,
		uint32_t N_rigidBodies,
		uint32_t N_rigidBoundaryParticles,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR,
		float* xTorqueR, float* yTorqueR, float* zTorqueR);

	void Launch_IntegrateRigidBodies(
		float timeStep,
		uint32_t N_rigidBodies,
		cudaStream_t stream,
		// destinations
		float* xCMR, float* yCMR, float* zCMR,
		float* xVelR, float* yVelR, float* zVelR,
		float* xOmegaR, float* yOmegaR, float* zOmegaR,
		float* xAngMomR, float* yAngMomR, float* zAngMomR,
		const float* xForceR, const float* yForceR, const float* zForceR,
		const float* xTorqueR, const float* yTorqueR, const float* zTorqueR,
		float* a00R, float* a01R, float* a02R,
		float* a10R, float* a11R, float* a12R,
		float* a20R, float* a21R, float* a22R,
		const float* massInvR,
		const float* iBodyInv00R, const float* iBodyInv01R, const float* iBodyInv02R,
		const float* iBodyInv10R, const float* iBodyInv11R, const float* iBodyInv12R,
		const float* iBodyInv20R, const float* iBodyInv21R, const float* iBodyInv22R);

} // namespace GPU01
