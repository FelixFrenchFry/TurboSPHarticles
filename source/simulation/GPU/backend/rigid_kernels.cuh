// rigid_kernels.cuh

#pragma once

// external
#include <cuda_runtime.h>
#include <cstdint>



namespace GPU
{
	// refresh before rigid force launches when the SPH parameters change
	void ReloadRigidConstants(
		float sizeInvF, float kernelRadiusSqu, float cubicSplineAlpha);

	void Launch_UpdateRigidParticles(
		const uint32_t* bodyIdR,
		const float* xRelPosR, const float* yRelPosR, const float* zRelPosR,
		const float* xCMRB, const float* yCMRB, const float* zCMRB,
		const float* xVelRB, const float* yVelRB, const float* zVelRB,
		const float* xOmegaRB, const float* yOmegaRB, const float* zOmegaRB,
		const float* a00RB, const float* a01RB, const float* a02RB,
		const float* a10RB, const float* a11RB, const float* a12RB,
		const float* a20RB, const float* a21RB, const float* a22RB,
		uint32_t N_R, uint32_t N_RB, cudaStream_t stream,
		// destinations
		float* xPosR, float* yPosR, float* zPosR,
		float* xVelR, float* yVelR, float* zVelR);

	void Launch_ClearRigidForcesAndAddGravity(
		const float* massRB,
		const float* massInvRB,
		float xGravity, float yGravity, float zGravity,
		uint32_t N_RB,
		cudaStream_t stream,
		// destinations
		float* xForceRB, float* yForceRB, float* zForceRB,
		float* xTorqueRB, float* yTorqueRB, float* zTorqueRB);

	void Launch_AccumulateRigidParticleNonPressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* xVelF, const float* yVelF, const float* zVelF,
		const float* xVelR, const float* yVelR, const float* zVelR,
		const float* volR,
		const uint16_t* nbrCountRF,
		const uint64_t* nbrOffsetsRF,
		const uint32_t* nbrListRF,
		float particleSizeSquF,
		float massF,
		float viscosityB,
		uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR);

	void Launch_AccumulateRigidParticlePressureForces(
		const float* xPosF, const float* yPosF, const float* zPosF,
		const float* xPosR, const float* yPosR, const float* zPosR,
		const float* pressureF,
		const float* psiR,
		const uint16_t* nbrCountRF,
		const uint64_t* nbrOffsetsRF,
		const uint32_t* nbrListRF,
		float restDensity,
		float massF,
		float pressGamma,
		uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceR, float* yForceR, float* zForceR);

	void Launch_ReduceRigidParticleForces(
		const uint32_t* rigidParticleBeginRB,
		const uint32_t* rigidParticleCountRB,
		const float* xRelPosR,
		const float* yRelPosR,
		const float* zRelPosR,
		const float* xForceR,
		const float* yForceR,
		const float* zForceR,
		const float* a00RB, const float* a01RB, const float* a02RB,
		const float* a10RB, const float* a11RB, const float* a12RB,
		const float* a20RB, const float* a21RB, const float* a22RB,
		uint32_t N_RB,
		uint32_t N_R,
		cudaStream_t stream,
		// destinations
		float* xForceRB, float* yForceRB, float* zForceRB,
		float* xTorqueRB, float* yTorqueRB, float* zTorqueRB);

	void Launch_IntegrateRigidBodies(
		float timeStep,
		uint32_t N_RB,
		cudaStream_t stream,
		// destinations
		float* xCMRB, float* yCMRB, float* zCMRB,
		float* xVelRB, float* yVelRB, float* zVelRB,
		float* xOmegaRB, float* yOmegaRB, float* zOmegaRB,
		float* xAngMomRB, float* yAngMomRB, float* zAngMomRB,
		const float* xForceRB, const float* yForceRB, const float* zForceRB,
		const float* xTorqueRB, const float* yTorqueRB, const float* zTorqueRB,
		float* a00RB, float* a01RB, float* a02RB,
		float* a10RB, float* a11RB, float* a12RB,
		float* a20RB, float* a21RB, float* a22RB,
		const float* massInvRB,
		const float* iBodyInv00RB, const float* iBodyInv01RB, const float* iBodyInv02RB,
		const float* iBodyInv10RB, const float* iBodyInv11RB, const float* iBodyInv12RB,
		const float* iBodyInv20RB, const float* iBodyInv21RB, const float* iBodyInv22RB);

} // namespace GPU
