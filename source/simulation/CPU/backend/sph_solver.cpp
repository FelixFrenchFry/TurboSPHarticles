// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

// internal
#include "source/simulation/core/solver_constants.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "cpu_backend.h"
#include "sph_solver.h"
#include "source/simulation/CPU/simd.h"
#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



using namespace Sim;

namespace
{
	void KernelGradient_CubicSpline_Scalar(
		const float dx, const float dy, const float dz,
		const float particleSizeInv,
		const float kernelRadiusSqu,
		const float cubicSplineAlpha,
		float& xkg, float& ykg, float& zkg)
	{
		const float distSqu = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
		if (distSqu >= kernelRadiusSqu || !(distSqu > CPU::KERNEL_GRADIENT_MIN_DIST_SQU_F))
		{
			xkg = 0.0f;
			ykg = 0.0f;
			zkg = 0.0f;
			return;
		}

		const float dist = std::sqrt(distSqu);
		const float distNorm = dist * particleSizeInv;
		float commonFac = 0.0f;
		if (distNorm < 1.0f)
		{
			// factor q * (9q - 12) and cancel q / dist to avoid cancellation near zero
			commonFac = cubicSplineAlpha * particleSizeInv * particleSizeInv
				* std::fmaf(9.0f, distNorm, -12.0f);
		}
		else
		{
			const float t2 = std::max(2.0f - distNorm, 0.0f);
			commonFac = cubicSplineAlpha
				* (-3.0f * t2 * t2)
				* (particleSizeInv / dist);
		}

		xkg = commonFac * dx;
		ykg = commonFac * dy;
		zkg = commonFac * dz;
	}

} // namespace anonymous

void SimBackend_CPU::Step_IISPH(const SimParams& params)
{
	// distributed execution path
	if (distributedFluidSolverSync_)
	{
		Step_IISPH_Distributed(params);
		return;
	}

	// --- [ temporary pre-computed constants ] ------------------------------------------------------

	// misc hoisted constants
	const SolverConstants solverConstants = BuildSolverConstants(params.particleSizeF, params.restDensity);
	const float timeStep = params.timeStep;
	const float sizeInvF = solverConstants.sizeInvF;
	const float sizeSquF = solverConstants.sizeSquF;
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const float cubicSplineAlpha = solverConstants.cubicSplineAlpha;
	const float restDensitySquInv = solverConstants.restDensitySquInv;
	const float massF = params.massF;

	// --- [ particle data re-ordering ] -------------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
	neighborSearch_->DataReOrder_SolverStep(params);
	ApplyLastParticleReOrderToDistributedFluidParticleLocalIdxs();
	affinityParts_ = std::make_unique<StepAffinityPartitioners>(); // reset mapping
	benchmarksTracker_.StopBench(Bench::BenchOp::ParticleData_Reordering);

	// restricted pointers for all the SoA particle properties
	float* RESTRICT xPosF = xPosF_.data();
	float* RESTRICT yPosF = yPosF_.data();
	float* RESTRICT zPosF = zPosF_.data();
	float* RESTRICT xPosB = xPosB_.data();
	float* RESTRICT yPosB = yPosB_.data();
	float* RESTRICT zPosB = zPosB_.data();
	float* RESTRICT xPosR = xPosR_.data();
	float* RESTRICT yPosR = yPosR_.data();
	float* RESTRICT zPosR = zPosR_.data();
	float* RESTRICT xVelF = xVelF_.data();
	float* RESTRICT yVelF = yVelF_.data();
	float* RESTRICT zVelF = zVelF_.data();
	float* RESTRICT xVelR = xVelR_.data();
	float* RESTRICT yVelR = yVelR_.data();
	float* RESTRICT zVelR = zVelR_.data();
	float* RESTRICT xVelPredF = xVelPredF_.data();
	float* RESTRICT yVelPredF = yVelPredF_.data();
	float* RESTRICT zVelPredF = zVelPredF_.data();
	float* RESTRICT xAccPressF = xAccPressF_.data();
	float* RESTRICT yAccPressF = yAccPressF_.data();
	float* RESTRICT zAccPressF = zAccPressF_.data();
	float* RESTRICT xForceR = xForceR_.data();
	float* RESTRICT yForceR = yForceR_.data();
	float* RESTRICT zForceR = zForceR_.data();
	float* RESTRICT sourceTermF = sourceTermF_.data();
	float* RESTRICT diagElemF = diagElemF_.data();
	float* RESTRICT pressureF = pressureF_.data();
	float* RESTRICT densityF = densityF_.data();
	float* RESTRICT volB = volB_.data();
	float* RESTRICT volR = volR_.data();
	float* RESTRICT psiB = psiB_.data();
	float* RESTRICT psiR = psiR_.data();
	I_NeighborListSELL& neighborList = *neighborList_;
	const bool runCountQuery = neighborList.RunCountQuery();
	const bool runFillQuery = neighborList.RunFillQuery();

	// --- [ neighbor search ] -----------------------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborSearch_->Construct(params);
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	if (runCountQuery)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Count(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborList.Finalize_Count();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	if (runFillQuery)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborList.Finalize_Fill();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	// --- [ density computations ] ------------------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Density_Computation);
	CPU::Density_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xPosR, yPosR, zPosR,
		psiB, psiR,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, N_F_,
		densityF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Density_Computation);

	// --- [ non-pressure accelerations ] ------------------------------------------------------------

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		ClearRigidForcesAndAddGravity(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::NonPressure_Acceleration);
	CPU::NonPressureAcceleration_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xPosR, yPosR, zPosR,
		xVelF, yVelF, zVelF,
		xVelR, yVelR, zVelR,
		densityF,
		volB, volR,
		neighborList,
		timeStep, sizeInvF, sizeSquF, kernelRadiusSqu,
		cubicSplineAlpha, massF,
		params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z, N_F_,
		xVelPredF, yVelPredF, zVelPredF);
	benchmarksTracker_.StopBench(Bench::BenchOp::NonPressure_Acceleration);

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		CPU::AccumulateRigidParticleNonPressureForces_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF,
			xVelR, yVelR, zVelR,
			volR,
			neighborList,
			sizeInvF, sizeSquF, kernelRadiusSqu,
			cubicSplineAlpha, massF, params.viscosityB, N_R_,
			xForceR, yForceR, zForceR);
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	// --- [ source-term (density invariance) and diagonal elements ] --------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
	CPU::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xPosR, yPosR, zPosR,
		xVelPredF, yVelPredF, zVelPredF,
		xVelR, yVelR, zVelR,
		densityF,
		psiB, psiR,
		neighborList,
		timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensity, restDensitySquInv,
		massF, params.pressGamma, params.jacobiOmega, N_F_,
		sourceTermF, diagElemF, pressureF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_SourceAndDiagonal);

	// --- [ relaxed Jacobi solver for pressure ] ----------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);

	// reset solver-related variables
	N_solverIters_ = 1;
	avgRelDensityErrorF_raw_ = 0.0f;
	avgRelDensityErrorF_pos_ = 0.0f;
	avgRelDensityErrorF_abs_ = 0.0f;

	do // Jacobi iteration loop
	{
		PressureSolveErrorSums errorSums{};

		// compute pressure acceleration from current pressure field
		CPU::PressureAcceleration_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			pressureF,
			psiB, psiR,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
			massF, params.pressGamma, N_F_, affinityParts_->full,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU::PressureRelaxedJacobiSolver_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF,
			psiB, psiR,
			neighborList,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensity, massF, params.jacobiOmega,
			N_F_, affinityParts_->full, pressureF, nullptr, errorSums);

		if (N_F_ > 0)
		{
			avgRelDensityErrorF_raw_ = static_cast<float>(errorSums.raw)
				/ static_cast<float>(N_F_);
			avgRelDensityErrorF_pos_ = static_cast<float>(errorSums.pos)
				/ static_cast<float>(N_F_);
			avgRelDensityErrorF_abs_ = static_cast<float>(errorSums.abs)
				/ static_cast<float>(N_F_);
		}

		N_solverIters_++;
	} while (!TerminationConditionSatisfied(params));

	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);

	// --- [ finalize pressure acceleration ] --------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_Acceleration);
	CPU::PressureAcceleration_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xPosR, yPosR, zPosR,
		pressureF,
		psiB, psiR,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
		massF, params.pressGamma, N_F_, affinityParts_->full,
		xAccPressF, yAccPressF, zAccPressF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_Acceleration);

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		CPU::AccumulateRigidParticlePressureForces_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosR, yPosR, zPosR,
			pressureF,
			psiR,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
			massF, params.pressGamma, N_R_,
			xForceR, yForceR, zForceR);

		// reduce rigid particle forces into body force and torque
		ReduceRigidParticleForces();
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	// --- [ integrate velocity and position ] -------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Particle_Integration);
	CPU::Integration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccPressF, yAccPressF, zAccPressF,
		timeStep, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Particle_Integration);

	if (N_RB_ > 0 && integrateRigidBodies_)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_StateUpdate);
		IntegrateRigidBodies(timeStep);
		UpdateRigidParticles();
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_StateUpdate);
	}
}

void SimBackend_CPU::Step_IISPH_Distributed(const SimParams& params)
{
	I_DistributedFluidSolverSync& sync = *distributedFluidSolverSync_;
	const bool computeFirstHaloLayerPressureAcc =
		params.distHaloMode == Base::DistHaloMode::DoubleHalo;

	// --- [ temporary pre-computed constants ] ------------------------------------------------------

	// misc hoisted constants
	const SolverConstants solverConstants = BuildSolverConstants(params.particleSizeF, params.restDensity);
	const float timeStep = params.timeStep;
	const float sizeInvF = solverConstants.sizeInvF;
	const float sizeSquF = solverConstants.sizeSquF;
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const float cubicSplineAlpha = solverConstants.cubicSplineAlpha;
	const float restDensitySquInv = solverConstants.restDensitySquInv;
	const float massF = params.massF;

	// --- [ particle data re-ordering ] -------------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
	neighborSearch_->DataReOrder_SolverStep(params);
	// update owned indices and the synchronization plan for the reordered storage
	ApplyLastParticleReOrderToDistributedFluidParticleLocalIdxs();
	affinityParts_ = std::make_unique<StepAffinityPartitioners>(); // reset mapping
	benchmarksTracker_.StopBench(Bench::BenchOp::ParticleData_Reordering);

	// restricted pointers for all the SoA particle properties
	float* RESTRICT xPosF = xPosF_.data();
	float* RESTRICT yPosF = yPosF_.data();
	float* RESTRICT zPosF = zPosF_.data();
	float* RESTRICT xPosB = xPosB_.data();
	float* RESTRICT yPosB = yPosB_.data();
	float* RESTRICT zPosB = zPosB_.data();
	float* RESTRICT xPosR = xPosR_.data();
	float* RESTRICT yPosR = yPosR_.data();
	float* RESTRICT zPosR = zPosR_.data();
	float* RESTRICT xVelF = xVelF_.data();
	float* RESTRICT yVelF = yVelF_.data();
	float* RESTRICT zVelF = zVelF_.data();
	float* RESTRICT xVelR = xVelR_.data();
	float* RESTRICT yVelR = yVelR_.data();
	float* RESTRICT zVelR = zVelR_.data();
	float* RESTRICT xVelPredF = xVelPredF_.data();
	float* RESTRICT yVelPredF = yVelPredF_.data();
	float* RESTRICT zVelPredF = zVelPredF_.data();
	float* RESTRICT xAccPressF = xAccPressF_.data();
	float* RESTRICT yAccPressF = yAccPressF_.data();
	float* RESTRICT zAccPressF = zAccPressF_.data();
	float* RESTRICT xForceR = xForceR_.data();
	float* RESTRICT yForceR = yForceR_.data();
	float* RESTRICT zForceR = zForceR_.data();
	float* RESTRICT sourceTermF = sourceTermF_.data();
	float* RESTRICT diagElemF = diagElemF_.data();
	float* RESTRICT pressureF = pressureF_.data();
	float* RESTRICT densityF = densityF_.data();
	float* RESTRICT volB = volB_.data();
	float* RESTRICT volR = volR_.data();
	float* RESTRICT psiB = psiB_.data();
	float* RESTRICT psiR = psiR_.data();
	I_NeighborListSELL& neighborList = *neighborList_;
	const bool runCountQuery = neighborList.RunCountQuery();
	const bool runFillQuery = neighborList.RunFillQuery();

	// --- [ neighbor search ] -----------------------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborSearch_->Construct(params);
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	if (runCountQuery)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Count(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborList.Finalize_Count();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	if (runFillQuery)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborList.Finalize_Fill();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	// --- [ density computations ] ------------------------------------------------------------------
	// exported outer values are complete before Begin; inner work cannot require halo results,
	// and every consumer of received values runs only after End

	benchmarksTracker_.StartBench(Bench::BenchOp::Density_Computation);
	const auto computeDensity = [&](const std::span<const uint32_t> chunkIdxs)
	{
		CPU::Density_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			psiB, psiR,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, N_ownedF_,
			densityF, chunkIdxs, true);
	};
	computeDensity(outerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::Density_Computation);
	sync.BeginSynchronizeFluidScalarField(
		std::span<float>(densityF_.data(), N_F_),
		Base::DistHaloMode::SingleHalo,
		Sim::FluidSolverSyncPhase::Density, 0);
	benchmarksTracker_.StartBench(Bench::BenchOp::Density_Computation);
	computeDensity(innerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::Density_Computation);
	sync.EndSynchronizeFluidFields();

	// --- [ non-pressure accelerations ] ------------------------------------------------------------

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		ClearRigidForcesAndAddGravity(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::NonPressure_Acceleration);
	const auto computeNonPressureAcceleration =
		[&](const std::span<const uint32_t> chunkIdxs)
	{
		CPU::NonPressureAcceleration_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF,
			xVelR, yVelR, zVelR,
			densityF,
			volB, volR,
			neighborList,
			timeStep, sizeInvF, sizeSquF, kernelRadiusSqu,
			cubicSplineAlpha, massF,
			params.viscosityF, params.viscosityB,
			params.gravity.x, params.gravity.y, params.gravity.z, N_ownedF_,
			xVelPredF, yVelPredF, zVelPredF, chunkIdxs, true);
	};
	computeNonPressureAcceleration(outerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::NonPressure_Acceleration);
	sync.BeginSynchronizeFluidVectorField(
		std::span<float>(xVelPredF_.data(), N_F_),
		std::span<float>(yVelPredF_.data(), N_F_),
		std::span<float>(zVelPredF_.data(), N_F_),
		Base::DistHaloMode::SingleHalo,
		Sim::FluidSolverSyncPhase::PredictedVelocity, 0);
	benchmarksTracker_.StartBench(Bench::BenchOp::NonPressure_Acceleration);
	computeNonPressureAcceleration(innerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::NonPressure_Acceleration);
	sync.EndSynchronizeFluidFields();

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		CPU::AccumulateRigidParticleNonPressureForces_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosR, yPosR, zPosR,
			xVelF, yVelF, zVelF,
			xVelR, yVelR, zVelR,
			volR,
			neighborList,
			sizeInvF, sizeSquF, kernelRadiusSqu,
			cubicSplineAlpha, massF, params.viscosityB, N_R_,
			xForceR, yForceR, zForceR);
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	// --- [ source-term (density invariance) and diagonal elements ] --------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
	const auto computeSourceAndDiagonal = [&](const std::span<const uint32_t> chunkIdxs)
	{
		CPU::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xPosR, yPosR, zPosR,
			xVelPredF, yVelPredF, zVelPredF,
			xVelR, yVelR, zVelR,
			densityF,
			psiB, psiR,
			neighborList,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensity, restDensitySquInv,
			massF, params.pressGamma, params.jacobiOmega, N_ownedF_,
			sourceTermF, diagElemF, pressureF, chunkIdxs, true);
	};
	computeSourceAndDiagonal(outerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
	sync.BeginSynchronizeFluidScalarField(
		std::span<float>(pressureF_.data(), N_F_), params.distHaloMode,
		Sim::FluidSolverSyncPhase::InitialPressure, 0);
	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
	computeSourceAndDiagonal(innerFluidChunkIdxs_);
	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
	sync.EndSynchronizeFluidFields();

	// --- [ relaxed Jacobi solver for pressure ] ----------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);

	// reset solver-related variables
	N_solverIters_ = 1;
	avgRelDensityErrorF_raw_ = 0.0f;
	avgRelDensityErrorF_pos_ = 0.0f;
	avgRelDensityErrorF_abs_ = 0.0f;

	do // Jacobi iteration loop
	{
		const auto computePressureAcceleration = [&](
			const std::span<const uint32_t> chunkIdxs,
			const uint32_t N_fluidCenters,
			const uint8_t* particleFlags,
			tbb::affinity_partitioner& affPart)
		{
			CPU::PressureAcceleration_ChunkedSELL(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR,
				pressureF,
				psiB, psiR,
				neighborList,
				sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
				massF, params.pressGamma, N_fluidCenters, affPart,
				xAccPressF, yAccPressF, zAccPressF,
				chunkIdxs, true, particleFlags);
		};

		if (computeFirstHaloLayerPressureAcc)
		{
			computePressureAcceleration(
				pressureAccChunkIdxs_, N_F_, pressureAccFluidParticleFlags_.data(),
				affinityParts_->pressureAcc);
		}
		else
		{
			computePressureAcceleration(
				outerFluidChunkIdxs_, N_ownedF_, nullptr, affinityParts_->outer);
			benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);
			sync.BeginSynchronizeFluidVectorField(
				std::span<float>(xAccPressF_.data(), N_F_),
				std::span<float>(yAccPressF_.data(), N_F_),
				std::span<float>(zAccPressF_.data(), N_F_),
				Base::DistHaloMode::SingleHalo,
				Sim::FluidSolverSyncPhase::PressureAcceleration, N_solverIters_);
			benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);
			computePressureAcceleration(
				innerFluidChunkIdxs_, N_ownedF_, nullptr, affinityParts_->inner);
			benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);
			sync.EndSynchronizeFluidFields();
			benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);
		}

		const auto updatePressure = [&](const std::span<const uint32_t> chunkIdxs,
			tbb::affinity_partitioner& affPart)
		{
			PressureSolveErrorSums errorSums{};
			CPU::PressureRelaxedJacobiSolver_ChunkedSELL(
				xPosF, yPosF, zPosF,
				xPosB, yPosB, zPosB,
				xPosR, yPosR, zPosR,
				xAccPressF, yAccPressF, zAccPressF,
				sourceTermF, diagElemF,
				psiB, psiR,
				neighborList,
				timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
				params.restDensity, massF, params.jacobiOmega,
				N_ownedF_, affPart, pressureF,
				nullptr, errorSums,
				chunkIdxs, true);
			return errorSums;
		};

		PressureSolveErrorSums localErrorSums = updatePressure(
			outerFluidChunkIdxs_, affinityParts_->outer);
		benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);
		sync.BeginSynchronizeFluidScalarField(
			std::span<float>(pressureF_.data(), N_F_), params.distHaloMode,
			Sim::FluidSolverSyncPhase::Pressure, N_solverIters_);
		benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);
		const PressureSolveErrorSums innerErrorSums = updatePressure(
			innerFluidChunkIdxs_, affinityParts_->inner);
		localErrorSums.raw += innerErrorSums.raw;
		localErrorSums.pos += innerErrorSums.pos;
		localErrorSums.abs += innerErrorSums.abs;
		benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);
		sync.EndSynchronizeFluidFields();
		benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);

		// make every rank use the same global convergence criterion
		const PressureSolveErrorSums errorSums =
			sync.GlobalizePressureSolveErrorSums(localErrorSums, N_solverIters_);
		const uint64_t N_globalF = sync.GetGlobalFluidParticleCount();
		if (N_globalF > 0)
		{
			const double N_globalF_inv = 1.0 / static_cast<double>(N_globalF);
			avgRelDensityErrorF_raw_ = static_cast<float>(errorSums.raw * N_globalF_inv);
			avgRelDensityErrorF_pos_ = static_cast<float>(errorSums.pos * N_globalF_inv);
			avgRelDensityErrorF_abs_ = static_cast<float>(errorSums.abs * N_globalF_inv);
		}
		else
		{
			avgRelDensityErrorF_raw_ = 0.0f;
			avgRelDensityErrorF_pos_ = 0.0f;
			avgRelDensityErrorF_abs_ = 0.0f;
		}

		N_solverIters_++;
	} while (!TerminationConditionSatisfied(params));

	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);

	// --- [ finalize pressure acceleration ] --------------------------------------------------------

	benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_Acceleration);
	CPU::PressureAcceleration_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xPosR, yPosR, zPosR,
		pressureF,
		psiB, psiR,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
		massF, params.pressGamma, N_ownedF_, affinityParts_->full,
		xAccPressF, yAccPressF, zAccPressF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_Acceleration);

	if (N_RB_ > 0)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
		CPU::AccumulateRigidParticlePressureForces_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosR, yPosR, zPosR,
			pressureF,
			psiR,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
			massF, params.pressGamma, N_R_,
			xForceR, yForceR, zForceR);

		// reduce rigid particle forces into body force and torque
		ReduceRigidParticleForces();
		benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
	}

	// --- [ integrate velocity and position ] -------------------------------------------------------

	// integrate owned fluid particles only
	benchmarksTracker_.StartBench(Bench::BenchOp::Particle_Integration);
	CPU::Integration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccPressF, yAccPressF, zAccPressF,
		timeStep, N_ownedF_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF);
	benchmarksTracker_.StopBench(Bench::BenchOp::Particle_Integration);

	// rigid bodies are integrated after their forces are reduced across ranks
}

bool SimBackend_CPU::TerminationConditionSatisfied(const SimParams& params) const
{
	// check min/max iteration condition for potential early-out
	if (N_solverIters_ < params.minSolverIters) { return false; }
	if (N_solverIters_ >= params.maxSolverIters) { return true; }

	bool satisfied = false;

	// check actual termination condition
	switch (params.solverTerminationCond)
	{
	case Base::SolverTerminationCond::FixedIters:
		if (N_solverIters_ >= params.fixedIters)
		{
			satisfied = true;
		}
		break;

	case Base::SolverTerminationCond::RelDensityError_Raw:
		if (avgRelDensityErrorF_raw_ <= params.densityDevTarget)
		{
			satisfied = true;
		}
		break;

	case Base::SolverTerminationCond::RelDensityError_Pos:
		if (avgRelDensityErrorF_pos_ <= params.densityDevTarget)
		{
			satisfied = true;
		}
		break;
	case Base::SolverTerminationCond::RelDensityError_Abs:
		if (avgRelDensityErrorF_abs_ <= params.densityDevTarget)
		{
			satisfied = true;
		}
		break;
	default:
		if (N_solverIters_ >= 3)
		{
			satisfied = true;
		}
		break;
	}

	return satisfied;
}

void CPU::Density_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT psiB,
	const float* RESTRICT psiR,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float massF,
	const uint32_t N_F,
	// destinations
	float* RESTRICT densityF,
	const std::span<const uint32_t> selectedChunkIdxs,
	const bool useSelectedChunks)
{
	const uint32_t N_chunks = useSelectedChunks
		? static_cast<uint32_t>(selectedChunkIdxs.size())
		: neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedChunks
					? selectedChunkIdxs[workIdx] : workIdx;
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFR = neighborList.FRChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = std::min(nbrFF.end, N_F);
				if (begin >= end) { continue; }

				// one SELL chunk, covering up to CPU::N_SIMD_WIDTH consecutive fluid particles
				const CPU::Float32xSIMD xi = CPU::LoadFloat32xSIMDPartial(xPosF, begin, end);
				const CPU::Float32xSIMD yi = CPU::LoadFloat32xSIMDPartial(yPosF, begin, end);
				const CPU::Float32xSIMD zi = CPU::LoadFloat32xSIMDPartial(zPosF, begin, end);

				CPU::Float32xSIMD tmp_densityF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_densityB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_densityR = CPU::ZeroFloat32xSIMD();

				// contributions from fluid neighbors
				{
					const CPU::UInt32xSIMD countFF = CPU::LoadUInt16AsUInt32xSIMD(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL column, storing neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFF);
						const CPU::UInt32xSIMD idxJ = nbrFF.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosF, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosF, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosF, idxJ);

						const CPU::Float32xSIMD kw = CPU::ZeroMaskedFloat32xSIMD(
							CPU::KernelWeight_CubicSpline_SIMD(
								dx, dy, dz, particleSizeInvF,
								kernelRadiusSqu, cubicSplineAlpha), mask);
						// padded lanes gathered valid indices but contribute zero
						tmp_densityF = tmp_densityF + kw;
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32xSIMD countFB = CPU::LoadUInt16AsUInt32xSIMD(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL column, storing neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFB);
						const CPU::UInt32xSIMD idxJ = nbrFB.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosB, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosB, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosB, idxJ);

						const CPU::Float32xSIMD kw = CPU::ZeroMaskedFloat32xSIMD(
							CPU::KernelWeight_CubicSpline_SIMD(
								dx, dy, dz, particleSizeInvF,
								kernelRadiusSqu, cubicSplineAlpha), mask);
						const CPU::Float32xSIMD psiBj = CPU::GatherFloat32xSIMD(psiB, idxJ);
						// padded lanes gathered valid indices but contribute zero
						tmp_densityB = xsimd::fma(psiBj, kw, tmp_densityB);
					}
				}

				// contributions from rigid neighbors
				{
					const CPU::UInt32xSIMD countFR = CPU::LoadUInt16AsUInt32xSIMD(nbrFR.counts + begin);

					for (uint32_t k = 0; k < nbrFR.pitch; k++)
					{
						// one SELL column, storing neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFR);
						const CPU::UInt32xSIMD idxJ = nbrFR.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosR, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosR, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosR, idxJ);

						const CPU::Float32xSIMD kw = CPU::ZeroMaskedFloat32xSIMD(
							CPU::KernelWeight_CubicSpline_SIMD(
								dx, dy, dz, particleSizeInvF,
								kernelRadiusSqu, cubicSplineAlpha), mask);
						const CPU::Float32xSIMD psiRj = CPU::GatherFloat32xSIMD(psiR, idxJ);
						// padded lanes gathered valid indices but contribute zero
						tmp_densityR = xsimd::fma(psiRj, kw, tmp_densityR);
					}
				}

				const CPU::Float32xSIMD density = massFV * tmp_densityF + tmp_densityB + tmp_densityR;
				// final chunk, maybe containing fewer than CPU::N_SIMD_WIDTH particles
				CPU::StoreFloat32xSIMDPartial(densityF, begin, end, density);
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU::NonPressureAcceleration_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT xVelR,
	const float* RESTRICT yVelR,
	const float* RESTRICT zVelR,
	const float* RESTRICT densityF,
	const float* RESTRICT volB,
	const float* RESTRICT volR,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float particleSizeSquF, const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	const float massF, const float viscosityF, const float viscosityB,
	const float xGravity, const float yGravity, const float zGravity,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xVelPredF,
	float* RESTRICT yVelPredF,
	float* RESTRICT zVelPredF,
	const std::span<const uint32_t> selectedChunkIdxs,
	const bool useSelectedChunks)
{
	const uint32_t N_chunks = useSelectedChunks
		? static_cast<uint32_t>(selectedChunkIdxs.size())
		: neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD eps = CPU::SetFloat32xSIMD(ZERO_EQU_LIM_F);
	const CPU::Float32xSIMD particleSizeSquEps = CPU::SetFloat32xSIMD(0.01f * particleSizeSquF);
	const CPU::Float32xSIMD viscosityFacF = CPU::SetFloat32xSIMD(10.0f * viscosityF);
	const CPU::Float32xSIMD viscosityFacB = CPU::SetFloat32xSIMD(10.0f * viscosityB);
	const CPU::Float32xSIMD timeStepV = CPU::SetFloat32xSIMD(timeStep);
	const CPU::Float32xSIMD xGravityV = CPU::SetFloat32xSIMD(xGravity);
	const CPU::Float32xSIMD yGravityV = CPU::SetFloat32xSIMD(yGravity);
	const CPU::Float32xSIMD zGravityV = CPU::SetFloat32xSIMD(zGravity);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedChunks
					? selectedChunkIdxs[workIdx] : workIdx;
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFR = neighborList.FRChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = std::min(nbrFF.end, N_F);
				if (begin >= end) { continue; }

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::LoadFloat32xSIMDPartial(xPosF, begin, end);
				const CPU::Float32xSIMD yi = CPU::LoadFloat32xSIMDPartial(yPosF, begin, end);
				const CPU::Float32xSIMD zi = CPU::LoadFloat32xSIMDPartial(zPosF, begin, end);
				const CPU::Float32xSIMD vxi = CPU::LoadFloat32xSIMDPartial(xVelF, begin, end);
				const CPU::Float32xSIMD vyi = CPU::LoadFloat32xSIMDPartial(yVelF, begin, end);
				const CPU::Float32xSIMD vzi = CPU::LoadFloat32xSIMDPartial(zVelF, begin, end);

				CPU::Float32xSIMD tmp_xAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_xAccViscB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccViscB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccViscB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_xAccViscR = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccViscR = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccViscR = CPU::ZeroFloat32xSIMD();

				// contributions from fluid neighbors
				{
					const CPU::UInt32xSIMD countFF = CPU::LoadUInt16AsUInt32xSIMD(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFF);
						const CPU::UInt32xSIMD idxJ = nbrFF.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosF, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosF, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosF, idxJ);
						const CPU::Float32xSIMD dvx = vxi - CPU::GatherFloat32xSIMD(xVelF, idxJ);
						const CPU::Float32xSIMD dvy = vyi - CPU::GatherFloat32xSIMD(yVelF, idxJ);
						const CPU::Float32xSIMD dvz = vzi - CPU::GatherFloat32xSIMD(zVelF, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dot = xsimd::fma(dvx, dx,
							xsimd::fma(dvy, dy, dvz * dz));
						const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
							xsimd::fma(dy, dy, dz * dz));
						const CPU::Float32xSIMD densityJ =
							xsimd::max(CPU::GatherFloat32xSIMD(densityF, idxJ), eps);
						const CPU::Float32xSIMD accCoeff =
							(massFV * dot) / (densityJ * (distSqu + particleSizeSquEps));
						const CPU::Float32xSIMD accCoeffMasked =
							CPU::ZeroMaskedFloat32xSIMD(accCoeff, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccViscF = xsimd::fma(accCoeffMasked, xkg, tmp_xAccViscF);
						tmp_yAccViscF = xsimd::fma(accCoeffMasked, ykg, tmp_yAccViscF);
						tmp_zAccViscF = xsimd::fma(accCoeffMasked, zkg, tmp_zAccViscF);
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32xSIMD countFB = CPU::LoadUInt16AsUInt32xSIMD(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFB);
						const CPU::UInt32xSIMD idxJ = nbrFB.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosB, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosB, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosB, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dvx = vxi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD dvy = vyi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD dvz = vzi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD dot = xsimd::fma(dvx, dx,
							xsimd::fma(dvy, dy, dvz * dz));
						const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
							xsimd::fma(dy, dy, dz * dz));
						const CPU::Float32xSIMD volBj = CPU::GatherFloat32xSIMD(volB, idxJ);
						const CPU::Float32xSIMD accCoeff =
							(volBj * dot) / (distSqu + particleSizeSquEps);
						const CPU::Float32xSIMD accCoeffMasked =
							CPU::ZeroMaskedFloat32xSIMD(accCoeff, mask);

						tmp_xAccViscB = xsimd::fma(accCoeffMasked, xkg, tmp_xAccViscB);
						tmp_yAccViscB = xsimd::fma(accCoeffMasked, ykg, tmp_yAccViscB);
						tmp_zAccViscB = xsimd::fma(accCoeffMasked, zkg, tmp_zAccViscB);
					}
				}

				// contributions from rigid neighbors
				{
					const CPU::UInt32xSIMD countFR = CPU::LoadUInt16AsUInt32xSIMD(nbrFR.counts + begin);

					for (uint32_t k = 0; k < nbrFR.pitch; k++)
					{
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFR);
						const CPU::UInt32xSIMD idxJ = nbrFR.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosR, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosR, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosR, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dvx = vxi - CPU::GatherFloat32xSIMD(xVelR, idxJ);
						const CPU::Float32xSIMD dvy = vyi - CPU::GatherFloat32xSIMD(yVelR, idxJ);
						const CPU::Float32xSIMD dvz = vzi - CPU::GatherFloat32xSIMD(zVelR, idxJ);
						const CPU::Float32xSIMD dot = xsimd::fma(dvx, dx,
							xsimd::fma(dvy, dy, dvz * dz));
						const CPU::Float32xSIMD distSqu = xsimd::fma(dx, dx,
							xsimd::fma(dy, dy, dz * dz));
						const CPU::Float32xSIMD volRj = CPU::GatherFloat32xSIMD(volR, idxJ);
						const CPU::Float32xSIMD accCoeff =
							(volRj * dot) / (distSqu + particleSizeSquEps);
						const CPU::Float32xSIMD accCoeffMasked =
							CPU::ZeroMaskedFloat32xSIMD(accCoeff, mask);

						tmp_xAccViscR = xsimd::fma(accCoeffMasked, xkg, tmp_xAccViscR);
						tmp_yAccViscR = xsimd::fma(accCoeffMasked, ykg, tmp_yAccViscR);
						tmp_zAccViscR = xsimd::fma(accCoeffMasked, zkg, tmp_zAccViscR);
					}
				}

				const CPU::Float32xSIMD xAccNonPressF =
					tmp_xAccViscF * viscosityFacF + (tmp_xAccViscB + tmp_xAccViscR) * viscosityFacB + xGravityV;
				const CPU::Float32xSIMD yAccNonPressF =
					tmp_yAccViscF * viscosityFacF + (tmp_yAccViscB + tmp_yAccViscR) * viscosityFacB + yGravityV;
				const CPU::Float32xSIMD zAccNonPressF =
					tmp_zAccViscF * viscosityFacF + (tmp_zAccViscB + tmp_zAccViscR) * viscosityFacB + zGravityV;

				// final chunk may contain fewer than CPU::N_SIMD_WIDTH active particles
				CPU::StoreFloat32xSIMDPartial(
					xVelPredF, begin, end, xsimd::fma(xAccNonPressF, timeStepV, vxi));
				CPU::StoreFloat32xSIMDPartial(
					yVelPredF, begin, end, xsimd::fma(yAccNonPressF, timeStepV, vyi));
				CPU::StoreFloat32xSIMDPartial(
					zVelPredF, begin, end, xsimd::fma(zAccNonPressF, timeStepV, vzi));
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU::AccumulateRigidParticleNonPressureForces_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT xVelR,
	const float* RESTRICT yVelR,
	const float* RESTRICT zVelR,
	const float* RESTRICT volR,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float particleSizeSquF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float massF, const float viscosityB,
	const uint32_t N_R,
	// destinations
	float* RESTRICT xForceR,
	float* RESTRICT yForceR,
	float* RESTRICT zForceR)
{
	const float particleSizeSquEps = 0.01f * particleSizeSquF;
	const float viscosityFacB = 10.0f * viscosityB;

	// accumulate one force per rigid particle without atomics
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float xForce = 0.0f;
				float yForce = 0.0f;
				float zForce = 0.0f;
				const NeighborSpan nbrRF = neighborList.RF(i);
				for (uint32_t k = 0; k < nbrRF.N; k++)
				{
					const uint32_t j = nbrRF.data[k];

					const float dx = xPosF[j] - xPosR[i];
					const float dy = yPosF[j] - yPosR[i];
					const float dz = zPosF[j] - zPosR[i];
					const float dvx = xVelF[j] - xVelR[i];
					const float dvy = yVelF[j] - yVelR[i];
					const float dvz = zVelF[j] - zVelR[i];

					float xkg, ykg, zkg;
					KernelGradient_CubicSpline_Scalar(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					const float dot = std::fmaf(dvx, dx, std::fmaf(dvy, dy, dvz * dz));
					const float distSqu = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
					const float accCoeff = (volR[i] * dot) / (distSqu + particleSizeSquEps);
					const float forceCoeff = -massF * viscosityFacB * accCoeff;

					xForce += forceCoeff * xkg;
					yForce += forceCoeff * ykg;
					zForce += forceCoeff * zkg;
				}
				xForceR[i] = xForce;
				yForceR[i] = yForce;
				zForceR[i] = zForce;
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT xVelR,
	const float* RESTRICT yVelR,
	const float* RESTRICT zVelR,
	const float* RESTRICT densityF,
	const float* RESTRICT psiB,
	const float* RESTRICT psiR,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float restDensitySquInv,
	const float massF, const float pressGamma, const float jacobiOmega,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sourceTermF,
	float* RESTRICT diagElemF,
	float* RESTRICT pressureF,
	const std::span<const uint32_t> selectedChunkIdxs,
	const bool useSelectedChunks)
{
	const uint32_t N_chunks = useSelectedChunks
		? static_cast<uint32_t>(selectedChunkIdxs.size())
		: neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD timeStepV = CPU::SetFloat32xSIMD(timeStep);
	const CPU::Float32xSIMD timeStepSquV = CPU::SetFloat32xSIMD(timeStep * timeStep);
	const CPU::Float32xSIMD restDensityV = CPU::SetFloat32xSIMD(restDensity);
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD cTermFacF = CPU::SetFloat32xSIMD(-massF * restDensitySquInv);
	const CPU::Float32xSIMD cTermFacB = CPU::SetFloat32xSIMD(-2.0f * pressGamma * restDensitySquInv);
	const CPU::Float32xSIMD diagBFac = CPU::SetFloat32xSIMD(-massF * restDensitySquInv);
	const CPU::Float32xSIMD jacobiOmegaV = CPU::SetFloat32xSIMD(jacobiOmega);
	const CPU::Float32xSIMD zero = CPU::ZeroFloat32xSIMD();
	const CPU::Float32xSIMD eps = CPU::SetFloat32xSIMD(ZERO_EQU_LIM_F);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedChunks
					? selectedChunkIdxs[workIdx] : workIdx;
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFR = neighborList.FRChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = std::min(nbrFF.end, N_F);
				if (begin >= end) { continue; }

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::LoadFloat32xSIMDPartial(xPosF, begin, end);
				const CPU::Float32xSIMD yi = CPU::LoadFloat32xSIMDPartial(yPosF, begin, end);
				const CPU::Float32xSIMD zi = CPU::LoadFloat32xSIMDPartial(zPosF, begin, end);
				const CPU::Float32xSIMD vpxi = CPU::LoadFloat32xSIMDPartial(xVelPredF, begin, end);
				const CPU::Float32xSIMD vpyi = CPU::LoadFloat32xSIMDPartial(yVelPredF, begin, end);
				const CPU::Float32xSIMD vpzi = CPU::LoadFloat32xSIMDPartial(zVelPredF, begin, end);

				CPU::Float32xSIMD tmp_SourceTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_SourceTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD xCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD yCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD zCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD gradSquSumF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD xCTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD yCTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD zCTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_SourceTermR = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD xCTermR = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD yCTermR = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD zCTermR = CPU::ZeroFloat32xSIMD();

				// contributions from fluid neighbors
				{
					const CPU::UInt32xSIMD countFF = CPU::LoadUInt16AsUInt32xSIMD(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFF);
						const CPU::UInt32xSIMD idxJ = nbrFF.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosF, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosF, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosF, idxJ);
						const CPU::Float32xSIMD dvpx = vpxi - CPU::GatherFloat32xSIMD(xVelPredF, idxJ);
						const CPU::Float32xSIMD dvpy = vpyi - CPU::GatherFloat32xSIMD(yVelPredF, idxJ);
						const CPU::Float32xSIMD dvpz = vpzi - CPU::GatherFloat32xSIMD(zVelPredF, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD sourceTerm =
							xsimd::fma(dvpx, xkg, xsimd::fma(dvpy, ykg, dvpz * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermF += CPU::ZeroMaskedFloat32xSIMD(sourceTerm, mask);
						xCTermF += CPU::ZeroMaskedFloat32xSIMD(xkg, mask);
						yCTermF += CPU::ZeroMaskedFloat32xSIMD(ykg, mask);
						zCTermF += CPU::ZeroMaskedFloat32xSIMD(zkg, mask);

						const CPU::Float32xSIMD gradSqu =
							xsimd::fma(xkg, xkg, xsimd::fma(ykg, ykg, zkg * zkg));
						gradSquSumF += CPU::ZeroMaskedFloat32xSIMD(gradSqu, mask);
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32xSIMD countFB = CPU::LoadUInt16AsUInt32xSIMD(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFB);
						const CPU::UInt32xSIMD idxJ = nbrFB.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosB, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosB, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosB, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dvpx = vpxi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD dvpy = vpyi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD dvpz = vpzi - CPU::ZeroFloat32xSIMD();
						const CPU::Float32xSIMD sourceTerm =
							xsimd::fma(dvpx, xkg, xsimd::fma(dvpy, ykg, dvpz * zkg));
						const CPU::Float32xSIMD psiBj = CPU::GatherFloat32xSIMD(psiB, idxJ);
						const CPU::Float32xSIMD psiBjMasked = CPU::ZeroMaskedFloat32xSIMD(psiBj, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermB = xsimd::fma(psiBjMasked, sourceTerm, tmp_SourceTermB);
						xCTermB = xsimd::fma(psiBjMasked, xkg, xCTermB);
						yCTermB = xsimd::fma(psiBjMasked, ykg, yCTermB);
						zCTermB = xsimd::fma(psiBjMasked, zkg, zCTermB);
					}
				}

				// contributions from rigid neighbors
				{
					const CPU::UInt32xSIMD countFR = CPU::LoadUInt16AsUInt32xSIMD(nbrFR.counts + begin);

					for (uint32_t k = 0; k < nbrFR.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFR);
						const CPU::UInt32xSIMD idxJ = nbrFR.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosR, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosR, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosR, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dvpx = vpxi - CPU::GatherFloat32xSIMD(xVelR, idxJ);
						const CPU::Float32xSIMD dvpy = vpyi - CPU::GatherFloat32xSIMD(yVelR, idxJ);
						const CPU::Float32xSIMD dvpz = vpzi - CPU::GatherFloat32xSIMD(zVelR, idxJ);
						const CPU::Float32xSIMD sourceTerm =
							xsimd::fma(dvpx, xkg, xsimd::fma(dvpy, ykg, dvpz * zkg));
						const CPU::Float32xSIMD psiRj = CPU::GatherFloat32xSIMD(psiR, idxJ);
						const CPU::Float32xSIMD psiRjMasked = CPU::ZeroMaskedFloat32xSIMD(psiRj, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermR = xsimd::fma(psiRjMasked, sourceTerm, tmp_SourceTermR);
						xCTermR = xsimd::fma(psiRjMasked, xkg, xCTermR);
						yCTermR = xsimd::fma(psiRjMasked, ykg, yCTermR);
						zCTermR = xsimd::fma(psiRjMasked, zkg, zCTermR);
					}
				}

				tmp_SourceTermF *= massFV;

				const CPU::Float32xSIMD fin_sourceTermF =
					restDensityV - CPU::LoadFloat32xSIMDPartial(densityF, begin, end)
					- timeStepV * (tmp_SourceTermF + tmp_SourceTermB + tmp_SourceTermR);

				const CPU::Float32xSIMD xCTerm = xCTermF * cTermFacF + (xCTermB + xCTermR) * cTermFacB;
				const CPU::Float32xSIMD yCTerm = yCTermF * cTermFacF + (yCTermB + yCTermR) * cTermFacB;
				const CPU::Float32xSIMD zCTerm = zCTermF * cTermFacF + (zCTermB + zCTermR) * cTermFacB;

				// part A of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemA = massFV *
					xsimd::fma(xCTerm, xCTermF,
						xsimd::fma(yCTerm, yCTermF, zCTerm * zCTermF));
				// part B of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemB = massFV * diagBFac * gradSquSumF;
				// part C of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemC =
					xsimd::fma(xCTerm, xCTermB + xCTermR,
						xsimd::fma(yCTerm, yCTermB + yCTermR, zCTerm * (zCTermB + zCTermR)));

				const CPU::Float32xSIMD fin_diagElemF =
					timeStepSquV * (tmp_DiagElemA + tmp_DiagElemB + tmp_DiagElemC);
				const CPU::Mask32xSIMD validDiag = CPU::AbsFloat32xSIMD(fin_diagElemF) > eps;
				const CPU::Float32xSIMD pressure = xsimd::max(zero,
					CPU::ZeroMaskedFloat32xSIMD(
						jacobiOmegaV * (fin_sourceTermF / fin_diagElemF), validDiag));

				// final chunk may contain fewer than CPU::N_SIMD_WIDTH active particles
				CPU::StoreFloat32xSIMDPartial(sourceTermF, begin, end, fin_sourceTermF);
				CPU::StoreFloat32xSIMDPartial(diagElemF, begin, end, fin_diagElemF);
				CPU::StoreFloat32xSIMDPartial(pressureF, begin, end, pressure);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: create separate optimized versions of this kernel, instead of having flags affect behavior?
void CPU::PressureAcceleration_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT pressureF,
	const float* RESTRICT psiB,
	const float* RESTRICT psiR,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float pressGamma,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT xAccPressF,
	float* RESTRICT yAccPressF,
	float* RESTRICT zAccPressF,
	const std::span<const uint32_t> selectedChunkIdxs,
	const bool useSelectedChunks,
	const uint8_t* RESTRICT activeParticleFlags)
{
	const uint32_t N_chunks = useSelectedChunks
		? static_cast<uint32_t>(selectedChunkIdxs.size())
		: (N_F + CPU::N_SIMD_WIDTH - 1) / CPU::N_SIMD_WIDTH;
	const float densiInvSqu = 1.0f / (restDensity * restDensity);
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD densiInvSquV = CPU::SetFloat32xSIMD(densiInvSqu);
	const CPU::Float32xSIMD accCoeffB_outer = CPU::SetFloat32xSIMD(2.0f * pressGamma * densiInvSqu);
	const CPU::Float32xSIMD zero = CPU::ZeroFloat32xSIMD();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_chunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedChunks
					? selectedChunkIdxs[workIdx]
					: workIdx;
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFR = neighborList.FRChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = std::min(nbrFF.end, N_F);
				if (begin >= end) { continue; }

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::LoadFloat32xSIMDPartial(xPosF, begin, end);
				const CPU::Float32xSIMD yi = CPU::LoadFloat32xSIMDPartial(yPosF, begin, end);
				const CPU::Float32xSIMD zi = CPU::LoadFloat32xSIMDPartial(zPosF, begin, end);
				const CPU::Float32xSIMD presi = CPU::LoadFloat32xSIMDPartial(pressureF, begin, end);
				const CPU::Float32xSIMD accCoeffF_outer = massFV * presi * densiInvSquV;

				CPU::Float32xSIMD tmp_xAccPressF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccPressF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccPressF = CPU::ZeroFloat32xSIMD();

				// contributions from fluid neighbors
				{
					const CPU::UInt32xSIMD countFF = CPU::LoadUInt16AsUInt32xSIMD(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFF);
						const CPU::UInt32xSIMD idxJ = nbrFF.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosF, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosF, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosF, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD presj = CPU::GatherFloat32xSIMD(pressureF, idxJ);
						const CPU::Float32xSIMD accCoeffF =
							accCoeffF_outer + massFV * presj * densiInvSquV;
						const CPU::Float32xSIMD accCoeffFMasked = CPU::ZeroMaskedFloat32xSIMD(accCoeffF, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffFMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffFMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffFMasked, zkg, tmp_zAccPressF);
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32xSIMD countFB = CPU::LoadUInt16AsUInt32xSIMD(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFB);
						const CPU::UInt32xSIMD idxJ = nbrFB.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosB, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosB, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosB, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD psiBj = CPU::GatherFloat32xSIMD(psiB, idxJ);
						const CPU::Float32xSIMD accCoeffB = accCoeffB_outer * psiBj * presi;
						const CPU::Float32xSIMD accCoeffBMasked = CPU::ZeroMaskedFloat32xSIMD(accCoeffB, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffBMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffBMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffBMasked, zkg, tmp_zAccPressF);
					}
				}

				// contributions from rigid neighbors
				{
					const CPU::UInt32xSIMD countFR = CPU::LoadUInt16AsUInt32xSIMD(nbrFR.counts + begin);

					for (uint32_t k = 0; k < nbrFR.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFR);
						const CPU::UInt32xSIMD idxJ = nbrFR.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosR, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosR, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosR, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD psiRj = CPU::GatherFloat32xSIMD(psiR, idxJ);
						const CPU::Float32xSIMD accCoeffB = accCoeffB_outer * psiRj * presi;
						const CPU::Float32xSIMD accCoeffBMasked = CPU::ZeroMaskedFloat32xSIMD(accCoeffB, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffBMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffBMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffBMasked, zkg, tmp_zAccPressF);
					}
				}

				// final chunk may contain fewer than CPU::N_SIMD_WIDTH active particles
				CPU::StoreFloat32xSIMDPartialSelected(
					xAccPressF, begin, end, zero - tmp_xAccPressF, activeParticleFlags);
				CPU::StoreFloat32xSIMDPartialSelected(
					yAccPressF, begin, end, zero - tmp_yAccPressF, activeParticleFlags);
				CPU::StoreFloat32xSIMDPartialSelected(
					zAccPressF, begin, end, zero - tmp_zAccPressF, activeParticleFlags);
			}
		},
		affPart
	);
}

void CPU::AccumulateRigidParticlePressureForces_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT pressureF,
	const float* RESTRICT psiR,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float pressGamma,
	const uint32_t N_R,
	// destinations
	float* RESTRICT xForceR,
	float* RESTRICT yForceR,
	float* RESTRICT zForceR)
{
	const float densiInvSqu = 1.0f / (restDensity * restDensity);
	const float accCoeffB_outer = 2.0f * pressGamma * densiInvSqu;

	// accumulate the opposite pressure force from the RF neighbor relation
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float xForce = 0.0f;
				float yForce = 0.0f;
				float zForce = 0.0f;
				const NeighborSpan nbrRF = neighborList.RF(i);
				for (uint32_t k = 0; k < nbrRF.N; k++)
				{
					const uint32_t j = nbrRF.data[k];

					const float dx = xPosF[j] - xPosR[i];
					const float dy = yPosF[j] - yPosR[i];
					const float dz = zPosF[j] - zPosR[i];

					float xkg, ykg, zkg;
					KernelGradient_CubicSpline_Scalar(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					const float forceCoeff = massF * accCoeffB_outer * psiR[i] * pressureF[j];

					xForce += forceCoeff * xkg;
					yForce += forceCoeff * ykg;
					zForce += forceCoeff * zkg;
				}
				xForceR[i] += xForce;
				yForceR[i] += yForce;
				zForceR[i] += zForce;
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: create separate optimized versions of this kernel, instead of having flags affect behavior?
void CPU::PressureRelaxedJacobiSolver_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xPosR,
	const float* RESTRICT yPosR,
	const float* RESTRICT zPosR,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT sourceTermF,
	const float* RESTRICT diagElemF,
	const float* RESTRICT psiB,
	const float* RESTRICT psiR,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float massF,
	const float jacobiOmega,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT pressureF,
	const uint8_t* RESTRICT errorParticleFlags,
	PressureSolveErrorSums& errorSums,
	const std::span<const uint32_t> selectedChunkIdxs,
	const bool useSelectedChunks)
{
	struct DensityErrorAccum
	{
		double sumRaw = 0.0;
		double sumPos = 0.0;
		double sumAbs = 0.0;
		DensityErrorAccum() = default;

		void join(const DensityErrorAccum& rhs)
		{
			sumRaw += rhs.sumRaw;
			sumPos += rhs.sumPos;
			sumAbs += rhs.sumAbs;
		}
	};

	const uint32_t N_chunks = useSelectedChunks
		? static_cast<uint32_t>(selectedChunkIdxs.size())
		: neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD timeStepSquV = CPU::SetFloat32xSIMD(timeStep * timeStep);
	const CPU::Float32xSIMD restDensityV = CPU::SetFloat32xSIMD(restDensity);
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD jacobiOmegaV = CPU::SetFloat32xSIMD(jacobiOmega);
	const CPU::Float32xSIMD zero = CPU::ZeroFloat32xSIMD();
	const CPU::Float32xSIMD eps = CPU::SetFloat32xSIMD(ZERO_EQU_LIM_F);

	DensityErrorAccum acc = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_chunks), DensityErrorAccum{},
		[&](const tbb::blocked_range<uint32_t>& r, DensityErrorAccum local)
		{
			alignas(CPU::N_SIMD_ALIGNMENT) float relErrTmp[CPU::N_SIMD_WIDTH]{};

			for (uint32_t workIdx = r.begin(); workIdx != r.end(); workIdx++)
			{
				const uint32_t chunk = useSelectedChunks
					? selectedChunkIdxs[workIdx]
					: workIdx;
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFR = neighborList.FRChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = std::min(nbrFF.end, N_F);
				if (begin >= end) { continue; }

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::LoadFloat32xSIMDPartial(xPosF, begin, end);
				const CPU::Float32xSIMD yi = CPU::LoadFloat32xSIMDPartial(yPosF, begin, end);
				const CPU::Float32xSIMD zi = CPU::LoadFloat32xSIMDPartial(zPosF, begin, end);
				const CPU::Float32xSIMD axi = CPU::LoadFloat32xSIMDPartial(xAccPressF, begin, end);
				const CPU::Float32xSIMD ayi = CPU::LoadFloat32xSIMDPartial(yAccPressF, begin, end);
				const CPU::Float32xSIMD azi = CPU::LoadFloat32xSIMDPartial(zAccPressF, begin, end);

				CPU::Float32xSIMD tmp_SysVecElemF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_SysVecElemB = CPU::ZeroFloat32xSIMD();

				// contributions from fluid neighbors
				{
					const CPU::UInt32xSIMD countFF = CPU::LoadUInt16AsUInt32xSIMD(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFF);
						const CPU::UInt32xSIMD idxJ = nbrFF.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosF, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosF, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosF, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32xSIMD dapx = axi - CPU::GatherFloat32xSIMD(xAccPressF, idxJ);
						const CPU::Float32xSIMD dapy = ayi - CPU::GatherFloat32xSIMD(yAccPressF, idxJ);
						const CPU::Float32xSIMD dapz = azi - CPU::GatherFloat32xSIMD(zAccPressF, idxJ);

						const CPU::Float32xSIMD sysVecElem = CPU::FmaFloat32xSIMD(
							dapx, xkg, CPU::FmaFloat32xSIMD(dapy, ykg, dapz * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SysVecElemF += CPU::AndMaskedFloat32xSIMD(sysVecElem, mask);
					}
				}
				tmp_SysVecElemF *= massFV;

				// contributions from boundary neighbors
				{
					const CPU::UInt32xSIMD countFB = CPU::LoadUInt16AsUInt32xSIMD(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFB);
						const CPU::UInt32xSIMD idxJ = nbrFB.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosB, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosB, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosB, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
						const CPU::Float32xSIMD sysVecElem = CPU::FmaFloat32xSIMD(
							axi, xkg, CPU::FmaFloat32xSIMD(ayi, ykg, azi * zkg));
						const CPU::Float32xSIMD psiBj = CPU::GatherFloat32xSIMD(psiB, idxJ);
						const CPU::Float32xSIMD sysVecElemWeighted = psiBj * sysVecElem;

						// padded lanes gathered valid indices but contribute zero
						tmp_SysVecElemB += CPU::AndMaskedFloat32xSIMD(sysVecElemWeighted, mask);
					}
				}

				// contributions from rigid neighbors
				{
					const CPU::UInt32xSIMD countFR = CPU::LoadUInt16AsUInt32xSIMD(nbrFR.counts + begin);

					for (uint32_t k = 0; k < nbrFR.pitch; k++)
					{
						// one SELL column stores neighbor k for all SIMD lanes
						const CPU::Mask32xSIMD mask = CPU::MaskKLessThanCounts32xSIMD(k, countFR);
						const CPU::UInt32xSIMD idxJ = nbrFR.LoadNeighborIdx32xSIMD(k);

						const CPU::Float32xSIMD dx = xi - CPU::GatherFloat32xSIMD(xPosR, idxJ);
						const CPU::Float32xSIMD dy = yi - CPU::GatherFloat32xSIMD(yPosR, idxJ);
						const CPU::Float32xSIMD dz = zi - CPU::GatherFloat32xSIMD(zPosR, idxJ);

						CPU::Float32xSIMD xkg;
						CPU::Float32xSIMD ykg;
						CPU::Float32xSIMD zkg;
						CPU::KernelGradient_CubicSpline_SIMD(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						// TODO: subtract actual rigid pressure acc when adding non-static boundaries?
						const CPU::Float32xSIMD sysVecElem = CPU::FmaFloat32xSIMD(
							axi, xkg, CPU::FmaFloat32xSIMD(ayi, ykg, azi * zkg));
						const CPU::Float32xSIMD psiRj = CPU::GatherFloat32xSIMD(psiR, idxJ);
						const CPU::Float32xSIMD sysVecElemWeighted = psiRj * sysVecElem;

						// padded lanes gathered valid indices but contribute zero
						tmp_SysVecElemB += CPU::AndMaskedFloat32xSIMD(sysVecElemWeighted, mask);
					}
				}

				const CPU::Float32xSIMD sysVecElemF = timeStepSquV * (tmp_SysVecElemF + tmp_SysVecElemB);
				const CPU::Float32xSIMD sourceTerm = CPU::LoadFloat32xSIMDPartial(sourceTermF, begin, end);
				const CPU::Float32xSIMD diagElem = CPU::LoadFloat32xSIMDPartial(diagElemF, begin, end);
				const CPU::Float32xSIMD solverResidual = sourceTerm - sysVecElemF;
				const CPU::Float32xSIMD predDensityError = sysVecElemF - sourceTerm;

				const CPU::Mask32xSIMD validDiag = CPU::AbsBitwiseFloat32xSIMD(diagElem) > eps;
				const CPU::Float32xSIMD pressure = CPU::LoadFloat32xSIMDPartial(pressureF, begin, end);
				const CPU::Float32xSIMD pressureNext = xsimd::max(zero,
					CPU::AndMaskedFloat32xSIMD(
						CPU::FmaFloat32xSIMD(jacobiOmegaV, solverResidual / diagElem, pressure),
						validDiag));

				// final chunk may contain fewer than CPU::N_SIMD_WIDTH active particles
				CPU::StoreFloat32xSIMDPartial(pressureF, begin, end, pressureNext);

				const CPU::Float32xSIMD relErr = predDensityError / restDensityV;
				const uint32_t validDiagMask = CPU::MaskBits32xSIMD(validDiag);
				relErr.store_aligned(relErrTmp);

				for (uint32_t lane = 0; lane < end - begin; lane++)
				{
					if ((validDiagMask & (1u << lane)) == 0u) { continue; }
					if (errorParticleFlags && errorParticleFlags[begin + lane] == 0) { continue; }

					const double relErrLane = relErrTmp[lane];
					const double relPosErrLane = std::max(0.0, relErrLane);
					const double relAbsErrLane = std::fabs(relErrLane);

					local.sumRaw += relErrLane;
					local.sumPos += relPosErrLane;
					local.sumAbs += relAbsErrLane;
				}
			}

			return local;
		},
		[](const DensityErrorAccum& a, const DensityErrorAccum& b) -> DensityErrorAccum
		{
			DensityErrorAccum c = a;
			c.join(b);
			return c;
		},
		affPart
	);

	errorSums.raw = acc.sumRaw;
	errorSums.pos = acc.sumPos;
	errorSums.abs = acc.sumAbs;
}

void CPU::Integration(
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float timeStep,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xPosF,
	float* RESTRICT yPosF,
	float* RESTRICT zPosF,
	float* RESTRICT xVelF,
	float* RESTRICT yVelF,
	float* RESTRICT zVelF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// explicit Euler integration
				// (refining predicted velocity with pressure acc)
				xVelF[i] = std::fmaf(timeStep, xAccPressF[i], xVelPredF[i]);
				yVelF[i] = std::fmaf(timeStep, yAccPressF[i], yVelPredF[i]);
				zVelF[i] = std::fmaf(timeStep, zAccPressF[i], zVelPredF[i]);

				// semi-implicit Euler integration
				// (using updated velocity for position update)
				xPosF[i] = std::fmaf(timeStep, xVelF[i], xPosF[i]);
				yPosF[i] = std::fmaf(timeStep, yVelF[i], yPosF[i]);
				zPosF[i] = std::fmaf(timeStep, zVelF[i], zPosF[i]);
			}
		},
		tbb::auto_partitioner{}
	);
}
