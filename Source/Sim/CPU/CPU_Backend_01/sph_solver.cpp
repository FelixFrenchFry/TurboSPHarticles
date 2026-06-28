// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>

// internal
#include "Source/Base/helpers.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "cpu_backend_01.h"
#include "sph_solver.h"
#include "Source/Sim/CPU/simd.h"
#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

void SimBackend_CPU01::Step_IISPH(const SimParams& params)
{
	// --- [ temporary pre-computed constants ] ------------------------------------------------------

	// misc hoisted constants
	const float timeStep = params.timeStep;
	const float sizeInvF = params.particleSizeInvF;
	const float sizeSquF = params.particleSizeF * params.particleSizeF;
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	// TODO: use particle size OR kernel supp radius / ratio ?
	const float cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * static_cast<double>(sizeInvF) *
		static_cast<double>(sizeInvF) * static_cast<double>(sizeInvF));
	const float restDensitySquInv = 1.0f / (params.restDensity * params.restDensity);
	const float massF = params.massF;

	// --- [ particle data re-ordering ] -------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Data_Ordering });
	neighborSearch_->DataReOrder(params);
	ApplyLastParticleReOrderToActiveFluidParticleLocalIdxs();
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
	benchmarksTracker_.StopBench({ Bench::BenchOp::Data_Ordering });

	// restricted pointers for all the SoA particle properties
	float* RESTRICT xPosF = xPosF_.data();
	float* RESTRICT yPosF = yPosF_.data();
	float* RESTRICT zPosF = zPosF_.data();
	float* RESTRICT xPosB = xPosB_.data();
	float* RESTRICT yPosB = yPosB_.data();
	float* RESTRICT zPosB = zPosB_.data();
	float* RESTRICT volB = volB_.data();
	float* RESTRICT psiB = psiB_.data();
	float* RESTRICT xVelF = xVelF_.data();
	float* RESTRICT yVelF = yVelF_.data();
	float* RESTRICT zVelF = zVelF_.data();
	float* RESTRICT xVelPredF = xVelPredF_.data();
	float* RESTRICT yVelPredF = yVelPredF_.data();
	float* RESTRICT zVelPredF = zVelPredF_.data();
	float* RESTRICT xAccPressF = xAccPressF_.data();
	float* RESTRICT yAccPressF = yAccPressF_.data();
	float* RESTRICT zAccPressF = zAccPressF_.data();
	float* RESTRICT sourceTermF = sourceTermF_.data();
	float* RESTRICT diagElemF = diagElemF_.data();
	float* RESTRICT pressureF = pressureF_.data();
	float* RESTRICT densityF = densityF_.data();
	I_NeighborListSELL& neighborList = *neighborList_;
	const bool runCountQuery = neighborList.RunCountQuery();
	const bool runFillQuery = neighborList.RunFillQuery();

	// --- [ neighbor search ] -----------------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearch_->Construct(params);
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	if (runCountQuery)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearch_->Query_Count(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborList.Finalize_Count();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	if (runFillQuery)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborList.Finalize_Fill();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ density computations ] ------------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Density_Solving });

	CPU01::Density_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		psiB,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, N_F_,
		densityF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] ------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	CPU01::NonPressureAcceleration_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		volB,
		xVelF, yVelF, zVelF,
		densityF,
		neighborList,
		timeStep, sizeInvF, sizeSquF, kernelRadiusSqu,
		cubicSplineAlpha, params.restDensity, massF,
		params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z, N_F_,
		xVelPredF, yVelPredF, zVelPredF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	// --- [ source-term (density invariance) and diagonal elements ] --------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		psiB,
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborList,
		timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensity, restDensitySquInv,
		massF, params.pressGamma, params.jacobiOmega, N_F_,
		sourceTermF, diagElemF, pressureF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

	// --- [ relaxed Jacobi solver for pressure ] ----------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// rest solver-related variables
	N_solverIterCount_ = 1;
	avgRelDensityErrorF_raw_ = 0.0f;
	avgRelDensityErrorF_pos_ = 0.0f;
	avgRelDensityErrorF_abs_ = 0.0f;

	do
	{
		// compute pressure acceleration from current pressure field
		CPU01::PressureAcceleration_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			psiB,
			pressureF,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
			massF, params.pressGamma, N_F_, *affinityPart_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver_ChunkedSELL(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			psiB,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF,
			neighborList,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensity, massF, params.jacobiOmega, N_F_,
			*affinityPart_, pressureF, avgRelDensityErrorF_raw_,
			avgRelDensityErrorF_pos_, avgRelDensityErrorF_abs_);

		N_solverIterCount_++;
	} while (!TerminationConditionSatisfied(params));

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration ] --------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_Acceleration });

	CPU01::PressureAcceleration_ChunkedSELL(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		psiB,
		pressureF,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensity,
		massF, params.pressGamma, N_F_, *affinityPart_,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ integrate velocity and position ] -------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::VelocityUpdate, Bench::BenchOp::PositionUpdate });

	if (useActiveFluidParticleLocalIdxs_)
	{
		CPU01::Integration_Indexed(
			xVelPredF, yVelPredF, zVelPredF,
			xAccPressF, yAccPressF, zAccPressF,
			activeFluidParticleLocalIdxs_.data(),
			timeStep, static_cast<uint32_t>(activeFluidParticleLocalIdxs_.size()),
			xPosF, yPosF, zPosF,
			xVelF, yVelF, zVelF);
	}
	else
	{
		CPU01::Integration(
			xVelPredF, yVelPredF, zVelPredF,
			xAccPressF, yAccPressF, zAccPressF,
			timeStep, N_F_,
			xPosF, yPosF, zPosF,
			xVelF, yVelF, zVelF);
	}

	benchmarksTracker_.StopBench({ Bench::BenchOp::VelocityUpdate, Bench::BenchOp::PositionUpdate });
}

bool SimBackend_CPU01::TerminationConditionSatisfied(const SimParams& params) const
{
	// check min/max iteration condition for potential early-out
	if (N_solverIterCount_ < params.minSolverIters) { return false; }
	if (N_solverIterCount_ >= params.maxSolverIters) { return true; }

	bool satisfied = false;

	// check actual termination condition
	switch (params.solverTerminationCond)
	{
	case Base::SolverTerminationCond::FixedIters:
		if (N_solverIterCount_ >= params.fixedIters)
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
		if (N_solverIterCount_ >= 3)
		{
			satisfied = true;
		}
		break;
	}

	return satisfied;
}

void CPU01::Density_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT psiB,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float massF,
	const uint32_t N_F,
	// destinations
	float* RESTRICT densityF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;

				// one SELL chunk, covering up to CPU::N_SIMD_WIDTH consecutive fluid particles
				const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);

				const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
				const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
				const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);

				CPU::Float32xSIMD tmp_densityF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_densityB = CPU::ZeroFloat32xSIMD();

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

				const CPU::Float32xSIMD density = massFV * tmp_densityF + tmp_densityB;
				// final chunk, maybe containing fewer than CPU::N_SIMD_WIDTH particles
				CPU::StoreFloat32xSIMDPartial(densityF, begin, end, density);
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU01::NonPressureAcceleration_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT volB,
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT densityF,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float particleSizeSquF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float viscosityF, const float viscosityB,
	const float xGravity, const float yGravity, const float zGravity,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xVelPredF,
	float* RESTRICT yVelPredF,
	float* RESTRICT zVelPredF)
{
	(void)restDensity;
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
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
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);

				const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
				const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
				const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
				const CPU::Float32xSIMD vxi = CPU::GatherFloat32xSIMD(xVelF, idxI);
				const CPU::Float32xSIMD vyi = CPU::GatherFloat32xSIMD(yVelF, idxI);
				const CPU::Float32xSIMD vzi = CPU::GatherFloat32xSIMD(zVelF, idxI);

				CPU::Float32xSIMD tmp_xAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccViscF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_xAccViscB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_yAccViscB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_zAccViscB = CPU::ZeroFloat32xSIMD();

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

						const CPU::Float32xSIMD dot = xsimd::fma(vxi, dx,
							xsimd::fma(vyi, dy, vzi * dz));
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

				const CPU::Float32xSIMD xAccNonPressF =
					tmp_xAccViscF * viscosityFacF + tmp_xAccViscB * viscosityFacB + xGravityV;
				const CPU::Float32xSIMD yAccNonPressF =
					tmp_yAccViscF * viscosityFacF + tmp_yAccViscB * viscosityFacB + yGravityV;
				const CPU::Float32xSIMD zAccNonPressF =
					tmp_zAccViscF * viscosityFacF + tmp_zAccViscB * viscosityFacB + zGravityV;

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

void CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT psiB,
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT densityF,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float restDensitySquInv,
	const float massF, const float pressGamma, const float jacobiOmega,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sourceTermF,
	float* RESTRICT diagElemF,
	float* RESTRICT pressureF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
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
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
				const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
				const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
				const CPU::Float32xSIMD vpxi = CPU::GatherFloat32xSIMD(xVelPredF, idxI);
				const CPU::Float32xSIMD vpyi = CPU::GatherFloat32xSIMD(yVelPredF, idxI);
				const CPU::Float32xSIMD vpzi = CPU::GatherFloat32xSIMD(zVelPredF, idxI);

				CPU::Float32xSIMD tmp_SourceTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD tmp_SourceTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD xCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD yCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD zCTermF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD gradSquSumF = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD xCTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD yCTermB = CPU::ZeroFloat32xSIMD();
				CPU::Float32xSIMD zCTermB = CPU::ZeroFloat32xSIMD();

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

						const CPU::Float32xSIMD sourceTerm =
							xsimd::fma(vpxi, xkg, xsimd::fma(vpyi, ykg, vpzi * zkg));
						const CPU::Float32xSIMD psiBj = CPU::GatherFloat32xSIMD(psiB, idxJ);
						const CPU::Float32xSIMD psiBjMasked =
							CPU::ZeroMaskedFloat32xSIMD(psiBj, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermB = xsimd::fma(psiBjMasked, sourceTerm, tmp_SourceTermB);
						xCTermB = xsimd::fma(psiBjMasked, xkg, xCTermB);
						yCTermB = xsimd::fma(psiBjMasked, ykg, yCTermB);
						zCTermB = xsimd::fma(psiBjMasked, zkg, zCTermB);
					}
				}

				tmp_SourceTermF *= massFV;

				const CPU::Float32xSIMD fin_sourceTermF =
					restDensityV - CPU::GatherFloat32xSIMD(densityF, idxI)
					- timeStepV * (tmp_SourceTermF + tmp_SourceTermB);

				const CPU::Float32xSIMD xCTerm = xCTermF * cTermFacF + xCTermB * cTermFacB;
				const CPU::Float32xSIMD yCTerm = yCTermF * cTermFacF + yCTermB * cTermFacB;
				const CPU::Float32xSIMD zCTerm = zCTermF * cTermFacF + zCTermB * cTermFacB;

				// part A of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemA = massFV *
					xsimd::fma(xCTerm, xCTermF,
						xsimd::fma(yCTerm, yCTermF, zCTerm * zCTermF));
				// part B of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemB = massFV * diagBFac * gradSquSumF;
				// part C of diagonal element
				const CPU::Float32xSIMD tmp_DiagElemC =
					xsimd::fma(xCTerm, xCTermB,
						xsimd::fma(yCTerm, yCTermB, zCTerm * zCTermB));

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

void CPU01::PressureAcceleration_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT psiB,
	const float* RESTRICT pressureF,
	const I_NeighborListSELL& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float pressGamma,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT xAccPressF,
	float* RESTRICT yAccPressF,
	float* RESTRICT zAccPressF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const float densiInvSqu = 1.0f / (restDensity * restDensity);
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD densiInvSquV = CPU::SetFloat32xSIMD(densiInvSqu);
	const CPU::Float32xSIMD accCoeffB_outer = CPU::SetFloat32xSIMD(2.0f * pressGamma * densiInvSqu);
	const CPU::Float32xSIMD zero = CPU::ZeroFloat32xSIMD();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
				const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
				const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
				const CPU::Float32xSIMD presi = CPU::GatherFloat32xSIMD(pressureF, idxI);
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
						const CPU::Float32xSIMD accCoeffFMasked =
							CPU::ZeroMaskedFloat32xSIMD(accCoeffF, mask);

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
						const CPU::Float32xSIMD accCoeffBMasked =
							CPU::ZeroMaskedFloat32xSIMD(accCoeffB, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffBMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffBMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffBMasked, zkg, tmp_zAccPressF);
					}
				}

				// final chunk may contain fewer than CPU::N_SIMD_WIDTH active particles
				CPU::StoreFloat32xSIMDPartial(xAccPressF, begin, end, zero - tmp_xAccPressF);
				CPU::StoreFloat32xSIMDPartial(yAccPressF, begin, end, zero - tmp_yAccPressF);
				CPU::StoreFloat32xSIMDPartial(zAccPressF, begin, end, zero - tmp_zAccPressF);
			}
		},
		affPart
	);
}

void CPU01::PressureRelaxedJacobiSolver_ChunkedSELL(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT psiB,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT sourceTermF,
	const float* RESTRICT diagElemF,
	const I_NeighborListSELL& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float massF,
	const float jacobiOmega,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT pressureF,
	float& avgRelDensityErrorF_raw,
	float& avgRelDensityErrorF_pos,
	float& avgRelDensityErrorF_abs)
{
	struct DensityErrorAccum
	{
		double sumRaw = 0.0;
		double sumPos = 0.0;
		double sumAbs = 0.0;
		// has neighbors and is considered in the solver update
		// TODO: use only "valid" particles or all for averaging?
		uint64_t NF_valid = 0;

		DensityErrorAccum() = default;

		void join(const DensityErrorAccum& rhs)
		{
			sumRaw += rhs.sumRaw;
			sumPos += rhs.sumPos;
			sumAbs += rhs.sumAbs;
			NF_valid += rhs.NF_valid;
		}
	};

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const CPU::Float32xSIMD timeStepSquV = CPU::SetFloat32xSIMD(timeStep * timeStep);
	const CPU::Float32xSIMD restDensityV = CPU::SetFloat32xSIMD(restDensity);
	const CPU::Float32xSIMD massFV = CPU::SetFloat32xSIMD(massF);
	const CPU::Float32xSIMD jacobiOmegaV = CPU::SetFloat32xSIMD(jacobiOmega);
	const CPU::Float32xSIMD zero = CPU::ZeroFloat32xSIMD();
	const CPU::Float32xSIMD eps = CPU::SetFloat32xSIMD(ZERO_EQU_LIM_F);

	DensityErrorAccum acc = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, numChunks), DensityErrorAccum{},
		[&](const tbb::blocked_range<uint32_t>& r, DensityErrorAccum local)
		{
			alignas(CPU::N_SIMD_ALIGNMENT) float relErrTmp[CPU::N_SIMD_WIDTH]{};

			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELLNeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELLNeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32xSIMD idxI = CPU::ParticleIdx32xSIMD(begin, end);

				// one SELL chunk covers up to CPU::N_SIMD_WIDTH consecutive i-particles
				const CPU::Float32xSIMD xi = CPU::GatherFloat32xSIMD(xPosF, idxI);
				const CPU::Float32xSIMD yi = CPU::GatherFloat32xSIMD(yPosF, idxI);
				const CPU::Float32xSIMD zi = CPU::GatherFloat32xSIMD(zPosF, idxI);
				const CPU::Float32xSIMD axi = CPU::GatherFloat32xSIMD(xAccPressF, idxI);
				const CPU::Float32xSIMD ayi = CPU::GatherFloat32xSIMD(yAccPressF, idxI);
				const CPU::Float32xSIMD azi = CPU::GatherFloat32xSIMD(zAccPressF, idxI);

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

				const CPU::Float32xSIMD sysVecElemF =
					timeStepSquV * (tmp_SysVecElemF + tmp_SysVecElemB);
				const CPU::Float32xSIMD sourceTerm = CPU::GatherFloat32xSIMD(sourceTermF, idxI);
				const CPU::Float32xSIMD diagElem = CPU::GatherFloat32xSIMD(diagElemF, idxI);
				const CPU::Float32xSIMD solverResidual = sourceTerm - sysVecElemF;
				const CPU::Float32xSIMD predDensityError = sysVecElemF - sourceTerm;

				const CPU::Mask32xSIMD validDiag = CPU::AbsBitwiseFloat32xSIMD(diagElem) > eps;
				const CPU::Float32xSIMD pressure = CPU::GatherFloat32xSIMD(pressureF, idxI);
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

					const double relErrLane = relErrTmp[lane];
					const double relPosErrLane = std::max(0.0, relErrLane);
					const double relAbsErrLane = std::fabs(relErrLane);

					local.sumRaw += relErrLane;
					local.sumPos += relPosErrLane;
					local.sumAbs += relAbsErrLane;
					local.NF_valid++;
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

	// finalize raw, pos, and abs density deviation averages
	if (N_F > 0)
	{
		avgRelDensityErrorF_raw = static_cast<float>(acc.sumRaw)
								/ static_cast<float>(N_F);

		avgRelDensityErrorF_pos = static_cast<float>(acc.sumPos)
								/ static_cast<float>(N_F);

		avgRelDensityErrorF_abs = static_cast<float>(acc.sumAbs)
								/ static_cast<float>(N_F);
	}
}

void CPU01::Integration(
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

void CPU01::Integration_Indexed(
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const uint32_t* RESTRICT fluidParticleLocalIdxs,
	const float timeStep,
	const uint32_t N_activeF,
	// destinations
	float* RESTRICT xPosF,
	float* RESTRICT yPosF,
	float* RESTRICT zPosF,
	float* RESTRICT xVelF,
	float* RESTRICT yVelF,
	float* RESTRICT zVelF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_activeF),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t ii = r.begin(); ii != r.end(); ii++)
			{
				// accessing only fluid particles in the provided index list
				const uint32_t i = fluidParticleLocalIdxs[ii];

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
