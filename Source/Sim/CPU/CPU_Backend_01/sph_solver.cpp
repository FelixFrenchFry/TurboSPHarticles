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
#include "Source/Sim/CPU/simd8.h"
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
	const float restDensitySquInvF = 1.0f / (params.restDensityF * params.restDensityF);
	const float massF = params.massF;
	const float massB = params.massB;

	// --- [ particle data re-ordering ] -------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Data_Ordering });
	neighborSearch_->DataReOrder(params);
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
	benchmarksTracker_.StopBench({ Bench::BenchOp::Data_Ordering });

	// restricted pointers for all the SoA particle properties
	float* RESTRICT xPosF = xPosF_.data();
	float* RESTRICT yPosF = yPosF_.data();
	float* RESTRICT zPosF = zPosF_.data();
	float* RESTRICT xPosB = xPosB_.data();
	float* RESTRICT yPosB = yPosB_.data();
	float* RESTRICT zPosB = zPosB_.data();
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
	I_NeighborListSELL8& neighborList = *neighborList_;
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

	CPU01::Density_ChunkedSELL8(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, massB, N_F_,
		densityF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] ------------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	CPU01::NonPressureAcceleration_ChunkedSELL8(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelF, yVelF, zVelF,
		densityF,
		neighborList,
		timeStep, sizeInvF, sizeSquF, kernelRadiusSqu,
		cubicSplineAlpha, params.restDensityF, params.restDensityB, massF, massB,
		params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z, N_F_,
		xVelPredF, yVelPredF, zVelPredF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	// --- [ source-term (density invariance) and diagonal elements ] --------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL8(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborList,
		timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensityF, restDensitySquInvF,
		massF, massB, params.pressGamma, params.jacobiOmega, N_F_,
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
		CPU01::PressureAcceleration_ChunkedSELL8(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			pressureF,
			neighborList,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
			massF, massB, params.pressGamma, N_F_, *affinityPart_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver_ChunkedSELL8(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF,
			neighborList,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensityF, massF, massB, params.jacobiOmega, N_F_,
			*affinityPart_, pressureF, avgRelDensityErrorF_raw_,
			avgRelDensityErrorF_pos_, avgRelDensityErrorF_abs_);

		N_solverIterCount_++;
	} while (!TerminationConditionSatisfied(params));

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration ] --------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_Acceleration });

	CPU01::PressureAcceleration_ChunkedSELL8(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		pressureF,
		neighborList,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
		massF, massB, params.pressGamma, N_F_, *affinityPart_,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ integrate velocity and position ] -------------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::VelocityUpdate, Bench::BenchOp::PositionUpdate });

	CPU01::Integration(
		xVelPredF, yVelPredF, zVelPredF,
		xAccPressF, yAccPressF, zAccPressF,
		timeStep, N_F_,
		xPosF, yPosF, zPosF,
		xVelF, yVelF, zVelF);

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

void CPU01::Density_ChunkedSELL8(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const I_NeighborListSELL8& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float massF, const float massB,
	const uint32_t N_F,
	// destinations
	float* RESTRICT densityF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const CPU::Float32x8 massFV = CPU::SetFloat32x8(massF);
	const CPU::Float32x8 massBV = CPU::SetFloat32x8(massB);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELL8NeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELL8NeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;

				// one SELL8 chunk, covering (up to) 8 consecutive fluid particles
				const CPU::UInt32x8 idxI = CPU::ParticleIdx32x8(begin, end);

				const CPU::Float32x8 xi = CPU::GatherFloat32x8(xPosF, idxI);
				const CPU::Float32x8 yi = CPU::GatherFloat32x8(yPosF, idxI);
				const CPU::Float32x8 zi = CPU::GatherFloat32x8(zPosF, idxI);

				CPU::Float32x8 tmp_densityF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_densityB = CPU::ZeroFloat32x8();

				// contributions from fluid neighbors
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column, storing neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);

						const CPU::Float32x8 kw = CPU::ZeroMaskedFloat32x8(
							CPU::KernelWeight_CubicSpline_SIMD8(
								dx, dy, dz, particleSizeInvF,
								kernelRadiusSqu, cubicSplineAlpha), mask);
						// padded lanes gathered valid indices but contribute zero
						tmp_densityF = tmp_densityF + kw;
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL8 column, storing neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						const CPU::Float32x8 kw = CPU::ZeroMaskedFloat32x8(
							CPU::KernelWeight_CubicSpline_SIMD8(
								dx, dy, dz, particleSizeInvF,
								kernelRadiusSqu, cubicSplineAlpha), mask);
						// padded lanes gathered valid indices but contribute zero
						tmp_densityB = tmp_densityB + kw;
					}
				}

				const CPU::Float32x8 density = massFV * tmp_densityF + massBV * tmp_densityB;
				// final chunk, maybe containing less than 8 particles
				CPU::StoreFloat32x8Partial(densityF, begin, end, density);
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU01::NonPressureAcceleration_ChunkedSELL8(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xVelF,
	const float* RESTRICT yVelF,
	const float* RESTRICT zVelF,
	const float* RESTRICT densityF,
	const I_NeighborListSELL8& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float particleSizeSquF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensityF,
	const float restDensityB, const float massF, const float massB,
	const float viscosityF, const float viscosityB,
	const float xGravity, const float yGravity, const float zGravity,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xVelPredF,
	float* RESTRICT yVelPredF,
	float* RESTRICT zVelPredF)
{
	(void)restDensityF;
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const CPU::Float32x8 massFV = CPU::SetFloat32x8(massF);
	const CPU::Float32x8 massBOverRestDensityBV = CPU::SetFloat32x8(massB / restDensityB);
	const CPU::Float32x8 eps = CPU::SetFloat32x8(ZERO_EQU_LIM_F);
	const CPU::Float32x8 particleSizeSquEps = CPU::SetFloat32x8(0.01f * particleSizeSquF);
	const CPU::Float32x8 viscosityFacF = CPU::SetFloat32x8(10.0f * viscosityF);
	const CPU::Float32x8 viscosityFacB = CPU::SetFloat32x8(10.0f * viscosityB);
	const CPU::Float32x8 timeStepV = CPU::SetFloat32x8(timeStep);
	const CPU::Float32x8 xGravityV = CPU::SetFloat32x8(xGravity);
	const CPU::Float32x8 yGravityV = CPU::SetFloat32x8(yGravity);
	const CPU::Float32x8 zGravityV = CPU::SetFloat32x8(zGravity);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELL8NeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELL8NeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;

				// one SELL8 chunk covers up to 8 consecutive i-particles
				const CPU::UInt32x8 idxI = CPU::ParticleIdx32x8(begin, end);

				const CPU::Float32x8 xi = CPU::GatherFloat32x8(xPosF, idxI);
				const CPU::Float32x8 yi = CPU::GatherFloat32x8(yPosF, idxI);
				const CPU::Float32x8 zi = CPU::GatherFloat32x8(zPosF, idxI);
				const CPU::Float32x8 vxi = CPU::GatherFloat32x8(xVelF, idxI);
				const CPU::Float32x8 vyi = CPU::GatherFloat32x8(yVelF, idxI);
				const CPU::Float32x8 vzi = CPU::GatherFloat32x8(zVelF, idxI);

				CPU::Float32x8 tmp_xAccViscF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_yAccViscF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_zAccViscF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_xAccViscB = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_yAccViscB = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_zAccViscB = CPU::ZeroFloat32x8();

				// contributions from fluid neighbors
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);
						const CPU::Float32x8 dvx = vxi - CPU::GatherFloat32x8(xVelF, idxJ);
						const CPU::Float32x8 dvy = vyi - CPU::GatherFloat32x8(yVelF, idxJ);
						const CPU::Float32x8 dvz = vzi - CPU::GatherFloat32x8(zVelF, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 dot = xsimd::fma(dvx, dx,
							xsimd::fma(dvy, dy, dvz * dz));
						const CPU::Float32x8 distSqu = xsimd::fma(dx, dx,
							xsimd::fma(dy, dy, dz * dz));
						const CPU::Float32x8 densityJ =
							xsimd::max(CPU::GatherFloat32x8(densityF, idxJ), eps);
						const CPU::Float32x8 accCoeff =
							(massFV * dot) / (densityJ * (distSqu + particleSizeSquEps));
						const CPU::Float32x8 accCoeffMasked =
							CPU::ZeroMaskedFloat32x8(accCoeff, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccViscF = xsimd::fma(accCoeffMasked, xkg, tmp_xAccViscF);
						tmp_yAccViscF = xsimd::fma(accCoeffMasked, ykg, tmp_yAccViscF);
						tmp_zAccViscF = xsimd::fma(accCoeffMasked, zkg, tmp_zAccViscF);
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 dot = xsimd::fma(vxi, dx,
							xsimd::fma(vyi, dy, vzi * dz));
						const CPU::Float32x8 distSqu = xsimd::fma(dx, dx,
							xsimd::fma(dy, dy, dz * dz));
						const CPU::Float32x8 accCoeff =
							(massBOverRestDensityBV * dot) / (distSqu + particleSizeSquEps);
						const CPU::Float32x8 accCoeffMasked =
							CPU::ZeroMaskedFloat32x8(accCoeff, mask);

						tmp_xAccViscB = xsimd::fma(accCoeffMasked, xkg, tmp_xAccViscB);
						tmp_yAccViscB = xsimd::fma(accCoeffMasked, ykg, tmp_yAccViscB);
						tmp_zAccViscB = xsimd::fma(accCoeffMasked, zkg, tmp_zAccViscB);
					}
				}

				const CPU::Float32x8 xAccNonPressF =
					tmp_xAccViscF * viscosityFacF + tmp_xAccViscB * viscosityFacB + xGravityV;
				const CPU::Float32x8 yAccNonPressF =
					tmp_yAccViscF * viscosityFacF + tmp_yAccViscB * viscosityFacB + yGravityV;
				const CPU::Float32x8 zAccNonPressF =
					tmp_zAccViscF * viscosityFacF + tmp_zAccViscB * viscosityFacB + zGravityV;

				// final chunk may contain fewer than 8 active particles
				CPU::StoreFloat32x8Partial(
					xVelPredF, begin, end, xsimd::fma(xAccNonPressF, timeStepV, vxi));
				CPU::StoreFloat32x8Partial(
					yVelPredF, begin, end, xsimd::fma(yAccNonPressF, timeStepV, vyi));
				CPU::StoreFloat32x8Partial(
					zVelPredF, begin, end, xsimd::fma(zAccNonPressF, timeStepV, vzi));
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit_ChunkedSELL8(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xVelPredF,
	const float* RESTRICT yVelPredF,
	const float* RESTRICT zVelPredF,
	const float* RESTRICT densityF,
	const I_NeighborListSELL8& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensityF, const float restDensitySquInvF,
	const float massF, const float massB,
	const float pressGamma, const float jacobiOmega,
	const uint32_t N_F,
	// destinations
	float* RESTRICT sourceTermF,
	float* RESTRICT diagElemF,
	float* RESTRICT pressureF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const CPU::Float32x8 timeStepV = CPU::SetFloat32x8(timeStep);
	const CPU::Float32x8 timeStepSquV = CPU::SetFloat32x8(timeStep * timeStep);
	const CPU::Float32x8 restDensityFV = CPU::SetFloat32x8(restDensityF);
	const CPU::Float32x8 massFV = CPU::SetFloat32x8(massF);
	const CPU::Float32x8 massBV = CPU::SetFloat32x8(massB);
	const CPU::Float32x8 cTermFacF = CPU::SetFloat32x8(-massF * restDensitySquInvF);
	const CPU::Float32x8 cTermFacB = CPU::SetFloat32x8(-2.0f * pressGamma * massB * restDensitySquInvF);
	const CPU::Float32x8 diagBFac = CPU::SetFloat32x8(-massF * restDensitySquInvF);
	const CPU::Float32x8 jacobiOmegaV = CPU::SetFloat32x8(jacobiOmega);
	const CPU::Float32x8 zero = CPU::ZeroFloat32x8();
	const CPU::Float32x8 eps = CPU::SetFloat32x8(ZERO_EQU_LIM_F);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELL8NeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELL8NeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32x8 idxI = CPU::ParticleIdx32x8(begin, end);

				// one SELL8 chunk covers up to 8 consecutive i-particles
				const CPU::Float32x8 xi = CPU::GatherFloat32x8(xPosF, idxI);
				const CPU::Float32x8 yi = CPU::GatherFloat32x8(yPosF, idxI);
				const CPU::Float32x8 zi = CPU::GatherFloat32x8(zPosF, idxI);
				const CPU::Float32x8 vpxi = CPU::GatherFloat32x8(xVelPredF, idxI);
				const CPU::Float32x8 vpyi = CPU::GatherFloat32x8(yVelPredF, idxI);
				const CPU::Float32x8 vpzi = CPU::GatherFloat32x8(zVelPredF, idxI);

				CPU::Float32x8 tmp_SourceTermF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_SourceTermB = CPU::ZeroFloat32x8();
				CPU::Float32x8 xCTermF = CPU::ZeroFloat32x8();
				CPU::Float32x8 yCTermF = CPU::ZeroFloat32x8();
				CPU::Float32x8 zCTermF = CPU::ZeroFloat32x8();
				CPU::Float32x8 xCTermB = CPU::ZeroFloat32x8();
				CPU::Float32x8 yCTermB = CPU::ZeroFloat32x8();
				CPU::Float32x8 zCTermB = CPU::ZeroFloat32x8();

				// contributions from fluid neighbors
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);
						const CPU::Float32x8 dvpx = vpxi - CPU::GatherFloat32x8(xVelPredF, idxJ);
						const CPU::Float32x8 dvpy = vpyi - CPU::GatherFloat32x8(yVelPredF, idxJ);
						const CPU::Float32x8 dvpz = vpzi - CPU::GatherFloat32x8(zVelPredF, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 sourceTerm =
							xsimd::fma(dvpx, xkg, xsimd::fma(dvpy, ykg, dvpz * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermF += CPU::ZeroMaskedFloat32x8(sourceTerm, mask);
						xCTermF += CPU::ZeroMaskedFloat32x8(xkg, mask);
						yCTermF += CPU::ZeroMaskedFloat32x8(ykg, mask);
						zCTermF += CPU::ZeroMaskedFloat32x8(zkg, mask);
					}
				}

				// contributions from boundary neighbors
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 sourceTerm =
							xsimd::fma(vpxi, xkg, xsimd::fma(vpyi, ykg, vpzi * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SourceTermB += CPU::ZeroMaskedFloat32x8(sourceTerm, mask);
						xCTermB += CPU::ZeroMaskedFloat32x8(xkg, mask);
						yCTermB += CPU::ZeroMaskedFloat32x8(ykg, mask);
						zCTermB += CPU::ZeroMaskedFloat32x8(zkg, mask);
					}
				}

				tmp_SourceTermF *= massFV;
				tmp_SourceTermB *= massBV;

				const CPU::Float32x8 fin_sourceTermF =
					restDensityFV - CPU::GatherFloat32x8(densityF, idxI)
					- timeStepV * (tmp_SourceTermF + tmp_SourceTermB);

				const CPU::Float32x8 xCTerm = xCTermF * cTermFacF + xCTermB * cTermFacB;
				const CPU::Float32x8 yCTerm = yCTermF * cTermFacF + yCTermB * cTermFacB;
				const CPU::Float32x8 zCTerm = zCTermF * cTermFacF + zCTermB * cTermFacB;

				CPU::Float32x8 tmp_DiagElemA = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_DiagElemB = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_DiagElemC = CPU::ZeroFloat32x8();

				// part A and B
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 diagElemA =
							xsimd::fma(xCTerm, xkg, xsimd::fma(yCTerm, ykg, zCTerm * zkg));

						const CPU::Float32x8 tmp_x = diagBFac * xkg;
						const CPU::Float32x8 tmp_y = diagBFac * ykg;
						const CPU::Float32x8 tmp_z = diagBFac * zkg;
						const CPU::Float32x8 diagElemB =
							xsimd::fma(tmp_x, xkg, xsimd::fma(tmp_y, ykg, tmp_z * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_DiagElemA += CPU::ZeroMaskedFloat32x8(diagElemA, mask);
						tmp_DiagElemB += CPU::ZeroMaskedFloat32x8(diagElemB, mask);
					}
				}
				tmp_DiagElemA *= massFV;
				tmp_DiagElemB *= massFV;

				// part C
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 diagElemC =
							xsimd::fma(xCTerm, xkg, xsimd::fma(yCTerm, ykg, zCTerm * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_DiagElemC += CPU::ZeroMaskedFloat32x8(diagElemC, mask);
					}
				}
				tmp_DiagElemC *= massBV;

				const CPU::Float32x8 fin_diagElemF =
					timeStepSquV * (tmp_DiagElemA + tmp_DiagElemB + tmp_DiagElemC);
				const CPU::Mask32x8 validDiag = CPU::AbsFloat32x8(fin_diagElemF) > eps;
				const CPU::Float32x8 pressure = xsimd::max(zero,
					CPU::ZeroMaskedFloat32x8(
						jacobiOmegaV * (fin_sourceTermF / fin_diagElemF), validDiag));

				// final chunk may contain fewer than 8 active particles
				CPU::StoreFloat32x8Partial(sourceTermF, begin, end, fin_sourceTermF);
				CPU::StoreFloat32x8Partial(diagElemF, begin, end, fin_diagElemF);
				CPU::StoreFloat32x8Partial(pressureF, begin, end, pressure);
			}
		},
		tbb::auto_partitioner{}
	);
}

void CPU01::PressureAcceleration_ChunkedSELL8(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT pressureF,
	const I_NeighborListSELL8& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensityF,
	const float massF, const float massB, const float pressGamma,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT xAccPressF,
	float* RESTRICT yAccPressF,
	float* RESTRICT zAccPressF)
{
	(void)N_F;

	const uint32_t numChunks = neighborList.ChunkNumTotal();
	const float densiInvSqu = 1.0f / (restDensityF * restDensityF);
	const CPU::Float32x8 massFV = CPU::SetFloat32x8(massF);
	const CPU::Float32x8 densiInvSquV = CPU::SetFloat32x8(densiInvSqu);
	const CPU::Float32x8 accCoeffB_outer = CPU::SetFloat32x8(2.0f * pressGamma * massB * densiInvSqu);
	const CPU::Float32x8 zero = CPU::ZeroFloat32x8();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELL8NeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELL8NeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32x8 idxI = CPU::ParticleIdx32x8(begin, end);

				// one SELL8 chunk covers up to 8 consecutive i-particles
				const CPU::Float32x8 xi = CPU::GatherFloat32x8(xPosF, idxI);
				const CPU::Float32x8 yi = CPU::GatherFloat32x8(yPosF, idxI);
				const CPU::Float32x8 zi = CPU::GatherFloat32x8(zPosF, idxI);
				const CPU::Float32x8 presi = CPU::GatherFloat32x8(pressureF, idxI);
				const CPU::Float32x8 accCoeffF_outer = massFV * presi * densiInvSquV;

				CPU::Float32x8 tmp_xAccPressF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_yAccPressF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_zAccPressF = CPU::ZeroFloat32x8();

				// contributions from fluid neighbors
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 presj = CPU::GatherFloat32x8(pressureF, idxJ);
						const CPU::Float32x8 accCoeffF =
							accCoeffF_outer + massFV * presj * densiInvSquV;
						const CPU::Float32x8 accCoeffFMasked =
							CPU::ZeroMaskedFloat32x8(accCoeffF, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffFMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffFMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffFMasked, zkg, tmp_zAccPressF);
					}
				}

				const CPU::Float32x8 accCoeffB = accCoeffB_outer * presi;

				// contributions from boundary neighbors
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 accCoeffBMasked =
							CPU::ZeroMaskedFloat32x8(accCoeffB, mask);

						// padded lanes gathered valid indices but contribute zero
						tmp_xAccPressF = xsimd::fma(accCoeffBMasked, xkg, tmp_xAccPressF);
						tmp_yAccPressF = xsimd::fma(accCoeffBMasked, ykg, tmp_yAccPressF);
						tmp_zAccPressF = xsimd::fma(accCoeffBMasked, zkg, tmp_zAccPressF);
					}
				}

				// final chunk may contain fewer than 8 active particles
				CPU::StoreFloat32x8Partial(xAccPressF, begin, end, zero - tmp_xAccPressF);
				CPU::StoreFloat32x8Partial(yAccPressF, begin, end, zero - tmp_yAccPressF);
				CPU::StoreFloat32x8Partial(zAccPressF, begin, end, zero - tmp_zAccPressF);
			}
		},
		affPart
	);
}

void CPU01::PressureRelaxedJacobiSolver_ChunkedSELL8(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT xAccPressF,
	const float* RESTRICT yAccPressF,
	const float* RESTRICT zAccPressF,
	const float* RESTRICT sourceTermF,
	const float* RESTRICT diagElemF,
	const I_NeighborListSELL8& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensityF, const float massF, const float massB,
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
	const CPU::Float32x8 timeStepSquV = CPU::SetFloat32x8(timeStep * timeStep);
	const CPU::Float32x8 restDensityFV = CPU::SetFloat32x8(restDensityF);
	const CPU::Float32x8 massFV = CPU::SetFloat32x8(massF);
	const CPU::Float32x8 massBV = CPU::SetFloat32x8(massB);
	const CPU::Float32x8 jacobiOmegaV = CPU::SetFloat32x8(jacobiOmega);
	const CPU::Float32x8 zero = CPU::ZeroFloat32x8();
	const CPU::Float32x8 eps = CPU::SetFloat32x8(ZERO_EQU_LIM_F);

	DensityErrorAccum acc = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, numChunks), DensityErrorAccum{},
		[&](const tbb::blocked_range<uint32_t>& r, DensityErrorAccum local)
		{
			alignas(32) float relErrTmp[CPU::N_SIMD_WIDTH]{};

			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const ChunkedSELL8NeighborSpan_Read nbrFF = neighborList.FFChunk(chunk);
				const ChunkedSELL8NeighborSpan_Read nbrFB = neighborList.FBChunk(chunk);
				const uint32_t begin = nbrFF.begin;
				const uint32_t end = nbrFF.end;
				const CPU::UInt32x8 idxI = CPU::ParticleIdx32x8(begin, end);

				// one SELL8 chunk covers up to 8 consecutive i-particles
				const CPU::Float32x8 xi = CPU::GatherFloat32x8(xPosF, idxI);
				const CPU::Float32x8 yi = CPU::GatherFloat32x8(yPosF, idxI);
				const CPU::Float32x8 zi = CPU::GatherFloat32x8(zPosF, idxI);
				const CPU::Float32x8 axi = CPU::GatherFloat32x8(xAccPressF, idxI);
				const CPU::Float32x8 ayi = CPU::GatherFloat32x8(yAccPressF, idxI);
				const CPU::Float32x8 azi = CPU::GatherFloat32x8(zAccPressF, idxI);

				CPU::Float32x8 tmp_SysVecElemF = CPU::ZeroFloat32x8();
				CPU::Float32x8 tmp_SysVecElemB = CPU::ZeroFloat32x8();

				// contributions from fluid neighbors
				{
					const CPU::UInt32x8 countFF = CPU::LoadUInt16AsUInt32x8(nbrFF.counts + begin);

					for (uint32_t k = 0; k < nbrFF.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFF);
						const CPU::UInt32x8 idxJ = nbrFF.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosF, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosF, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosF, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						const CPU::Float32x8 dapx = axi - CPU::GatherFloat32x8(xAccPressF, idxJ);
						const CPU::Float32x8 dapy = ayi - CPU::GatherFloat32x8(yAccPressF, idxJ);
						const CPU::Float32x8 dapz = azi - CPU::GatherFloat32x8(zAccPressF, idxJ);

						const CPU::Float32x8 sysVecElem = CPU::FmaFloat32x8(
							dapx, xkg, CPU::FmaFloat32x8(dapy, ykg, dapz * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SysVecElemF += CPU::AndMaskedFloat32x8(sysVecElem, mask);
					}
				}
				tmp_SysVecElemF *= massFV;

				// contributions from boundary neighbors
				{
					const CPU::UInt32x8 countFB = CPU::LoadUInt16AsUInt32x8(nbrFB.counts + begin);

					for (uint32_t k = 0; k < nbrFB.pitch; k++)
					{
						// one SELL8 column stores neighbor k for all 8 lanes
						const CPU::Mask32x8 mask = CPU::MaskKLessThanCounts32x8(k, countFB);
						const CPU::UInt32x8 idxJ = nbrFB.LoadNeighborIdx32x8(k);

						const CPU::Float32x8 dx = xi - CPU::GatherFloat32x8(xPosB, idxJ);
						const CPU::Float32x8 dy = yi - CPU::GatherFloat32x8(yPosB, idxJ);
						const CPU::Float32x8 dz = zi - CPU::GatherFloat32x8(zPosB, idxJ);

						CPU::Float32x8 xkg;
						CPU::Float32x8 ykg;
						CPU::Float32x8 zkg;
						CPU::KernelGradient_CubicSpline_SIMD8(
							dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
							cubicSplineAlpha, xkg, ykg, zkg);

						// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
						const CPU::Float32x8 sysVecElem = CPU::FmaFloat32x8(
							axi, xkg, CPU::FmaFloat32x8(ayi, ykg, azi * zkg));

						// padded lanes gathered valid indices but contribute zero
						tmp_SysVecElemB += CPU::AndMaskedFloat32x8(sysVecElem, mask);
					}
				}
				tmp_SysVecElemB *= massBV;

				const CPU::Float32x8 sysVecElemF =
					timeStepSquV * (tmp_SysVecElemF + tmp_SysVecElemB);
				const CPU::Float32x8 sourceTerm = CPU::GatherFloat32x8(sourceTermF, idxI);
				const CPU::Float32x8 diagElem = CPU::GatherFloat32x8(diagElemF, idxI);
				const CPU::Float32x8 solverResidual = sourceTerm - sysVecElemF;
				const CPU::Float32x8 predDensityError = sysVecElemF - sourceTerm;

				const CPU::Mask32x8 validDiag = CPU::AbsBitwiseFloat32x8(diagElem) > eps;
				const CPU::Float32x8 pressure = CPU::GatherFloat32x8(pressureF, idxI);
				const CPU::Float32x8 pressureNext = xsimd::max(zero,
					CPU::AndMaskedFloat32x8(
						CPU::FmaFloat32x8(jacobiOmegaV, solverResidual / diagElem, pressure),
						validDiag));

				// final chunk may contain fewer than 8 active particles
				CPU::StoreFloat32x8Partial(pressureF, begin, end, pressureNext);

				const CPU::Float32x8 relErr = predDensityError / restDensityFV;
				const uint32_t validDiagMask = CPU::MaskBits32x8(validDiag);
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
