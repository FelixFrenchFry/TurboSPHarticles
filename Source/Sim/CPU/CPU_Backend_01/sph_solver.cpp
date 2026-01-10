// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <numbers>
#include "spdlog/spdlog.h"
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>

// internal
#include "Source/Base/helpers.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"
#include "cpu_backend_01.h"
#include "sph_solver.h"
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
	// --- [ temporary pre-computed constants ] --------------------------------

	// misc hoisted constants
	const float timeStep = params.timeStep;
	const float sizeInvF = params.particleSizeInvF;
	const float sizeSquF = params.particleSizeF * params.particleSizeF;
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const float cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * static_cast<double>(sizeInvF) *
		static_cast<double>(sizeInvF) * static_cast<double>(sizeInvF));
	const float restDensitySquInvF = 1.0f / (params.restDensityF * params.restDensityF);
	const float massF = params.massF;
	const float massB = params.massB;

	// --- [ particle data re-ordering ] ---------------------------------------

	if (params.dataReOrderInterval > 0 && simStep_ % params.dataReOrderInterval == 0)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Data_Ordering });
		neighborSearchStrat_->DataReOrder(params);
		affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
		benchmarksTracker_.StopBench({ Bench::BenchOp::Data_Ordering });
	}

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
	float* RESTRICT xAccNonPressF = xAccNonPressF_.data();
	float* RESTRICT yAccNonPressF = yAccNonPressF_.data();
	float* RESTRICT zAccNonPressF = zAccNonPressF_.data();
	float* RESTRICT sourceTermF = sourceTermF_.data();
	float* RESTRICT diagElemF = diagElemF_.data();
	float* RESTRICT pressureF = pressureF_.data();
	float* RESTRICT densityF = densityF_.data();
	I_NeighborLists& neighborLists = *neighborLists_;

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	if (params.runCountQuery)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query_Count(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborLists_->Finalize_Count();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	if (params.runFillQuery)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearchStrat_->Query_Fill(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborLists_->Finalize_Fill();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ density computations ] --------------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Density_Solving });

	CPU01::Density(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		neighborLists,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, massF, massB, N_F_,
		densityF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] --------------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	CPU01::NonPressureAcceleration(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelF, yVelF, zVelF,
		densityF,
		neighborLists,
		timeStep, sizeInvF, sizeSquF, kernelRadiusSqu,
		cubicSplineAlpha, params.restDensityF, params.restDensityB, massF, massB,
		params.viscosityF, params.viscosityB,
		params.gravity.x, params.gravity.y, params.gravity.z, N_F_,
		xVelPredF, yVelPredF, zVelPredF,
		xAccNonPressF, yAccNonPressF, zAccNonPressF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Viscosity_Acceleration, Bench::BenchOp::External_Acceleration });

	// --- [ source-term (density invariance) and diagonal elements ] ----------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

	CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		xVelPredF, yVelPredF, zVelPredF,
		densityF,
		neighborLists,
		timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
		params.restDensityF, restDensitySquInvF,
		massF, massB, params.pressGamma, params.jacobiOmega, N_F_,
		sourceTermF, diagElemF, pressureF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_SourceTerm, Bench::BenchOp::Pressure_DiagonalElem });

	// --- [ relaxed Jacobi solver for pressure ] ------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// rest solver-related variables
	N_solverIterCount_ = 1;
	avgRelDensityErrorF_raw_ = 0.0f;
	avgRelDensityErrorF_pos_ = 0.0f;
	avgRelDensityErrorF_abs_ = 0.0f;

	do
	{
		// compute pressure acceleration from current pressure field
		CPU01::PressureAcceleration(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			pressureF,
			neighborLists,
			sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
			massF, massB, params.pressGamma, N_F_, *affinityPart_,
			xAccPressF, yAccPressF, zAccPressF);

		// update system vector and pressure
		CPU01::PressureRelaxedJacobiSolver(
			xPosF, yPosF, zPosF,
			xPosB, yPosB, zPosB,
			xAccPressF, yAccPressF, zAccPressF,
			sourceTermF, diagElemF,
			neighborLists,
			timeStep, sizeInvF, kernelRadiusSqu, cubicSplineAlpha,
			params.restDensityF, massF, massB, params.jacobiOmega, N_F_,
			*affinityPart_, pressureF, avgRelDensityErrorF_raw_,
			avgRelDensityErrorF_pos_, avgRelDensityErrorF_abs_);

		N_solverIterCount_++;
	} while (!TerminationConditionSatisfied(params));

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration ] ----------------------------------

	benchmarksTracker_.StartBench({ Bench::BenchOp::Pressure_Acceleration });

	CPU01::PressureAcceleration(
		xPosF, yPosF, zPosF,
		xPosB, yPosB, zPosB,
		pressureF,
		neighborLists,
		sizeInvF, kernelRadiusSqu, cubicSplineAlpha, params.restDensityF,
		massF, massB, params.pressGamma, N_F_, *affinityPart_,
		xAccPressF, yAccPressF, zAccPressF);

	benchmarksTracker_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ integrate velocity and position ] ---------------------------------

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

inline void CPU01::KernelWeight_CubicSpline(
	const float dx, const float dy, const float dz,
	const float particleSizeInv, const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	// destinations
	float& kw) noexcept
{
	const float distSqu = std::fmaf(dx, dx, 
		std::fmaf(dy, dy, dz * dz));

	// outside support radius -> no weight
	if (distSqu >= kernelRadiusSqu) [[unlikely]]
	{
		kw = 0.0f;
		return;
	}

	// no distance -> shortcut computation
	if (distSqu < ZERO_EQU_LIM) [[unlikely]]
	{
		kw = 4.0f * cubicSplineAlpha;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp t1 to [0, 1]
	// no clamp for t2 needed, distNorm < 2 here
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	const float t2 = 2.0f - distNorm;

	// common sub-expressions
	const float t1_2 = t1 * t1;
	const float t1_3 = t1_2 * t1;
	const float t2_2 = t2 * t2;
	const float t2_3 = t2_2 * t2;

	// kernel weight
	kw = cubicSplineAlpha * (t2_3 - 4.0f * t1_3);
}

inline void CPU01::KernelGradient_CubicSpline(
	const float dx, const float dy, const float dz,
	const float particleSizeInv, const float kernelRadiusSqu,
	const float cubicSplineAlpha,
	// destinations
	float& xkg, float& ykg, float& zkg) noexcept
{
	const float distSqu = std::fmaf(dx, dx, 
		std::fmaf(dy, dy, dz * dz));

	// outside support radius -> no gradient
	// no distance -> also no gradient
	if (distSqu >= kernelRadiusSqu ||
		distSqu < ZERO_EQU_LIM) [[unlikely]]
	{
		xkg = 0.0f;
		ykg = 0.0f;
		zkg = 0.0f;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp t1 to [0, 1]
	// no clamp for t2 needed, distNorm < 2 here
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	const float t2 = 2.0f - distNorm;

	// common sub-expressions
	const float t1_2 = t1 * t1;
	const float t2_2 = t2 * t2;

	float commonFac = cubicSplineAlpha
					* (-3.0f * t2_2 + 12.0f * t1_2)
					* (particleSizeInv / dist);

	// kernel gradient
	xkg = commonFac * dx;
	ykg = commonFac * dy;
	zkg = commonFac * dz;
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::Density(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const I_NeighborLists& neighborLists,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float massF, const float massB,
	const uint32_t N_F,
	// destinations
	float* RESTRICT densityF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				float tmp_densityF = 0.0f;
				float tmp_densityB = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_densityF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel weight
					float kw = 0.0f;
					KernelWeight_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, kw);

					tmp_densityF += kw;
				}
				tmp_densityF *= massF;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_densityB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel weight
					float kw = 0.0f;
					KernelWeight_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu, cubicSplineAlpha, kw);

					tmp_densityB += kw;
				}
				tmp_densityB *= massB;

				// finalize density
				densityF[i] = tmp_densityF + tmp_densityB;
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::NonPressureAcceleration(
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
	const I_NeighborLists& neighborLists,
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
	float* RESTRICT zVelPredF,
	float* RESTRICT xAccNonPressF,
	float* RESTRICT yAccNonPressF,
	float* RESTRICT zAccNonPressF)
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float vxi = xVelF[i];
				const float vyi = yVelF[i];
				const float vzi = zVelF[i];

				float tmp_xAccViscF = 0.0f;
				float tmp_yAccViscF = 0.0f;
				float tmp_zAccViscF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_xAccViscF,tmp_yAccViscF,tmp_zAccViscF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// velocity difference
					const float dvx = vxi - xVelF[j];
					const float dvy = vyi - yVelF[j];
					const float dvz = vzi - zVelF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF[j], 1e-12f))
										 * (dvx * dx + dvy * dy + dvz * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeSquF);

					tmp_xAccViscF = std::fmaf(accCoeff, xkg, tmp_xAccViscF);
					tmp_yAccViscF = std::fmaf(accCoeff, ykg, tmp_yAccViscF);
					tmp_zAccViscF = std::fmaf(accCoeff, zkg, tmp_zAccViscF);
				}

				float tmp_xAccViscB = 0.0f;
				float tmp_yAccViscB = 0.0f;
				float tmp_zAccViscB = 0.0f;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_xAccViscB,tmp_yAccViscB,tmp_zAccViscB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / restDensityB)
										 * (vxi * dx + vyi * dy + vzi * dz)
										 / (dx * dx + dy * dy + dz * dz
										 + 0.01f * particleSizeSquF);

					tmp_xAccViscB = std::fmaf(accCoeff, xkg, tmp_xAccViscB);
					tmp_yAccViscB = std::fmaf(accCoeff, ykg, tmp_yAccViscB);
					tmp_zAccViscB = std::fmaf(accCoeff, zkg, tmp_zAccViscB);
				}

				// finalize accelerations
				// (3D -> factor 2 * (dim + 2))
				tmp_xAccViscF *= 10.0f * viscosityF;
				tmp_yAccViscF *= 10.0f * viscosityF;
				tmp_zAccViscF *= 10.0f * viscosityF;
				tmp_xAccViscB *= 10.0f * viscosityB;
				tmp_yAccViscB *= 10.0f * viscosityB;
				tmp_zAccViscB *= 10.0f * viscosityB;

				// TODO: add surface tension acc contribution
				xAccNonPressF[i] = tmp_xAccViscF + tmp_xAccViscB + xGravity;
				yAccNonPressF[i] = tmp_yAccViscF + tmp_yAccViscB + yGravity;
				zAccNonPressF[i] = tmp_zAccViscF + tmp_zAccViscB + zGravity;

				// compute predicted velocity from non-pressure accelerations
				xVelPredF[i] = std::fmaf(xAccNonPressF[i], timeStep, xVelF[i]);
				yVelPredF[i] = std::fmaf(yAccNonPressF[i], timeStep, yVelF[i]);
				zVelPredF[i] = std::fmaf(zAccNonPressF[i], timeStep, zVelF[i]);
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::SourceTermAndDiagonalElementAndPressureJacobiInit(
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
	const Sim::I_NeighborLists& neighborLists,
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
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float vpxi = xVelPredF[i];
				const float vpyi = yVelPredF[i];
				const float vpzi = zVelPredF[i];

				// source term accumulators
				float tmp_SourceTermF = 0.0f;
				float tmp_SourceTermB = 0.0f;

				// c-term accumulators
				float xCTermF = 0.0f;
				float yCTermF = 0.0f;
				float zCTermF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_SourceTermF,xCTermF,yCTermF,zCTermF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// predicted velocity difference
					const float dvpx = vpxi - xVelPredF[j];
					const float dvpy = vpyi - yVelPredF[j];
					const float dvpz = vpzi - zVelPredF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// source term contribution from fluid neighbor
					tmp_SourceTermF += dvpx * xkg
									 + dvpy * ykg
									 + dvpz * zkg;

					xCTermF += xkg;
					yCTermF += ykg;
					zCTermF += zkg;
				}
				tmp_SourceTermF *= massF;

				float xCTermB = 0.0f;
				float yCTermB = 0.0f;
				float zCTermB = 0.0f;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_SourceTermB,xCTermB,yCTermB,zCTermB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// source term contribution from boundary neighbor
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTermB += vpxi * xkg
									 + vpyi * ykg
									 + vpzi * zkg;

					xCTermB += xkg;
					yCTermB += ykg;
					zCTermB += zkg;
				}
				tmp_SourceTermB *= massB;

				// finalize source term
				const float fin_sourceTermF = restDensityF - densityF[i] - timeStep
										    * (tmp_SourceTermF + tmp_SourceTermB);

				// finalize c-term
				// TODO: pre-compute the constant factors?
				const float xCTerm = -1.0f * (xCTermF * restDensitySquInvF * massF)
					- 2.0f * pressGamma * (xCTermB * restDensitySquInvF * massB);
				const float yCTerm = -1.0f * (yCTermF * restDensitySquInvF * massF)
					- 2.0f * pressGamma * (yCTermB * restDensitySquInvF * massB);
				const float zCTerm = -1.0f * (zCTermF * restDensitySquInvF * massF)
					- 2.0f * pressGamma * (zCTermB * restDensitySquInvF * massB);

				float tmp_DiagElemA = 0.0f;
				float tmp_DiagElemB = 0.0f;
				float tmp_DiagElemC = 0.0f;

				// part A and B
				#pragma omp simd reduction(+:tmp_DiagElemA,tmp_DiagElemB)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					tmp_DiagElemA += fmaf(xCTerm, xkg,
						fmaf(yCTerm, ykg, zCTerm * zkg));

					// TODO: pre-compute the constant factors?
					const float tmp_x = (massF * restDensitySquInvF) * (-xkg);
					const float tmp_y = (massF * restDensitySquInvF) * (-ykg);
					const float tmp_z = (massF * restDensitySquInvF) * (-zkg);

					tmp_DiagElemB += fmaf(tmp_x, xkg, 
						fmaf(tmp_y, ykg, tmp_z * zkg));
				}
				tmp_DiagElemA *= massF;
				tmp_DiagElemB *= massF;

				// part C
				#pragma omp simd reduction(+:tmp_DiagElemC)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					tmp_DiagElemC += fmaf(xCTerm, xkg, 
						fmaf(yCTerm, ykg, zCTerm * zkg));
				}
				tmp_DiagElemC *= massB;

				// finalize diagonal element
				const float fin_diagElemF = timeStep * timeStep
										  * (tmp_DiagElemA + tmp_DiagElemB + tmp_DiagElemC);

				float tmp_pressureF = 0.0f;

				// only update pressure if particle has neighbors
				if (!(fabsf(fin_diagElemF) <= 1e-6f))
				{
					tmp_pressureF = jacobiOmega * (fin_sourceTermF / fin_diagElemF);
				}

				// clamp pressure to non-negative values
				// (and perform final global memory writes)
				pressureF[i] = fmaxf(tmp_pressureF, 0.0f);
				sourceTermF[i] = fin_sourceTermF;
				diagElemF[i] = fin_diagElemF;
			}
		},
		tbb::auto_partitioner{}
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureAcceleration(
	const float* RESTRICT xPosF,
	const float* RESTRICT yPosF,
	const float* RESTRICT zPosF,
	const float* RESTRICT xPosB,
	const float* RESTRICT yPosB,
	const float* RESTRICT zPosB,
	const float* RESTRICT pressureF,
	const I_NeighborLists& neighborLists,
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
	// hoisted constants
	const float densiInvSqu = 1.0f / (restDensityF * restDensityF);
	const float accCoeffB_outer = 2.0f * pressGamma * massB * densiInvSqu;

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float presi = pressureF[i];
				const float accCoeffF_outer = massF * presi * densiInvSqu;

				float tmp_xAccPressF = 0.0f;
				float tmp_yAccPressF = 0.0f;
				float tmp_zAccPressF = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					const float accCoeffF = accCoeffF_outer + massF
										  * pressureF[j] * densiInvSqu;

					// pressure acc contribution from fluid neighbor j
					tmp_xAccPressF = fmaf(accCoeffF, xkg, tmp_xAccPressF);
					tmp_yAccPressF = fmaf(accCoeffF, ykg, tmp_yAccPressF);
					tmp_zAccPressF = fmaf(accCoeffF, zkg, tmp_zAccPressF);
				}

				const float accCoeffB = accCoeffB_outer * presi;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_xAccPressF,tmp_yAccPressF,tmp_zAccPressF)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// pressure acc contribution from boundary neighbor j
					tmp_xAccPressF = fmaf(accCoeffB, xkg, tmp_xAccPressF);
					tmp_yAccPressF = fmaf(accCoeffB, ykg, tmp_yAccPressF);
					tmp_zAccPressF = fmaf(accCoeffB, zkg, tmp_zAccPressF);
				}

				// finalize pressure acceleration
				xAccPressF[i] = -tmp_xAccPressF;
				yAccPressF[i] = -tmp_yAccPressF;
				zAccPressF[i] = -tmp_zAccPressF;
			}
		},
		affPart
	);
}

// TODO: use more explicit SIMD hints, maybe xsimd library?
void CPU01::PressureRelaxedJacobiSolver(
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
	const I_NeighborLists& neighborLists,
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

	DensityErrorAccum acc = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_F), DensityErrorAccum{},
		[&](const tbb::blocked_range<uint32_t>& r, DensityErrorAccum local)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
			    const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				const float axi = xAccPressF[i];
				const float ayi = yAccPressF[i];
				const float azi = zAccPressF[i];

				float tmp_SysVecElemF = 0.0f;
				float tmp_SysVecElemB = 0.0f;

				// contributions from fluid neighbors
				NeighborSpan nbrListFF = neighborLists.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.count;
				#pragma omp simd reduction(+:tmp_SysVecElemF)
				for (uint32_t k = 0; k < N_FF; ++k)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFF[k];

					// position difference
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// pressure acc difference
					const float dapx = axi - xAccPressF[j];
					const float dapy = ayi - yAccPressF[j];
					const float dapz = azi - zAccPressF[j];

					tmp_SysVecElemF += std::fmaf(dapx, xkg, 
						std::fmaf(dapy, ykg, dapz * zkg));
				}
				tmp_SysVecElemF *= massF;

				// contributions from boundary neighbors
				NeighborSpan nbrListFB = neighborLists.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.count;
				#pragma omp simd reduction(+:tmp_SysVecElemB)
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = nbrFB[k];

					// position difference
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// kernel gradient
					float xkg = 0.0f;
					float ykg = 0.0f;
					float zkg = 0.0f;
					KernelGradient_CubicSpline(
						dx, dy, dz, particleSizeInvF, kernelRadiusSqu,
						cubicSplineAlpha, xkg, ykg, zkg);

					// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
					tmp_SysVecElemB += std::fmaf(axi, xkg, 
						std::fmaf(ayi, ykg, azi * zkg));
				}
				tmp_SysVecElemB *= massB;

				// TODO: only save this if needed later?
				// sysVecElemF[i] = timeStep * timeStep * (tmp_SysVecElemF + tmp_SysVecElemB);

				// finalize system vector element
				const float sysVecElemF_i = timeStep * timeStep
										  * (tmp_SysVecElemF + tmp_SysVecElemB);

				const float solverResidual = sourceTermF[i] - sysVecElemF_i;
				const float predDensityError = sysVecElemF_i - sourceTermF[i];
				float tmp_PressureF = 0.0f;

				// only update pressure and error if particle has neighbors
				if (!Base::ApproxEqu(diagElemF[i], 0.0f))
				{
					// relaxed Jacobi solver update
					tmp_PressureF = pressureF[i] + jacobiOmega
								  * (solverResidual / diagElemF[i]);

					// relative density error
					const double relErr = predDensityError / restDensityF;
					const double relPosErr = std::max(0.0, relErr);
					const double relAbsErr = std::fabs(relErr);

					local.sumRaw += relErr;
					local.sumPos += relPosErr;
					local.sumAbs += relAbsErr;
					local.NF_valid++;
				}

				// clamp pressure to non-negative values
				pressureF[i] = fmaxf(tmp_PressureF, 0.0f);
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
