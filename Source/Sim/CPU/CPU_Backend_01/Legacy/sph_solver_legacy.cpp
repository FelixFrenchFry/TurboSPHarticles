// sph_solver_legacy.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

// internal
#include "sph_solver_legacy.h"



using namespace Sim;

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
	if (distSqu < ZERO_EQU_LIM_F) [[unlikely]]
	{
		kw = 4.0f * cubicSplineAlpha;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp to [0, 1]
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	float t2 = 2.0f - distNorm;
	t2 = std::max(t2, 0.0f);

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
		distSqu < ZERO_EQU_LIM_F) [[unlikely]]
	{
		xkg = 0.0f;
		ykg = 0.0f;
		zkg = 0.0f;
		return;
	}

	const float dist = std::sqrt(distSqu);
	const float distNorm = dist * particleSizeInv;

	// clamp to [0, 1]
	float t1 = 1.0f - distNorm;
	t1 = std::max(t1, 0.0f);
	float t2 = 2.0f - distNorm;
	t2 = std::max(t2, 0.0f);

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
	const I_NeighborList& neighborList,
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
				NeighborSpan nbrListFF = neighborList.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.N;
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
				NeighborSpan nbrListFB = neighborList.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.N;
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
	const I_NeighborList& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float particleSizeSquF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float massB,
	const float viscosityF, const float viscosityB,
	const float xGravity, const float yGravity, const float zGravity,
	const uint32_t N_F,
	// destinations
	float* RESTRICT xVelPredF,
	float* RESTRICT yVelPredF,
	float* RESTRICT zVelPredF)
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
				NeighborSpan nbrListFF = neighborList.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.N;
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
					const float accCoeff = (massF / fmaxf(densityF[j], ZERO_EQU_LIM_F))
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
				NeighborSpan nbrListFB = neighborList.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.N;
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
					const float accCoeff = (massB / restDensity)
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
				const float xAccNonPressF = tmp_xAccViscF + tmp_xAccViscB + xGravity;
				const float yAccNonPressF = tmp_yAccViscF + tmp_yAccViscB + yGravity;
				const float zAccNonPressF = tmp_zAccViscF + tmp_zAccViscB + zGravity;

				// compute predicted velocity from non-pressure accelerations
				xVelPredF[i] = std::fmaf(xAccNonPressF, timeStep, xVelF[i]);
				yVelPredF[i] = std::fmaf(yAccNonPressF, timeStep, yVelF[i]);
				zVelPredF[i] = std::fmaf(zAccNonPressF, timeStep, zVelF[i]);
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
	const Sim::I_NeighborList& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float restDensitySquInv,
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
				NeighborSpan nbrListFF = neighborList.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.N;
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
				NeighborSpan nbrListFB = neighborList.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.N;
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
				const float fin_sourceTermF = restDensity - densityF[i] - timeStep
										    * (tmp_SourceTermF + tmp_SourceTermB);

				// finalize c-term
				// TODO: pre-compute the constant factors?
				const float xCTerm = -1.0f * (xCTermF * restDensitySquInv * massF)
					- 2.0f * pressGamma * (xCTermB * restDensitySquInv * massB);
				const float yCTerm = -1.0f * (yCTermF * restDensitySquInv * massF)
					- 2.0f * pressGamma * (yCTermB * restDensitySquInv * massB);
				const float zCTerm = -1.0f * (zCTermF * restDensitySquInv * massF)
					- 2.0f * pressGamma * (zCTermB * restDensitySquInv * massB);

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
					const float tmp_x = (massF * restDensitySquInv) * (-xkg);
					const float tmp_y = (massF * restDensitySquInv) * (-ykg);
					const float tmp_z = (massF * restDensitySquInv) * (-zkg);

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
				if (!(fabsf(fin_diagElemF) <= ZERO_EQU_LIM_F))
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
	const I_NeighborList& neighborList,
	const float particleSizeInvF, const float kernelRadiusSqu,
	const float cubicSplineAlpha, const float restDensity,
	const float massF, const float massB, const float pressGamma,
	const uint32_t N_F,
	tbb::affinity_partitioner& affPart,
	// destinations
	float* RESTRICT xAccPressF,
	float* RESTRICT yAccPressF,
	float* RESTRICT zAccPressF)
{
	// hoisted constants
	const float densiInvSqu = 1.0f / (restDensity * restDensity);
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
				NeighborSpan nbrListFF = neighborList.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.N;
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
				NeighborSpan nbrListFB = neighborList.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.N;
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
	const I_NeighborList& neighborList,
	const float timeStep, const float particleSizeInvF,
	const float kernelRadiusSqu, const float cubicSplineAlpha,
	const float restDensity, const float massF, const float massB,
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
				NeighborSpan nbrListFF = neighborList.FF(i);
				const uint32_t* RESTRICT nbrFF = nbrListFF.data;
				const uint32_t N_FF = nbrListFF.N;
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
				NeighborSpan nbrListFB = neighborList.FB(i);
				const uint32_t* RESTRICT nbrFB = nbrListFB.data;
				const uint32_t N_FB = nbrListFB.N;
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
				if (!(fabsf(diagElemF[i]) <= ZERO_EQU_LIM_F))
				{
					// relaxed Jacobi solver update
					tmp_PressureF = pressureF[i] + jacobiOmega
								  * (solverResidual / diagElemF[i]);

					// relative density error
					const double relErr = predDensityError / restDensity;
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
