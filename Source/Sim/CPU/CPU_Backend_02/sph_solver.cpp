// sph_solver.cpp

// external
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <numbers>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>

// internal
#include "Source/Base/helpers.h"
#include "neighbor_search.h"
#include "cpu_backend_02.h"



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

void SimBackend_CPU02::Step_IISPH(const SimParams& params)
{
	// --- [ temporary pre-computed constants ] --------------------------------

	// cubic spline alpha (computed in double precision for accuracy)
	const double size = static_cast<double>(params.particleSizeF);
	const double sizeInv = 1.0 / size;
	const float cubicSplineAlpha = static_cast<float>(
		(1.0 / (4.0 * static_cast<double>(PI))) * sizeInv * sizeInv * sizeInv);

	// misc hoisted constants
	const float sizeF = params.particleSizeF;
	const float massF = params.massF;
	const float massB = params.massB;
	const float restDensitySquInvF = 1.0f / (params.restDensityF
								   * params.restDensityF);

	// --- [ neighbor search ] -------------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborSearchStrat_->Construct(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	// --- [ neighbor search query ] -------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Neighbors_Query });
	neighborSearchStrat_->Query(params);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Neighbors_Query });

	// --- [ kernel weight and gradient computations ] -------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Kernel_Weights,
									Bench::BenchOp::Kernel_Gradients });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// resize kernel weight/gradient storage to current number of neighbors
				kernelWeightsFF_[i].resize(neighborsFF_[i].size());
				kernelWeightsFB_[i].resize(neighborsFB_[i].size());
				kernelGradFF_[i].resize(neighborsFF_[i].size());
				kernelGradFB_[i].resize(neighborsFB_[i].size());

				const Eigen::Vector3f posi = posF_[i];

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// position difference
					const Eigen::Vector3f dposij = posi - posF_[j];
					const float dist = dposij.norm();
					const float distNorm = dist / sizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFF_[i][k] = cubicSplineAlpha * 4.0f;
						kernelGradFF_[i][k].setZero();

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFF_[i][k] = cubicSplineAlpha
										   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * sizeF * sizeF);

					kernelGradFF_[i][k] = gradCoeff * dposij;
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					// position difference
					const Eigen::Vector3f dposij = posi - posB_[j];
					const float dist = dposij.norm();
					const float distNorm = dist / sizeF;

					// avoid division by zero in gradient computation
					if (Base::ApproxEqu(distNorm, 0.0f)) [[unlikely]]
					{
						kernelWeightsFB_[i][k] = cubicSplineAlpha * 4.0f;
						kernelGradFB_[i][k].setZero();

						continue;
					}

					const float t1 = distNorm < 1.0f ? 1.0f - distNorm : 0.0f;
					const float t2 = distNorm < 2.0f ? 2.0f - distNorm : 0.0f;

					// kernel weight
					kernelWeightsFB_[i][k] = cubicSplineAlpha
										   * (t2 * t2 * t2 - 4.0f * t1 * t1 * t1);

					// kernel gradient
					// TODO: multiply distNorm with size once or twice?
					const float gradCoeff = cubicSplineAlpha
										  * (-3.0f * t2 * t2 + 12.0f * t1 * t1)
										  / (distNorm * sizeF * sizeF);

					kernelGradFB_[i][k] = gradCoeff * dposij;
				}
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Kernel_Weights,
								   Bench::BenchOp::Kernel_Gradients });

	// --- [ density computations ] --------------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Density_Solving });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float density = 0.0f;

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					density += massF * kernelWeightsFF_[i][k];
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					density += massB * kernelWeightsFB_[i][k];
				}

				densityF_[i] = density;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Density_Solving });

	// --- [ non-pressure accelerations ] --------------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Viscosity_Acceleration,
									Bench::BenchOp::External_Acceleration });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const Eigen::Vector3f posi = posF_[i];
				const Eigen::Vector3f veli = velF_[i];

				Eigen::Vector3f tmp_AccViscF = Eigen::Vector3f::Zero();

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// position difference
					const Eigen::Vector3f dposij = posi - posF_[j];

					// velocity difference
					const Eigen::Vector3f dvelij = veli - velF_[j];

					// viscosity acc contribution from fluid neighbor j
					const float accCoeff = (massF / fmaxf(densityF_[j], 1e-12f))
										 * dvelij.dot(dposij) / (dposij.dot(dposij) 
										 + 0.01f * sizeF * sizeF);

					tmp_AccViscF += accCoeff * kernelGradFF_[i][k];
				}

				Eigen::Vector3f tmp_AccViscB = Eigen::Vector3f::Zero();

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFB_[i][k];

					// position difference
					const Eigen::Vector3f dposij = posi - posB_[j];

					// viscosity acc contribution from boundary neighbor j
					const float accCoeff = (massB / params.restDensityB)
										 * veli.dot(dposij) / (dposij.dot(dposij) 
										 + 0.01f * sizeF * sizeF);

					tmp_AccViscB += accCoeff * kernelGradFB_[i][k];
				}

				// finalize accelerations
				// (3D -> factor 2 * (dim + 2))
				tmp_AccViscF *= 2.0f * (3 + 2) * params.viscosityF;
				tmp_AccViscB *= 2.0f * (3 + 2) * params.viscosityB;

				accViscF_[i] = tmp_AccViscF + tmp_AccViscB;

				// TODO: add surface tension acc contribution
				accExtF_[i] = Eigen::Vector3f{ params.gravity.x,
											   params.gravity.y,
											   params.gravity.z };

				// compute predicted velocity from non-pressure accelerations
				velPredF_[i] = velF_[i] + params.timeStep * (accViscF_[i] + accExtF_[i]);
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Viscosity_Acceleration,
								   Bench::BenchOp::External_Acceleration });

	// --- [ source-term for density invariance ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_SourceTerm });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmp_SourceTerm = 0.0f;

				const Eigen::Vector3f velpi = velPredF_[i];

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// predicted velocity difference
					const Eigen::Vector3f dvelpij = velpi - velPredF_[j];

					tmp_SourceTerm += massF * dvelpij.dot(kernelGradFF_[i][k]);
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// TODO: subtract actual boundary velocity when adding non-static boundaries?
					tmp_SourceTerm += massB * velpi.dot(kernelGradFB_[i][k]);
				}

				// finalize source term
				sourceTermF_[i] = params.restDensityF - densityF_[i]
								- params.timeStep * tmp_SourceTerm;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_SourceTerm });

	// --- [ diagonal elements of system matrix ] ------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_DiagonalElem });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmp_DiagElem = 0.0f;

				Eigen::Vector3f cTermF = Eigen::Vector3f::Zero();

				// add c-term contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					cTermF += massF * kernelGradFF_[i][k];
				}

				Eigen::Vector3f cTermB = Eigen::Vector3f::Zero();

				// add c-term contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					cTermB += massB * kernelGradFB_[i][k];
				}

				// finalize c-term
				const Eigen::Vector3f cTerm = -1.0f * (cTermF * restDensitySquInvF)
					- 2.0f * params.gammaPress * (cTermB * restDensitySquInvF);

				// part 1
				for (uint32_t k = 0; k < N_FF; k++)
				{
					const Eigen::Vector3f& grad = kernelGradFF_[i][k];
					tmp_DiagElem += massF * cTerm.dot(grad);
				}

				// part 2
				for (uint32_t k = 0; k < N_FF; k++)
				{
					const Eigen::Vector3f& grad = kernelGradFF_[i][k];
					const Eigen::Vector3f tmp = (massF * restDensitySquInvF) * (-grad);

					tmp_DiagElem += massF * tmp.dot(grad);
				}

				// part 3
				for (uint32_t k = 0; k < N_FB; k++)
				{
					tmp_DiagElem += massB * cTerm.dot(kernelGradFB_[i][k]);
				}

				// finalize diagonal element
				diagElemF_[i] = params.timeStep * params.timeStep * tmp_DiagElem;
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_DiagonalElem });

	// --- [ initialize pressure for jacobi solver ] -----------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_JacobiSolver });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float tmpPressure = 0.0f;

				// only update pressure if particle has neighbors
				if (!Base::ApproxEqu(diagElemF_[i], 0.0f))
				{
					tmpPressure = params.jacobiOmega * (sourceTermF_[i] / diagElemF_[i]);
				}

				// clamp pressure to non-negative values
				pressureF_[i] = fmaxf(tmpPressure, 0.0f);
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ relaxed jacobi solver for pressure ] --------------------------

	int cnt = 0;
	while (cnt < 2) // TODO: add proper termination condition
	{
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const float presi = pressureF_[i];
					const float densi = params.restDensityF;

					Eigen::Vector3f tmp_AccPressF = Eigen::Vector3f::Zero();

					// contributions from fluid neighbors
					const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
					for (uint32_t k = 0; k < N_FF; k++)
					{
						// global index of neighbor particle
						const uint32_t j = neighborsFF_[i][k];

						// pressure acc contribution from fluid neighbor j
						const float accCoeff = massF * (presi / (densi * densi)
											 + pressureF_[j] / (densi * densi));

						tmp_AccPressF += accCoeff * kernelGradFF_[i][k];
					}

					// contributions from boundary neighbors
					const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
					for (uint32_t k = 0; k < N_FB; k++)
					{
						// pressure acc contribution from boundary neighbor j
						const float accCoeff = params.gammaPress * massB
											 * 2.0f * (presi / (densi * densi));

						tmp_AccPressF += accCoeff * kernelGradFB_[i][k];
					}

					// finalize pressure acceleration
					tmp_AccPressF *= -1.0f;

					accPressF_[i] = tmp_AccPressF;
				}
			},
			tbb::auto_partitioner{}
		);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					float tmp_SysVecElem = 0.0f;

					const Eigen::Vector3f accpi = accPressF_[i];

					// contributions from fluid neighbors
					const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
					for (uint32_t k = 0; k < N_FF; k++)
					{
						// global index of neighbor particle
						const uint32_t j = neighborsFF_[i][k];

						// pressure acc difference
						const Eigen::Vector3f daccpij = accpi - accPressF_[j];

						tmp_SysVecElem += massF * daccpij.dot(kernelGradFF_[i][k]);
					}

					// contributions from boundary neighbors
					const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
					for (uint32_t k = 0; k < N_FB; k++)
					{
						// TODO: subtract actual boundary pressure acc when adding non-static boundaries?
						tmp_SysVecElem += massB * accpi.dot(kernelGradFB_[i][k]);
					}

					// finalize system vector element
					sysVecElemF_[i] = params.timeStep * params.timeStep * tmp_SysVecElem;

					float tmpPressure = 0.0f;

					// only update pressure if particle has neighbors
					if (!Base::ApproxEqu(diagElemF_[i], 0.0f))
					{
						tmpPressure = pressureF_[i] + params.jacobiOmega
									* ((sourceTermF_[i] - sysVecElemF_[i]) / diagElemF_[i]);
					}

					// clamp pressure to non-negative values
					pressureF_[i] = fmaxf(tmpPressure, 0.0f);
				}
			},
			tbb::auto_partitioner{}
		);

		cnt++;
	}
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_JacobiSolver });

	// --- [ finalize pressure acceleration for integration ] ------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::Pressure_Acceleration });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const float presi = pressureF_[i];
				const float densi = params.restDensityF;

				Eigen::Vector3f tmp_AccPressF = Eigen::Vector3f::Zero();

				// contributions from fluid neighbors
				const uint32_t N_FF = static_cast<uint32_t>(neighborsFF_[i].size());
				for (uint32_t k = 0; k < N_FF; k++)
				{
					// global index of neighbor particle
					const uint32_t j = neighborsFF_[i][k];

					// pressure acc contribution from fluid neighbor j
					const float accCoeff = massF * (presi / (densi * densi)
										 + pressureF_[j] / (densi * densi));

					tmp_AccPressF += accCoeff * kernelGradFF_[i][k];
				}

				// contributions from boundary neighbors
				const uint32_t N_FB = static_cast<uint32_t>(neighborsFB_[i].size());
				for (uint32_t k = 0; k < N_FB; k++)
				{
					// pressure acc contribution from boundary neighbor j
					const float accCoeff = params.gammaPress * massB
										 * 2.0f * (presi / (densi * densi));

					tmp_AccPressF += accCoeff * kernelGradFB_[i][k];
				}

				// finalize pressure acceleration
				tmp_AccPressF *= -1.0f;

				accPressF_[i] = tmp_AccPressF;

				// finalize total acceleration
				accF_[i] = tmp_AccPressF + accViscF_[i] + accExtF_[i];
			}
		},
		tbb::auto_partitioner{}
	);
	benchmarksMgr_.StopBench({ Bench::BenchOp::Pressure_Acceleration });

	// --- [ velocity and position updates ] -----------------------------------

	benchmarksMgr_.StartBench({ Bench::BenchOp::VelocityUpdate,
									Bench::BenchOp::PositionUpdate });
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// explicit Euler integration
				// (now only refining predicted velocity with pressure acc)
				velF_[i] = velPredF_[i] + params.timeStep * accPressF_[i];

				// semi-implicit Euler integration
				// (using updated velocity for position update)
				posF_[i] = posF_[i] + params.timeStep * velF_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	benchmarksMgr_.StopBench({ Bench::BenchOp::VelocityUpdate,
								   Bench::BenchOp::PositionUpdate });
}
