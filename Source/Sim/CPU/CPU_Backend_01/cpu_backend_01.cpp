// cpu_backend_01.h

// external
#include <algorithm>
#include <cmath>
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <numbers>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "Source/Base/helpers.h"
#include "cpu_backend_01.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"



using namespace Sim;

SimBackend_CPU01::SimBackend_CPU01(Bench::BenchmarksMgr& benchmarksMgr)
	: benchmarksMgr_{ benchmarksMgr }
{
	//neighborSearchStrat_ = std::make_unique<NS_BruteForce_CPU01>(*this);
	//neighborSearchStrat_ = std::make_unique<NS_SpatialHashing_Basic_CPU01>(*this);
	//neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Basic_CPU01>(*this);
	neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Compact_CPU01>(*this);

	//neighborLists_ = std::make_unique<NeighborLists_SmallBuffers_CPU01>();
	//neighborLists_ = std::make_unique<NeighborLists_FixedArrays_CPU01>();
	neighborLists_ = std::make_unique<NeighborLists_CompactVectors_CPU01>();
}

void SimBackend_CPU01::SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat)
{
	neighborSearchStrat_ = std::move(strat);
}

void SimBackend_CPU01::PullFromUniversalSimState(const Sync::UniversalSimState& src)
{
	const auto& F = src.fluidParticles;
	const auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	simStep_ = src.step;

	// resize backend fluid data structures
	xPosF_.resize(N_F_);
	yPosF_.resize(N_F_);
	zPosF_.resize(N_F_);

	xVelF_.resize(N_F_);
	yVelF_.resize(N_F_);
	zVelF_.resize(N_F_);

	xVelPredF_.resize(N_F_);
	yVelPredF_.resize(N_F_);
	zVelPredF_.resize(N_F_);

	xAccF_.resize(N_F_);
	yAccF_.resize(N_F_);
	zAccF_.resize(N_F_);

	xAccPressF_.resize(N_F_);
	yAccPressF_.resize(N_F_);
	zAccPressF_.resize(N_F_);

	xAccViscF_.resize(N_F_);
	yAccViscF_.resize(N_F_);
	zAccViscF_.resize(N_F_);

	xAccExtF_.resize(N_F_);
	yAccExtF_.resize(N_F_);
	zAccExtF_.resize(N_F_);

	sourceTermF_.resize(N_F_);
	diagElemF_.resize(N_F_);
	sysVecElemF_.resize(N_F_);
	pressureF_.resize(N_F_);
	densityF_.resize(N_F_);
	idF_.resize(N_F_);

	// resize backend boundary data structures
	xPosB_.resize(N_B_); yPosB_.resize(N_B_); zPosB_.resize(N_B_);
	xVelB_.resize(N_B_); yVelB_.resize(N_B_); zVelB_.resize(N_B_);
	idB_.resize(N_B_);

	// fluid particles
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				xPosF_[i] = F.xPos[i];
				yPosF_[i] = F.yPos[i];
				zPosF_[i] = F.zPos[i];

				xVelF_[i] = F.xVel[i];
				yVelF_[i] = F.yVel[i];
				zVelF_[i] = F.zVel[i];

				xVelPredF_[i] = F.xVelPred[i];
				yVelPredF_[i] = F.yVelPred[i];
				zVelPredF_[i] = F.zVelPred[i];

				xAccF_[i] = F.xAcc[i];
				yAccF_[i] = F.yAcc[i];
				zAccF_[i] = F.zAcc[i];

				xAccPressF_[i] = F.xAccPress[i];
				yAccPressF_[i] = F.yAccPress[i];
				zAccPressF_[i] = F.zAccPress[i];

				xAccViscF_[i] = F.xAccVisc[i];
				yAccViscF_[i] = F.yAccVisc[i];
				zAccViscF_[i] = F.zAccVisc[i];

				xAccExtF_[i] = F.xAccExt[i];
				yAccExtF_[i] = F.yAccExt[i];
				zAccExtF_[i] = F.zAccExt[i];

				sourceTermF_[i] = F.sourceTerm[i];
				diagElemF_[i] = F.diagElem[i];
				sysVecElemF_[i] = F.sysVecElem[i];
				pressureF_[i] = F.pressure[i];
				densityF_[i] = F.density[i];
				idF_[i] = F.id[i];
			}
		},
		tbb::auto_partitioner{}
	);

	// boundary particles
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				xPosB_[i] = B.xPos[i];
				yPosB_[i] = B.yPos[i];
				zPosB_[i] = B.zPos[i];

				xVelB_[i] = B.xVel[i];
				yVelB_[i] = B.yVel[i];
				zVelB_[i] = B.zVel[i];

				idB_[i] = B.id[i];
			}
		},
		tbb::auto_partitioner{}
	);

	// neighbor lists
	neighborLists_->Resize(N_F_, N_B_);

	//SPDLOG_INFO("Pulled universal sim state with {} fluid and "
	//			"{} boundary particles into data structures of backend CPU01",
	//			N_F_, N_B_);
}

void SimBackend_CPU01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;

	// make sure the destination data structures have enough capacity
	dst.ResizeFluid(N_F_);
	dst.ResizeBoundary(N_B_);

	// fluid particles
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// TODO: also re-order the ones that get overwritten anyway?
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				F.xPos[i] = xPosF_[i];
				F.yPos[i] = yPosF_[i];
				F.zPos[i] = zPosF_[i];

				F.xVel[i] = xVelF_[i];
				F.yVel[i] = yVelF_[i];
				F.zVel[i] = zVelF_[i];

				F.xVelPred[i] = xVelPredF_[i];
				F.yVelPred[i] = yVelPredF_[i];
				F.zVelPred[i] = zVelPredF_[i];

				F.xAcc[i] = xAccF_[i];
				F.yAcc[i] = yAccF_[i];
				F.zAcc[i] = zAccF_[i];

				F.xAccPress[i] = xAccPressF_[i];
				F.yAccPress[i] = yAccPressF_[i];
				F.zAccPress[i] = zAccPressF_[i];

				F.xAccVisc[i] = xAccViscF_[i];
				F.yAccVisc[i] = yAccViscF_[i];
				F.zAccVisc[i] = zAccViscF_[i];

				F.xAccExt[i] = xAccExtF_[i];
				F.yAccExt[i] = yAccExtF_[i];
				F.zAccExt[i] = zAccExtF_[i];

				F.sourceTerm[i] = sourceTermF_[i];
				F.diagElem[i] = diagElemF_[i];
				F.sysVecElem[i] = sysVecElemF_[i];
				F.pressure[i] = pressureF_[i];
				F.density[i] = densityF_[i];
				F.id[i] = idF_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	// boundary particles
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				B.xPos[i] = xPosB_[i];
				B.yPos[i] = yPosB_[i];
				B.zPos[i] = zPosB_[i];

				B.xVel[i] = xVelB_[i];
				B.yVel[i] = yVelB_[i];
				B.zVel[i] = zVelB_[i];

				B.id[i] = idB_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	F.N_ = N_F_;
	B.N_ = N_B_;

	//SPDLOG_INFO("Pushed from data structures of backend CPU01 with "
	//			"{} fluid and {} boundary particles to universal sim state",
	//			N_F_, N_B_);
}

void SimBackend_CPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	// TODO: only export surface particles?

	if (N_F_ == 0)
	{
		snap.simStep = simStep_;
		snap.simTime = simTime_;
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// classify surface particles
	std::vector<uint8_t> surfaceFlags(N_F_);

	const uint32_t N_surface = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_F_), uint32_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint32_t localCount)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// particle considered part of the surface
				// if touching boundary OR low fluid neighbor count
				bool surface = neighborLists_->FB(i).count > 0 ||
							   neighborLists_->FF(i).count < 28;

				// if not surface itself, check its fluid neighbors
				if (!surface)
				{
					NeighborSpan nbrListFF = neighborLists_->FF(i);
					const uint32_t* nbrFF = nbrListFF.data;
					const uint32_t N_FF = nbrListFF.count;

					for (uint32_t k = 0; k < N_FF; ++k)
					{
						const uint32_t j = nbrFF[k];

						if (neighborLists_->FB(j).count > 0 ||
							neighborLists_->FF(j).count < 24)
						{
							surface = true;
							break;
						}
					}
				}

				surfaceFlags[i] = static_cast<uint8_t>(surface);
				if (surface) { localCount++; }
			}

			return localCount;
		},
		[](uint32_t a, uint32_t b) -> uint32_t
		{
			return a + b;
		}
	);

	if (N_surface == 0) // no particles to export
	{
		snap.simStep = simStep_;
		snap.simTime = simTime_;
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// prepare snapshot buffers
	snap.simStep = simStep_;
	snap.simTime = simTime_;
	snap.N_F = N_surface;

	snap.xPosF.resize(N_surface);
	snap.yPosF.resize(N_surface);
	snap.zPosF.resize(N_surface);
	snap.velMagF.resize(N_surface);

	// copy only surface particles
	uint32_t dst = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (!surfaceFlags[i]) { continue; }

		snap.xPosF[dst] = xPosF_[i];
		snap.yPosF[dst] = yPosF_[i];
		snap.zPosF[dst] = zPosF_[i];

		const float vx = xVelF_[i];
		const float vy = yVelF_[i];
		const float vz = zVelF_[i];

		snap.velMagF[dst] = glm::sqrt(vx * vx + vy * vy + vz * vz);

		dst++;
	}

	/*
	if (N_F_ == 0) { return; }

	snap.simStep = simStep_;
	snap.simTime = simTime_;
	snap.N_F = N_F_;

	snap.xPosF.resize(N_F_);
	snap.yPosF.resize(N_F_);
	snap.zPosF.resize(N_F_);
	snap.velMagF.resize(N_F_);

	// pack into the export snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.xPosF[i] = xPosF_[i];
				snap.yPosF[i] = yPosF_[i];
				snap.zPosF[i] = zPosF_[i];
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
				// magnitude of velocity vector
				snap.velMagF[i] = glm::sqrt(xVelF_[i] * xVelF_[i]
											 + yVelF_[i] * yVelF_[i]
											 + zVelF_[i] * zVelF_[i]);
			}
		},
		tbb::auto_partitioner{}
	);
	*/
}

void SimBackend_CPU01::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	// TODO: add boundary particles

	if (N_F_ == 0) { return; }

	snap.simStep = simStep_;
	snap.N_F = N_F_;

	snap.posF.resize(N_F_);
	snap.scalarDataF.resize(N_F_);

	// pack into the visual snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posF[i] = glm::vec3{ xPosF_[i], yPosF_[i], zPosF_[i] };
			}
		},
		tbb::auto_partitioner{}
	);

	FillScalarData(snap, snap.scalarModeF);
}

void SimBackend_CPU01::FillScalarData(Sync::VisualSnapshot& snap,
									  Base::ColorCodingMode mode) const
{
	if (N_F_ == 0) // empty sim
	{
		snap.scalarDataF.clear();

		// TODO: add boundary particles

		return;
	}

	switch (mode)
	{
	case Base::ColorCodingMode::None:
		{
			snap.scalarDataF.clear();
			return;
		}
	case Base::ColorCodingMode::VelocityMag:
		{
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_F_),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						// magnitude of velocity vector
						snap.scalarDataF[i] = glm::sqrt(xVelF_[i] * xVelF_[i]
							+ yVelF_[i] * yVelF_[i]
							+ zVelF_[i] * zVelF_[i]);
					}
				},
				tbb::auto_partitioner{}
			);
			return;
		}
	case Base::ColorCodingMode::Density:
		{
			snap.scalarDataF = densityF_;
			return;
		}
	case Base::ColorCodingMode::Pressure:
		{
			snap.scalarDataF = pressureF_;
			return;
		}
	}

	SPDLOG_ERROR("Unsupported scalar data mode in backend CPU01");
	snap.scalarDataF.clear();
}

void SimBackend_CPU01::FillStatisticsRecord(Stats::StatRecordStep& record,
											const SimParams& simParams) const
{
	using namespace Stats;

	const uint32_t NF = N_F_;
	const uint32_t NB = N_B_;

	Set(record, StatMetric::NumF, static_cast<float>(NF));
	Set(record, StatMetric::NumB, static_cast<float>(NB));

	const float invNF = NF ? 1.0f / static_cast<float>(NF) : 0.0f;

	// --- [ particle counts ] -------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		benchmarksMgr_.StartBench({ Bench::BenchOp::Statistics_Particles });

		float sumFF = 0.0f;
		float nbrFF_p95 = 0.0f;
		float nbrFF_p99 = 0.0f;
		float minFF = std::numeric_limits<float>::max();
		float maxFF = 0.0f;

		float sumFB = 0.0f;
		float nbrFB_p95 = 0.0f;
		float nbrFB_p99 = 0.0f;
		float minFB = std::numeric_limits<float>::max();
		float maxFB = 0.0f;

		if (NF > 0)
		{
			struct NeighStatsLocal
			{
				float sumFF;
				float minFF;
				float maxFF;
				float sumFB;
				float minFB;
				float maxFB;
			};

			NeighStatsLocal globalNeigh = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, NF),
				NeighStatsLocal
				{
					.sumFF = 0.0f,
					.minFF = std::numeric_limits<float>::max(),
					.maxFF = 0.0f,
					.sumFB = 0.0f,
					.minFB = std::numeric_limits<float>::max(),
					.maxFB = 0.0f
				},
				[&](const tbb::blocked_range<uint32_t>& r, NeighStatsLocal local) -> NeighStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const float N_FF = static_cast<float>(neighborLists_->FF(i).count);
						const float N_FB = static_cast<float>(neighborLists_->FB(i).count);

						local.sumFF += N_FF;
						if (N_FF < local.minFF) [[unlikely]] { local.minFF = N_FF; }
						if (N_FF > local.maxFF) [[unlikely]] { local.maxFF = N_FF; }

						local.sumFB += N_FB;
						if (N_FB < local.minFB) [[unlikely]] { local.minFB = N_FB; }
						if (N_FB > local.maxFB) [[unlikely]] { local.maxFB = N_FB; }
					}
					return local;
				},
				[](const NeighStatsLocal& a, const NeighStatsLocal& b) -> NeighStatsLocal
				{
					NeighStatsLocal r;

					r.sumFF = a.sumFF + b.sumFF;
					r.minFF = std::min(a.minFF, b.minFF);
					r.maxFF = std::max(a.maxFF, b.maxFF);
					r.sumFB = a.sumFB + b.sumFB;
					r.minFB = std::min(a.minFB, b.minFB);
					r.maxFB = std::max(a.maxFB, b.maxFB);

					return r;
				},
				tbb::auto_partitioner{}
			);

			sumFF = globalNeigh.sumFF;
			minFF = globalNeigh.minFF;
			maxFF = globalNeigh.maxFF;
			sumFB = globalNeigh.sumFB;
			minFB = globalNeigh.minFB;
			maxFB = globalNeigh.maxFB;
		}
		if (NF == 0) // reasonable defaults for empty sim
		{
			minFF = 0.0f;
			minFB = 0.0f;
		}

		if (NF > 0) // determine percentiles
		{
			// reuse workBuffer_ for neighbor counts
			workBuffer_.resize(NF);

			// FF
			for (uint32_t i = 0; i < NF; i++)
			{
				workBuffer_[i] = static_cast<float>(neighborLists_->FF(i).count);
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBuffer_.begin();
				auto end = workBuffer_.end();
				auto nth95 = begin + i95;
				auto nth99 = begin + i99;

				// select p95 within the full range
				std::nth_element(begin, nth95, end);
				nbrFF_p95 = *nth95;

				if (i95 == i99) // handle same index edge case
				{
					nbrFF_p99 = *nth95;
				}
				else
				{
					// select p99 only within the upper part
					std::nth_element(nth95, nth99, end);
					nbrFF_p99 = *nth99;
				}
			}

			// FB
			for (uint32_t i = 0; i < NF; ++i)
			{
				workBuffer_[i] = static_cast<float>(neighborLists_->FB(i).count);
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBuffer_.begin();
				auto end = workBuffer_.end();
				auto nth95 = begin + i95;
				auto nth99 = begin + i99;

				// select p95 within the full range
				std::nth_element(begin, nth95, end);
				nbrFB_p95 = *nth95;

				if (i95 == i99) // handle same index edge case
				{
					nbrFB_p99 = *nth95;
				}
				else
				{
					// select p99 only within the upper part
					std::nth_element(nth95, nth99, end);
					nbrFB_p99 = *nth99;
				}
			}
		}
		else // reasonable defaults for empty sim
		{
			nbrFF_p95 = nbrFF_p99 = 0.0f;
			nbrFB_p95 = nbrFB_p99 = 0.0f;
		}

		Set(record, StatMetric::NumNbrFF_Avg, sumFF * invNF);
		Set(record, StatMetric::NumNbrFF_P95, nbrFF_p95);
		Set(record, StatMetric::NumNbrFF_P99, nbrFF_p99);
		Set(record, StatMetric::NumNbrFF_Min, minFF);
		Set(record, StatMetric::NumNbrFF_Max, maxFF);

		Set(record, StatMetric::NumNbrFB_Avg, sumFB * invNF);
		Set(record, StatMetric::NumNbrFB_P95, nbrFB_p95);
		Set(record, StatMetric::NumNbrFB_P99, nbrFB_p99);
		Set(record, StatMetric::NumNbrFB_Min, minFB);
		Set(record, StatMetric::NumNbrFB_Max, maxFB);

		benchmarksMgr_.StopBench({ Bench::BenchOp::Statistics_Particles });
	}

	// --- [ density stats ] ---------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)])
	{
		benchmarksMgr_.StartBench({ Bench::BenchOp::Statistics_Density });

		// TODO: use double for the accumulations?
		float sumRelErr_raw = 0.0f;
		float sumRelErr_pos = 0.0f;
		float sumRelErr_abs = 0.0f;

		if (NF > 0)
		{
			struct DensStatsLocal
			{
				float sumRelErr_raw;
				float sumRelErr_pos;
				float sumRelErr_abs;
			};

			DensStatsLocal globalDens = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, NF),
				DensStatsLocal
				{
					.sumRelErr_raw = 0.0f,
					.sumRelErr_pos = 0.0f,
					.sumRelErr_abs = 0.0f
				},
				[&](const tbb::blocked_range<uint32_t>& r, DensStatsLocal local) -> DensStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						// relative density error (aka deviation)
						const float relErr = (densityF_[i] / simParams.restDensityF) - 1.0f;
						const float relPosErr = std::max(0.0f, relErr);
						const float relAbsErr = std::fabs(relErr);

						local.sumRelErr_raw += relErr;
						local.sumRelErr_pos += relPosErr;
						local.sumRelErr_abs += relAbsErr;
					}

					return local;
				},
				[](const DensStatsLocal& a, const DensStatsLocal& b) -> DensStatsLocal
				{
					DensStatsLocal r;

					r.sumRelErr_raw = a.sumRelErr_raw + b.sumRelErr_raw;
					r.sumRelErr_pos = a.sumRelErr_pos + b.sumRelErr_pos;
					r.sumRelErr_abs = a.sumRelErr_abs + b.sumRelErr_abs;

					return r;
				},
				tbb::auto_partitioner{}
			);

			sumRelErr_raw = globalDens.sumRelErr_raw;
			sumRelErr_pos = globalDens.sumRelErr_pos;
			sumRelErr_abs = globalDens.sumRelErr_abs;
		}

		Set(record, StatMetric::DensRelError_Raw_Avg, sumRelErr_raw * invNF);
		Set(record, StatMetric::DensRelError_Pos_Avg, sumRelErr_pos * invNF);
		Set(record, StatMetric::DensRelError_Abs_Avg, sumRelErr_abs * invNF);

		benchmarksMgr_.StopBench({ Bench::BenchOp::Statistics_Density });
	}

	// --- [ pressure stats ] --------------------------------------------------

	Set(record, StatMetric::NumSolverIters, N_solverIterCount_);

	// --- [ velocity stats ] --------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
		benchmarksMgr_.StartBench({ Bench::BenchOp::Statistics_Velocity });

		float velMagSum = 0.0f;
		float velMagMin = std::numeric_limits<float>::max();
		float velMagMax = 0.0f;

		if (NF > 0)
		{
			struct VelStatsLocal
			{
				float velMagSum;
				float velMagMin;
				float velMagMax;
			};

			VelStatsLocal globalVel = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, NF),
				VelStatsLocal
				{
					.velMagSum = 0.0f,
					.velMagMin = std::numeric_limits<float>::max(),
					.velMagMax = 0.0f
				},
				[&](const tbb::blocked_range<uint32_t>& r, VelStatsLocal local) -> VelStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const float vx = xVelF_[i];
						const float vy = yVelF_[i];
						const float vz = zVelF_[i];

						const float velMag = std::sqrt(vx * vx + vy * vy + vz * vz);

						local.velMagSum += velMag;
						if (velMag < local.velMagMin) [[unlikely]] { local.velMagMin = velMag; }
						if (velMag > local.velMagMax) [[unlikely]] { local.velMagMax = velMag; }
					}

					return local;
				},
				[](const VelStatsLocal& a, const VelStatsLocal& b) -> VelStatsLocal
				{
					VelStatsLocal r;

					r.velMagSum = a.velMagSum + b.velMagSum;
					r.velMagMin = std::min(a.velMagMin, b.velMagMin);
					r.velMagMax = std::max(a.velMagMax, b.velMagMax);

					return r;
				},
				tbb::auto_partitioner{}
			);

			velMagSum = globalVel.velMagSum;
			velMagMin = globalVel.velMagMin;
			velMagMax = globalVel.velMagMax;
		}
		if (NF == 0) // reasonable defaults for empty sim
		{
			velMagMin = 0.0f;
			velMagMax = 0.0f;
		}

		Set(record, StatMetric::VelMag_Avg, velMagSum * invNF);
		Set(record, StatMetric::VelMag_Min, velMagMin);
		Set(record, StatMetric::VelMag_Max, velMagMax);

		benchmarksMgr_.StopBench({ Bench::BenchOp::Statistics_Velocity });
	}
}

void SimBackend_CPU01::ExecuteSimStep(const SimParams& params, uint32_t N_steps)
{
	if (N_F_ == 0) // empty sim
	{
		simStep_ += N_steps;
		simTime_ += static_cast<double>(N_steps) * params.timeStep;

		return;
	}

	Step_IISPH(params);

	simStep_ += N_steps;
	simTime_ += static_cast<double>(N_steps) * params.timeStep;
}
