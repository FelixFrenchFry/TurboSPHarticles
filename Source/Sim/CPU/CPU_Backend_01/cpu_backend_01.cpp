// cpu_backend_01.h

// external
#include <algorithm>
#include <cmath>
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "Source/Base/helpers.h"
#include "cpu_backend_01.h"
#include "neighbor_lists.h"
#include "neighbor_search.h"



using namespace Sim;

// ---------------------------------------------------------------------------------------------------

SimBackend_CPU01::SimBackend_CPU01(
	Sim::SimParamsMgr& simParamsMgr,
	Bench::BenchmarksTracker& benchmarksTracker,
	const SimParams& params)
	: simParamsMgr_{ simParamsMgr }, benchmarksTracker_{ benchmarksTracker }
{
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>();

	// select neighbor search method
	if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::BruteForce)
	{
		neighborSearch_ = std::make_unique<NS_BruteForce_CPU01>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::CellLinkedList_Compact)
	{
		neighborSearch_ = std::make_unique<NS_CellLinkedList_Compact_CPU01>(*this);
	}
	else
	{
		throw std::runtime_error("Unsupported 'NeighborSearchMethod' requested.");
	}
	neighborSearch_->Init(params);

	// select neighbor list method
	if (params.neighborListMethod ==
		Base::NeighborListMethod::ChunkedSELLX_Explicit)
	{
		neighborList_ = std::make_unique<NeighborLists_ChunkedSELL_Explicit_CPU01>();
	}
	else if (params.neighborListMethod ==
		Base::NeighborListMethod::ChunkedSELLX_AdaptCompress)
	{
		neighborList_ = std::make_unique<NeighborLists_ChunkedSELL_AdaptCompress_CPU01>();
	}
	else
	{
		throw std::runtime_error(
			"CPU01 supports only ChunkedSELL neighbor-list storage.");
	}
}

void SimBackend_CPU01::PullFromUniversalSimState(
	Sync::UniversalSimState& src, const SimParams& params)
{
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping

	auto& F = src.fluidParticles;
	auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	simStep_ = src.step;
	simTime_ = src.time;
	F.ensureParticleIdsPresent();
	B.ensureParticleIdsPresent();
	B.EnsureBoundaryCoefficientsPresent(params.massB, params.restDensity);

	// transfer to backend fluid data structures
	idF_ = std::move(F.id);
	xPosF_ = std::move(F.xPos);
	yPosF_ = std::move(F.yPos);
	zPosF_ = std::move(F.zPos);

	xVelF_ = std::move(F.xVel);
	yVelF_ = std::move(F.yVel);
	zVelF_ = std::move(F.zVel);

	xVelPredF_ = std::move(F.xVelPred);
	yVelPredF_ = std::move(F.yVelPred);
	zVelPredF_ = std::move(F.zVelPred);

	xAccPressF_ = std::move(F.xAccPress);
	yAccPressF_ = std::move(F.yAccPress);
	zAccPressF_ = std::move(F.zAccPress);

	sourceTermF_ = std::move(F.sourceTerm);
	diagElemF_ = std::move(F.diagElem);
	pressureF_ = std::move(F.pressure);
	densityF_ = std::move(F.density);

	// transfer to boundary data structures
	idB_ = std::move(B.id);
	xPosB_ = std::move(B.xPos);
	yPosB_ = std::move(B.yPos);
	zPosB_ = std::move(B.zPos);
	volB_ = std::move(B.volB);
	psiB_ = std::move(B.psiB);

	// init neighbor list sizes
	neighborList_->Resize(N_F_, N_B_, params);
}

// TODO: re-implement this to free up memory as soon as copied out!
void SimBackend_CPU01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;

	dst.step = simStep_;
	dst.time = simTime_;

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
				F.id[i] = idF_[i];

				F.xPos[i] = xPosF_[i];
				F.yPos[i] = yPosF_[i];
				F.zPos[i] = zPosF_[i];

				F.xVel[i] = xVelF_[i];
				F.yVel[i] = yVelF_[i];
				F.zVel[i] = zVelF_[i];

				F.xVelPred[i] = xVelPredF_[i];
				F.yVelPred[i] = yVelPredF_[i];
				F.zVelPred[i] = zVelPredF_[i];

				F.xAccPress[i] = xAccPressF_[i];
				F.yAccPress[i] = yAccPressF_[i];
				F.zAccPress[i] = zAccPressF_[i];

				F.sourceTerm[i] = sourceTermF_[i];
				F.diagElem[i] = diagElemF_[i];
				F.pressure[i] = pressureF_[i];
				F.density[i] = densityF_[i];
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
				B.id[i] = idB_[i];
				B.xPos[i] = xPosB_[i];
				B.yPos[i] = yPosB_[i];
				B.zPos[i] = zPosB_[i];
				B.volB[i] = volB_[i];
				B.psiB[i] = psiB_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	F.N_ = N_F_;
	B.N_ = N_B_;
}

void SimBackend_CPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	snap.simStep = simStep_;
	snap.simTime = simTime_;

	snap.N_B = N_B_;
	snap.idB.resize(N_B_);
	snap.xPosB.resize(N_B_);
	snap.yPosB.resize(N_B_);
	snap.zPosB.resize(N_B_);
	snap.volB.resize(N_B_);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.idB[i] = idB_[i];
				snap.xPosB[i] = xPosB_[i];
				snap.yPosB[i] = yPosB_[i];
				snap.zPosB[i] = zPosB_[i];
				snap.volB[i] = volB_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	if (N_F_ == 0)
	{
		snap.N_F = 0;
		snap.idF.clear();
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// classify surface particles
	auto& surfaceFlags = expSurfaceFlagsScratch_;
	surfaceFlags.resize(N_F_);

	const uint32_t N_surface = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N_F_), uint32_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint32_t localCount)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// particle considered part of the surface
				// if touching boundary OR low fluid neighbor count
				bool surface = neighborList_->CountFB(i) > 0 ||
							   neighborList_->CountFF(i) < 28;

				// if not surface itself, check its fluid neighbors
				if (!surface)
				{
					NeighborSpan nbrListFF = neighborList_->FF(i);
					const uint32_t* nbrFF = nbrListFF.data;
					const uint32_t N_FF = nbrListFF.count;

					for (uint32_t k = 0; k < N_FF; k++)
					{
						const uint32_t j = nbrFF[k];

						if (neighborList_->CountFB(j) > 0 ||
							neighborList_->CountFF(j) < 24)
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
		snap.N_F = 0;
		snap.idF.clear();
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// prepare snapshot buffers
	snap.N_F = N_surface;

	snap.idF.resize(N_surface);
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
		snap.idF[dst] = idF_[i];

		const float vx = xVelF_[i];
		const float vy = yVelF_[i];
		const float vz = zVelF_[i];

		snap.velMagF[dst] = glm::sqrt(vx * vx + vy * vy + vz * vz);

		dst++;
	}
}

void SimBackend_CPU01::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	snap.simStep = simStep_;
	snap.N_F = N_F_;
	snap.N_B = N_B_;

	if (N_F_ == 0)
	{
		snap.posF.clear();
		snap.scalarDataF.clear();
	}
	else
	{
		snap.posF.resize(N_F_);
		snap.scalarDataF.resize(N_F_);

		// pack fluid positions into the visual snapshot's data structures
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

	snap.posB.resize(N_B_);

	// pack boundary positions into the visual snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posB[i] = glm::vec3{ xPosB_[i], yPosB_[i], zPosB_[i] };
			}
		},
		tbb::auto_partitioner{}
	);
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

void SimBackend_CPU01::FillMemoryRecord(
	Mem::MemRecordStep& record,
	const SimParams& params) const
{
	using namespace Mem;
	constexpr MemSpace memSpace = MemSpace::RAM;

	// --- [ particle properties ] -------------------------------------------------------------------

	// fluid
	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(idF_));
		Add(usage, MemUsageOf_Vector(xPosF_));
		Add(usage, MemUsageOf_Vector(yPosF_));
		Add(usage, MemUsageOf_Vector(zPosF_));

		Add(usage, MemUsageOf_Vector(xVelF_));
		Add(usage, MemUsageOf_Vector(yVelF_));
		Add(usage, MemUsageOf_Vector(zVelF_));

		Add(usage, MemUsageOf_Vector(xVelPredF_));
		Add(usage, MemUsageOf_Vector(yVelPredF_));
		Add(usage, MemUsageOf_Vector(zVelPredF_));

		Add(usage, MemUsageOf_Vector(xAccPressF_));
		Add(usage, MemUsageOf_Vector(yAccPressF_));
		Add(usage, MemUsageOf_Vector(zAccPressF_));

		Add(usage, MemUsageOf_Vector(sourceTermF_));
		Add(usage, MemUsageOf_Vector(diagElemF_));
		Add(usage, MemUsageOf_Vector(pressureF_));
		Add(usage, MemUsageOf_Vector(densityF_));

		// finalize
		Set(record, memSpace, MemBucket::ParticlePropertiesF, usage);
	}

	// boundary
	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(idB_));
		Add(usage, MemUsageOf_Vector(xPosB_));
		Add(usage, MemUsageOf_Vector(yPosB_));
		Add(usage, MemUsageOf_Vector(zPosB_));
		Add(usage, MemUsageOf_Vector(volB_));
		Add(usage, MemUsageOf_Vector(psiB_));

		// finalize
		Set(record, memSpace, MemBucket::ParticlePropertiesB, usage);
	}

	// --- [ neighbor search data structures ] -------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		std::pair<MemUsage, MemUsage> usageFB = neighborSearch_->GetMemUsage(params);

		Add(usageF, usageFB.first);
		Add(usageB, usageFB.second);

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructF, usageF);
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructB, usageB);
	}

	// --- [ neighbor list storage ] -----------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		std::pair<MemUsage, MemUsage> usageFB = neighborList_->GetMemUsage(params);

		Add(usageF, usageFB.first);
		Add(usageB, usageFB.second);

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_ListStorageF, usageF);
		Set(record, memSpace, MemBucket::Neighbors_ListStorageB, usageB);
	}

	// --- [ misc ] ----------------------------------------------------------------------------------

	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(workBufferF_));
		Add(usage, MemUsageOf_Vector(workBufferB_));
		Add(usage, MemUsageOf_Vector(workBufferIdF_));
		Add(usage, MemUsageOf_Vector(workBufferIdB_));
		Add(usage, MemUsageOf_Vector(expSurfaceFlagsScratch_));

		// finalize
		Set(record, memSpace, MemBucket::Misc, usage);
	}
}

void SimBackend_CPU01::FillStatisticsRecord(
	Stats::StatRecordStep& record,
	const SimParams& simParams) const
{
	using namespace Stats;

	const uint32_t NF = N_F_;
	const uint32_t NB = N_B_;

	Set(record, StatMetric::NumF, static_cast<float>(NF));
	Set(record, StatMetric::NumB, static_cast<float>(NB));

	const float invNF = NF ? 1.0f / static_cast<float>(NF) : 0.0f;

	benchmarksTracker_.StartBench({
		Bench::BenchOp::Statistics_Particles,
		Bench::BenchOp::Statistics_Density,
		Bench::BenchOp::Statistics_Velocity });

	// --- [ particle counts ] -----------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
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
						const float N_FF = static_cast<float>(neighborList_->CountFF(i));
						const float N_FB = static_cast<float>(neighborList_->CountFB(i));

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
			// reuse workBufferF_ for neighbor counts
			workBufferF_.resize(NF);

			// FF
			for (uint32_t i = 0; i < NF; i++)
			{
				workBufferF_[i] = static_cast<float>(neighborList_->CountFF(i));
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBufferF_.begin();
				auto end = workBufferF_.end();
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
				workBufferF_[i] = static_cast<float>(neighborList_->CountFB(i));
			}

			{
				// indices of the 95th and 99th percentiles
				const size_t i95 = (95ull * (NF - 1)) / 100ull;
				const size_t i99 = (99ull * (NF - 1)) / 100ull;

				auto begin = workBufferF_.begin();
				auto end = workBufferF_.end();
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
	}

	// --- [ density stats ] -------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)])
	{
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
						const float relErr = (densityF_[i] / simParams.restDensity) - 1.0f;
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
	}

	// --- [ pressure stats ] ------------------------------------------------------------------------

	Set(record, StatMetric::NumSolverIters, N_solverIterCount_);

	// --- [ velocity stats ] ------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
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

		Set(record, StatMetric::CFL_Number, velMagMax* simParams.timeStep / simParams.particleSizeF);
		Set(record, StatMetric::VelMag_Avg, velMagSum * invNF);
		Set(record, StatMetric::VelMag_Min, velMagMin);
		Set(record, StatMetric::VelMag_Max, velMagMax);
	}

	benchmarksTracker_.StopBench({
		Bench::BenchOp::Statistics_Particles,
		Bench::BenchOp::Statistics_Density,
		Bench::BenchOp::Statistics_Velocity });
}

void SimBackend_CPU01::ExecuteNeighborSearch(const SimParams& params, const bool bench)
{
	const bool runCountQuery = neighborList_->RunCountQuery();
	const bool runFillQuery = neighborList_->RunFillQuery();

	// initial re-ordering of the data before the initial neighbor search
	if (bench)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Data_Ordering });
	}
	neighborSearch_->DataReOrder(params);
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
	if (bench)
	{
		benchmarksTracker_.StopBench({ Bench::BenchOp::Data_Ordering });
	}

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
	neighborList_->Finalize_Count();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });

	if (runFillQuery)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Query });
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Query });
	}

	benchmarksTracker_.StartBench({ Bench::BenchOp::Neighbors_Construction });
	neighborList_->Finalize_Fill();
	benchmarksTracker_.StopBench({ Bench::BenchOp::Neighbors_Construction });
}

bool SimBackend_CPU01::TryGetLastParticleReOrder(
	std::vector<uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
	std::vector<uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
{
	if (!neighborSearch_) { return false; }

	return neighborSearch_->TryGetLastParticleReOrder(
		oldFluidLocalIdx_FromNewLocalIdx,
		oldBoundaryLocalIdx_FromNewLocalIdx);
}

bool SimBackend_CPU01::SetActiveFluidParticleLocalIdxs(
	const std::vector<int>& localIdxs)
{
	activeFluidParticleLocalIdxs_.clear();
	activeFluidParticleLocalIdxs_.reserve(localIdxs.size());

	std::vector<uint8_t> activeIdxUsed(N_F_, 0);
	for (const int localIdx : localIdxs)
	{
		if (localIdx < 0 || static_cast<uint32_t>(localIdx) >= N_F_)
		{
			SPDLOG_ERROR("Invalid active fluid particle local index ({})", localIdx);
			useActiveFluidParticleLocalIdxs_ = false;
			activeFluidParticleLocalIdxs_.clear();
			return false;
		}

		const uint32_t localIdxU = static_cast<uint32_t>(localIdx);
		if (activeIdxUsed[localIdxU] != 0)
		{
			SPDLOG_ERROR("Duplicate active fluid particle local index ({})", localIdx);
			useActiveFluidParticleLocalIdxs_ = false;
			activeFluidParticleLocalIdxs_.clear();
			return false;
		}

		activeIdxUsed[localIdxU] = 1;
		activeFluidParticleLocalIdxs_.push_back(localIdxU);
	}

	useActiveFluidParticleLocalIdxs_ = true;
	return true;
}

void SimBackend_CPU01::ClearActiveFluidParticleLocalIdxs()
{
	useActiveFluidParticleLocalIdxs_ = false;
	activeFluidParticleLocalIdxs_.clear();
}

void SimBackend_CPU01::ApplyLastParticleReOrderToActiveFluidParticleLocalIdxs()
{
	if (!useActiveFluidParticleLocalIdxs_) { return; }
	if (!neighborSearch_) { return; }

	std::vector<uint32_t> oldFluidLocalIdx_FromNewLocalIdx;
	std::vector<uint32_t> oldBoundaryLocalIdx_FromNewLocalIdx;
	if (!neighborSearch_->TryGetLastParticleReOrder(
		oldFluidLocalIdx_FromNewLocalIdx,
		oldBoundaryLocalIdx_FromNewLocalIdx))
	{
		return;
	}

	if (oldFluidLocalIdx_FromNewLocalIdx.size() != static_cast<uint64_t>(N_F_))
	{
		throw std::runtime_error("Invalid CPU01 fluid particle re-order size.");
	}

	std::vector<uint32_t> newFluidLocalIdx_FromOldLocalIdx(N_F_);
	for (uint64_t newLocalIdx = 0; newLocalIdx < oldFluidLocalIdx_FromNewLocalIdx.size(); newLocalIdx++)
	{
		const uint32_t oldLocalIdx = oldFluidLocalIdx_FromNewLocalIdx[newLocalIdx];
		if (oldLocalIdx >= N_F_)
		{
			throw std::runtime_error("Invalid CPU01 fluid particle re-order index.");
		}

		newFluidLocalIdx_FromOldLocalIdx[oldLocalIdx] =
			static_cast<uint32_t>(newLocalIdx);
	}

	for (uint32_t& localIdx : activeFluidParticleLocalIdxs_)
	{
		if (localIdx >= N_F_)
		{
			throw std::runtime_error("Invalid active CPU01 fluid particle local index.");
		}

		localIdx = newFluidLocalIdx_FromOldLocalIdx[localIdx];
	}
}

void SimBackend_CPU01::ExecuteFullSimStep(const SimParams& params, uint32_t N_steps)
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
