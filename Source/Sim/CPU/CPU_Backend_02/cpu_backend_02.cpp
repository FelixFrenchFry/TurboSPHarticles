// cpu_backend_02.cpp

// external
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <numbers>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "Source/Base/helpers.h"
#include "cpu_backend_02.h"
#include "neighbor_search.h"



constexpr float PI = std::numbers::pi_v<float>;
using namespace Sim;

SimBackend_CPU02::SimBackend_CPU02(Bench::BenchmarksMgr& benchmarksMgr)
	: benchmarksMgr_{ benchmarksMgr }
{
	//neighborSearchStrat_ = std::make_unique<NS_BruteForce_CPU02>(*this);
	//neighborSearchStrat_ = std::make_unique<NS_SpatialHashing_Basic_CPU02>(*this);
	//neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Basic_CPU02>(*this);
	neighborSearchStrat_ = std::make_unique<NS_CellLinkedList_Compact_CPU02>(*this);
}

void SimBackend_CPU02::SetNeighborSearchStrategy(std::unique_ptr<I_NeighborSearchStrat> strat)
{
	neighborSearchStrat_ = std::move(strat);
}

void SimBackend_CPU02::PullFromUniversalSimState(const Sync::UniversalSimState& src)
{
	const auto& F = src.fluidParticles;
	const auto& B = src.boundaryParticles;
	N_F_ = F.N_;
	N_B_ = B.N_;
	simStep_ = src.step;

	// resize backend fluid data structures
	posF_.resize(N_F_);
	velF_.resize(N_F_);
	velPredF_.resize(N_F_);
	accF_.resize(N_F_);
	accPressF_.resize(N_F_);
	accViscF_.resize(N_F_);
	accExtF_.resize(N_F_);

	sourceTermF_.resize(N_F_);
	diagElemF_.resize(N_F_);
	sysVecElemF_.resize(N_F_);
	pressureF_.resize(N_F_);
	densityF_.resize(N_F_);
	idF_.resize(N_F_);

	// resize backend boundary data structures
	posB_.resize(N_B_);
	velB_.resize(N_B_);
	idB_.resize(N_B_);

	// fluid particles
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				posF_[i] = Eigen::Vector3f{ F.xPos[i], F.yPos[i], F.zPos[i] };
				velF_[i] = Eigen::Vector3f{ F.xVel[i], F.yVel[i], F.zVel[i] };
				velPredF_[i] = Eigen::Vector3f{ F.xVelPred[i], F.yVelPred[i], F.zVelPred[i] };
				accF_[i] = Eigen::Vector3f{ F.xAcc[i], F.yAcc[i], F.zAcc[i] };
				accPressF_[i] = Eigen::Vector3f{ F.xAccPress[i], F.yAccPress[i], F.zAccPress[i] };
				accViscF_[i] = Eigen::Vector3f{ F.xAccVisc[i], F.yAccVisc[i], F.zAccVisc[i] };
				accExtF_[i] = Eigen::Vector3f{ F.xAccExt[i], F.yAccExt[i], F.zAccExt[i] };

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
				posB_[i] = Eigen::Vector3f{ B.xPos[i], B.yPos[i], B.zPos[i] };
				velB_[i] = Eigen::Vector3f{ B.xVel[i], B.yVel[i], B.zVel[i] };
				idB_[i] = B.id[i];
			}
		},
		tbb::auto_partitioner{}
	);

	//SPDLOG_INFO("Pulled universal sim state with {} fluid and "
	//			"{} boundary particles into data structures of backend CPU02",
	//			N_F_, N_B_);
}

void SimBackend_CPU02::PushToUniversalSimState(Sync::UniversalSimState& dst) const
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
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const Eigen::Vector3f p = posF_[i];
				const Eigen::Vector3f v = velF_[i];
				const Eigen::Vector3f vp = velPredF_[i];
				const Eigen::Vector3f a = accF_[i];
				const Eigen::Vector3f ap = accPressF_[i];
				const Eigen::Vector3f av = accViscF_[i];
				const Eigen::Vector3f ae = accExtF_[i];

				F.xPos[i] = p.x(); F.yPos[i] = p.y(); F.zPos[i] = p.z();

				F.xVel[i] = v.x(); F.yVel[i] = v.y(); F.zVel[i] = v.z();

				F.xVelPred[i] = vp.x(); F.yVelPred[i] = vp.y(); F.zVelPred[i] = vp.z();

				F.xAcc[i] = a.x(); F.yAcc[i] = a.y(); F.zAcc[i] = a.z();

				F.xAccPress[i] = ap.x(); F.yAccPress[i] = ap.y(); F.zAccPress[i] = ap.z();

				F.xAccVisc[i] = av.x(); F.yAccVisc[i] = av.y(); F.zAccVisc[i] = av.z();

				F.xAccExt[i] = ae.x(); F.yAccExt[i] = ae.y(); F.zAccExt[i] = ae.z();

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
				const Eigen::Vector3f p = posB_[i];
				const Eigen::Vector3f v = velB_[i];

				B.xPos[i] = p.x(); B.yPos[i] = p.y(); B.zPos[i] = p.z();
				B.xVel[i] = v.x(); B.yVel[i] = v.y(); B.zVel[i] = v.z();
				B.id[i] = idB_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	F.N_ = N_F_;
	B.N_ = N_B_;

	//SPDLOG_INFO("Pushed from data structures of backend CPU02 with "
	//			"{} fluid and {} boundary particles to universal sim state",
	//			N_F_, N_B_);
}

void SimBackend_CPU02::PackExportSnapshot(Sync::ExportSnapshot& snap) const
{
	// TODO: implement
}

void SimBackend_CPU02::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	// TODO: add boundary particles

	snap.N_F = N_F_;
	snap.simStep = simStep_;

	if (N_F_ == 0) // empty sim
	{
		snap.posF.clear();
		snap.scalarDataF.clear();

		// TODO: add boundary particles

		return;
	}

	snap.posF.resize(N_F_);
	snap.scalarDataF.resize(N_F_);

	// pack into the visual snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const Eigen::Vector3f p = posF_[i];

				snap.posF[i] = glm::vec3{ p.x(), p.y(), p.z() };
			}
		},
		tbb::auto_partitioner{}
	);

	FillScalarData(snap, snap.scalarModeF);
}

void SimBackend_CPU02::FillScalarData(Sync::VisualSnapshot& snap,
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
	case Base::None:
	{
		snap.scalarDataF.clear();
		return;
	}
	case Base::VelocityMag:
	{
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const Eigen::Vector3f v = velF_[i];

					// magnitude of velocity vector
					snap.scalarDataF[i] = v.norm();
				}
			},
			tbb::auto_partitioner{}
		);
		return;
	}
	case Base::Density:
	{
		snap.scalarDataF = densityF_;
		return;
	}
	case Base::Pressure:
	{
		snap.scalarDataF = pressureF_;
		return;
	}
	default:
		SPDLOG_ERROR("Unsupported scalar data mode in backend CPU02");
		snap.scalarDataF.clear();
		return;
	}
}

// TODO: add flags to enable/disable certain stats?
void SimBackend_CPU02::FillStatisticsRecord(Stats::StatRecordStep& record,
											const SimParams& simParams) const
{
	using namespace Stats;

	// --- [ particle counts ] -------------------------------------------------

	const uint32_t NF = N_F_;
	const uint32_t NB = N_B_;

	Set(record, StatMetric::NumF, static_cast<float>(NF));
	Set(record, StatMetric::NumB, static_cast<float>(NB));

	const float invNF = NF ? 1.0f / static_cast<float>(NF) : 0.0f;

	benchmarksMgr_.StartBench({ Bench::BenchOp::Statistics_Particles,
									Bench::BenchOp::Statistics_Density,
									Bench::BenchOp::Statistics_Velocity });

	float sumFF = 0.0f;
	float sumFB = 0.0f;
	float minFF = std::numeric_limits<float>::max();
	float minFB = std::numeric_limits<float>::max();
	float maxFF = 0.0f;
	float maxFB = 0.0f;

	if (NF > 0)
	{
		struct NeighStatsLocal
		{
			float sumFF;
			float sumFB;
			float minFF;
			float minFB;
			float maxFF;
			float maxFB;
		};

		NeighStatsLocal globalNeigh = tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, NF),
			NeighStatsLocal
			{
				.sumFF = 0.0f,
				.sumFB = 0.0f,
				.minFF = std::numeric_limits<float>::max(),
				.minFB = std::numeric_limits<float>::max(),
				.maxFF = 0.0f,
				.maxFB = 0.0f
			},
			[&](const tbb::blocked_range<uint32_t>& r, NeighStatsLocal local) -> NeighStatsLocal
			{
				for (uint32_t i = r.begin(); i != r.end(); ++i)
				{
					const float nf = static_cast<float>(neighborsFF_[i].size());
					const float nb = static_cast<float>(neighborsFB_[i].size());

					local.sumFF += nf;
					local.sumFB += nb;

					if (nf < local.minFF) [[unlikely]] { local.minFF = nf; }
					if (nb < local.minFB) [[unlikely]] { local.minFB = nb; }

					if (nf > local.maxFF) [[unlikely]] { local.maxFF = nf; }
					if (nb > local.maxFB) [[unlikely]] { local.maxFB = nb; }
				}
				return local;
			},
			[](const NeighStatsLocal& a, const NeighStatsLocal& b) -> NeighStatsLocal
			{
				NeighStatsLocal r;

				r.sumFF = a.sumFF + b.sumFF;
				r.sumFB = a.sumFB + b.sumFB;
				r.minFF = std::min(a.minFF, b.minFF);
				r.minFB = std::min(a.minFB, b.minFB);
				r.maxFF = std::max(a.maxFF, b.maxFF);
				r.maxFB = std::max(a.maxFB, b.maxFB);

				return r;
			},
			tbb::auto_partitioner{}
		);

		sumFF = globalNeigh.sumFF;
		sumFB = globalNeigh.sumFB;
		minFF = globalNeigh.minFF;
		minFB = globalNeigh.minFB;
		maxFF = globalNeigh.maxFF;
		maxFB = globalNeigh.maxFB;
	}
	if (NF == 0) // reasonable defaults for empty sim
	{
		minFF = 0.0f;
		minFB = 0.0f;
	}

	// --- [ density stats ] ---------------------------------------------------

	workBuffer_.resize(NF);

	float sumRelErr = 0.0f;
	float minRelErr = std::numeric_limits<float>::max();
	float maxRelErr = 0.0f;

	if (NF > 0)
	{
		struct DensStatsLocal
		{
			float sumRelErr;
			float minRelErr;
			float maxRelErr;
		};

		DensStatsLocal globalDens = tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, NF),
			DensStatsLocal
			{
				.sumRelErr = 0.0f,
				.minRelErr = std::numeric_limits<float>::max(),
				.maxRelErr = 0.0f
			},
			[&](const tbb::blocked_range<uint32_t>& r, DensStatsLocal local) -> DensStatsLocal
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const float dev = (densityF_[i] / simParams.restDensityF) - 1.0f;
					const float relErr = std::fabs(dev) * 100.0f; // deviation in percent

					workBuffer_[i] = relErr;
					local.sumRelErr += relErr;

					if (relErr < local.minRelErr) [[unlikely]] { local.minRelErr = relErr; }
					if (relErr > local.maxRelErr) [[unlikely]] { local.maxRelErr = relErr; }
				}

				return local;
			},
			[](const DensStatsLocal& a, const DensStatsLocal& b) -> DensStatsLocal
			{
				DensStatsLocal r;

				r.sumRelErr = a.sumRelErr + b.sumRelErr;
				r.minRelErr = std::min(a.minRelErr, b.minRelErr);
				r.maxRelErr = std::max(a.maxRelErr, b.maxRelErr);

				return r;
			},
			tbb::auto_partitioner{}
		);

		sumRelErr = globalDens.sumRelErr;
		minRelErr = globalDens.minRelErr;
		maxRelErr = globalDens.maxRelErr;
	}

	float p95 = 0.0f;
	float p99 = 0.0f;

	if (NF) // determine percentiles
	{
		// indices of the 95th and 99th percentiles
		const ptrdiff_t i95 = static_cast<ptrdiff_t>(0.95f * (NF - 1));
		const ptrdiff_t i99 = static_cast<ptrdiff_t>(0.99f * (NF - 1));

		auto nth95 = workBuffer_.begin() + i95;
		auto nth99 = workBuffer_.begin() + i99;

		// partially sort so that the desired percentiles can be extracted
		std::nth_element(workBuffer_.begin(), nth95, workBuffer_.end());
		std::nth_element(workBuffer_.begin(), nth99, workBuffer_.end());

		p95 = *nth95;
		p99 = *nth99;
	}
	else // reasonable defaults for empty sim
	{
		minRelErr = 0.0f;
		maxRelErr = 0.0f;
	}

	// --- [ velocity stats ] --------------------------------------------------

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
					const Eigen::Vector3f v = velF_[i];
					const float velMag = v.norm();

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

	benchmarksMgr_.StopBench({ Bench::BenchOp::Statistics_Particles,
								   Bench::BenchOp::Statistics_Density,
								   Bench::BenchOp::Statistics_Velocity });

	// --- [ writes to record ] ------------------------------------------------

	Set(record, StatMetric::NumNeighFF_Avg, sumFF * invNF);
	Set(record, StatMetric::NumNeighFB_Avg, sumFB * invNF);
	Set(record, StatMetric::NumNeighFF_Min, minFF);
	Set(record, StatMetric::NumNeighFB_Min, minFB);
	Set(record, StatMetric::NumNeighFF_Max, maxFF);
	Set(record, StatMetric::NumNeighFB_Max, maxFB);

	Set(record, StatMetric::DensRelError_Avg, sumRelErr * invNF);
	Set(record, StatMetric::DensRelError_P95, p95);
	Set(record, StatMetric::DensRelError_P99, p99);
	Set(record, StatMetric::DensRelError_Min, minRelErr);
	Set(record, StatMetric::DensRelError_Max, maxRelErr);

	Set(record, StatMetric::VelMag_Avg, velMagSum * invNF);
	Set(record, StatMetric::VelMag_Min, velMagMin);
	Set(record, StatMetric::VelMag_Max, velMagMax);
}

void SimBackend_CPU02::ExecuteSimStep(SimParams params, int64_t N_steps)
{
	if (N_F_ == 0) // empty sim
	{
		simStep_ += N_steps;
		simTime_ += N_steps * params.timeStep;

		return;
	}

	// --- [ initialization and checks ] ---------------------------------------

	// ensure neighbor storage has enough capacity
	if (N_F_ != static_cast<uint32_t>(neighborsFF_.size()) ||
		N_F_ != static_cast<uint32_t>(neighborsFB_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelWeightsFF_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelWeightsFB_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelGradFF_.size()) ||
		N_F_ != static_cast<uint32_t>(kernelGradFB_.size()))
	{
		neighborsFF_.resize(N_F_);
		neighborsFB_.resize(N_F_);
		kernelWeightsFF_.resize(N_F_);
		kernelWeightsFB_.resize(N_F_);
		kernelGradFF_.resize(N_F_);
		kernelGradFB_.resize(N_F_);

		// expected number of fluid and boundary neighbors
		const uint32_t N_expectedNumFF = 
			std::ceilf(2.0f * PI * std::powf(params.kernelRadius, 3.0f));

		const uint32_t N_expectedNumFB =
			std::ceilf(0.5f * PI * std::powf(params.kernelRadius, 3.0f));

		// heuristic reserves (kept and re-used across steps)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					neighborsFF_[i].reserve(N_expectedNumFF);
					neighborsFB_[i].reserve(N_expectedNumFB);
					kernelWeightsFF_[i].reserve(N_expectedNumFF);
					kernelWeightsFB_[i].reserve(N_expectedNumFB);
					kernelGradFF_[i].reserve(N_expectedNumFF);
					kernelGradFB_[i].reserve(N_expectedNumFB);
				}
			},
			tbb::auto_partitioner{}
		);
	}

	Step_IISPH(params);

	simStep_ += N_steps;
	simTime_ += N_steps * params.timeStep;
}
