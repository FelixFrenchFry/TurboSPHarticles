// cpu_backend_01.cpp

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

namespace
{
	constexpr float RigidOrthonormalEps = 1e-12f;

	float Dot3(
		const float ax, const float ay, const float az,
		const float bx, const float by, const float bz)
	{
		return std::fmaf(ax, bx, std::fmaf(ay, by, az * bz));
	}

	float Length3(const float x, const float y, const float z)
	{
		return std::sqrt(Dot3(x, y, z, x, y, z));
	}

	void Normalize3(float& x, float& y, float& z)
	{
		const float len = Length3(x, y, z);
		if (len <= RigidOrthonormalEps || !std::isfinite(len))
		{
			throw std::runtime_error("Degenerate CPU01 rigid body orientation.");
		}

		const float lenInv = 1.0f / len;
		x *= lenInv;
		y *= lenInv;
		z *= lenInv;
	}

	struct RigidForceTorqueAccum
	{
		float xForce = 0.0f;
		float yForce = 0.0f;
		float zForce = 0.0f;
		float xTorque = 0.0f;
		float yTorque = 0.0f;
		float zTorque = 0.0f;

		void Add(const RigidForceTorqueAccum& rhs)
		{
			xForce += rhs.xForce;
			yForce += rhs.yForce;
			zForce += rhs.zForce;
			xTorque += rhs.xTorque;
			yTorque += rhs.yTorque;
			zTorque += rhs.zTorque;
		}
	};
}

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
	auto& R = src.rigidParticles;
	auto& RB = src.rigidBodies;
	N_F_ = F.N_;
	N_B_ = B.N_;
	N_R_ = R.N_;
	N_RB_ = RB.N_;
	simStep_ = src.step;
	simTime_ = src.time;
	F.ensureParticleIdsPresent();
	B.ensureParticleIdsPresent();
	B.ensureBoundaryCoefficientsPresent(params.massB, params.restDensity);
	R.ensureParticleIdsPresent();
	R.ensureRigidCoefficientsPresent(params.massB, params.restDensity);
	RB.ensureRigidBodyIdsPresent();
	src.ValidateRigidAttachments();

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

	// transfer to backend boundary data structures
	idB_ = std::move(B.id);
	xPosB_ = std::move(B.xPos);
	yPosB_ = std::move(B.yPos);
	zPosB_ = std::move(B.zPos);
	volB_ = std::move(B.volB);
	psiB_ = std::move(B.psiB);

	// transfer to backend rigid particle data structures
	idR_ = std::move(R.id);
	bodyIdR_ = std::move(R.bodyId);
	xPosR_.resize(N_R_);
	yPosR_.resize(N_R_);
	zPosR_.resize(N_R_);
	xRelPosR_ = std::move(R.xRelPos);
	yRelPosR_ = std::move(R.yRelPos);
	zRelPosR_ = std::move(R.zRelPos);
	xVelR_.resize(N_R_);
	yVelR_.resize(N_R_);
	zVelR_.resize(N_R_);
	xForceR_.assign(N_R_, 0.0f);
	yForceR_.assign(N_R_, 0.0f);
	zForceR_.assign(N_R_, 0.0f);
	volR_ = std::move(R.volR);
	psiR_ = std::move(R.psiR);

	// transfer to backend rigid body data structures
	idRB_ = std::move(RB.id);
	rigidParticleBeginRB_ = std::move(RB.rigidParticleBegin);
	rigidParticleCountRB_ = std::move(RB.rigidParticleCount);
	xCMRB_ = std::move(RB.xCM);
	yCMRB_ = std::move(RB.yCM);
	zCMRB_ = std::move(RB.zCM);
	xVelRB_ = std::move(RB.xVel);
	yVelRB_ = std::move(RB.yVel);
	zVelRB_ = std::move(RB.zVel);
	xOmegaRB_ = std::move(RB.xOmega);
	yOmegaRB_ = std::move(RB.yOmega);
	zOmegaRB_ = std::move(RB.zOmega);
	xAngMomRB_ = std::move(RB.xAngMom);
	yAngMomRB_ = std::move(RB.yAngMom);
	zAngMomRB_ = std::move(RB.zAngMom);

	xForceRB_.assign(N_RB_, 0.0f);
	yForceRB_.assign(N_RB_, 0.0f);
	zForceRB_.assign(N_RB_, 0.0f);
	xTorqueRB_.assign(N_RB_, 0.0f);
	yTorqueRB_.assign(N_RB_, 0.0f);
	zTorqueRB_.assign(N_RB_, 0.0f);

	a00RB_ = std::move(RB.a00);
	a01RB_ = std::move(RB.a01);
	a02RB_ = std::move(RB.a02);
	a10RB_ = std::move(RB.a10);
	a11RB_ = std::move(RB.a11);
	a12RB_ = std::move(RB.a12);
	a20RB_ = std::move(RB.a20);
	a21RB_ = std::move(RB.a21);
	a22RB_ = std::move(RB.a22);
	massRB_ = std::move(RB.mass);
	massInvRB_ = std::move(RB.massInv);
	iBodyInv00RB_ = std::move(RB.iBodyInv00);
	iBodyInv01RB_ = std::move(RB.iBodyInv01);
	iBodyInv02RB_ = std::move(RB.iBodyInv02);
	iBodyInv10RB_ = std::move(RB.iBodyInv10);
	iBodyInv11RB_ = std::move(RB.iBodyInv11);
	iBodyInv12RB_ = std::move(RB.iBodyInv12);
	iBodyInv20RB_ = std::move(RB.iBodyInv20);
	iBodyInv21RB_ = std::move(RB.iBodyInv21);
	iBodyInv22RB_ = std::move(RB.iBodyInv22);

	F = {};
	B = {};
	R = {};
	RB = {};

	// initialize derived rigid state and neighbor storage
	UpdateRigidParticles();
	neighborList_->Resize(N_F_, N_B_, N_R_, params);
}

// TODO: re-implement this to free up memory as soon as copied out?
void SimBackend_CPU01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;
	auto& R = dst.rigidParticles;
	auto& RB = dst.rigidBodies;

	dst.step = simStep_;
	dst.time = simTime_;

	// (no resize, because it is included in the copy operations)

	// copy fluid particles
	F.id = idF_;
	F.xPos = xPosF_; F.yPos = yPosF_; F.zPos = zPosF_;
	F.xVel = xVelF_; F.yVel = yVelF_; F.zVel = zVelF_;
	F.xVelPred = xVelPredF_; F.yVelPred = yVelPredF_; F.zVelPred = zVelPredF_;
	F.xAccPress = xAccPressF_; F.yAccPress = yAccPressF_; F.zAccPress = zAccPressF_;
	F.sourceTerm = sourceTermF_;
	F.diagElem = diagElemF_;
	F.pressure = pressureF_;
	F.density = densityF_;
	F.N_ = N_F_;

	// copy boundary particles
	B.id = idB_;
	B.xPos = xPosB_; B.yPos = yPosB_; B.zPos = zPosB_;
	B.volB = volB_; B.psiB = psiB_;
	B.N_ = N_B_;

	// copy rigid particles
	R.id = idR_;
	R.bodyId = bodyIdR_;
	R.xRelPos = xRelPosR_; R.yRelPos = yRelPosR_; R.zRelPos = zRelPosR_;
	R.volR = volR_; R.psiR = psiR_;
	R.N_ = N_R_;

	// copy rigid bodies
	RB.id = idRB_;
	RB.rigidParticleBegin = rigidParticleBeginRB_;
	RB.rigidParticleCount = rigidParticleCountRB_;
	RB.xCM = xCMRB_; RB.yCM = yCMRB_; RB.zCM = zCMRB_;
	RB.xVel = xVelRB_; RB.yVel = yVelRB_; RB.zVel = zVelRB_;
	RB.xOmega = xOmegaRB_; RB.yOmega = yOmegaRB_; RB.zOmega = zOmegaRB_;
	RB.xAngMom = xAngMomRB_; RB.yAngMom = yAngMomRB_; RB.zAngMom = zAngMomRB_;
	RB.a00 = a00RB_; RB.a01 = a01RB_; RB.a02 = a02RB_;
	RB.a10 = a10RB_; RB.a11 = a11RB_; RB.a12 = a12RB_;
	RB.a20 = a20RB_; RB.a21 = a21RB_; RB.a22 = a22RB_;
	RB.mass = massRB_; RB.massInv = massInvRB_;
	RB.iBodyInv00 = iBodyInv00RB_; RB.iBodyInv01 = iBodyInv01RB_; RB.iBodyInv02 = iBodyInv02RB_;
	RB.iBodyInv10 = iBodyInv10RB_; RB.iBodyInv11 = iBodyInv11RB_; RB.iBodyInv12 = iBodyInv12RB_;
	RB.iBodyInv20 = iBodyInv20RB_; RB.iBodyInv21 = iBodyInv21RB_; RB.iBodyInv22 = iBodyInv22RB_;
	RB.N_ = N_RB_;
}

void SimBackend_CPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	snap.simStep = simStep_;
	snap.simTime = simTime_;

	// copy boundary particles
	snap.N_B = N_B_;
	snap.idB = idB_;
	snap.xPosB = xPosB_; snap.yPosB = yPosB_; snap.zPosB = zPosB_;
	snap.volB = volB_;

	// copy rigid particles
	snap.N_R = N_R_;
	snap.idR = idR_;
	snap.bodyIdR = bodyIdR_;
	snap.xPosR = xPosR_; snap.yPosR = yPosR_; snap.zPosR = zPosR_;
	snap.volR = volR_;

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

	// classify fluid particles wanted for export
	FillFluidParticleNearSurfaceFlags(expNearSurfaceFlagsScratch_);
	ClassifyExportSurfaceParticles(expNearSurfaceFlagsScratch_);
	const uint32_t N_surface = static_cast<uint32_t>(
		std::ranges::count(expSurfaceFlagsScratch_, uint8_t{ 1 }));

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
	// TODO: parallelize with TBB?
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (!expSurfaceFlagsScratch_[i]) { continue; }

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

bool SimBackend_CPU01::FillFluidParticleNearSurfaceFlags(
	std::vector<uint8_t>& nearSurfaceFlags) const
{
	if (!neighborList_) { return false; }

	nearSurfaceFlags.resize(N_F_);
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// particle considered near the surface
				// if touching boundary OR low fluid neighbor count
				nearSurfaceFlags[i] = static_cast<uint8_t>(
					(neighborList_->CountFB(i) > 0 ||
					 neighborList_->CountFR(i) > 0) ||
					 neighborList_->CountFF(i) < 24);
			}
		},
		tbb::auto_partitioner{});

	return true;
}

void SimBackend_CPU01::ClassifyExportSurfaceParticles(
	const std::vector<uint8_t>& nearSurfaceFlags) const
{
	if (!neighborList_ || nearSurfaceFlags.size() != static_cast<uint64_t>(N_F_))
	{
		expSurfaceFlagsScratch_.clear();
		return;
	}

	auto& surfaceFlags = expSurfaceFlagsScratch_;
	surfaceFlags.resize(N_F_);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// particle considered part of the surface
				// if touching boundary OR low fluid neighbor count
				bool surface = (neighborList_->CountFB(i) > 0 ||
							   neighborList_->CountFR(i) > 0) ||
							   neighborList_->CountFF(i) < 28;

				// if not surface itself, check its fluid neighbors
				if (!surface)
				{
					const NeighborSpan nbrListFF = neighborList_->FF(i);
					const uint32_t* nbrFF = nbrListFF.data;
					const uint32_t N_FF = nbrListFF.count;

					for (uint32_t k = 0; k < N_FF; k++)
					{
						if (nearSurfaceFlags[nbrFF[k]] != 0)
						{
							surface = true;
							break;
						}
					}
				}

				surfaceFlags[i] = static_cast<uint8_t>(surface);
			}
		},
		tbb::auto_partitioner{});
}

bool SimBackend_CPU01::BuildExportFluidParticleLocalIdxs(
	const std::vector<uint8_t>& nearSurfaceFlags,
	std::vector<int>& localIdxs) const
{
	if (!neighborList_ || nearSurfaceFlags.size() != static_cast<uint64_t>(N_F_))
	{
		return false;
	}

	ClassifyExportSurfaceParticles(nearSurfaceFlags);
	const uint32_t surfaceParticleCount = static_cast<uint32_t>(
		std::ranges::count(expSurfaceFlagsScratch_, uint8_t{ 1 }));

	localIdxs.clear();
	localIdxs.reserve(surfaceParticleCount);
	// TODO: parallelize with TBB?
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (expSurfaceFlagsScratch_[i] != 0)
		{
			localIdxs.push_back(static_cast<int>(i));
		}
	}

	return true;
}

void SimBackend_CPU01::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	snap.simStep = simStep_;
	snap.N_F = N_F_;
	snap.N_B = N_B_;
	snap.N_R = N_R_;
	snap.N_RB = N_RB_;

	// copy fluid particles
	if (N_F_ == 0)
	{
		snap.posF.clear();
		snap.scalarDataF.clear();
	}
	else
	{
		snap.posF.resize(N_F_);
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

	// copy boundary particles
	snap.posB.resize(snap.N_B);
	snap.volB = volB_;
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

	// copy rigid particles
	snap.posR.resize(snap.N_R);
	snap.volR = volR_;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posR[i] = glm::vec3{ xPosR_[i], yPosR_[i], zPosR_[i] };
			}
		},
		tbb::auto_partitioner{}
	);

	// copy rigid body state
	snap.centerOfMassRB.resize(N_RB_);
	snap.orientationRB.resize(N_RB_);
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
	{
		snap.centerOfMassRB[bodyIdx] = { xCMRB_[bodyIdx], yCMRB_[bodyIdx], zCMRB_[bodyIdx] };
		snap.orientationRB[bodyIdx][0] = { a00RB_[bodyIdx], a10RB_[bodyIdx], a20RB_[bodyIdx] };
		snap.orientationRB[bodyIdx][1] = { a01RB_[bodyIdx], a11RB_[bodyIdx], a21RB_[bodyIdx] };
		snap.orientationRB[bodyIdx][2] = { a02RB_[bodyIdx], a12RB_[bodyIdx], a22RB_[bodyIdx] };
	}
}

void SimBackend_CPU01::FillScalarData(
	Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const
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
			snap.scalarDataF.resize(N_F_);
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

	throw std::runtime_error("Unsupported scalar data mode in CPU01.");
}

void SimBackend_CPU01::FillMemoryRecord(
	Mem::MemRecordStep& record,
	const SimParams& params) const
{
	using namespace Mem;
	constexpr MemSpace memSpace = MemSpace::RAM;

	// --- [ particle properties ] -------------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		MemUsage usageR{};

		// fluid
		Add(usageF, MemUsageOf_Vector(idF_));
		Add(usageF, MemUsageOf_Vector(xPosF_));
		Add(usageF, MemUsageOf_Vector(yPosF_));
		Add(usageF, MemUsageOf_Vector(zPosF_));
		Add(usageF, MemUsageOf_Vector(xVelF_));
		Add(usageF, MemUsageOf_Vector(yVelF_));
		Add(usageF, MemUsageOf_Vector(zVelF_));
		Add(usageF, MemUsageOf_Vector(xVelPredF_));
		Add(usageF, MemUsageOf_Vector(yVelPredF_));
		Add(usageF, MemUsageOf_Vector(zVelPredF_));
		Add(usageF, MemUsageOf_Vector(xAccPressF_));
		Add(usageF, MemUsageOf_Vector(yAccPressF_));
		Add(usageF, MemUsageOf_Vector(zAccPressF_));
		Add(usageF, MemUsageOf_Vector(sourceTermF_));
		Add(usageF, MemUsageOf_Vector(diagElemF_));
		Add(usageF, MemUsageOf_Vector(pressureF_));
		Add(usageF, MemUsageOf_Vector(densityF_));

		// boundary
		Add(usageB, MemUsageOf_Vector(idB_));
		Add(usageB, MemUsageOf_Vector(xPosB_));
		Add(usageB, MemUsageOf_Vector(yPosB_));
		Add(usageB, MemUsageOf_Vector(zPosB_));
		Add(usageB, MemUsageOf_Vector(volB_));
		Add(usageB, MemUsageOf_Vector(psiB_));

		// rigid
		Add(usageR, MemUsageOf_Vector(idR_));
		Add(usageR, MemUsageOf_Vector(bodyIdR_));
		Add(usageR, MemUsageOf_Vector(xPosR_));
		Add(usageR, MemUsageOf_Vector(yPosR_));
		Add(usageR, MemUsageOf_Vector(zPosR_));
		Add(usageR, MemUsageOf_Vector(xRelPosR_));
		Add(usageR, MemUsageOf_Vector(yRelPosR_));
		Add(usageR, MemUsageOf_Vector(zRelPosR_));
		Add(usageR, MemUsageOf_Vector(xVelR_));
		Add(usageR, MemUsageOf_Vector(yVelR_));
		Add(usageR, MemUsageOf_Vector(zVelR_));
		Add(usageR, MemUsageOf_Vector(xForceR_));
		Add(usageR, MemUsageOf_Vector(yForceR_));
		Add(usageR, MemUsageOf_Vector(zForceR_));
		Add(usageR, MemUsageOf_Vector(volR_));
		Add(usageR, MemUsageOf_Vector(psiR_));

		// finalize
		Set(record, memSpace, MemBucket::ParticlePropertiesF, usageF);
		Set(record, memSpace, MemBucket::ParticlePropertiesB, usageB);
		Set(record, memSpace, MemBucket::ParticlePropertiesR, usageR);
	}

	// --- [ neighbor search data structures ] -------------------------------------------------------

	{
		const ParticleClassMemUsage usage = neighborSearch_->GetMemUsage(params);

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructF, usage.fluid);
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructB, usage.boundary);
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructR, usage.rigid);
	}

	// --- [ neighbor list storage ] -----------------------------------------------------------------

	{
		const ParticleClassMemUsage usage = neighborList_->GetMemUsage(params);

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_ListStorageF, usage.fluid);
		Set(record, memSpace, MemBucket::Neighbors_ListStorageB, usage.boundary);
		Set(record, memSpace, MemBucket::Neighbors_ListStorageR, usage.rigid);
	}

	// --- [ misc ] ----------------------------------------------------------------------------------

	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(workBufferF_));
		Add(usage, MemUsageOf_Vector(workBufferB_));
		Add(usage, MemUsageOf_Vector(workBufferIdF_));
		Add(usage, MemUsageOf_Vector(workBufferIdB_));
		Add(usage, MemUsageOf_Vector(expSurfaceFlagsScratch_));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFF.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFB.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFR.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRF.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRB.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRR.histogram));
		Add(usage, MemUsageOf_Vector(activeRigidParticleLocalIdxs_));
		Add(usage, MemUsageOf_Vector(idRB_));
		Add(usage, MemUsageOf_Vector(rigidParticleBeginRB_));
		Add(usage, MemUsageOf_Vector(rigidParticleCountRB_));
		Add(usage, MemUsageOf_Vector(xCMRB_));
		Add(usage, MemUsageOf_Vector(yCMRB_));
		Add(usage, MemUsageOf_Vector(zCMRB_));
		Add(usage, MemUsageOf_Vector(xVelRB_));
		Add(usage, MemUsageOf_Vector(yVelRB_));
		Add(usage, MemUsageOf_Vector(zVelRB_));
		Add(usage, MemUsageOf_Vector(xOmegaRB_));
		Add(usage, MemUsageOf_Vector(yOmegaRB_));
		Add(usage, MemUsageOf_Vector(zOmegaRB_));
		Add(usage, MemUsageOf_Vector(xAngMomRB_));
		Add(usage, MemUsageOf_Vector(yAngMomRB_));
		Add(usage, MemUsageOf_Vector(zAngMomRB_));
		Add(usage, MemUsageOf_Vector(xForceRB_));
		Add(usage, MemUsageOf_Vector(yForceRB_));
		Add(usage, MemUsageOf_Vector(zForceRB_));
		Add(usage, MemUsageOf_Vector(xTorqueRB_));
		Add(usage, MemUsageOf_Vector(yTorqueRB_));
		Add(usage, MemUsageOf_Vector(zTorqueRB_));
		Add(usage, MemUsageOf_Vector(a00RB_));
		Add(usage, MemUsageOf_Vector(a01RB_));
		Add(usage, MemUsageOf_Vector(a02RB_));
		Add(usage, MemUsageOf_Vector(a10RB_));
		Add(usage, MemUsageOf_Vector(a11RB_));
		Add(usage, MemUsageOf_Vector(a12RB_));
		Add(usage, MemUsageOf_Vector(a20RB_));
		Add(usage, MemUsageOf_Vector(a21RB_));
		Add(usage, MemUsageOf_Vector(a22RB_));
		Add(usage, MemUsageOf_Vector(massRB_));
		Add(usage, MemUsageOf_Vector(massInvRB_));
		Add(usage, MemUsageOf_Vector(iBodyInv00RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv01RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv02RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv10RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv11RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv12RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv20RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv21RB_));
		Add(usage, MemUsageOf_Vector(iBodyInv22RB_));

		// finalize
		Set(record, memSpace, MemBucket::Misc, usage);
	}
}

void SimBackend_CPU01::FillStatisticsRecord(
	Stats::StatRecordStep& record,
	const SimParams& simParams) const
{
	auto& partial = statisticsPartialScratch_;
	ComputeStatisticsPartial(partial, simParams, false);
	Stats::StatisticsTracker::FinalizeStatisticsRecord(record, partial, simParams);
}

bool SimBackend_CPU01::FillStatisticsPartial(
	Stats::StatisticsPartial& partial,
	const SimParams& simParams) const
{
	ComputeStatisticsPartial(partial, simParams, true);
	return true;
}

void SimBackend_CPU01::ComputeStatisticsPartial(
	Stats::StatisticsPartial& partial,
	const SimParams& simParams,
	const bool useActiveParticleLocalIdxs) const
{
	using namespace Stats;

	partial.Clear();

	const uint32_t N_F = useActiveParticleLocalIdxs
		? static_cast<uint32_t>(activeFluidParticleLocalIdxs_.size())
		: N_F_;
	const uint32_t N_R = useActiveParticleLocalIdxs
		? static_cast<uint32_t>(activeRigidParticleLocalIdxs_.size())
		: N_R_;

	partial.N_F = N_F;
	partial.N_B = useActiveParticleLocalIdxs ? 0 : N_B_;
	partial.N_R = N_R;
	partial.solverIterCount = N_solverIterCount_;

	const auto getFluidLocalIdx = [&](const uint32_t i)
	{
		return useActiveParticleLocalIdxs
			? activeFluidParticleLocalIdxs_[i]
			: i;
	};
	const auto getRigidLocalIdx = [&](const uint32_t i)
	{
		return useActiveParticleLocalIdxs
			? activeRigidParticleLocalIdxs_[i]
			: i;
	};

	benchmarksTracker_.StartBench(Bench::BenchOp::Statistics_Computation);

	// --- [ particle counts ] -----------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		const auto fillNbrCountPartial =
			[&]<typename LocalIdxFunc, typename NbrCountFunc>(
				const uint32_t N,
				LocalIdxFunc getLocalIdx,
				NbrCountFunc getNbrCount,
				DistributionPartial& distribution)
		{
			distribution.Clear();
			distribution.N = N;
			if (N == 0) { return; }

			struct NbrCountStatsLocal
			{
				uint64_t sum = 0;
				uint32_t min = std::numeric_limits<uint32_t>::max();
				uint32_t max = 0;
			};

			const NbrCountStatsLocal nbrCountStats = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, N),
				NbrCountStatsLocal{},
				[&](const tbb::blocked_range<uint32_t>& r,
					NbrCountStatsLocal local) -> NbrCountStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const uint32_t N_nbrs =
							getNbrCount(getLocalIdx(i));
						local.sum += N_nbrs;
						local.min = std::min(local.min, N_nbrs);
						local.max = std::max(local.max, N_nbrs);
					}
					return local;
				},
				[](const NbrCountStatsLocal& a,
					const NbrCountStatsLocal& b) -> NbrCountStatsLocal
				{
					return NbrCountStatsLocal
					{
						a.sum + b.sum,
						std::min(a.min, b.min),
						std::max(a.max, b.max)
					};
				},
				tbb::auto_partitioner{});

			distribution.sum = static_cast<double>(nbrCountStats.sum);
			distribution.min = static_cast<float>(nbrCountStats.min);
			distribution.max = static_cast<float>(nbrCountStats.max);
			distribution.histogram.assign(static_cast<uint64_t>(nbrCountStats.max) + 1, 0);

			for (uint32_t i = 0; i < N; i++)
			{
				const uint32_t N_nbrs = getNbrCount(getLocalIdx(i));
				distribution.histogram[N_nbrs]++;
			}
		};

		fillNbrCountPartial(
			N_F,
			getFluidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountFF(localIdx);
			},
			partial.nbrCountFF);
		fillNbrCountPartial(
			N_F,
			getFluidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountFB(localIdx);
			},
			partial.nbrCountFB);
		fillNbrCountPartial(
			N_F,
			getFluidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountFR(localIdx);
			},
			partial.nbrCountFR);
		fillNbrCountPartial(
			N_R,
			getRigidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountRF(localIdx);
			},
			partial.nbrCountRF);
		fillNbrCountPartial(
			N_R,
			getRigidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountRB(localIdx);
			},
			partial.nbrCountRB);
		fillNbrCountPartial(
			N_R,
			getRigidLocalIdx,
			[&](const uint32_t localIdx)
			{
				return neighborList_->CountRR(localIdx);
			},
			partial.nbrCountRR);
	}

	// --- [ density stats ] -------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)] &&
		N_F > 0)
	{
		struct DensityStatsLocal
		{
			double rawSum = 0.0;
			double positiveSum = 0.0;
			double absoluteSum = 0.0;
		};

		const DensityStatsLocal densityStats = tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, N_F),
			DensityStatsLocal{},
			[&](const tbb::blocked_range<uint32_t>& r,
				DensityStatsLocal local) -> DensityStatsLocal
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const uint32_t localIdx =
						getFluidLocalIdx(i);
					const float relErr =
						(densityF_[localIdx] / simParams.restDensity) - 1.0f;
					local.rawSum += relErr;
					local.positiveSum += std::max(0.0f, relErr);
					local.absoluteSum += std::fabs(relErr);
				}
				return local;
			},
			[](const DensityStatsLocal& a,
				const DensityStatsLocal& b) -> DensityStatsLocal
			{
				return DensityStatsLocal
				{
					a.rawSum + b.rawSum,
					a.positiveSum + b.positiveSum,
					a.absoluteSum + b.absoluteSum
				};
			},
			tbb::auto_partitioner{});

		partial.densRelErrorRawSum = densityStats.rawSum;
		partial.densRelErrorPosSum = densityStats.positiveSum;
		partial.densRelErrorAbsSum = densityStats.absoluteSum;
	}

	// --- [ velocity stats ] ------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
		auto& velMag = partial.velMag;
		velMag.N = N_F;

		if (N_F > 0)
		{
			struct VelMagStatsLocal
			{
				double sum = 0.0;
				float min = std::numeric_limits<float>::max();
				float max = 0.0f;
			};

			const VelMagStatsLocal velMagStats = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, N_F),
				VelMagStatsLocal{},
				[&](const tbb::blocked_range<uint32_t>& r,
					VelMagStatsLocal local) -> VelMagStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const uint32_t localIdx =
							getFluidLocalIdx(i);
						const float vx = xVelF_[localIdx];
						const float vy = yVelF_[localIdx];
						const float vz = zVelF_[localIdx];
						const float velMagValue =
							std::sqrt(vx * vx + vy * vy + vz * vz);

						local.sum += velMagValue;
						local.min = std::min(local.min, velMagValue);
						local.max = std::max(local.max, velMagValue);
					}
					return local;
				},
				[](const VelMagStatsLocal& a,
					const VelMagStatsLocal& b) -> VelMagStatsLocal
				{
					return VelMagStatsLocal
					{
						a.sum + b.sum,
						std::min(a.min, b.min),
						std::max(a.max, b.max)
					};
				},
				tbb::auto_partitioner{});

			velMag.sum = velMagStats.sum;
			velMag.min = velMagStats.min;
			velMag.max = velMagStats.max;
		}
	}

	benchmarksTracker_.StopBench(Bench::BenchOp::Statistics_Computation);
}

void SimBackend_CPU01::ExecuteNeighborSearch(const SimParams& params, const bool bench)
{
	const bool runCountQuery = neighborList_->RunCountQuery();
	const bool runFillQuery = neighborList_->RunFillQuery();

	// initial re-ordering of the data before the initial neighbor search
	if (bench)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
	}
	neighborSearch_->DataReOrder(params);
	ApplyLastParticleReOrderToActiveFluidParticleLocalIdxs();
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
	if (bench)
	{
		benchmarksTracker_.StopBench(Bench::BenchOp::ParticleData_Reordering);
	}

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
	neighborList_->Finalize_Count();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

	if (runFillQuery)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}

	benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	neighborList_->Finalize_Fill();
	benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);
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

bool SimBackend_CPU01::ConfigureDistributedStep(
	const std::vector<int>& activeFluidParticleLocalIdxs,
	const std::vector<int>& activeRigidParticleLocalIdxs)
{
	std::vector<uint32_t> activeFluidParticleLocalIdxsU;
	activeFluidParticleLocalIdxsU.reserve(activeFluidParticleLocalIdxs.size());

	std::vector<uint8_t> activeIdxUsed(N_F_, 0);
	for (const int localIdx : activeFluidParticleLocalIdxs)
	{
		if (localIdx < 0 || static_cast<uint32_t>(localIdx) >= N_F_)
		{
			SPDLOG_ERROR("Invalid active fluid particle local index {}.", localIdx);
			return false;
		}

		const uint32_t localIdxU = static_cast<uint32_t>(localIdx);
		if (activeIdxUsed[localIdxU] != 0)
		{
			SPDLOG_ERROR("Duplicate active fluid particle local index {}.", localIdx);
			return false;
		}

		activeIdxUsed[localIdxU] = 1;
		activeFluidParticleLocalIdxsU.push_back(localIdxU);
	}

	std::vector<uint32_t> activeRigidParticleLocalIdxsU;
	activeRigidParticleLocalIdxsU.reserve(activeRigidParticleLocalIdxs.size());
	std::vector<uint8_t> activeRigidParticleFlags(N_R_, 0);
	for (const int localIdx : activeRigidParticleLocalIdxs)
	{
		if (localIdx < 0 || static_cast<uint32_t>(localIdx) >= N_R_)
		{
			SPDLOG_ERROR("Invalid active rigid particle local index {}.", localIdx);
			return false;
		}

		const uint32_t localIdxU = static_cast<uint32_t>(localIdx);
		if (activeRigidParticleFlags[localIdxU] != 0)
		{
			SPDLOG_ERROR("Duplicate active rigid particle local index {}.", localIdx);
			return false;
		}

		activeRigidParticleFlags[localIdxU] = 1;
		activeRigidParticleLocalIdxsU.push_back(localIdxU);
	}

	activeFluidParticleLocalIdxs_ = std::move(activeFluidParticleLocalIdxsU);
	activeRigidParticleLocalIdxs_ = std::move(activeRigidParticleLocalIdxsU);
	activeRigidParticleFlags_ = std::move(activeRigidParticleFlags);
	useActiveFluidParticleLocalIdxs_ = true;
	useActiveRigidParticleLocalIdxs_ = true;
	integrateRigidBodies_ = false;
	applyRigidBodyExternalForces_ = false;
	return true;
}

bool SimBackend_CPU01::GetRigidBodyForceTorque(
	RigidBodyForceTorque& dst) const
{
	dst.Resize(N_RB_);
	dst.xForce = xForceRB_;
	dst.yForce = yForceRB_;
	dst.zForce = zForceRB_;
	dst.xTorque = xTorqueRB_;
	dst.yTorque = yTorqueRB_;
	dst.zTorque = zTorqueRB_;
	return true;
}

bool SimBackend_CPU01::ApplyRigidBodyForceTorque(
	const RigidBodyForceTorque& forceTorque,
	const float timeStep)
{
	xForceRB_ = forceTorque.xForce;
	yForceRB_ = forceTorque.yForce;
	zForceRB_ = forceTorque.zForce;
	xTorqueRB_ = forceTorque.xTorque;
	yTorqueRB_ = forceTorque.yTorque;
	zTorqueRB_ = forceTorque.zTorque;

	benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_StateUpdate);
	IntegrateRigidBodies(timeStep);
	UpdateRigidParticles();
	benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_StateUpdate);
	return true;
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
	std::vector<uint8_t> oldFluidLocalIdxUsed(N_F_, 0);
	for (uint64_t newLocalIdx = 0; newLocalIdx < oldFluidLocalIdx_FromNewLocalIdx.size(); newLocalIdx++)
	{
		const uint32_t oldLocalIdx = oldFluidLocalIdx_FromNewLocalIdx[newLocalIdx];
		if (oldLocalIdx >= N_F_ || oldFluidLocalIdxUsed[oldLocalIdx] != 0)
		{
			throw std::runtime_error("Invalid CPU01 fluid particle re-order index.");
		}

		oldFluidLocalIdxUsed[oldLocalIdx] = 1;
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

void SimBackend_CPU01::UpdateRigidParticles()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_R_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint32_t bodyIdx = bodyIdR_[i];
				if (bodyIdx >= N_RB_) // TODO: hide check behind debug compiler flag?
				{
					throw std::runtime_error("Invalid CPU01 rigid particle attachment.");
				}

				const float lx = xRelPosR_[i];
				const float ly = yRelPosR_[i];
				const float lz = zRelPosR_[i];

				const float rx = std::fmaf(a00RB_[bodyIdx], lx,
					std::fmaf(a01RB_[bodyIdx], ly, a02RB_[bodyIdx] * lz));
				const float ry = std::fmaf(a10RB_[bodyIdx], lx,
					std::fmaf(a11RB_[bodyIdx], ly, a12RB_[bodyIdx] * lz));
				const float rz = std::fmaf(a20RB_[bodyIdx], lx,
					std::fmaf(a21RB_[bodyIdx], ly, a22RB_[bodyIdx] * lz));

				xPosR_[i] = xCMRB_[bodyIdx] + rx;
				yPosR_[i] = yCMRB_[bodyIdx] + ry;
				zPosR_[i] = zCMRB_[bodyIdx] + rz;

				const float wx = xOmegaRB_[bodyIdx];
				const float wy = yOmegaRB_[bodyIdx];
				const float wz = zOmegaRB_[bodyIdx];

				xVelR_[i] = xVelRB_[bodyIdx] + wy * rz - wz * ry;
				yVelR_[i] = yVelRB_[bodyIdx] + wz * rx - wx * rz;
				zVelR_[i] = zVelRB_[bodyIdx] + wx * ry - wy * rx;
			}
		},
		tbb::auto_partitioner{}
	);
}

void SimBackend_CPU01::ClearRigidForcesAndAddGravity(const SimParams& params)
{
	std::ranges::fill(xForceRB_, 0.0f);
	std::ranges::fill(yForceRB_, 0.0f);
	std::ranges::fill(zForceRB_, 0.0f);
	std::ranges::fill(xTorqueRB_, 0.0f);
	std::ranges::fill(yTorqueRB_, 0.0f);
	std::ranges::fill(zTorqueRB_, 0.0f);

	if (!applyRigidBodyExternalForces_) { return; }

	// gravity acts at the center of mass
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
	{
		const float mass = massRB_[bodyIdx];
		const float massInv = massInvRB_[bodyIdx];
		if (mass <= 0.0f || massInv <= 0.0f) { continue; }

		xForceRB_[bodyIdx] += mass * params.gravity.x;
		yForceRB_[bodyIdx] += mass * params.gravity.y;
		zForceRB_[bodyIdx] += mass * params.gravity.z;
	}
}

void SimBackend_CPU01::ReduceRigidParticleForces()
{
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
	{
		const uint32_t begin = rigidParticleBeginRB_[bodyIdx];
		const uint32_t end = begin + rigidParticleCountRB_[bodyIdx];

		const float a00 = a00RB_[bodyIdx];
		const float a01 = a01RB_[bodyIdx];
		const float a02 = a02RB_[bodyIdx];
		const float a10 = a10RB_[bodyIdx];
		const float a11 = a11RB_[bodyIdx];
		const float a12 = a12RB_[bodyIdx];
		const float a20 = a20RB_[bodyIdx];
		const float a21 = a21RB_[bodyIdx];
		const float a22 = a22RB_[bodyIdx];

		// reduce rigid-particle forces of one rigid body into body force and torque
		const RigidForceTorqueAccum acc = tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(begin, end),
			RigidForceTorqueAccum{},
			[&](const tbb::blocked_range<uint32_t>& r, RigidForceTorqueAccum local) -> RigidForceTorqueAccum
			{
				for (uint32_t rigidParticleIdx = r.begin(); rigidParticleIdx != r.end(); rigidParticleIdx++)
				{
					if (useActiveRigidParticleLocalIdxs_ &&
						activeRigidParticleFlags_[rigidParticleIdx] == 0)
					{
						continue;
					}

					const float lx = xRelPosR_[rigidParticleIdx];
					const float ly = yRelPosR_[rigidParticleIdx];
					const float lz = zRelPosR_[rigidParticleIdx];

					const float xForce = xForceR_[rigidParticleIdx];
					const float yForce = yForceR_[rigidParticleIdx];
					const float zForce = zForceR_[rigidParticleIdx];

					const float rx = std::fmaf(a00, lx, std::fmaf(a01, ly, a02 * lz));
					const float ry = std::fmaf(a10, lx, std::fmaf(a11, ly, a12 * lz));
					const float rz = std::fmaf(a20, lx, std::fmaf(a21, ly, a22 * lz));

					local.xForce += xForce;
					local.yForce += yForce;
					local.zForce += zForce;
					local.xTorque += ry * zForce - rz * yForce;
					local.yTorque += rz * xForce - rx * zForce;
					local.zTorque += rx * yForce - ry * xForce;
				}

				return local;
			},
			[](RigidForceTorqueAccum a, const RigidForceTorqueAccum& b) -> RigidForceTorqueAccum
			{
				a.Add(b);
				return a;
			}
		);

		xForceRB_[bodyIdx] += acc.xForce;
		yForceRB_[bodyIdx] += acc.yForce;
		zForceRB_[bodyIdx] += acc.zForce;
		xTorqueRB_[bodyIdx] += acc.xTorque;
		yTorqueRB_[bodyIdx] += acc.yTorque;
		zTorqueRB_[bodyIdx] += acc.zTorque;
	}
}

void SimBackend_CPU01::IntegrateRigidBodies(const float timeStep)
{
	// TODO: parallelize with TBB?
	for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
	{
		const float massInv = massInvRB_[bodyIdx];

		xVelRB_[bodyIdx] += timeStep * xForceRB_[bodyIdx] * massInv;
		yVelRB_[bodyIdx] += timeStep * yForceRB_[bodyIdx] * massInv;
		zVelRB_[bodyIdx] += timeStep * zForceRB_[bodyIdx] * massInv;

		xCMRB_[bodyIdx] += timeStep * xVelRB_[bodyIdx];
		yCMRB_[bodyIdx] += timeStep * yVelRB_[bodyIdx];
		zCMRB_[bodyIdx] += timeStep * zVelRB_[bodyIdx];

		xAngMomRB_[bodyIdx] += timeStep * xTorqueRB_[bodyIdx];
		yAngMomRB_[bodyIdx] += timeStep * yTorqueRB_[bodyIdx];
		zAngMomRB_[bodyIdx] += timeStep * zTorqueRB_[bodyIdx];

		const float a00 = a00RB_[bodyIdx], a01 = a01RB_[bodyIdx], a02 = a02RB_[bodyIdx];
		const float a10 = a10RB_[bodyIdx], a11 = a11RB_[bodyIdx], a12 = a12RB_[bodyIdx];
		const float a20 = a20RB_[bodyIdx], a21 = a21RB_[bodyIdx], a22 = a22RB_[bodyIdx];

		const float b00 = iBodyInv00RB_[bodyIdx], b01 = iBodyInv01RB_[bodyIdx], b02 = iBodyInv02RB_[bodyIdx];
		const float b10 = iBodyInv10RB_[bodyIdx], b11 = iBodyInv11RB_[bodyIdx], b12 = iBodyInv12RB_[bodyIdx];
		const float b20 = iBodyInv20RB_[bodyIdx], b21 = iBodyInv21RB_[bodyIdx], b22 = iBodyInv22RB_[bodyIdx];

		const float t00 = a00 * b00 + a01 * b10 + a02 * b20;
		const float t01 = a00 * b01 + a01 * b11 + a02 * b21;
		const float t02 = a00 * b02 + a01 * b12 + a02 * b22;
		const float t10 = a10 * b00 + a11 * b10 + a12 * b20;
		const float t11 = a10 * b01 + a11 * b11 + a12 * b21;
		const float t12 = a10 * b02 + a11 * b12 + a12 * b22;
		const float t20 = a20 * b00 + a21 * b10 + a22 * b20;
		const float t21 = a20 * b01 + a21 * b11 + a22 * b21;
		const float t22 = a20 * b02 + a21 * b12 + a22 * b22;

		const float i00 = t00 * a00 + t01 * a01 + t02 * a02;
		const float i01 = t00 * a10 + t01 * a11 + t02 * a12;
		const float i02 = t00 * a20 + t01 * a21 + t02 * a22;
		const float i10 = t10 * a00 + t11 * a01 + t12 * a02;
		const float i11 = t10 * a10 + t11 * a11 + t12 * a12;
		const float i12 = t10 * a20 + t11 * a21 + t12 * a22;
		const float i20 = t20 * a00 + t21 * a01 + t22 * a02;
		const float i21 = t20 * a10 + t21 * a11 + t22 * a12;
		const float i22 = t20 * a20 + t21 * a21 + t22 * a22;

		const float lx = xAngMomRB_[bodyIdx];
		const float ly = yAngMomRB_[bodyIdx];
		const float lz = zAngMomRB_[bodyIdx];
		const float wx = std::fmaf(i00, lx, std::fmaf(i01, ly, i02 * lz));
		const float wy = std::fmaf(i10, lx, std::fmaf(i11, ly, i12 * lz));
		const float wz = std::fmaf(i20, lx, std::fmaf(i21, ly, i22 * lz));

		xOmegaRB_[bodyIdx] = wx;
		yOmegaRB_[bodyIdx] = wy;
		zOmegaRB_[bodyIdx] = wz;

		a00RB_[bodyIdx] += timeStep * (-wz * a10 + wy * a20);
		a01RB_[bodyIdx] += timeStep * (-wz * a11 + wy * a21);
		a02RB_[bodyIdx] += timeStep * (-wz * a12 + wy * a22);
		a10RB_[bodyIdx] += timeStep * ( wz * a00 - wx * a20);
		a11RB_[bodyIdx] += timeStep * ( wz * a01 - wx * a21);
		a12RB_[bodyIdx] += timeStep * ( wz * a02 - wx * a22);
		a20RB_[bodyIdx] += timeStep * (-wy * a00 + wx * a10);
		a21RB_[bodyIdx] += timeStep * (-wy * a01 + wx * a11);
		a22RB_[bodyIdx] += timeStep * (-wy * a02 + wx * a12);

		// use Gram-Schmidt algo on orientation columns
		float b1x = a00RB_[bodyIdx];
		float b1y = a10RB_[bodyIdx];
		float b1z = a20RB_[bodyIdx];
		Normalize3(b1x, b1y, b1z);

		float b2x = a01RB_[bodyIdx];
		float b2y = a11RB_[bodyIdx];
		float b2z = a21RB_[bodyIdx];
		const float b1DotA2 = Dot3(b1x, b1y, b1z, b2x, b2y, b2z);
		b2x -= b1DotA2 * b1x;
		b2y -= b1DotA2 * b1y;
		b2z -= b1DotA2 * b1z;
		Normalize3(b2x, b2y, b2z);

		float b3x = a02RB_[bodyIdx];
		float b3y = a12RB_[bodyIdx];
		float b3z = a22RB_[bodyIdx];
		const float b1DotA3 = Dot3(b1x, b1y, b1z, b3x, b3y, b3z);
		const float b2DotA3 = Dot3(b2x, b2y, b2z, b3x, b3y, b3z);
		b3x -= b1DotA3 * b1x + b2DotA3 * b2x;
		b3y -= b1DotA3 * b1y + b2DotA3 * b2y;
		b3z -= b1DotA3 * b1z + b2DotA3 * b2z;
		Normalize3(b3x, b3y, b3z);

		a00RB_[bodyIdx] = b1x; a01RB_[bodyIdx] = b2x; a02RB_[bodyIdx] = b3x;
		a10RB_[bodyIdx] = b1y; a11RB_[bodyIdx] = b2y; a12RB_[bodyIdx] = b3y;
		a20RB_[bodyIdx] = b1z; a21RB_[bodyIdx] = b2z; a22RB_[bodyIdx] = b3z;
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
