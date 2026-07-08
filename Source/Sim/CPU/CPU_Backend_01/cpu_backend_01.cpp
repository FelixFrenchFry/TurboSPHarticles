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
	auto& R = src.rigidBodies;
	N_F_ = F.N_;
	N_B_ = B.N_;
	N_rigidBodies_ = R.N_;
	N_rigidBoundaryParticles_ = R.N_rigidBoundaryParticles_;
	simStep_ = src.step;
	simTime_ = src.time;
	F.ensureParticleIdsPresent();
	B.ensureParticleIdsPresent();
	B.EnsureBoundaryKinematicsPresent();
	B.EnsureBoundaryCoefficientsPresent(params.massB, params.restDensity);
	R.ensureRigidBodyIdsPresent();
	src.ValidateRigidBoundaryAttachments();

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
	bodyIdB_ = std::move(B.bodyId);
	rigidBoundaryParticleIdxB_ = std::move(B.rigidBoundaryParticleIdx);
	xPosB_ = std::move(B.xPos);
	yPosB_ = std::move(B.yPos);
	zPosB_ = std::move(B.zPos);
	xVelB_ = std::move(B.xVel);
	yVelB_ = std::move(B.yVel);
	zVelB_ = std::move(B.zVel);
	volB_ = std::move(B.volB);
	psiB_ = std::move(B.psiB);

	// transfer to rigid body data structures
	idR_ = std::move(R.id);
	rigidBoundaryParticleBeginR_ = std::move(R.rigidBoundaryParticleBegin);
	rigidBoundaryParticleCountR_ = std::move(R.rigidBoundaryParticleCount);

	xCMR_ = std::move(R.xCM);
	yCMR_ = std::move(R.yCM);
	zCMR_ = std::move(R.zCM);
	xVelR_ = std::move(R.xVel);
	yVelR_ = std::move(R.yVel);
	zVelR_ = std::move(R.zVel);
	xOmegaR_ = std::move(R.xOmega);
	yOmegaR_ = std::move(R.yOmega);
	zOmegaR_ = std::move(R.zOmega);
	xAngMomR_ = std::move(R.xAngMom);
	yAngMomR_ = std::move(R.yAngMom);
	zAngMomR_ = std::move(R.zAngMom);
	xForceR_.assign(N_rigidBodies_, 0.0f);
	yForceR_.assign(N_rigidBodies_, 0.0f);
	zForceR_.assign(N_rigidBodies_, 0.0f);
	xTorqueR_.assign(N_rigidBodies_, 0.0f);
	yTorqueR_.assign(N_rigidBodies_, 0.0f);
	zTorqueR_.assign(N_rigidBodies_, 0.0f);
	xForceRigidBoundaryParticle_.assign(N_rigidBoundaryParticles_, 0.0f);
	yForceRigidBoundaryParticle_.assign(N_rigidBoundaryParticles_, 0.0f);
	zForceRigidBoundaryParticle_.assign(N_rigidBoundaryParticles_, 0.0f);

	a00R_ = std::move(R.a00);
	a01R_ = std::move(R.a01);
	a02R_ = std::move(R.a02);
	a10R_ = std::move(R.a10);
	a11R_ = std::move(R.a11);
	a12R_ = std::move(R.a12);
	a20R_ = std::move(R.a20);
	a21R_ = std::move(R.a21);
	a22R_ = std::move(R.a22);

	massR_ = std::move(R.mass);
	massInvR_ = std::move(R.massInv);

	iBodyInv00R_ = std::move(R.iBodyInv00);
	iBodyInv01R_ = std::move(R.iBodyInv01);
	iBodyInv02R_ = std::move(R.iBodyInv02);
	iBodyInv10R_ = std::move(R.iBodyInv10);
	iBodyInv11R_ = std::move(R.iBodyInv11);
	iBodyInv12R_ = std::move(R.iBodyInv12);
	iBodyInv20R_ = std::move(R.iBodyInv20);
	iBodyInv21R_ = std::move(R.iBodyInv21);
	iBodyInv22R_ = std::move(R.iBodyInv22);

	xLocalRigidBoundaryParticle_ = std::move(R.xLocalRigidBoundaryParticle);
	yLocalRigidBoundaryParticle_ = std::move(R.yLocalRigidBoundaryParticle);
	zLocalRigidBoundaryParticle_ = std::move(R.zLocalRigidBoundaryParticle);

	// init neighbor list sizes
	neighborList_->Resize(N_F_, N_B_, params);
}

// TODO: re-implement this to free up memory as soon as copied out!
void SimBackend_CPU01::PushToUniversalSimState(Sync::UniversalSimState& dst) const
{
	auto& F = dst.fluidParticles;
	auto& B = dst.boundaryParticles;
	auto& R = dst.rigidBodies;

	dst.step = simStep_;
	dst.time = simTime_;

	// make sure the destination data structures have enough capacity
	dst.ResizeFluid(N_F_);
	dst.ResizeBoundary(N_B_);
	dst.ResizeRigidBodies(N_rigidBodies_);
	R.resizeRigidBoundaryParticles(N_rigidBoundaryParticles_);

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
				B.bodyId[i] = bodyIdB_[i];
				B.rigidBoundaryParticleIdx[i] = rigidBoundaryParticleIdxB_[i];
				B.xPos[i] = xPosB_[i];
				B.yPos[i] = yPosB_[i];
				B.zPos[i] = zPosB_[i];
				B.xVel[i] = xVelB_[i];
				B.yVel[i] = yVelB_[i];
				B.zVel[i] = zVelB_[i];
				B.volB[i] = volB_[i];
				B.psiB[i] = psiB_[i];
			}
		},
		tbb::auto_partitioner{}
	);

	F.N_ = N_F_;
	B.N_ = N_B_;
	R.N_ = N_rigidBodies_;
	R.N_rigidBoundaryParticles_ = N_rigidBoundaryParticles_;

	R.id = idR_;
	R.rigidBoundaryParticleBegin = rigidBoundaryParticleBeginR_;
	R.rigidBoundaryParticleCount = rigidBoundaryParticleCountR_;

	R.xCM = xCMR_;
	R.yCM = yCMR_;
	R.zCM = zCMR_;
	R.xVel = xVelR_;
	R.yVel = yVelR_;
	R.zVel = zVelR_;
	R.xOmega = xOmegaR_;
	R.yOmega = yOmegaR_;
	R.zOmega = zOmegaR_;
	R.xAngMom = xAngMomR_;
	R.yAngMom = yAngMomR_;
	R.zAngMom = zAngMomR_;

	R.a00 = a00R_;
	R.a01 = a01R_;
	R.a02 = a02R_;
	R.a10 = a10R_;
	R.a11 = a11R_;
	R.a12 = a12R_;
	R.a20 = a20R_;
	R.a21 = a21R_;
	R.a22 = a22R_;

	R.mass = massR_;
	R.massInv = massInvR_;

	R.iBodyInv00 = iBodyInv00R_;
	R.iBodyInv01 = iBodyInv01R_;
	R.iBodyInv02 = iBodyInv02R_;
	R.iBodyInv10 = iBodyInv10R_;
	R.iBodyInv11 = iBodyInv11R_;
	R.iBodyInv12 = iBodyInv12R_;
	R.iBodyInv20 = iBodyInv20R_;
	R.iBodyInv21 = iBodyInv21R_;
	R.iBodyInv22 = iBodyInv22R_;

	R.xLocalRigidBoundaryParticle = xLocalRigidBoundaryParticle_;
	R.yLocalRigidBoundaryParticle = yLocalRigidBoundaryParticle_;
	R.zLocalRigidBoundaryParticle = zLocalRigidBoundaryParticle_;
}

void SimBackend_CPU01::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	snap.simStep = simStep_;
	snap.simTime = simTime_;

	snap.N_B = N_B_;
	snap.idB.resize(N_B_);
	snap.bodyIdB.resize(N_B_);
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
				snap.bodyIdB[i] = bodyIdB_[i];
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
	snap.bodyIdB.resize(N_B_);

	// pack boundary positions into the visual snapshot's data structures
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posB[i] = glm::vec3{ xPosB_[i], yPosB_[i], zPosB_[i] };
				snap.bodyIdB[i] = bodyIdB_[i];
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
		Add(usage, MemUsageOf_Vector(bodyIdB_));
		Add(usage, MemUsageOf_Vector(rigidBoundaryParticleIdxB_));
		Add(usage, MemUsageOf_Vector(xPosB_));
		Add(usage, MemUsageOf_Vector(yPosB_));
		Add(usage, MemUsageOf_Vector(zPosB_));
		Add(usage, MemUsageOf_Vector(xVelB_));
		Add(usage, MemUsageOf_Vector(yVelB_));
		Add(usage, MemUsageOf_Vector(zVelB_));
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
		Add(usage, MemUsageOf_Vector(idR_));
		Add(usage, MemUsageOf_Vector(rigidBoundaryParticleBeginR_));
		Add(usage, MemUsageOf_Vector(rigidBoundaryParticleCountR_));
		Add(usage, MemUsageOf_Vector(xCMR_));
		Add(usage, MemUsageOf_Vector(yCMR_));
		Add(usage, MemUsageOf_Vector(zCMR_));
		Add(usage, MemUsageOf_Vector(xVelR_));
		Add(usage, MemUsageOf_Vector(yVelR_));
		Add(usage, MemUsageOf_Vector(zVelR_));
		Add(usage, MemUsageOf_Vector(xOmegaR_));
		Add(usage, MemUsageOf_Vector(yOmegaR_));
		Add(usage, MemUsageOf_Vector(zOmegaR_));
		Add(usage, MemUsageOf_Vector(xAngMomR_));
		Add(usage, MemUsageOf_Vector(yAngMomR_));
		Add(usage, MemUsageOf_Vector(zAngMomR_));
		Add(usage, MemUsageOf_Vector(xForceR_));
		Add(usage, MemUsageOf_Vector(yForceR_));
		Add(usage, MemUsageOf_Vector(zForceR_));
		Add(usage, MemUsageOf_Vector(xTorqueR_));
		Add(usage, MemUsageOf_Vector(yTorqueR_));
		Add(usage, MemUsageOf_Vector(zTorqueR_));
		Add(usage, MemUsageOf_Vector(xForceRigidBoundaryParticle_));
		Add(usage, MemUsageOf_Vector(yForceRigidBoundaryParticle_));
		Add(usage, MemUsageOf_Vector(zForceRigidBoundaryParticle_));
		Add(usage, MemUsageOf_Vector(a00R_));
		Add(usage, MemUsageOf_Vector(a01R_));
		Add(usage, MemUsageOf_Vector(a02R_));
		Add(usage, MemUsageOf_Vector(a10R_));
		Add(usage, MemUsageOf_Vector(a11R_));
		Add(usage, MemUsageOf_Vector(a12R_));
		Add(usage, MemUsageOf_Vector(a20R_));
		Add(usage, MemUsageOf_Vector(a21R_));
		Add(usage, MemUsageOf_Vector(a22R_));
		Add(usage, MemUsageOf_Vector(massR_));
		Add(usage, MemUsageOf_Vector(massInvR_));
		Add(usage, MemUsageOf_Vector(iBodyInv00R_));
		Add(usage, MemUsageOf_Vector(iBodyInv01R_));
		Add(usage, MemUsageOf_Vector(iBodyInv02R_));
		Add(usage, MemUsageOf_Vector(iBodyInv10R_));
		Add(usage, MemUsageOf_Vector(iBodyInv11R_));
		Add(usage, MemUsageOf_Vector(iBodyInv12R_));
		Add(usage, MemUsageOf_Vector(iBodyInv20R_));
		Add(usage, MemUsageOf_Vector(iBodyInv21R_));
		Add(usage, MemUsageOf_Vector(iBodyInv22R_));
		Add(usage, MemUsageOf_Vector(xLocalRigidBoundaryParticle_));
		Add(usage, MemUsageOf_Vector(yLocalRigidBoundaryParticle_));
		Add(usage, MemUsageOf_Vector(zLocalRigidBoundaryParticle_));

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

	benchmarksTracker_.StartBench({ Bench::BenchOp::Statistics_Computation });

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

	benchmarksTracker_.StopBench({ Bench::BenchOp::Statistics_Computation });
}

void SimBackend_CPU01::ExecuteNeighborSearch(const SimParams& params, const bool bench)
{
	const bool runCountQuery = neighborList_->RunCountQuery();
	const bool runFillQuery = neighborList_->RunFillQuery();

	// initial re-ordering of the data before the initial neighbor search
	if (bench)
	{
		benchmarksTracker_.StartBench({ Bench::BenchOp::ParticleData_Reordering });
	}
	neighborSearch_->DataReOrder(params);
	affinityPart_ = std::make_unique<tbb::affinity_partitioner>(); // reset mapping
	if (bench)
	{
		benchmarksTracker_.StopBench({ Bench::BenchOp::ParticleData_Reordering });
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

void SimBackend_CPU01::UpdateRigidBoundaryParticles()
{
	// TODO: parallelize with TBB?
	for (uint32_t i = 0; i < N_B_; i++)
	{
		const uint32_t bodyIdx = bodyIdB_[i];
		if (bodyIdx == Sync::InvalidRigidBodyId)
		{
			xVelB_[i] = 0.0f;
			yVelB_[i] = 0.0f;
			zVelB_[i] = 0.0f;
			continue;
		}

		const uint32_t rigidBoundaryParticleIdx = rigidBoundaryParticleIdxB_[i];
		if (bodyIdx >= N_rigidBodies_ || rigidBoundaryParticleIdx >= N_rigidBoundaryParticles_)
		{
			throw std::runtime_error("Invalid CPU01 rigid boundary attachment.");
		}
		const uint32_t rigidBoundaryParticleBegin = rigidBoundaryParticleBeginR_[bodyIdx];
		const uint32_t rigidBoundaryParticleCount = rigidBoundaryParticleCountR_[bodyIdx];

		// TODO: gate behind debug compiler flag?
		if (rigidBoundaryParticleBegin > N_rigidBoundaryParticles_ ||
			rigidBoundaryParticleCount > N_rigidBoundaryParticles_ - rigidBoundaryParticleBegin)
		{
			throw std::runtime_error("Invalid CPU01 rigid boundary particle range.");
		}
		if (rigidBoundaryParticleIdx < rigidBoundaryParticleBegin || rigidBoundaryParticleIdx >= rigidBoundaryParticleBegin + rigidBoundaryParticleCount)
		{
			throw std::runtime_error("CPU01 rigid boundary particle belongs to a different body.");
		}

		const float lx = xLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];
		const float ly = yLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];
		const float lz = zLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];

		const float rx = std::fmaf(a00R_[bodyIdx], lx,
			std::fmaf(a01R_[bodyIdx], ly, a02R_[bodyIdx] * lz));
		const float ry = std::fmaf(a10R_[bodyIdx], lx,
			std::fmaf(a11R_[bodyIdx], ly, a12R_[bodyIdx] * lz));
		const float rz = std::fmaf(a20R_[bodyIdx], lx,
			std::fmaf(a21R_[bodyIdx], ly, a22R_[bodyIdx] * lz));

		xPosB_[i] = xCMR_[bodyIdx] + rx;
		yPosB_[i] = yCMR_[bodyIdx] + ry;
		zPosB_[i] = zCMR_[bodyIdx] + rz;

		const float wx = xOmegaR_[bodyIdx];
		const float wy = yOmegaR_[bodyIdx];
		const float wz = zOmegaR_[bodyIdx];

		xVelB_[i] = xVelR_[bodyIdx] + wy * rz - wz * ry;
		yVelB_[i] = yVelR_[bodyIdx] + wz * rx - wx * rz;
		zVelB_[i] = zVelR_[bodyIdx] + wx * ry - wy * rx;
	}
}

void SimBackend_CPU01::ClearRigidForcesAndAddGravity(const SimParams& params)
{
	std::ranges::fill(xForceR_, 0.0f);
	std::ranges::fill(yForceR_, 0.0f);
	std::ranges::fill(zForceR_, 0.0f);
	std::ranges::fill(xTorqueR_, 0.0f);
	std::ranges::fill(yTorqueR_, 0.0f);
	std::ranges::fill(zTorqueR_, 0.0f);

	// gravity acts at the center of mass
	for (uint32_t bodyIdx = 0; bodyIdx < N_rigidBodies_; bodyIdx++)
	{
		if (massInvR_[bodyIdx] <= 0.0f) { continue; }

		const float mass = massR_[bodyIdx];
		if (mass <= 0.0f) { continue; }

		xForceR_[bodyIdx] += mass * params.gravity.x;
		yForceR_[bodyIdx] += mass * params.gravity.y;
		zForceR_[bodyIdx] += mass * params.gravity.z;
	}
}

void SimBackend_CPU01::ClearRigidBoundaryParticleForces()
{
	std::ranges::fill(xForceRigidBoundaryParticle_, 0.0f);
	std::ranges::fill(yForceRigidBoundaryParticle_, 0.0f);
	std::ranges::fill(zForceRigidBoundaryParticle_, 0.0f);
}

void SimBackend_CPU01::ReduceRigidBoundaryParticleForces()
{
	// TODO: parallelize with TBB?
	for (uint32_t bodyIdx = 0; bodyIdx < N_rigidBodies_; bodyIdx++)
	{
		const uint32_t begin = rigidBoundaryParticleBeginR_[bodyIdx];
		const uint32_t end = begin + rigidBoundaryParticleCountR_[bodyIdx];

		for (uint32_t rigidBoundaryParticleIdx = begin; rigidBoundaryParticleIdx < end; rigidBoundaryParticleIdx++)
		{
			const float xForce = xForceRigidBoundaryParticle_[rigidBoundaryParticleIdx];
			const float yForce = yForceRigidBoundaryParticle_[rigidBoundaryParticleIdx];
			const float zForce = zForceRigidBoundaryParticle_[rigidBoundaryParticleIdx];

			const float lx = xLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];
			const float ly = yLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];
			const float lz = zLocalRigidBoundaryParticle_[rigidBoundaryParticleIdx];

			const float rx = std::fmaf(a00R_[bodyIdx], lx,
				std::fmaf(a01R_[bodyIdx], ly, a02R_[bodyIdx] * lz));
			const float ry = std::fmaf(a10R_[bodyIdx], lx,
				std::fmaf(a11R_[bodyIdx], ly, a12R_[bodyIdx] * lz));
			const float rz = std::fmaf(a20R_[bodyIdx], lx,
				std::fmaf(a21R_[bodyIdx], ly, a22R_[bodyIdx] * lz));

			xForceR_[bodyIdx] += xForce;
			yForceR_[bodyIdx] += yForce;
			zForceR_[bodyIdx] += zForce;
			xTorqueR_[bodyIdx] += ry * zForce - rz * yForce;
			yTorqueR_[bodyIdx] += rz * xForce - rx * zForce;
			zTorqueR_[bodyIdx] += rx * yForce - ry * xForce;
		}
	}
}

void SimBackend_CPU01::IntegrateRigidBodies(const float timeStep)
{
	// TODO: parallelize with TBB?
	for (uint32_t bodyIdx = 0; bodyIdx < N_rigidBodies_; bodyIdx++)
	{
		const float massInv = massInvR_[bodyIdx];

		xVelR_[bodyIdx] += timeStep * xForceR_[bodyIdx] * massInv;
		yVelR_[bodyIdx] += timeStep * yForceR_[bodyIdx] * massInv;
		zVelR_[bodyIdx] += timeStep * zForceR_[bodyIdx] * massInv;

		xCMR_[bodyIdx] += timeStep * xVelR_[bodyIdx];
		yCMR_[bodyIdx] += timeStep * yVelR_[bodyIdx];
		zCMR_[bodyIdx] += timeStep * zVelR_[bodyIdx];

		xAngMomR_[bodyIdx] += timeStep * xTorqueR_[bodyIdx];
		yAngMomR_[bodyIdx] += timeStep * yTorqueR_[bodyIdx];
		zAngMomR_[bodyIdx] += timeStep * zTorqueR_[bodyIdx];

		const float a00 = a00R_[bodyIdx], a01 = a01R_[bodyIdx], a02 = a02R_[bodyIdx];
		const float a10 = a10R_[bodyIdx], a11 = a11R_[bodyIdx], a12 = a12R_[bodyIdx];
		const float a20 = a20R_[bodyIdx], a21 = a21R_[bodyIdx], a22 = a22R_[bodyIdx];

		const float b00 = iBodyInv00R_[bodyIdx], b01 = iBodyInv01R_[bodyIdx], b02 = iBodyInv02R_[bodyIdx];
		const float b10 = iBodyInv10R_[bodyIdx], b11 = iBodyInv11R_[bodyIdx], b12 = iBodyInv12R_[bodyIdx];
		const float b20 = iBodyInv20R_[bodyIdx], b21 = iBodyInv21R_[bodyIdx], b22 = iBodyInv22R_[bodyIdx];

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

		const float lx = xAngMomR_[bodyIdx];
		const float ly = yAngMomR_[bodyIdx];
		const float lz = zAngMomR_[bodyIdx];
		const float wx = std::fmaf(i00, lx, std::fmaf(i01, ly, i02 * lz));
		const float wy = std::fmaf(i10, lx, std::fmaf(i11, ly, i12 * lz));
		const float wz = std::fmaf(i20, lx, std::fmaf(i21, ly, i22 * lz));

		xOmegaR_[bodyIdx] = wx;
		yOmegaR_[bodyIdx] = wy;
		zOmegaR_[bodyIdx] = wz;

		a00R_[bodyIdx] += timeStep * (-wz * a10 + wy * a20);
		a01R_[bodyIdx] += timeStep * (-wz * a11 + wy * a21);
		a02R_[bodyIdx] += timeStep * (-wz * a12 + wy * a22);
		a10R_[bodyIdx] += timeStep * ( wz * a00 - wx * a20);
		a11R_[bodyIdx] += timeStep * ( wz * a01 - wx * a21);
		a12R_[bodyIdx] += timeStep * ( wz * a02 - wx * a22);
		a20R_[bodyIdx] += timeStep * (-wy * a00 + wx * a10);
		a21R_[bodyIdx] += timeStep * (-wy * a01 + wx * a11);
		a22R_[bodyIdx] += timeStep * (-wy * a02 + wx * a12);

		// use Gram-Schmidt algo on orientation columns
		float b1x = a00R_[bodyIdx];
		float b1y = a10R_[bodyIdx];
		float b1z = a20R_[bodyIdx];
		Normalize3(b1x, b1y, b1z);

		float b2x = a01R_[bodyIdx];
		float b2y = a11R_[bodyIdx];
		float b2z = a21R_[bodyIdx];
		const float b1DotA2 = Dot3(b1x, b1y, b1z, b2x, b2y, b2z);
		b2x -= b1DotA2 * b1x;
		b2y -= b1DotA2 * b1y;
		b2z -= b1DotA2 * b1z;
		Normalize3(b2x, b2y, b2z);

		float b3x = a02R_[bodyIdx];
		float b3y = a12R_[bodyIdx];
		float b3z = a22R_[bodyIdx];
		const float b1DotA3 = Dot3(b1x, b1y, b1z, b3x, b3y, b3z);
		const float b2DotA3 = Dot3(b2x, b2y, b2z, b3x, b3y, b3z);
		b3x -= b1DotA3 * b1x + b2DotA3 * b2x;
		b3y -= b1DotA3 * b1y + b2DotA3 * b2y;
		b3z -= b1DotA3 * b1z + b2DotA3 * b2z;
		Normalize3(b3x, b3y, b3z);

		a00R_[bodyIdx] = b1x; a01R_[bodyIdx] = b2x; a02R_[bodyIdx] = b3x;
		a10R_[bodyIdx] = b1y; a11R_[bodyIdx] = b2y; a12R_[bodyIdx] = b3y;
		a20R_[bodyIdx] = b1z; a21R_[bodyIdx] = b2z; a22R_[bodyIdx] = b3z;
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
