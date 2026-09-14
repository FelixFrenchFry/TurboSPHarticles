// cpu_backend.cpp

// external
#include <algorithm>
#include <atomic>
#include <cmath>
#include <glm/exponential.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <span>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "source/base/helpers.h"
#include "source/base/parallel_checks.h"
#include "source/simulation/CPU/simd.h"
#include "cpu_backend.h"
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
			throw std::runtime_error("Degenerate CPU rigid body orientation.");
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

	// compacts a particle block, reading retained particles at scattered
	// old local indices and writing them to their contiguous new local indices
	template <typename CopyParticleOldToNew>
	void CompactParticles(
		const std::vector<uint32_t>& oldLocalIdxs,
		const tbb::blocked_range<uint32_t>& blockRange,
		const CopyParticleOldToNew& copyParticleOldToNew)
	{
		for (uint32_t newLocalIdx = blockRange.begin(); newLocalIdx < blockRange.end(); newLocalIdx++)
		{
			const uint32_t oldLocalIdx = oldLocalIdxs[newLocalIdx];
			if (oldLocalIdx == newLocalIdx) { continue; }

			copyParticleOldToNew(oldLocalIdx, newLocalIdx);
		}
	}

	// copies one field of a particle block behind the retained prefix of the storage
	template <typename DstValues, typename SrcValues>
	void AppendParticleField(
		DstValues& dstValues,
		const SrcValues& srcValues,
		const uint32_t appendBegin,
		const tbb::blocked_range<uint32_t>& blockRange)
	{
		std::copy(
			srcValues.begin() + blockRange.begin(),
			srcValues.begin() + blockRange.end(),
			dstValues.begin() + appendBegin + blockRange.begin());
	}

	// reserves capacity for a particle field, plus extra X% to avoid repeated reallocations during growth
	template <typename Values>
	void ReserveParticleFieldWithExtra(
		Values& values,
		const uint32_t N_particles,
		const float extraAllocCapacityPct)
	{
		if (values.capacity() >= N_particles) { return; }

		const uint64_t capacityTarget = std::min(
			Sim::CheckedParticleAdd(
				N_particles,
				Base::ExtraPercent(N_particles, extraAllocCapacityPct),
				"CPU particle reserve"),
			static_cast<uint64_t>(Sim::MaxLocalParticleCount));
		values.reserve(capacityTarget);
	}

} // namespace anonymous

// ---------------------------------------------------------------------------------------------------

SimBackend_CPU::SimBackend_CPU(
	Bench::BenchmarksTracker& benchmarksTracker,
	const SimParams& params)
	: benchmarksTracker_{ benchmarksTracker }
{
	affinityParts_ = std::make_unique<StepAffinityPartitioners>();

	// select neighbor search method
	if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::BruteForce)
	{
		neighborSearch_ = std::make_unique<NS_BruteForce_CPU>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::CellLinkedList_Compact)
	{
		neighborSearch_ = std::make_unique<NS_CellLinkedList_Compact_CPU>(*this);
	}
	else
	{
		throw std::runtime_error("Unsupported 'NeighborSearchMethod' requested.");
	}
	neighborSearch_->Init();

	// select neighbor list method
	if (params.neighborListMethod ==
		Base::NeighborListMethod::ChunkedSELLX_Explicit)
	{
		neighborList_ = std::make_unique<NeighborLists_ChunkedSELL_Explicit_CPU>();
	}
	else if (params.neighborListMethod ==
		Base::NeighborListMethod::ChunkedSELLX_AdaptCompress)
	{
		neighborList_ = std::make_unique<NeighborLists_ChunkedSELL_AdaptCompress_CPU>();
	}
	else
	{
		throw std::runtime_error(
			"CPU supports only ChunkedSELL neighbor-list storage.");
	}
}

void SimBackend_CPU::PullFromUniversalSimState(
	Sync::UniversalSimState& src, const SimParams& params)
{
	affinityParts_ = std::make_unique<StepAffinityPartitioners>(); // reset mapping
	distributedFluidSolverSync_ = nullptr;
	useDistributedParticleLayout_ = false;
	N_ownedF_ = 0;
	N_ownedB_ = 0;
	outerFluidParticleLocalIdxs_.clear();
	pressureAccFluidParticleLocalIdxs_.clear();
	pressureAccFluidParticleFlags_.clear();
	pressureAccChunkIdxs_.clear();
	innerFluidChunkIdxs_.clear();
	outerFluidChunkIdxs_.clear();
	pressureAccIsOwnedPrefix_ = false;
	activeRigidParticleLocalIdxs_.clear();
	activeRigidParticleFlags_.clear();
	newFluidLocalIdx_FromOldLocalIdx_.clear();
	chunkFlagsScratch_.clear();
	particleFlagsScratch_.clear();
	useDistributedStepWork_ = false;
	useActiveRigidParticleLocalIdxs_ = false;
	integrateRigidBodies_ = true;
	applyRigidBodyExternalForces_ = true;

	auto& F = src.particlesF;
	auto& B = src.particlesB;
	auto& R = src.particlesR;
	auto& RB = src.rigidBodies;
	(void)CheckedLocalParticleCount(F.N_, "CPU fluid state");
	(void)CheckedLocalParticleCount(B.N_, "CPU boundary state");
	(void)CheckedLocalParticleCount(R.N_, "CPU rigid state");
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

	// (no resize, because it is included in the transfer operations)

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

void SimBackend_CPU::PushToUniversalSimState(
	Sync::UniversalSimState& dst,
	const bool backendWillBeRetired)
{
	auto& F = dst.particlesF;
	auto& B = dst.particlesB;
	auto& R = dst.particlesR;
	auto& RB = dst.rigidBodies;
	F.N_ = N_F_;
	B.N_ = N_B_;
	R.N_ = N_R_;
	RB.N_ = N_RB_;
	dst.step = simStep_;
	dst.time = simTime_;

	const auto transferField = [backendWillBeRetired](auto& dstField, auto& srcField)
	{
		if (backendWillBeRetired)
		{
			dstField = std::move(srcField);
		}
		else
		{
			dstField = srcField;
		}
	};

	// (no resize, because it is included in the transfer operations)

	// transfer from backend fluid data structures
	transferField(F.id, idF_);
	transferField(F.xPos, xPosF_);
	transferField(F.yPos, yPosF_);
	transferField(F.zPos, zPosF_);
	transferField(F.xVel, xVelF_);
	transferField(F.yVel, yVelF_);
	transferField(F.zVel, zVelF_);
	transferField(F.xVelPred, xVelPredF_);
	transferField(F.yVelPred, yVelPredF_);
	transferField(F.zVelPred, zVelPredF_);
	transferField(F.xAccPress, xAccPressF_);
	transferField(F.yAccPress, yAccPressF_);
	transferField(F.zAccPress, zAccPressF_);
	transferField(F.sourceTerm, sourceTermF_);
	transferField(F.diagElem, diagElemF_);
	transferField(F.pressure, pressureF_);
	transferField(F.density, densityF_);

	// transfer from backend boundary data structures
	transferField(B.id, idB_);
	transferField(B.xPos, xPosB_);
	transferField(B.yPos, yPosB_);
	transferField(B.zPos, zPosB_);
	transferField(B.volB, volB_);
	transferField(B.psiB, psiB_);

	// transfer from backend rigid particle data structures
	transferField(R.id, idR_);
	transferField(R.bodyId, bodyIdR_);
	transferField(R.xRelPos, xRelPosR_);
	transferField(R.yRelPos, yRelPosR_);
	transferField(R.zRelPos, zRelPosR_);
	transferField(R.volR, volR_);
	transferField(R.psiR, psiR_);

	// transfer from backend rigid body data structures
	transferField(RB.id, idRB_);
	transferField(RB.rigidParticleBegin, rigidParticleBeginRB_);
	transferField(RB.rigidParticleCount, rigidParticleCountRB_);
	transferField(RB.xCM, xCMRB_);
	transferField(RB.yCM, yCMRB_);
	transferField(RB.zCM, zCMRB_);
	transferField(RB.xVel, xVelRB_);
	transferField(RB.yVel, yVelRB_);
	transferField(RB.zVel, zVelRB_);
	transferField(RB.xOmega, xOmegaRB_);
	transferField(RB.yOmega, yOmegaRB_);
	transferField(RB.zOmega, zOmegaRB_);
	transferField(RB.xAngMom, xAngMomRB_);
	transferField(RB.yAngMom, yAngMomRB_);
	transferField(RB.zAngMom, zAngMomRB_);
	transferField(RB.a00, a00RB_);
	transferField(RB.a01, a01RB_);
	transferField(RB.a02, a02RB_);
	transferField(RB.a10, a10RB_);
	transferField(RB.a11, a11RB_);
	transferField(RB.a12, a12RB_);
	transferField(RB.a20, a20RB_);
	transferField(RB.a21, a21RB_);
	transferField(RB.a22, a22RB_);
	transferField(RB.mass, massRB_);
	transferField(RB.massInv, massInvRB_);
	transferField(RB.iBodyInv00, iBodyInv00RB_);
	transferField(RB.iBodyInv01, iBodyInv01RB_);
	transferField(RB.iBodyInv02, iBodyInv02RB_);
	transferField(RB.iBodyInv10, iBodyInv10RB_);
	transferField(RB.iBodyInv11, iBodyInv11RB_);
	transferField(RB.iBodyInv12, iBodyInv12RB_);
	transferField(RB.iBodyInv20, iBodyInv20RB_);
	transferField(RB.iBodyInv21, iBodyInv21RB_);
	transferField(RB.iBodyInv22, iBodyInv22RB_);
}

Sync::FluidConstView SimBackend_CPU::GetFluidParticleView() const
{
	return
	{
		.N_ = N_F_,
		.id = idF_,
		.xPos = xPosF_, .yPos = yPosF_, .zPos = zPosF_,
		.xVel = xVelF_, .yVel = yVelF_, .zVel = zVelF_,
		.xVelPred = xVelPredF_, .yVelPred = yVelPredF_, .zVelPred = zVelPredF_,
		.xAccPress = xAccPressF_, .yAccPress = yAccPressF_, .zAccPress = zAccPressF_,
		.sourceTerm = sourceTermF_,
		.diagElem = diagElemF_,
		.pressure = pressureF_,
		.density = densityF_
	};
}

Sync::BoundaryConstView SimBackend_CPU::GetBoundaryParticleView() const
{
	return
	{
		.N_ = N_B_,
		.id = idB_,
		.xPos = xPosB_, .yPos = yPosB_, .zPos = zPosB_,
		.volB = volB_,
		.psiB = psiB_
	};
}

Sync::RigidConstView SimBackend_CPU::GetRigidParticleView() const
{
	return
	{
		.N_ = N_R_,
		.id = idR_,
		.bodyId = bodyIdR_,
		.xRelPos = xRelPosR_, .yRelPos = yRelPosR_, .zRelPos = zRelPosR_,
		.volR = volR_,
		.psiR = psiR_
	};
}

bool SimBackend_CPU::ApplyParticleRedistribution(
	const std::vector<uint32_t>& retainedFluidOldLocalIdx_FromNewLocalIdx,
	const std::vector<uint32_t>& retainedBoundaryOldLocalIdx_FromNewLocalIdx,
	const Sync::FluidParticles& appendedParticlesF,
	const Sync::BoundaryParticles& appendedParticlesB,
	const bool redistributeParticlesB,
	const SimParams& params)
{
	const auto fluidSizeValid = [&appendedParticlesF](const auto& values)
	{
		return values.size() == appendedParticlesF.N_;
	};
	const auto fluidFieldOmitted = [](const auto& values)
	{
		return values.empty();
	};
	const auto boundarySizeValid = [&appendedParticlesB](const auto& values)
	{
		return values.size() == appendedParticlesB.N_;
	};

	// check that the appended particle blocks have the expected sizes
	// (the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
	// because the following solver step re-writes them before reading them)
	// TODO: keep it like this or copy them too, if solver step behavior changes?
	if (!fluidSizeValid(appendedParticlesF.id) ||
		!fluidSizeValid(appendedParticlesF.xPos) ||
		!fluidSizeValid(appendedParticlesF.yPos) ||
		!fluidSizeValid(appendedParticlesF.zPos) ||
		!fluidSizeValid(appendedParticlesF.xVel) ||
		!fluidSizeValid(appendedParticlesF.yVel) ||
		!fluidSizeValid(appendedParticlesF.zVel) ||
		!fluidFieldOmitted(appendedParticlesF.xVelPred) ||
		!fluidFieldOmitted(appendedParticlesF.yVelPred) ||
		!fluidFieldOmitted(appendedParticlesF.zVelPred) ||
		!fluidFieldOmitted(appendedParticlesF.xAccPress) ||
		!fluidFieldOmitted(appendedParticlesF.yAccPress) ||
		!fluidFieldOmitted(appendedParticlesF.zAccPress) ||
		!fluidFieldOmitted(appendedParticlesF.sourceTerm) ||
		!fluidFieldOmitted(appendedParticlesF.diagElem) ||
		!fluidFieldOmitted(appendedParticlesF.pressure) ||
		!fluidFieldOmitted(appendedParticlesF.density) ||
		(redistributeParticlesB &&
			(!boundarySizeValid(appendedParticlesB.id) ||
			 !boundarySizeValid(appendedParticlesB.xPos) ||
			 !boundarySizeValid(appendedParticlesB.yPos) ||
			 !boundarySizeValid(appendedParticlesB.zPos) ||
			 !boundarySizeValid(appendedParticlesB.volB) ||
			 !boundarySizeValid(appendedParticlesB.psiB))))
	{
		return false;
	}

	const auto compactionMapValid = [](
		const std::vector<uint32_t>& oldLocalIdx_FromNewLocalIdx,
		const uint32_t N_old)
	{
		if (oldLocalIdx_FromNewLocalIdx.size() > N_old) { return false; }

		#if defined(_DEBUG)
		const uint32_t N_retained = static_cast<uint32_t>(oldLocalIdx_FromNewLocalIdx.size());

		// exhaustively validate they do not source from the retained prefix and do not contain duplicates
		std::vector<uint8_t> oldLocalIdxUsed(N_old, 0);
		for (uint32_t newLocalIdx = 0; newLocalIdx < N_retained; newLocalIdx++)
		{
			const uint32_t oldLocalIdx = oldLocalIdx_FromNewLocalIdx[newLocalIdx];
			if (oldLocalIdx >= N_old ||
				(oldLocalIdx != newLocalIdx &&
				oldLocalIdx < N_retained))
			{
				return false;
			}

			if (oldLocalIdxUsed[oldLocalIdx] != 0) { return false; }
			oldLocalIdxUsed[oldLocalIdx] = 1;
		}
		#endif

		return true;
	};

	// check that the compaction maps are valid
	if (!compactionMapValid(retainedFluidOldLocalIdx_FromNewLocalIdx, N_F_) ||
		(redistributeParticlesB &&
		 !compactionMapValid(retainedBoundaryOldLocalIdx_FromNewLocalIdx, N_B_)))
	{
		return false;
	}

	// check that the final particle counts do not exceed the local storage limit
	const uint32_t N_retainedF = static_cast<uint32_t>(retainedFluidOldLocalIdx_FromNewLocalIdx.size());
	const uint32_t N_retainedB = redistributeParticlesB
		? static_cast<uint32_t>(retainedBoundaryOldLocalIdx_FromNewLocalIdx.size())
		: N_B_;
	const uint64_t N_finalF_U64 = static_cast<uint64_t>(N_retainedF) + appendedParticlesF.N_;
	const uint64_t N_finalB_U64 = redistributeParticlesB
		? static_cast<uint64_t>(N_retainedB) + appendedParticlesB.N_
		: N_B_;
	if (N_finalF_U64 > Sim::MaxLocalParticleCount ||
		N_finalB_U64 > Sim::MaxLocalParticleCount)
	{
		return false;
	}

	const uint32_t N_finalF = static_cast<uint32_t>(N_finalF_U64);
	const uint32_t N_finalB = static_cast<uint32_t>(N_finalB_U64);

	// reserve all required field capacity before compacting particles in place,
	// so subsequent resize calls can then use this capacity without reallocating
	ReserveParticleFieldWithExtra(idF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(xPosF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(yPosF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(zPosF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(xVelF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(yVelF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(zVelF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(xVelPredF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(yVelPredF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(zVelPredF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(xAccPressF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(yAccPressF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(zAccPressF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(sourceTermF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(diagElemF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(pressureF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	ReserveParticleFieldWithExtra(densityF_, N_finalF, particleFieldExtraAllocCapacityPct_);
	if (redistributeParticlesB)
	{
		ReserveParticleFieldWithExtra(idB_, N_finalB, particleFieldExtraAllocCapacityPct_);
		ReserveParticleFieldWithExtra(xPosB_, N_finalB, particleFieldExtraAllocCapacityPct_);
		ReserveParticleFieldWithExtra(yPosB_, N_finalB, particleFieldExtraAllocCapacityPct_);
		ReserveParticleFieldWithExtra(zPosB_, N_finalB, particleFieldExtraAllocCapacityPct_);
		ReserveParticleFieldWithExtra(volB_, N_finalB, particleFieldExtraAllocCapacityPct_);
		ReserveParticleFieldWithExtra(psiB_, N_finalB, particleFieldExtraAllocCapacityPct_);
	}

	const auto copyFluidParticleOldToNew = [&](
		const uint32_t oldLocalIdx,
		const uint32_t newLocalIdx)
	{
		idF_[newLocalIdx] = idF_[oldLocalIdx];
		xPosF_[newLocalIdx] = xPosF_[oldLocalIdx];
		yPosF_[newLocalIdx] = yPosF_[oldLocalIdx];
		zPosF_[newLocalIdx] = zPosF_[oldLocalIdx];
		xVelF_[newLocalIdx] = xVelF_[oldLocalIdx];
		yVelF_[newLocalIdx] = yVelF_[oldLocalIdx];
		zVelF_[newLocalIdx] = zVelF_[oldLocalIdx];
		xVelPredF_[newLocalIdx] = xVelPredF_[oldLocalIdx];
		yVelPredF_[newLocalIdx] = yVelPredF_[oldLocalIdx];
		zVelPredF_[newLocalIdx] = zVelPredF_[oldLocalIdx];
		xAccPressF_[newLocalIdx] = xAccPressF_[oldLocalIdx];
		yAccPressF_[newLocalIdx] = yAccPressF_[oldLocalIdx];
		zAccPressF_[newLocalIdx] = zAccPressF_[oldLocalIdx];
		sourceTermF_[newLocalIdx] = sourceTermF_[oldLocalIdx];
		diagElemF_[newLocalIdx] = diagElemF_[oldLocalIdx];
		pressureF_[newLocalIdx] = pressureF_[oldLocalIdx];
		densityF_[newLocalIdx] = densityF_[oldLocalIdx];
	};

	// compact retained fluid particles into a dense prefix, filling its holes with particles from the old tail
	// (index blocks processed in parallel, and all fields of a moved particle together)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_retainedF, particleCompactionBlockSize_),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
		{
			CompactParticles(
				retainedFluidOldLocalIdx_FromNewLocalIdx,
				blockRange,
				copyFluidParticleOldToNew);
		},
		tbb::auto_partitioner{});

	if (redistributeParticlesB)
	{
		const auto copyBoundaryParticleOldToNew = [&](
			const uint32_t oldLocalIdx,
			const uint32_t newLocalIdx)
		{
			idB_[newLocalIdx] = idB_[oldLocalIdx];
			xPosB_[newLocalIdx] = xPosB_[oldLocalIdx];
			yPosB_[newLocalIdx] = yPosB_[oldLocalIdx];
			zPosB_[newLocalIdx] = zPosB_[oldLocalIdx];
			volB_[newLocalIdx] = volB_[oldLocalIdx];
			psiB_[newLocalIdx] = psiB_[oldLocalIdx];
		};

		// compact retained boundary particles into a dense prefix, filling its holes with particles from the old tail
		// (index blocks processed in parallel, and all fields of a moved particle together)
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_retainedB, particleCompactionBlockSize_),
			[&](const tbb::blocked_range<uint32_t>& blockRange)
			{
				CompactParticles(
					retainedBoundaryOldLocalIdx_FromNewLocalIdx,
					blockRange,
					copyBoundaryParticleOldToNew);
			},
			tbb::auto_partitioner{});
	}

	// resize fluid fields to their final particle count after compaction,
	// keeping the old tail while retained particles are moved into the prefix
	N_F_ = N_finalF;
	idF_.resize(N_F_);
	xPosF_.resize(N_F_);
	yPosF_.resize(N_F_);
	zPosF_.resize(N_F_);
	xVelF_.resize(N_F_);
	yVelF_.resize(N_F_);
	zVelF_.resize(N_F_);
	xVelPredF_.resize(N_F_);
	yVelPredF_.resize(N_F_);
	zVelPredF_.resize(N_F_);
	xAccPressF_.resize(N_F_);
	yAccPressF_.resize(N_F_);
	zAccPressF_.resize(N_F_);
	sourceTermF_.resize(N_F_);
	diagElemF_.resize(N_F_);
	pressureF_.resize(N_F_);
	densityF_.resize(N_F_);
	if (redistributeParticlesB)
	{
		N_B_ = N_finalB;
		idB_.resize(N_B_);
		xPosB_.resize(N_B_);
		yPosB_.resize(N_B_);
		zPosB_.resize(N_B_);
		volB_.resize(N_B_);
		psiB_.resize(N_B_);
	}

	// append migrated and halo fluid particles after the retained prefix to complete the new local layout
	// (index blocks processed in parallel, and fields sequentially within each block)
	const uint32_t fluidAppendBegin = N_retainedF;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, appendedParticlesF.N_, particleAppendBlockSize_),
		[&](const tbb::blocked_range<uint32_t>& blockRange)
		{
			const Sync::FluidParticles& src = appendedParticlesF;

			AppendParticleField(idF_, src.id, fluidAppendBegin, blockRange);
			AppendParticleField(xPosF_, src.xPos, fluidAppendBegin, blockRange);
			AppendParticleField(yPosF_, src.yPos, fluidAppendBegin, blockRange);
			AppendParticleField(zPosF_, src.zPos, fluidAppendBegin, blockRange);
			AppendParticleField(xVelF_, src.xVel, fluidAppendBegin, blockRange);
			AppendParticleField(yVelF_, src.yVel, fluidAppendBegin, blockRange);
			AppendParticleField(zVelF_, src.zVel, fluidAppendBegin, blockRange);

			// the fields velPred, accPress, sourceTerm, diagElem, pressure, and density are omitted,
			// because the following solver step re-writes them before reading them
			// TODO: keep it like this or copy them too, if solver step behavior changes?
		},
		tbb::auto_partitioner{});

	if (redistributeParticlesB)
	{
		// append migrated and halo boundary particles after the retained prefix to complete the new local layout
		// (index blocks processed in parallel, and fields sequentially within each block)
		const uint32_t boundaryAppendBegin = N_retainedB;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, appendedParticlesB.N_, particleAppendBlockSize_),
			[&](const tbb::blocked_range<uint32_t>& blockRange)
			{
				const Sync::BoundaryParticles& src = appendedParticlesB;

				AppendParticleField(idB_, src.id, boundaryAppendBegin, blockRange);
				AppendParticleField(xPosB_, src.xPos, boundaryAppendBegin, blockRange);
				AppendParticleField(yPosB_, src.yPos, boundaryAppendBegin, blockRange);
				AppendParticleField(zPosB_, src.zPos, boundaryAppendBegin, blockRange);
				AppendParticleField(volB_, src.volB, boundaryAppendBegin, blockRange);
				AppendParticleField(psiB_, src.psiB, boundaryAppendBegin, blockRange);
			},
			tbb::auto_partitioner{});
	}

	useDistributedParticleLayout_ = false;
	N_ownedF_ = 0;
	N_ownedB_ = 0;
	outerFluidParticleLocalIdxs_.clear();
	pressureAccFluidParticleLocalIdxs_.clear();
	pressureAccFluidParticleFlags_.clear();
	pressureAccChunkIdxs_.clear();
	innerFluidChunkIdxs_.clear();
	outerFluidChunkIdxs_.clear();
	pressureAccIsOwnedPrefix_ = false;
	chunkFlagsScratch_.clear();
	particleFlagsScratch_.clear();
	useDistributedStepWork_ = false;
	neighborList_->Resize(N_F_, N_B_, N_R_, params);
	return true;
}

void SimBackend_CPU::CopyRigidBodyStateTo(
	Sync::RigidBodies& rigidBodies) const
{
	rigidBodies.N_ = N_RB_;
	rigidBodies.xCM = xCMRB_; rigidBodies.yCM = yCMRB_; rigidBodies.zCM = zCMRB_;
	rigidBodies.xVel = xVelRB_; rigidBodies.yVel = yVelRB_; rigidBodies.zVel = zVelRB_;
	rigidBodies.xOmega = xOmegaRB_; rigidBodies.yOmega = yOmegaRB_; rigidBodies.zOmega = zOmegaRB_;
	rigidBodies.xAngMom = xAngMomRB_; rigidBodies.yAngMom = yAngMomRB_; rigidBodies.zAngMom = zAngMomRB_;
	rigidBodies.a00 = a00RB_; rigidBodies.a01 = a01RB_; rigidBodies.a02 = a02RB_;
	rigidBodies.a10 = a10RB_; rigidBodies.a11 = a11RB_; rigidBodies.a12 = a12RB_;
	rigidBodies.a20 = a20RB_; rigidBodies.a21 = a21RB_; rigidBodies.a22 = a22RB_;
}

bool SimBackend_CPU::ApplyRigidBodyStateFrom(
	const Sync::RigidBodies& rigidBodies)
{
	const auto bodySizeValid = [&rigidBodies](const auto& values)
	{
		return values.size() == rigidBodies.N_;
	};

	if (rigidBodies.N_ != N_RB_ ||
		!bodySizeValid(rigidBodies.xCM) ||
		!bodySizeValid(rigidBodies.yCM) ||
		!bodySizeValid(rigidBodies.zCM) ||
		!bodySizeValid(rigidBodies.xVel) ||
		!bodySizeValid(rigidBodies.yVel) ||
		!bodySizeValid(rigidBodies.zVel) ||
		!bodySizeValid(rigidBodies.xOmega) ||
		!bodySizeValid(rigidBodies.yOmega) ||
		!bodySizeValid(rigidBodies.zOmega) ||
		!bodySizeValid(rigidBodies.xAngMom) ||
		!bodySizeValid(rigidBodies.yAngMom) ||
		!bodySizeValid(rigidBodies.zAngMom) ||
		!bodySizeValid(rigidBodies.a00) ||
		!bodySizeValid(rigidBodies.a01) ||
		!bodySizeValid(rigidBodies.a02) ||
		!bodySizeValid(rigidBodies.a10) ||
		!bodySizeValid(rigidBodies.a11) ||
		!bodySizeValid(rigidBodies.a12) ||
		!bodySizeValid(rigidBodies.a20) ||
		!bodySizeValid(rigidBodies.a21) ||
		!bodySizeValid(rigidBodies.a22))
	{
		return false;
	}

	xCMRB_ = rigidBodies.xCM; yCMRB_ = rigidBodies.yCM; zCMRB_ = rigidBodies.zCM;
	xVelRB_ = rigidBodies.xVel; yVelRB_ = rigidBodies.yVel; zVelRB_ = rigidBodies.zVel;
	xOmegaRB_ = rigidBodies.xOmega; yOmegaRB_ = rigidBodies.yOmega; zOmegaRB_ = rigidBodies.zOmega;
	xAngMomRB_ = rigidBodies.xAngMom; yAngMomRB_ = rigidBodies.yAngMom; zAngMomRB_ = rigidBodies.zAngMom;
	a00RB_ = rigidBodies.a00; a01RB_ = rigidBodies.a01; a02RB_ = rigidBodies.a02;
	a10RB_ = rigidBodies.a10; a11RB_ = rigidBodies.a11; a12RB_ = rigidBodies.a12;
	a20RB_ = rigidBodies.a20; a21RB_ = rigidBodies.a21; a22RB_ = rigidBodies.a22;
	UpdateRigidParticles();
	return true;
}

void SimBackend_CPU::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	snap.simStep = simStep_;
	snap.simTime = simTime_;

	// copy boundary particles
	snap.N_B = N_B_;
	snap.xPosB = xPosB_; snap.yPosB = yPosB_; snap.zPosB = zPosB_;
	snap.volB = volB_;

	// copy rigid particles
	snap.N_R = N_R_;
	snap.bodyIdR = bodyIdR_;
	snap.xPosR = xPosR_; snap.yPosR = yPosR_; snap.zPosR = zPosR_;
	snap.volR = volR_;

	if (N_F_ == 0)
	{
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// classify fluid particles wanted for export
	FillFluidParticleNearSurfaceFlags(simParams, expNearSurfaceFlagsScratch_);
	ClassifyExportSurfaceParticles(expNearSurfaceFlagsScratch_);
	const uint32_t N_surface = static_cast<uint32_t>(
		std::ranges::count(expSurfaceFlagsScratch_, uint8_t{ 1 }));

	if (N_surface == 0) // no particles to export
	{
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// prepare snapshot buffers
	snap.N_F = N_surface;
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

		const float vx = xVelF_[i];
		const float vy = yVelF_[i];
		const float vz = zVelF_[i];

		snap.velMagF[dst] = glm::sqrt(vx * vx + vy * vy + vz * vz);

		dst++;
	}
}

bool SimBackend_CPU::FillFluidParticleNearSurfaceFlags(
	const SimParams& simParams,
	std::vector<uint8_t>& nearSurfaceFlags) const
{
	if (!neighborList_) { return false; }

	const uint32_t nbrCountThreshold = simParams.surfaceNbrCountThreshold;

	Base::ResizeScratchVector(nearSurfaceFlags, N_F_);
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nearSurfaceFlags[i] = static_cast<uint8_t>(
					IsNearSurfaceFluidParticle(
						neighborList_->CountFF(i),
						neighborList_->CountFB(i),
						neighborList_->CountFR(i),
						nbrCountThreshold));
			}
		},
		tbb::auto_partitioner{});

	return true;
}

void SimBackend_CPU::ClassifyExportSurfaceParticles(
	const std::vector<uint8_t>& nearSurfaceFlags) const
{
	if (!neighborList_ || nearSurfaceFlags.size() != N_F_)
	{
		expSurfaceFlagsScratch_.clear();
		return;
	}

	auto& surfaceFlags = expSurfaceFlagsScratch_;
	surfaceFlags.assign(N_F_, 0);

	// only near-surface particles are traversed, each marking itself and its fluid neighbors
	// (equivalent to checking every particle for near-surface neighbors, since FF is symmetric)
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// neighbors of different near-surface particles overlap, so the marks are
			// written as relaxed atomic stores (plain byte stores, no synchronization)
			const auto Mark = [&](const uint32_t i) noexcept
			{
				std::atomic_ref<uint8_t>{ surfaceFlags[i] }.store(1, std::memory_order_relaxed);
			};

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (nearSurfaceFlags[i] == 0) { continue; }

				Mark(i);

				const NeighborSpan nbrListFF = neighborList_->FF(i);
				const uint32_t* nbrFF = nbrListFF.data;
				const uint32_t N_nbrsFF = nbrListFF.N;

				for (uint32_t k = 0; k < N_nbrsFF; k++)
				{
					Mark(nbrFF[k]);
				}
			}
		},
		tbb::auto_partitioner{});
}

bool SimBackend_CPU::BuildExportFluidParticleLocalIdxs(
	const std::vector<uint8_t>& nearSurfaceFlags,
	std::vector<uint32_t>& localIdxs) const
{
	if (!neighborList_ || nearSurfaceFlags.size() != N_F_)
	{
		return false;
	}

	ClassifyExportSurfaceParticles(nearSurfaceFlags);
	const uint32_t N_surface = static_cast<uint32_t>(
		std::ranges::count(expSurfaceFlagsScratch_, uint8_t{ 1 }));

	localIdxs.clear();
	localIdxs.reserve(N_surface);
	// TODO: parallelize with TBB?
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (expSurfaceFlagsScratch_[i] != 0)
		{
			localIdxs.push_back(i);
		}
	}

	return true;
}

void SimBackend_CPU::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
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

void SimBackend_CPU::FillScalarData(
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

	throw std::runtime_error("Unsupported scalar data mode in CPU.");
}

void SimBackend_CPU::FillMemoryRecord(Mem::MemRecordStep& record) const
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
		const ParticleClassMemUsage usage = neighborSearch_->GetMemUsage();

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructF, usage.fluid);
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructB, usage.boundary);
		Set(record, memSpace, MemBucket::Neighbors_SearchDataStructR, usage.rigid);
	}

	// --- [ neighbor list storage ] -----------------------------------------------------------------

	{
		const ParticleClassMemUsage usage = neighborList_->GetMemUsage();

		// finalize
		Set(record, memSpace, MemBucket::Neighbors_ListStorageF, usage.fluid);
		Set(record, memSpace, MemBucket::Neighbors_ListStorageB, usage.boundary);
		Set(record, memSpace, MemBucket::Neighbors_ListStorageR, usage.rigid);
	}

	// --- [ misc ] ----------------------------------------------------------------------------------

	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(workBufferF_));
		Add(usage, MemUsageOf_Vector(workBufferF2_));
		Add(usage, MemUsageOf_Vector(workBufferF3_));
		Add(usage, MemUsageOf_Vector(workBufferB_));
		Add(usage, MemUsageOf_Vector(workBufferB2_));
		Add(usage, MemUsageOf_Vector(workBufferB3_));
		Add(usage, MemUsageOf_Vector(workBufferIdF_));
		Add(usage, MemUsageOf_Vector(workBufferIdB_));
		Add(usage, MemUsageOf_Vector(expSurfaceFlagsScratch_));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFF.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFB.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountFR.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRF.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRB.histogram));
		Add(usage, MemUsageOf_Vector(statisticsPartialScratch_.nbrCountRR.histogram));
		Add(usage, MemUsageOf_Vector(outerFluidParticleLocalIdxs_));
		Add(usage, MemUsageOf_Vector(pressureAccFluidParticleLocalIdxs_));
		Add(usage, MemUsageOf_Vector(pressureAccFluidParticleFlags_));
		Add(usage, MemUsageOf_Vector(pressureAccChunkIdxs_));
		Add(usage, MemUsageOf_Vector(innerFluidChunkIdxs_));
		Add(usage, MemUsageOf_Vector(outerFluidChunkIdxs_));
		Add(usage, MemUsageOf_Vector(activeRigidParticleLocalIdxs_));
		Add(usage, MemUsageOf_Vector(activeRigidParticleFlags_));
		Add(usage, MemUsageOf_Vector(newFluidLocalIdx_FromOldLocalIdx_));
		Add(usage, MemUsageOf_Vector(chunkFlagsScratch_));
		Add(usage, MemUsageOf_Vector(particleFlagsScratch_));
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

void SimBackend_CPU::FillStatisticsRecord(
	Stats::StatRecordStep& record,
	const SimParams& simParams) const
{
	auto& partial = statisticsPartialScratch_;
	ComputeStatisticsPartial(partial, simParams, false);
	Stats::StatisticsTracker::FinalizeStatisticsRecord(record, partial, simParams);
}

bool SimBackend_CPU::FillStatisticsPartial(
	Stats::StatisticsPartial& partial,
	const SimParams& simParams) const
{
	ComputeStatisticsPartial(partial, simParams, true);
	return true;
}

void SimBackend_CPU::ComputeStatisticsPartial(
	Stats::StatisticsPartial& partial,
	const SimParams& simParams,
	const bool useActiveParticleLocalIdxs) const
{
	using namespace Stats;

	partial.Clear();

	const uint32_t N_F = useActiveParticleLocalIdxs ? N_ownedF_ : N_F_;
	const uint32_t N_R = useActiveParticleLocalIdxs
		? static_cast<uint32_t>(activeRigidParticleLocalIdxs_.size())
		: N_R_;

	partial.N_F = N_F;
	partial.N_B = useActiveParticleLocalIdxs ? 0 : N_B_;
	partial.N_R = N_R;
	partial.N_solverIters = N_solverIters_;

	const auto getFluidLocalIdx = [](const uint32_t i) { return i; };
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
				const uint32_t N_particles,
				LocalIdxFunc getLocalIdx,
				NbrCountFunc getNbrCount,
				DistributionPartial& distribution)
		{
			distribution.Clear();
			distribution.N = N_particles;
			if (N_particles == 0) { return; }

			struct NbrCountStatsLocal
			{
				uint64_t sum = 0;
				uint32_t min = std::numeric_limits<uint32_t>::max();
				uint32_t max = 0;
			};

			const NbrCountStatsLocal nbrCountStats = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, N_particles),
				NbrCountStatsLocal{},
				[&](const tbb::blocked_range<uint32_t>& r,
					NbrCountStatsLocal local) -> NbrCountStatsLocal
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const uint32_t N_nbrs = getNbrCount(getLocalIdx(i));
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

			for (uint32_t i = 0; i < N_particles; i++)
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

void SimBackend_CPU::ExecuteNeighborSearch(const SimParams& params, const bool bench)
{
	const bool runCountQuery = neighborList_->RunCountQuery();
	const bool runFillQuery = neighborList_->RunFillQuery();

	// initial re-ordering of the data before the initial neighbor search
	if (bench)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
	}
	neighborSearch_->DataReOrder(params);
	ApplyLastParticleReOrderToDistributedFluidParticleLocalIdxs();
	affinityParts_ = std::make_unique<StepAffinityPartitioners>(); // reset mapping
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

bool SimBackend_CPU::TryGetLastParticleReOrder(
	std::span<const uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
	std::span<const uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
{
	if (!neighborSearch_) { return false; }

	return neighborSearch_->TryGetLastParticleReOrder(
		oldFluidLocalIdx_FromNewLocalIdx,
		oldBoundaryLocalIdx_FromNewLocalIdx);
}

bool SimBackend_CPU::ConfigureDistributedParticleLayout(
	const uint32_t N_ownedFluidParticles,
	const uint32_t N_ownedBoundaryParticles)
{
	if (N_ownedFluidParticles > N_F_ || N_ownedBoundaryParticles > N_B_)
	{
		SPDLOG_ERROR(
			"Invalid CPU distributed particle layout (owned fluid={}, boundary={}).",
			N_ownedFluidParticles, N_ownedBoundaryParticles);
		return false;
	}

	N_ownedF_ = N_ownedFluidParticles;
	N_ownedB_ = N_ownedBoundaryParticles;
	useDistributedParticleLayout_ = true;
	return true;
}

bool SimBackend_CPU::ConfigureDistributedStep(
	const std::vector<uint32_t>& ownedFluidParticleLocalIdxs,
	const std::vector<uint32_t>& outerFluidParticleLocalIdxs,
	const std::vector<uint32_t>& pressureAccelerationFluidParticleLocalIdxs,
	const bool pressureAccIsOwnedPrefix,
	const uint32_t N_ownedBoundaryParticles,
	const std::vector<uint32_t>& activeRigidParticleLocalIdxs,
	I_DistributedFluidSolverSync& distributedFluidSolverSync)
{
	const uint32_t N_ownedF = static_cast<uint32_t>(ownedFluidParticleLocalIdxs.size());
	const auto isInvalidOwnedIdx = [&](const uint32_t ownedIdx)
	{
		const uint32_t localIdx = ownedFluidParticleLocalIdxs[ownedIdx];
		return localIdx != ownedIdx || localIdx >= N_F_;
	};
	if (Check::CountViolations(0, N_ownedF, isInvalidOwnedIdx) != 0)
	{
		const uint32_t ownedIdx = Check::FindFirstViolation(0, N_ownedF, isInvalidOwnedIdx);
		SPDLOG_ERROR(
			"Invalid owned fluid particle local index {} at prefix position {}.",
			ownedFluidParticleLocalIdxs[ownedIdx], ownedIdx);
		return false;
	}

	const uint32_t N_outerF = static_cast<uint32_t>(outerFluidParticleLocalIdxs.size());
	const auto isInvalidOuterIdx = [&](const uint32_t outerIdx)
	{
		return outerFluidParticleLocalIdxs[outerIdx] >= N_ownedF;
	};
	if (Check::CountViolations(0, N_outerF, isInvalidOuterIdx) != 0)
	{
		const uint32_t outerIdx = Check::FindFirstViolation(0, N_outerF, isInvalidOuterIdx);
		SPDLOG_ERROR(
			"Invalid outer fluid particle local index {}.",
			outerFluidParticleLocalIdxs[outerIdx]);
		return false;
	}

	const uint32_t duplicateOuterIdx = Check::FindFirstDuplicateIdx(
		outerFluidParticleLocalIdxs, N_ownedF, particleFlagsScratch_);
	if (duplicateOuterIdx != N_outerF)
	{
		SPDLOG_ERROR(
			"Duplicate outer fluid particle local index {}.",
			outerFluidParticleLocalIdxs[duplicateOuterIdx]);
		return false;
	}

	const uint32_t N_pressureAccF = static_cast<uint32_t>(
		pressureAccelerationFluidParticleLocalIdxs.size());
	if (!pressureAccIsOwnedPrefix)
	{
		const auto isInvalidPressureAccIdx = [&](const uint32_t pressureAccIdx)
		{
			return pressureAccelerationFluidParticleLocalIdxs[pressureAccIdx] >= N_F_;
		};
		if (Check::CountViolations(0, N_pressureAccF, isInvalidPressureAccIdx) != 0)
		{
			const uint32_t pressureAccIdx = Check::FindFirstViolation(
				0, N_pressureAccF, isInvalidPressureAccIdx);
			SPDLOG_ERROR(
				"Invalid pressure-acceleration fluid particle local index {}.",
				pressureAccelerationFluidParticleLocalIdxs[pressureAccIdx]);
			return false;
		}

		const uint32_t duplicatePressureAccIdx = Check::FindFirstDuplicateIdx(
			pressureAccelerationFluidParticleLocalIdxs, N_F_, particleFlagsScratch_);
		if (duplicatePressureAccIdx != N_pressureAccF)
		{
			SPDLOG_ERROR(
				"Duplicate pressure-acceleration fluid particle local index {}.",
				pressureAccelerationFluidParticleLocalIdxs[duplicatePressureAccIdx]);
			return false;
		}
	}

	if (N_ownedBoundaryParticles > N_B_)
	{
		SPDLOG_ERROR(
			"Invalid owned boundary particle count {} for {} local particles.",
			N_ownedBoundaryParticles, N_B_);
		return false;
	}

	const uint32_t N_activeR = static_cast<uint32_t>(activeRigidParticleLocalIdxs.size());
	const auto isInvalidActiveRigidIdx = [&](const uint32_t activeIdx)
	{
		return activeRigidParticleLocalIdxs[activeIdx] >= N_R_;
	};
	if (Check::CountViolations(0, N_activeR, isInvalidActiveRigidIdx) != 0)
	{
		const uint32_t activeIdx = Check::FindFirstViolation(
			0, N_activeR, isInvalidActiveRigidIdx);
		SPDLOG_ERROR(
			"Invalid active rigid particle local index {}.",
			activeRigidParticleLocalIdxs[activeIdx]);
		return false;
	}

	const uint32_t duplicateActiveRigidIdx = Check::FindFirstDuplicateIdx(
		activeRigidParticleLocalIdxs, N_R_, particleFlagsScratch_);
	if (duplicateActiveRigidIdx != N_activeR)
	{
		SPDLOG_ERROR(
			"Duplicate active rigid particle local index {}.",
			activeRigidParticleLocalIdxs[duplicateActiveRigidIdx]);
		return false;
	}

	N_ownedF_ = N_ownedF;
	N_ownedB_ = N_ownedBoundaryParticles;
	useDistributedParticleLayout_ = true;
	outerFluidParticleLocalIdxs_ = outerFluidParticleLocalIdxs;
	pressureAccIsOwnedPrefix_ = pressureAccIsOwnedPrefix;
	if (pressureAccIsOwnedPrefix_)
	{
		pressureAccFluidParticleLocalIdxs_.clear();
	}
	else
	{
		pressureAccFluidParticleLocalIdxs_ = pressureAccelerationFluidParticleLocalIdxs;
	}
	activeRigidParticleLocalIdxs_ = activeRigidParticleLocalIdxs;
	activeRigidParticleFlags_.assign(N_R_, 0);
	for (const uint32_t localIdx : activeRigidParticleLocalIdxs)
	{
		activeRigidParticleFlags_[localIdx] = 1;
	}
	distributedFluidSolverSync_ = &distributedFluidSolverSync;
	useDistributedStepWork_ = true;
	useActiveRigidParticleLocalIdxs_ = true;
	integrateRigidBodies_ = false;
	applyRigidBodyExternalForces_ = false;
	return true;
}

bool SimBackend_CPU::GetRigidBodyForceTorque(
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

bool SimBackend_CPU::ApplyRigidBodyForceTorque(
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

void SimBackend_CPU::ApplyLastParticleReOrderToDistributedFluidParticleLocalIdxs()
{
	if (!useDistributedStepWork_) { return; }
	if (!neighborSearch_) { return; }

	std::span<const uint32_t> oldFluidLocalIdx_FromNewLocalIdx;
	std::span<const uint32_t> oldBoundaryLocalIdx_FromNewLocalIdx;
	if (!neighborSearch_->TryGetLastParticleReOrder(
		oldFluidLocalIdx_FromNewLocalIdx,
		oldBoundaryLocalIdx_FromNewLocalIdx))
	{
		if (distributedFluidSolverSync_)
		{
			distributedFluidSolverSync_->ApplyFluidParticleReOrder({});
		}
		BuildDistributedSolverChunkIdxs();
		return;
	}

	if (oldFluidLocalIdx_FromNewLocalIdx.size() != N_F_)
	{
		throw std::runtime_error("Invalid CPU fluid particle re-order size.");
	}
	if (oldBoundaryLocalIdx_FromNewLocalIdx.size() != N_B_)
	{
		throw std::runtime_error("Invalid CPU boundary particle re-order size.");
	}

	#if defined(_DEBUG)
	// exhaustively validate that the re-order contains no duplicates, so the inversion
	// below can write every old local index without claiming it first
	// TODO: parallelize with TBB?
	std::vector<uint8_t> oldFluidLocalIdxUsed(N_F_, 0);
	for (uint32_t newLocalIdx = 0; newLocalIdx != N_F_; newLocalIdx++)
	{
		const uint32_t oldLocalIdx = oldFluidLocalIdx_FromNewLocalIdx[newLocalIdx];
		if (oldLocalIdx >= N_F_ || oldFluidLocalIdxUsed[oldLocalIdx] != 0)
		{
			throw std::runtime_error("Invalid CPU fluid particle re-order index.");
		}
		if ((newLocalIdx < N_ownedF_) != (oldLocalIdx < N_ownedF_))
		{
			throw std::runtime_error(
				"CPU fluid particle re-order crosses the owned/halo boundary.");
		}

		oldFluidLocalIdxUsed[oldLocalIdx] = 1;
	}

	std::vector<uint8_t> oldBoundaryLocalIdxUsed(N_B_, 0);
	for (uint32_t newLocalIdx = 0; newLocalIdx != N_B_; newLocalIdx++)
	{
		const uint32_t oldLocalIdx = oldBoundaryLocalIdx_FromNewLocalIdx[newLocalIdx];
		if (oldLocalIdx >= N_B_ || oldBoundaryLocalIdxUsed[oldLocalIdx] != 0)
		{
			throw std::runtime_error("Invalid CPU boundary particle re-order index.");
		}
		if ((newLocalIdx < N_ownedB_) != (oldLocalIdx < N_ownedB_))
		{
			throw std::runtime_error(
				"CPU boundary particle re-order crosses the owned/halo boundary.");
		}

		oldBoundaryLocalIdxUsed[oldLocalIdx] = 1;
	}
	#endif

	// invert the re-order, so remapping the active indices below is one lookup per particle
	Base::ResizeScratchVector(newFluidLocalIdx_FromOldLocalIdx_, N_F_);
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t newLocalIdx = r.begin(); newLocalIdx != r.end(); newLocalIdx++)
			{
				const uint32_t oldLocalIdx = oldFluidLocalIdx_FromNewLocalIdx[newLocalIdx];
				if (oldLocalIdx >= N_F_)
				{
					throw std::runtime_error(
						"Invalid CPU fluid particle re-order index.");
				}

				newFluidLocalIdx_FromOldLocalIdx_[oldLocalIdx] = newLocalIdx;
			}
		},
		tbb::auto_partitioner{}
	);

	auto remapLocalIdxs = [this](
		std::vector<uint32_t>& localIdxs,
		const uint32_t N_validLocalIdxs,
		const char* failMsg)
	{
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(localIdxs.size())),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					uint32_t& localIdx = localIdxs[i];
					if (localIdx >= N_F_) { throw std::runtime_error(failMsg); }

					const uint32_t newLocalIdx = newFluidLocalIdx_FromOldLocalIdx_[localIdx];
					if (newLocalIdx >= N_validLocalIdxs) { throw std::runtime_error(failMsg); }

					localIdx = newLocalIdx;
				}
			},
			tbb::auto_partitioner{}
		);
	};

	remapLocalIdxs(outerFluidParticleLocalIdxs_, N_ownedF_,
		"Invalid outer CPU fluid particle local index.");
	if (!pressureAccIsOwnedPrefix_)
	{
		remapLocalIdxs(pressureAccFluidParticleLocalIdxs_, N_F_,
			"Invalid pressure-acceleration CPU fluid particle local index.");
	}

	BuildDistributedSolverChunkIdxs();

	if (distributedFluidSolverSync_)
	{
		distributedFluidSolverSync_->ApplyFluidParticleReOrder(
			newFluidLocalIdx_FromOldLocalIdx_);
	}
}

void SimBackend_CPU::BuildDistributedSolverChunkIdxs()
{
	// marking chunks that hold at least one of the particles yields the chunk indices in ascending order
	// (the callers validate the local indices against the fluid particle count)
	const uint32_t N_chunks = (N_F_ + CPU::N_SIMD_WIDTH - 1) / CPU::N_SIMD_WIDTH;
	const uint32_t N_ownedChunks = (N_ownedF_ + CPU::N_SIMD_WIDTH - 1) / CPU::N_SIMD_WIDTH;

	// TODO: parallelize with TBB?
	auto buildChunkIdxs = [this, N_chunks](const std::vector<uint32_t>& particleLocalIdxs,
		std::vector<uint32_t>& chunkIdxs)
	{
		chunkFlagsScratch_.assign(N_chunks, 0);
		for (const uint32_t localIdx : particleLocalIdxs)
		{
			chunkFlagsScratch_[localIdx / CPU::N_SIMD_WIDTH] = 1;
		}

		chunkIdxs.clear();
		chunkIdxs.reserve(N_chunks);
		for (uint32_t chunkIdx = 0; chunkIdx != N_chunks; chunkIdx++)
		{
			if (chunkFlagsScratch_[chunkIdx] != 0) { chunkIdxs.push_back(chunkIdx); }
		}
	};

	if (pressureAccIsOwnedPrefix_)
	{
		pressureAccFluidParticleFlags_.assign(N_ownedF_, 1);
		pressureAccFluidParticleFlags_.resize(N_F_, 0);

		pressureAccChunkIdxs_.resize(N_ownedChunks);
		for (uint32_t chunkIdx = 0; chunkIdx != N_ownedChunks; chunkIdx++)
		{
			pressureAccChunkIdxs_[chunkIdx] = chunkIdx;
		}
	}
	else
	{
		pressureAccFluidParticleFlags_.assign(N_F_, 0);
		for (const uint32_t localIdx : pressureAccFluidParticleLocalIdxs_)
		{
			pressureAccFluidParticleFlags_[localIdx] = 1;
		}

		buildChunkIdxs(
			pressureAccFluidParticleLocalIdxs_,
			pressureAccChunkIdxs_);
	}

	#if defined(_DEBUG)
	// exhaustively validate that the chunk indices are strictly ascending and in range,
	// so the count query can treat a full-length list as covering every chunk
	const uint32_t N_pressureAccChunks = static_cast<uint32_t>(pressureAccChunkIdxs_.size());
	for (uint32_t i = 0; i != N_pressureAccChunks; i++)
	{
		if (pressureAccChunkIdxs_[i] >= N_chunks
			|| (i > 0 && pressureAccChunkIdxs_[i] <= pressureAccChunkIdxs_[i - 1]))
		{
			throw std::runtime_error(
				"Invalid pressure-acceleration CPU fluid chunk index.");
		}
	}
	#endif

	chunkFlagsScratch_.assign(N_ownedChunks, 0);
	for (const uint32_t localIdx : outerFluidParticleLocalIdxs_)
	{
		chunkFlagsScratch_[localIdx / CPU::N_SIMD_WIDTH] = 1;
	}

	innerFluidChunkIdxs_.clear();
	outerFluidChunkIdxs_.clear();
	innerFluidChunkIdxs_.reserve(N_ownedChunks);
	outerFluidChunkIdxs_.reserve(N_ownedChunks);
	for (uint32_t chunkIdx = 0; chunkIdx != N_ownedChunks; chunkIdx++)
	{
		std::vector<uint32_t>& dst = chunkFlagsScratch_[chunkIdx] != 0
			? outerFluidChunkIdxs_ : innerFluidChunkIdxs_;
		dst.push_back(chunkIdx);
	}
}

void SimBackend_CPU::UpdateRigidParticles()
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
					throw std::runtime_error(
						"Invalid CPU rigid particle attachment.");
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

void SimBackend_CPU::ClearRigidForcesAndAddGravity(const SimParams& params)
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

void SimBackend_CPU::ReduceRigidParticleForces()
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

void SimBackend_CPU::IntegrateRigidBodies(const float timeStep)
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

void SimBackend_CPU::ExecuteFullSimStep(
	const SimParams& params,
	const uint32_t N_steps)
{
	if (N_F_ == 0 && !distributedFluidSolverSync_) // empty serial sim
	{
		simStep_ += N_steps;
		simTime_ += static_cast<double>(N_steps) * params.timeStep;

		return;
	}

	Step_IISPH(params);

	simStep_ += N_steps;
	simTime_ += static_cast<double>(N_steps) * params.timeStep;
}
