// gpu_backend.cpp

// external
#include <algorithm>
#include <atomic>
#include <glm/vec3.hpp>
#include <cmath>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/solver_constants.h"
#include "source/simulation/GPU/gpu_utils.h"
#include "source/simulation/sync/mailboxes_cuda.h"
#include "gpu_backend.h"
#include "neighbor_search.cuh"
#include "rigid_kernels.cuh"
#include "sph_kernels.cuh"
#include "statistics_kernels.cuh"
#include "snapshot_kernels.cuh"



using namespace Sim;

namespace
{
	template <typename T>
	inline Mem::MemUsage MemUsageOf_DeviceBuffer(
		const DeviceBuffer<T>& src) noexcept
	{
		const uint64_t N_bytes = static_cast<uint64_t>(src.Size()) * sizeof(T);

		return Mem::MemUsage{ N_bytes, N_bytes };
	}

	inline uint32_t SellChunkNumTotal(const uint32_t N_F) noexcept
	{
		return (N_F == 0u) ? 0u : ((N_F - 1u) / GPU::N_SELL_WIDTH) + 1u;
	}

	inline uint32_t ReadCompressedSell32Neighbor(
		const std::vector<uint64_t>& chunkOffsets,
		const std::vector<uint32_t>& chunkBaseIdx,
		const std::vector<uint8_t>& chunkFormat,
		const std::vector<uint8_t>& nbrListDelta8,
		const std::vector<uint16_t>& nbrListDelta16,
		const std::vector<uint32_t>& nbrListAbs32,
		const uint32_t i, const uint32_t k) noexcept
	{
		const uint32_t chunk = i / GPU::N_SELL_WIDTH;
		const uint32_t lane = i % GPU::N_SELL_WIDTH;
		const uint64_t linearIdx =
			static_cast<uint64_t>(k) * GPU::N_SELL_WIDTH + lane;
		const uint64_t base = chunkOffsets[chunk];

		if (chunkFormat[chunk] == static_cast<uint8_t>(GPU::NbrChunkFormat::Delta8))
		{
			return chunkBaseIdx[chunk] + static_cast<uint32_t>(nbrListDelta8[base + linearIdx]);
		}

		if (chunkFormat[chunk] == static_cast<uint8_t>(GPU::NbrChunkFormat::Delta16))
		{
			return chunkBaseIdx[chunk] + static_cast<uint32_t>(nbrListDelta16[base + linearIdx]);
		}

		return nbrListAbs32[base + linearIdx];
	}

} // namespace anonymous

SimBackend_GPU::SimBackend_GPU(
	Bench::BenchmarksTracker& benchmarksTracker, const SimParams& params)
	: benchmarksTracker_ { benchmarksTracker }
{
	if (params.neighborListMethod != Base::NeighborListMethod::ChunkedSELLX_AdaptCompress)
	{
		throw std::runtime_error(
			"GPU supports only neighborListStorage=1 (ChunkedSELL32_AdaptCompress); "
			"neighborListStorage=0 (ChunkedSELL32_Explicit) is not implemented.");
	}

	GPU::Check(cudaStreamCreate(&stream_), "cudaStreamCreate()");

	// select neighbor search method
	if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::BruteForce)
	{
		neighborSearch_ = std::make_unique<NS_BruteForce_GPU>(*this);
	}
	else if (params.neighborSearchMethod ==
		Base::NeighborSearchMethod::CellLinkedList_Compact)
	{
		neighborSearch_ = std::make_unique<NS_CellLinkedList_Compact_GPU>(*this);
	}
	else
	{
		throw std::runtime_error("Unsupported 'NeighborSearchMethod' requested.");
	}
	neighborSearch_->Init();

	// TODO: select neighbor list method once implemented
}

void SimBackend_GPU::SetSimDomainBounds(
	const glm::vec3& min,
	const glm::vec3& max)
{
	simDomainMin_ = Vec3f{ min.x, min.y, min.z };
	simDomainMax_ = Vec3f{ max.x, max.y, max.z };
	simDomainBoundsValid_ = max.x > min.x && max.y > min.y && max.z > min.z;
}

SimBackend_GPU::~SimBackend_GPU()
{
	if (stream_)
	{
		GPU::Check(cudaStreamDestroy(stream_), "cudaStreamDestroy()");
	}
}

void SimBackend_GPU::PullFromUniversalSimState(
	Sync::UniversalSimState& src, const SimParams& params)
{
	auto& F = src.particlesF;
	auto& B = src.particlesB;
	auto& R = src.particlesR;
	auto& RB = src.rigidBodies;
	(void)CheckedLocalParticleCount(F.N_, "GPU fluid state");
	(void)CheckedLocalParticleCount(B.N_, "GPU boundary state");
	(void)CheckedLocalParticleCount(R.N_, "GPU rigid state");
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

	EnsureCapacity(N_F_, N_B_);
	EnsureRigidCapacity(N_RB_, N_R_);

	// copy fluid particle data to device
	if (N_F_ > 0)
	{
		GPU::Copy_HostToDevice(d_idF_.Data(), F.id.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_xPosF_.Data(), F.xPos.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yPosF_.Data(), F.yPos.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zPosF_.Data(), F.zPos.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_xVelF_.Data(), F.xVel.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yVelF_.Data(), F.yVel.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zVelF_.Data(), F.zVel.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_xVelPredF_.Data(), F.xVelPred.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yVelPredF_.Data(), F.yVelPred.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zVelPredF_.Data(), F.zVelPred.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_xAccPressF_.Data(), F.xAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_yAccPressF_.Data(), F.yAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_zAccPressF_.Data(), F.zAccPress.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_sourceTermF_.Data(), F.sourceTerm.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_diagElemF_.Data(), F.diagElem.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_pressureF_.Data(), F.pressure.data(), N_F_, stream_);
		GPU::Copy_HostToDevice(d_densityF_.Data(), F.density.data(), N_F_, stream_);
	}

	// copy boundary particle data to device
	if (N_B_ > 0)
	{
		GPU::Copy_HostToDevice(d_idB_.Data(), B.id.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_xPosB_.Data(), B.xPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_yPosB_.Data(), B.yPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_zPosB_.Data(), B.zPos.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_volB_.Data(), B.volB.data(), N_B_, stream_);
		GPU::Copy_HostToDevice(d_psiB_.Data(), B.psiB.data(), N_B_, stream_);
	}

	// copy rigid particle data to device
	if (N_R_ > 0)
	{
		GPU::Copy_HostToDevice(d_idR_.Data(), R.id.data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_bodyIdR_.Data(), R.bodyId.data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_xRelPosR_.Data(), R.xRelPos.data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_yRelPosR_.Data(), R.yRelPos.data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_zRelPosR_.Data(), R.zRelPos.data(), N_R_, stream_);
		GPU::MemsetZero(d_xForceR_.Data(), N_R_, stream_);
		GPU::MemsetZero(d_yForceR_.Data(), N_R_, stream_);
		GPU::MemsetZero(d_zForceR_.Data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_volR_.Data(), R.volR.data(), N_R_, stream_);
		GPU::Copy_HostToDevice(d_psiR_.Data(), R.psiR.data(), N_R_, stream_);
	}

	// copy rigid body data to device
	if (N_RB_ > 0)
	{
		GPU::Copy_HostToDevice(d_idRB_.Data(), RB.id.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_rigidParticleBeginRB_.Data(), RB.rigidParticleBegin.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_rigidParticleCountRB_.Data(), RB.rigidParticleCount.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_xCMRB_.Data(), RB.xCM.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_yCMRB_.Data(), RB.yCM.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_zCMRB_.Data(), RB.zCM.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_xVelRB_.Data(), RB.xVel.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_yVelRB_.Data(), RB.yVel.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_zVelRB_.Data(), RB.zVel.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_xOmegaRB_.Data(), RB.xOmega.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_yOmegaRB_.Data(), RB.yOmega.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_zOmegaRB_.Data(), RB.zOmega.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_xAngMomRB_.Data(), RB.xAngMom.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_yAngMomRB_.Data(), RB.yAngMom.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_zAngMomRB_.Data(), RB.zAngMom.data(), N_RB_, stream_);
		GPU::MemsetZero(d_xForceRB_.Data(), N_RB_, stream_);
		GPU::MemsetZero(d_yForceRB_.Data(), N_RB_, stream_);
		GPU::MemsetZero(d_zForceRB_.Data(), N_RB_, stream_);
		GPU::MemsetZero(d_xTorqueRB_.Data(), N_RB_, stream_);
		GPU::MemsetZero(d_yTorqueRB_.Data(), N_RB_, stream_);
		GPU::MemsetZero(d_zTorqueRB_.Data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a00RB_.Data(), RB.a00.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a01RB_.Data(), RB.a01.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a02RB_.Data(), RB.a02.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a10RB_.Data(), RB.a10.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a11RB_.Data(), RB.a11.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a12RB_.Data(), RB.a12.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a20RB_.Data(), RB.a20.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a21RB_.Data(), RB.a21.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_a22RB_.Data(), RB.a22.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_massRB_.Data(), RB.mass.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_massInvRB_.Data(), RB.massInv.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv00RB_.Data(), RB.iBodyInv00.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv01RB_.Data(), RB.iBodyInv01.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv02RB_.Data(), RB.iBodyInv02.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv10RB_.Data(), RB.iBodyInv10.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv11RB_.Data(), RB.iBodyInv11.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv12RB_.Data(), RB.iBodyInv12.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv20RB_.Data(), RB.iBodyInv20.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv21RB_.Data(), RB.iBodyInv21.data(), N_RB_, stream_);
		GPU::Copy_HostToDevice(d_iBodyInv22RB_.Data(), RB.iBodyInv22.data(), N_RB_, stream_);
	}

	// initialize rigid particle world state and neighbor counts
	UpdateRigidParticles();
	GPU::MemsetZero(d_nbrCountFF_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_nbrCountFB_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_nbrCountFR_.Data(), N_F_, stream_);
	GPU::MemsetZero(d_nbrCountRF_.Data(), N_R_, stream_);
	GPU::MemsetZero(d_nbrCountRB_.Data(), N_R_, stream_);
	GPU::MemsetZero(d_nbrCountRR_.Data(), N_R_, stream_);
	GPU::MemsetZero(d_nbrOverflowFlag_.Data(), 1, stream_);
	N_totalFF_ = N_totalFB_ = N_totalFR_ = N_totalRF_ = N_totalRB_ = N_totalRR_ = 0;
	N_packedFF_ = N_packedFB_ = N_packedFR_ = 0;
	N_delta8FF_ = N_delta8FB_ = N_delta8FR_ = 0;
	N_delta16FF_ = N_delta16FB_ = N_delta16FR_ = 0;
	N_abs32FF_ = N_abs32FB_ = N_abs32FR_ = 0;

	GPU::StreamSync(stream_);
}

void SimBackend_GPU::PushToUniversalSimState(
	Sync::UniversalSimState& dst,
	const bool backendWillBeRetired)
{
	auto& F = dst.particlesF;
	auto& B = dst.particlesB;
	auto& R = dst.particlesR;
	auto& RB = dst.rigidBodies;
	(void)backendWillBeRetired; // device storage is released by the owning manager after synchronization

	dst.ResizeFluid(N_F_);
	dst.ResizeBoundary(N_B_);
	dst.ResizeRigid(N_R_);
	dst.ResizeRigidBodies(N_RB_);
	dst.step = simStep_;
	dst.time = simTime_;

	// copy fluid particle data to host
	if (N_F_ > 0)
	{
		GPU::Copy_DeviceToHost(F.id.data(), d_idF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.xPos.data(), d_xPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yPos.data(), d_yPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zPos.data(), d_zPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.xVel.data(), d_xVelF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yVel.data(), d_yVelF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zVel.data(), d_zVelF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.xVelPred.data(), d_xVelPredF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yVelPred.data(), d_yVelPredF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zVelPred.data(), d_zVelPredF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.xAccPress.data(), d_xAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.yAccPress.data(), d_yAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.zAccPress.data(), d_zAccPressF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.sourceTerm.data(), d_sourceTermF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.diagElem.data(), d_diagElemF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.pressure.data(), d_pressureF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(F.density.data(), d_densityF_.Data(), N_F_, stream_);
	}

	// copy boundary particle data to host
	if (N_B_ > 0)
	{
		GPU::Copy_DeviceToHost(B.id.data(), d_idB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.xPos.data(), d_xPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.yPos.data(), d_yPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.zPos.data(), d_zPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.volB.data(), d_volB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(B.psiB.data(), d_psiB_.Data(), N_B_, stream_);
	}

	// copy rigid particle data to host
	if (N_R_ > 0)
	{
		GPU::Copy_DeviceToHost(R.id.data(), d_idR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.bodyId.data(), d_bodyIdR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.xRelPos.data(), d_xRelPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.yRelPos.data(), d_yRelPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.zRelPos.data(), d_zRelPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.volR.data(), d_volR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(R.psiR.data(), d_psiR_.Data(), N_R_, stream_);
	}

	// copy rigid body data to host
	if (N_RB_ > 0)
	{
		GPU::Copy_DeviceToHost(RB.id.data(), d_idRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.rigidParticleBegin.data(), d_rigidParticleBeginRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.rigidParticleCount.data(), d_rigidParticleCountRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.xCM.data(), d_xCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.yCM.data(), d_yCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.zCM.data(), d_zCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.xVel.data(), d_xVelRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.yVel.data(), d_yVelRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.zVel.data(), d_zVelRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.xOmega.data(), d_xOmegaRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.yOmega.data(), d_yOmegaRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.zOmega.data(), d_zOmegaRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.xAngMom.data(), d_xAngMomRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.yAngMom.data(), d_yAngMomRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.zAngMom.data(), d_zAngMomRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a00.data(), d_a00RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a01.data(), d_a01RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a02.data(), d_a02RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a10.data(), d_a10RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a11.data(), d_a11RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a12.data(), d_a12RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a20.data(), d_a20RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a21.data(), d_a21RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.a22.data(), d_a22RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.mass.data(), d_massRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.massInv.data(), d_massInvRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv00.data(), d_iBodyInv00RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv01.data(), d_iBodyInv01RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv02.data(), d_iBodyInv02RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv10.data(), d_iBodyInv10RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv11.data(), d_iBodyInv11RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv12.data(), d_iBodyInv12RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv20.data(), d_iBodyInv20RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv21.data(), d_iBodyInv21RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(RB.iBodyInv22.data(), d_iBodyInv22RB_.Data(), N_RB_, stream_);
	}

	GPU::StreamSync(stream_);
}

void SimBackend_GPU::PackExportSnapshot(
	Sync::ExportSnapshot& snap, const SimParams& simParams) const
{
	snap.simStep = simStep_;
	snap.simTime = simTime_;

	// copy boundary particles
	snap.N_B = N_B_;
	snap.xPosB.resize(snap.N_B);
	snap.yPosB.resize(snap.N_B);
	snap.zPosB.resize(snap.N_B);
	snap.volB.resize(snap.N_B);
	if (N_B_ > 0)
	{
		GPU::Copy_DeviceToHost(snap.xPosB.data(), d_xPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(snap.yPosB.data(), d_yPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(snap.zPosB.data(), d_zPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(snap.volB.data(), d_volB_.Data(), N_B_, stream_);
	}

	// copy rigid particles
	snap.N_R = N_R_;
	snap.bodyIdR.resize(snap.N_R);
	snap.xPosR.resize(snap.N_R);
	snap.yPosR.resize(snap.N_R);
	snap.zPosR.resize(snap.N_R);
	snap.volR.resize(snap.N_R);
	if (N_R_ > 0)
	{
		GPU::Copy_DeviceToHost(snap.bodyIdR.data(), d_bodyIdR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(snap.xPosR.data(), d_xPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(snap.yPosR.data(), d_yPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(snap.zPosR.data(), d_zPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(snap.volR.data(), d_volR_.Data(), N_R_, stream_);
	}
	GPU::StreamSync(stream_);

	if (N_F_ == 0)
	{
		snap.N_F = 0;
		snap.xPosF.clear();
		snap.yPosF.clear();
		snap.zPosF.clear();
		snap.velMagF.clear();

		return;
	}

	// host staging for fluid neighbor list data
	auto& h_nbrCountFF = h_expNbrCountFF_;
	auto& h_nbrCountFB = h_expNbrCountFB_;
	auto& h_nbrCountFR = h_expNbrCountFR_;
	auto& h_nbrChunkOffsetsFF = h_expNbrChunkOffsetsFF_;
	auto& h_nbrChunkBaseIdxFF = h_expNbrChunkBaseIdxFF_;
	auto& h_nbrChunkFormatFF = h_expNbrChunkFormatFF_;
	auto& h_nbrListDelta8FF = h_expNbrListDelta8FF_;
	auto& h_nbrListDelta16FF = h_expNbrListDelta16FF_;
	auto& h_nbrListAbs32FF = h_expNbrListAbs32FF_;
	const uint32_t N_sellChunks = SellChunkNumTotal(N_F_);

	h_nbrCountFF.resize(N_F_);
	h_nbrCountFB.resize(N_F_);
	h_nbrCountFR.resize(N_F_);
	h_nbrChunkOffsetsFF.resize(static_cast<size_t>(N_sellChunks) + 1ull);
	h_nbrChunkBaseIdxFF.resize(N_sellChunks);
	h_nbrChunkFormatFF.resize(N_sellChunks);
	h_nbrListDelta8FF.resize(static_cast<size_t>(N_delta8FF_));
	h_nbrListDelta16FF.resize(static_cast<size_t>(N_delta16FF_));
	h_nbrListAbs32FF.resize(static_cast<size_t>(N_abs32FF_));

	// host staging for fluid positions and velocities
	auto& h_xPosF = h_expXPosF_;
	auto& h_yPosF = h_expYPosF_;
	auto& h_zPosF = h_expZPosF_;
	auto& h_xVelF = h_expXVelF_;
	auto& h_yVelF = h_expYVelF_;
	auto& h_zVelF = h_expZVelF_;

	h_xPosF.resize(N_F_);
	h_yPosF.resize(N_F_);
	h_zPosF.resize(N_F_);
	h_xVelF.resize(N_F_);
	h_yVelF.resize(N_F_);
	h_zVelF.resize(N_F_);

	// launch async device to host copies
	GPU::Copy_DeviceToHost(h_nbrCountFF.data(), d_nbrCountFF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrCountFB.data(), d_nbrCountFB_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrCountFR.data(), d_nbrCountFR_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_nbrChunkOffsetsFF.data(), d_nbrChunkOffsetsFF_.Data(),
						   static_cast<size_t>(N_sellChunks) + 1ull, stream_);
	GPU::Copy_DeviceToHost(h_nbrChunkBaseIdxFF.data(), d_nbrChunkBaseIdxFF_.Data(),
						   N_sellChunks, stream_);
	GPU::Copy_DeviceToHost(h_nbrChunkFormatFF.data(), d_nbrChunkFormatFF_.Data(),
						   N_sellChunks, stream_);
	GPU::Copy_DeviceToHost(h_nbrListDelta8FF.data(), d_nbrListDelta8FF_.Data(),
						   N_delta8FF_, stream_);
	GPU::Copy_DeviceToHost(h_nbrListDelta16FF.data(), d_nbrListDelta16FF_.Data(),
						   N_delta16FF_, stream_);
	GPU::Copy_DeviceToHost(h_nbrListAbs32FF.data(), d_nbrListAbs32FF_.Data(),
						   N_abs32FF_, stream_);

	GPU::Copy_DeviceToHost(h_xPosF.data(), d_xPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_yPosF.data(), d_yPosF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_zPosF.data(), d_zPosF_.Data(), N_F_, stream_);

	GPU::Copy_DeviceToHost(h_xVelF.data(), d_xVelF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_yVelF.data(), d_yVelF_.Data(), N_F_, stream_);
	GPU::Copy_DeviceToHost(h_zVelF.data(), d_zVelF_.Data(), N_F_, stream_);

	// wait for all copies to finish
	GPU::StreamSync(stream_);

	// classify surface particles
	const uint32_t nbrCountThreshold = simParams.surfaceNbrCountThreshold;

	auto& surfaceFlags = h_expSurfaceFlagsScratch_;
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
				if (!IsNearSurfaceFluidParticle(
					h_nbrCountFF[i], h_nbrCountFB[i], h_nbrCountFR[i], nbrCountThreshold))
				{
					continue;
				}

				Mark(i);

				const uint32_t N_FF = h_nbrCountFF[i];

				for (uint32_t k = 0; k < N_FF; k++)
				{
					Mark(ReadCompressedSell32Neighbor(
						h_nbrChunkOffsetsFF, h_nbrChunkBaseIdxFF,
						h_nbrChunkFormatFF,
						h_nbrListDelta8FF, h_nbrListDelta16FF, h_nbrListAbs32FF,
						i, k));
				}
			}
		},
		tbb::auto_partitioner{});

	const uint32_t N_surface = static_cast<uint32_t>(
		std::ranges::count(surfaceFlags, uint8_t{ 1 }));

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
	for (uint32_t i = 0; i < N_F_; i++)
	{
		if (!surfaceFlags[i]) { continue; }

		snap.xPosF[dst] = h_xPosF[i];
		snap.yPosF[dst] = h_yPosF[i];
		snap.zPosF[dst] = h_zPosF[i];

		const float vx = h_xVelF[i];
		const float vy = h_yVelF[i];
		const float vz = h_zVelF[i];

		snap.velMagF[dst] = std::sqrt(vx * vx + vy * vy + vz * vz);

		dst++;
	}
}

void SimBackend_GPU::PackVisualSnapshot(Sync::VisualSnapshot& snap) const
{
	snap.N_F = N_F_;
	snap.N_B = N_B_;
	snap.N_R = N_R_;
	snap.N_RB = N_RB_;
	snap.simStep = simStep_;

	if (N_F_ == 0)
	{
		snap.posF.clear();
		snap.scalarDataF.clear();
	}
	else
	{
		// copy fluid particles
		std::vector<float> xPosF(N_F_), yPosF(N_F_), zPosF(N_F_);
		GPU::Copy_DeviceToHost(xPosF.data(), d_xPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(yPosF.data(), d_yPosF_.Data(), N_F_, stream_);
		GPU::Copy_DeviceToHost(zPosF.data(), d_zPosF_.Data(), N_F_, stream_);
		GPU::StreamSync(stream_);

		snap.posF.resize(N_F_);
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					snap.posF[i] = glm::vec3{ xPosF[i], yPosF[i], zPosF[i] };
				}
			},
			tbb::auto_partitioner{});

		FillScalarData(snap, snap.scalarModeF);
	}

	// copy boundary particles
	snap.posB.resize(snap.N_B);
	snap.volB.resize(snap.N_B);
	std::vector<float> xPosB(N_B_), yPosB(N_B_), zPosB(N_B_);
	if (N_B_ > 0)
	{
		GPU::Copy_DeviceToHost(xPosB.data(), d_xPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(yPosB.data(), d_yPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(zPosB.data(), d_zPosB_.Data(), N_B_, stream_);
		GPU::Copy_DeviceToHost(snap.volB.data(), d_volB_.Data(), N_B_, stream_);
	}

	// copy rigid particles
	snap.posR.resize(snap.N_R);
	snap.volR.resize(snap.N_R);
	std::vector<float> xPosR(N_R_), yPosR(N_R_), zPosR(N_R_);
	if (N_R_ > 0)
	{
		GPU::Copy_DeviceToHost(xPosR.data(), d_xPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(yPosR.data(), d_yPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(zPosR.data(), d_zPosR_.Data(), N_R_, stream_);
		GPU::Copy_DeviceToHost(snap.volR.data(), d_volR_.Data(), N_R_, stream_);
	}
	GPU::StreamSync(stream_);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, snap.N_B),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posB[i] = glm::vec3{ xPosB[i], yPosB[i], zPosB[i] };
			}
		},
		tbb::auto_partitioner{});

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, snap.N_R),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				snap.posR[i] = glm::vec3{ xPosR[i], yPosR[i], zPosR[i] };
			}
		},
		tbb::auto_partitioner{});

	PackVisualRigidBodyState(snap);
}

Sync::VisualPackResultCUDA SimBackend_GPU::TryPackVisualSnapshotCUDA(
	Sync::VisualFrameMailboxCUDA& visualFrameMailboxCUDA,
	std::shared_ptr<Sync::VisualSnapshot> auxiliarySnap) const
{
	// acquire reusable CUDA mailbox slot for particle data
	Sync::VisualFrameWriteCUDA write{};
	if (!visualFrameMailboxCUDA.TryAcquireWrite(N_F_, N_R_, write))
	{
		return visualFrameMailboxCUDA.IsEnabled()
			? Sync::VisualPackResultCUDA::Skipped
			: Sync::VisualPackResultCUDA::Unsupported;
	}

	// pack fluid and rigid particles into CUDA-device buffers
	try
	{
		GPU::Launch_PackVisualSnapshotCUDA(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data(),
			d_densityF_.Data(), d_pressureF_.Data(),
			d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(), d_volR_.Data(),
			auxiliarySnap->scalarModeF, N_F_, N_R_, stream_,
			write.fluid, write.rigid);

		// pack remaining rigid body state into auxiliary host snapshot
		auxiliarySnap->N_F = 0;
		auxiliarySnap->N_B = 0;
		auxiliarySnap->N_R = 0;
		auxiliarySnap->N_RB = N_RB_;
		PackVisualRigidBodyState(*auxiliarySnap);
	}
	catch (const std::exception& ex)
	{
		visualFrameMailboxCUDA.Disable(ex.what());
		return Sync::VisualPackResultCUDA::Unsupported;
	}

	// publish packed frame to visualization thread
	if (!visualFrameMailboxCUDA.Publish(write, stream_, std::move(auxiliarySnap)))
	{
		return Sync::VisualPackResultCUDA::Unsupported;
	}
	return Sync::VisualPackResultCUDA::Published;
}

void SimBackend_GPU::PackVisualRigidBodyState(
	Sync::VisualSnapshot& snap) const
{
	snap.centerOfMassRB.resize(N_RB_);
	snap.orientationRB.resize(N_RB_);
	if (N_RB_ > 0)
	{
		std::vector<float> h_xCMR(N_RB_), h_yCMR(N_RB_), h_zCMR(N_RB_);
		std::vector<float> h_a00R(N_RB_), h_a01R(N_RB_), h_a02R(N_RB_);
		std::vector<float> h_a10R(N_RB_), h_a11R(N_RB_), h_a12R(N_RB_);
		std::vector<float> h_a20R(N_RB_), h_a21R(N_RB_), h_a22R(N_RB_);

		GPU::Copy_DeviceToHost(h_xCMR.data(), d_xCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_yCMR.data(), d_yCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_zCMR.data(), d_zCMRB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a00R.data(), d_a00RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a01R.data(), d_a01RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a02R.data(), d_a02RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a10R.data(), d_a10RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a11R.data(), d_a11RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a12R.data(), d_a12RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a20R.data(), d_a20RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a21R.data(), d_a21RB_.Data(), N_RB_, stream_);
		GPU::Copy_DeviceToHost(h_a22R.data(), d_a22RB_.Data(), N_RB_, stream_);
		GPU::StreamSync(stream_);

		for (uint32_t bodyIdx = 0; bodyIdx < N_RB_; bodyIdx++)
		{
			snap.centerOfMassRB[bodyIdx] =
				{ h_xCMR[bodyIdx], h_yCMR[bodyIdx], h_zCMR[bodyIdx] };
			snap.orientationRB[bodyIdx][0] =
				{ h_a00R[bodyIdx], h_a10R[bodyIdx], h_a20R[bodyIdx] };
			snap.orientationRB[bodyIdx][1] =
				{ h_a01R[bodyIdx], h_a11R[bodyIdx], h_a21R[bodyIdx] };
			snap.orientationRB[bodyIdx][2] =
				{ h_a02R[bodyIdx], h_a12R[bodyIdx], h_a22R[bodyIdx] };
		}
	}
}

void SimBackend_GPU::FillScalarData(
	Sync::VisualSnapshot& snap,
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
			snap.scalarDataF.resize(N_F_);

			// host staging for velocities
			std::vector<float> h_xVelF(N_F_);
			std::vector<float> h_yVelF(N_F_);
			std::vector<float> h_zVelF(N_F_);

			// launch async device to host copies
			GPU::Copy_DeviceToHost(h_xVelF.data(), d_xVelF_.Data(), N_F_, stream_);
			GPU::Copy_DeviceToHost(h_yVelF.data(), d_yVelF_.Data(), N_F_, stream_);
			GPU::Copy_DeviceToHost(h_zVelF.data(), d_zVelF_.Data(), N_F_, stream_);

			// wait for all copies to finish
			GPU::StreamSync(stream_);

			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_F_),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const float vxi = h_xVelF[i];
						const float vyi = h_yVelF[i];
						const float vzi = h_zVelF[i];

						// magnitude of velocity vector
						snap.scalarDataF[i] = std::sqrt(vxi * vxi
							+ vyi * vyi
							+ vzi * vzi);
					}
				},
				tbb::auto_partitioner{}
			);
			return;
		}
	case Base::ColorCodingMode::Density:
		{
			snap.scalarDataF.resize(N_F_);

			// copy density data directly into the host snapshot
			GPU::Copy_DeviceToHost(snap.scalarDataF.data(), d_densityF_.Data(), N_F_, stream_);
			GPU::StreamSync(stream_);
			return;
		}
	case Base::ColorCodingMode::Pressure:
		{
			snap.scalarDataF.resize(N_F_);

			// copy pressure data directly into the host snapshot
			GPU::Copy_DeviceToHost(snap.scalarDataF.data(), d_pressureF_.Data(), N_F_, stream_);
			GPU::StreamSync(stream_);
			return;
		}
	}
	
	throw std::runtime_error("Unsupported scalar data mode in GPU.");
}

void SimBackend_GPU::FillMemoryRecord(Mem::MemRecordStep& record) const
{
	using namespace Mem;
	constexpr MemSpace memSpaceRAM = MemSpace::RAM;
	constexpr MemSpace memSpaceVRAM = MemSpace::VRAM;

	// --- [ RAM: particle properties ] --------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		MemUsage usageR{};

		// fluid
		Add(usageF, MemUsageOf_Vector(h_expXPosF_));
		Add(usageF, MemUsageOf_Vector(h_expYPosF_));
		Add(usageF, MemUsageOf_Vector(h_expZPosF_));
		Add(usageF, MemUsageOf_Vector(h_expXVelF_));
		Add(usageF, MemUsageOf_Vector(h_expYVelF_));
		Add(usageF, MemUsageOf_Vector(h_expZVelF_));

		// boundary

		// rigid

		// finalize
		Set(record, memSpaceRAM, MemBucket::ParticlePropertiesF, usageF);
		Set(record, memSpaceRAM, MemBucket::ParticlePropertiesB, usageB);
		Set(record, memSpaceRAM, MemBucket::ParticlePropertiesR, usageR);
	}

	// --- [ RAM: neighbor list storage ] ------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		MemUsage usageR{};

		// fluid
		Add(usageF, MemUsageOf_Vector(h_expNbrCountFF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrChunkOffsetsFF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrChunkBaseIdxFF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrChunkFormatFF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrListDelta8FF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrListDelta16FF_));
		Add(usageF, MemUsageOf_Vector(h_expNbrListAbs32FF_));

		// boundary
		Add(usageB, MemUsageOf_Vector(h_expNbrCountFB_));

		// rigid
		Add(usageR, MemUsageOf_Vector(h_expNbrCountFR_));

		// finalize
		Set(record, memSpaceRAM, MemBucket::Neighbors_ListStorageF, usageF);
		Set(record, memSpaceRAM, MemBucket::Neighbors_ListStorageB, usageB);
		Set(record, memSpaceRAM, MemBucket::Neighbors_ListStorageR, usageR);
	}

	// --- [ RAM: misc ] -----------------------------------------------------------------------------

	{
		MemUsage usage{};

		Add(usage, MemUsageOf_Vector(h_expSurfaceFlagsScratch_));

		// finalize
		Set(record, memSpaceRAM, MemBucket::Misc, usage);
	}

	// --- [ VRAM: particle properties ] -------------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		MemUsage usageR{};

		// fluid
		Add(usageF, MemUsageOf_DeviceBuffer(d_idF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_xPosF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_yPosF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_zPosF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_xVelF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_yVelF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_zVelF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_xVelPredF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_yVelPredF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_zVelPredF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_xAccPressF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_yAccPressF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_zAccPressF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_sourceTermF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_diagElemF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_pressureF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_densityF_));

		// boundary
		Add(usageB, MemUsageOf_DeviceBuffer(d_idB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_xPosB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_yPosB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_zPosB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_volB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_psiB_));

		// rigid
		Add(usageR, MemUsageOf_DeviceBuffer(d_idR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_bodyIdR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_xPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_yPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_zPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_xRelPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_yRelPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_zRelPosR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_xVelR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_yVelR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_zVelR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_xForceR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_yForceR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_zForceR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_volR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_psiR_));

		// finalize
		Set(record, memSpaceVRAM, MemBucket::ParticlePropertiesF, usageF);
		Set(record, memSpaceVRAM, MemBucket::ParticlePropertiesB, usageB);
		Set(record, memSpaceVRAM, MemBucket::ParticlePropertiesR, usageR);
	}

	// --- [ VRAM: neighbor search data structures ] -------------------------------------------------

	{
		const ParticleClassMemUsage usage = neighborSearch_->GetMemUsage();

		// finalize
		Set(record, memSpaceVRAM, MemBucket::Neighbors_SearchDataStructF, usage.fluid);
		Set(record, memSpaceVRAM, MemBucket::Neighbors_SearchDataStructB, usage.boundary);
		Set(record, memSpaceVRAM, MemBucket::Neighbors_SearchDataStructR, usage.rigid);
	}

	// --- [ VRAM: neighbor list storage ] -----------------------------------------------------------

	{
		MemUsage usageF{};
		MemUsage usageB{};
		MemUsage usageR{};

		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrCountFF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrChunkMaxCountFF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrChunkBaseIdxFF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrChunkFormatFF_));
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrChunkOffsetsFF_));

		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrCountFB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrChunkMaxCountFB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrChunkBaseIdxFB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrChunkFormatFB_));
		Add(usageB, MemUsageOf_DeviceBuffer(d_nbrChunkOffsetsFB_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrCountFR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrChunkMaxCountFR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrChunkBaseIdxFR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrChunkFormatFR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrChunkOffsetsFR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrCountRF_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrCountRB_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrCountRR_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrOffsetsRF_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrOffsetsRB_));
		Add(usageR, MemUsageOf_DeviceBuffer(d_nbrOffsetsRR_));

		// compressed SELL32 storage used for FF, FB, and FR neighbors
		MemUsage listFF{};
		listFF.N_usedBytes = static_cast<uint64_t>(N_packedFF_);
		listFF.N_allocBytes =
			static_cast<uint64_t>(d_nbrListDelta8FF_.Size()) * sizeof(uint8_t)
			+ static_cast<uint64_t>(d_nbrListDelta16FF_.Size()) * sizeof(uint16_t)
			+ static_cast<uint64_t>(d_nbrListAbs32FF_.Size()) * sizeof(uint32_t);
		Add(usageF, listFF);

		MemUsage listFB{};
		listFB.N_usedBytes = static_cast<uint64_t>(N_packedFB_);
		listFB.N_allocBytes =
			static_cast<uint64_t>(d_nbrListDelta8FB_.Size()) * sizeof(uint8_t)
			+ static_cast<uint64_t>(d_nbrListDelta16FB_.Size()) * sizeof(uint16_t)
			+ static_cast<uint64_t>(d_nbrListAbs32FB_.Size()) * sizeof(uint32_t);
		Add(usageB, listFB);

		MemUsage listFR{};
		listFR.N_usedBytes = static_cast<uint64_t>(N_packedFR_);
		listFR.N_allocBytes =
			static_cast<uint64_t>(d_nbrListDelta8FR_.Size()) * sizeof(uint8_t)
			+ static_cast<uint64_t>(d_nbrListDelta16FR_.Size()) * sizeof(uint16_t)
			+ static_cast<uint64_t>(d_nbrListAbs32FR_.Size()) * sizeof(uint32_t);
		Add(usageR, listFR);

		MemUsage listRF{};
		listRF.N_usedBytes = N_totalRF_ * sizeof(uint32_t);
		listRF.N_allocBytes = static_cast<uint64_t>(d_nbrListRF_.Size()) * sizeof(uint32_t);
		Add(usageR, listRF);

		MemUsage listRB{};
		listRB.N_usedBytes = N_totalRB_ * sizeof(uint32_t);
		listRB.N_allocBytes = static_cast<uint64_t>(d_nbrListRB_.Size()) * sizeof(uint32_t);
		Add(usageR, listRB);

		MemUsage listRR{};
		listRR.N_usedBytes = N_totalRR_ * sizeof(uint32_t);
		listRR.N_allocBytes = static_cast<uint64_t>(d_nbrListRR_.Size()) * sizeof(uint32_t);
		Add(usageR, listRR);

		// shared overflow flag is accounted in the fluid-side bucket to avoid double-counting
		Add(usageF, MemUsageOf_DeviceBuffer(d_nbrOverflowFlag_));

		// finalize
		Set(record, memSpaceVRAM, MemBucket::Neighbors_ListStorageF, usageF);
		Set(record, memSpaceVRAM, MemBucket::Neighbors_ListStorageB, usageB);
		Set(record, memSpaceVRAM, MemBucket::Neighbors_ListStorageR, usageR);
	}

	// --- [ VRAM: misc ] ----------------------------------------------------------------------------

	{
		MemUsage usage{};

		Add(usage, MemUsageOf_DeviceBuffer(d_reorderIdScratch_));
		Add(usage, MemUsageOf_DeviceBuffer(d_sortParticleIdxScratch_));
		Add(usage, MemUsageOf_DeviceBuffer(d_idRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_rigidParticleBeginRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_rigidParticleCountRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xCMRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yCMRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zCMRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xVelRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yVelRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zVelRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xOmegaRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yOmegaRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zOmegaRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xAngMomRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yAngMomRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zAngMomRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xForceRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yForceRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zForceRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_xTorqueRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_yTorqueRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_zTorqueRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a00RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a01RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a02RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a10RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a11RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a12RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a20RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a21RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_a22RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_massRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_massInvRB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv00RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv01RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv02RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv10RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv11RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv12RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv20RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv21RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_iBodyInv22RB_));
		Add(usage, MemUsageOf_DeviceBuffer(d_nbrCountHist_));
		Add(usage, MemUsageOf_DeviceBuffer(d_nbrCountHistChunk_));
		Add(usage, MemUsageOf_DeviceBuffer(d_nbrCountStats_));
		Add(usage, MemUsageOf_DeviceBuffer(d_densityStatsPartials_));
		Add(usage, MemUsageOf_DeviceBuffer(d_densityStats_));
		Add(usage, MemUsageOf_DeviceBuffer(d_velocityStatsPartials_));
		Add(usage, MemUsageOf_DeviceBuffer(d_velocityStats_));

		Set(record, memSpaceVRAM, MemBucket::Misc, usage);
	}
}

void SimBackend_GPU::FillStatisticsRecord(
	Stats::StatRecordStep& record,
	const SimParams& simParams) const
{
	using namespace Stats;

	const uint32_t NF = N_F_;
	const uint32_t NB = N_B_;
	const uint32_t NR = N_R_;

	Set(record, StatMetric::NumF, static_cast<double>(NF));
	Set(record, StatMetric::NumB, static_cast<double>(NB));
	Set(record, StatMetric::NumR, static_cast<double>(NR));

	const double invNF = NF ? 1.0 / static_cast<double>(NF) : 0.0;
	const double invNR = NR ? 1.0 / static_cast<double>(NR) : 0.0;

	benchmarksTracker_.StartBench(Bench::BenchOp::Statistics_Computation);

	// --- [ particle counts ] -----------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Particles)])
	{
		// compute neighbor-count extrema and percentiles on device
		constexpr uint32_t N_relations = 6;
		d_nbrCountStats_.Resize(N_relations);
		if (NF > 0 || NR > 0)
		{
			d_nbrCountHist_.Resize(GPU::N_NBR_COUNT_HISTOGRAM_BINS);
			d_nbrCountHistChunk_.Resize(GPU::N_NBR_COUNT_HISTOGRAM_CHUNKS);
		}

		const std::array<const uint16_t*, N_relations> counts
		{
			d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
			d_nbrCountRF_.Data(), d_nbrCountRB_.Data(), d_nbrCountRR_.Data()
		};
		const std::array<uint32_t, N_relations> sourceCounts
		{
			NF, NF, NF, NR, NR, NR
		};
		for (uint32_t relation = 0; relation < N_relations; relation++)
		{
			GPU::Launch_ComputeNeighborCountStats(
				counts[relation], sourceCounts[relation], stream_,
				d_nbrCountHist_.Data(), d_nbrCountHistChunk_.Data(),
				d_nbrCountStats_.Data() + relation);
		}

		GPU::Copy_DeviceToHost(
			h_nbrCountStats_.data(), d_nbrCountStats_.Data(), N_relations, stream_);
		GPU::StreamSync(stream_);

		const auto setNeighborStats = [&](const GPU::NeighborCountStats& stats,
			const uint64_t total, const double N_sourcesInv,
			const StatMetric avg, const StatMetric min, const StatMetric max,
			const StatMetric p99)
		{
			Set(record, avg, static_cast<double>(total) * N_sourcesInv);
			Set(record, min, static_cast<double>(stats.min));
			Set(record, max, static_cast<double>(stats.max));
			Set(record, p99, static_cast<double>(stats.p99));
		};

		setNeighborStats(h_nbrCountStats_[0], N_totalFF_, invNF,
			StatMetric::NumNbrFF_Avg, StatMetric::NumNbrFF_Min, StatMetric::NumNbrFF_Max,
			StatMetric::NumNbrFF_P99);
		setNeighborStats(h_nbrCountStats_[1], N_totalFB_, invNF,
			StatMetric::NumNbrFB_Avg, StatMetric::NumNbrFB_Min, StatMetric::NumNbrFB_Max,
			StatMetric::NumNbrFB_P99);
		setNeighborStats(h_nbrCountStats_[2], N_totalFR_, invNF,
			StatMetric::NumNbrFR_Avg, StatMetric::NumNbrFR_Min, StatMetric::NumNbrFR_Max,
			StatMetric::NumNbrFR_P99);
		setNeighborStats(h_nbrCountStats_[3], N_totalRF_, invNR,
			StatMetric::NumNbrRF_Avg, StatMetric::NumNbrRF_Min, StatMetric::NumNbrRF_Max,
			StatMetric::NumNbrRF_P99);
		setNeighborStats(h_nbrCountStats_[4], N_totalRB_, invNR,
			StatMetric::NumNbrRB_Avg, StatMetric::NumNbrRB_Min, StatMetric::NumNbrRB_Max,
			StatMetric::NumNbrRB_P99);
		setNeighborStats(h_nbrCountStats_[5], N_totalRR_, invNR,
			StatMetric::NumNbrRR_Avg, StatMetric::NumNbrRR_Min, StatMetric::NumNbrRR_Max,
			StatMetric::NumNbrRR_P99);
	}

	// --- [ density stats ] -------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Density)])
	{
		d_densityStats_.Resize(1);
		if (NF > 0)
		{
			d_densityStatsPartials_.Resize(GPU::StatisticsReductionBlockCount(NF));
		}

		GPU::Launch_ComputeDensityStats(
			d_densityF_.Data(),
			simParams.restDensity,
			NF,
			stream_,
			d_densityStatsPartials_.Data(),
			d_densityStats_.Data());

		GPU::Copy_DeviceToHost(&h_densityStats_, d_densityStats_.Data(), 1, stream_);
		GPU::StreamSync(stream_);

		Set(record, StatMetric::DensRelError_Raw_Avg, h_densityStats_.sumRelErrRaw * invNF);
		Set(record, StatMetric::DensRelError_Pos_Avg, h_densityStats_.sumRelErrPos * invNF);
		Set(record, StatMetric::DensRelError_Abs_Avg, h_densityStats_.sumRelErrAbs * invNF);
	}

	// --- [ pressure stats ] ------------------------------------------------------------------------

	Set(record, StatMetric::NumSolverIters, static_cast<double>(N_solverIters_));

	// --- [ velocity stats ] ------------------------------------------------------------------------

	if (simParams.statGroupsEnabled[GetIdx(StatGroup::Velocity)])
	{
		d_velocityStats_.Resize(1);
		if (NF > 0)
		{
			d_velocityStatsPartials_.Resize(GPU::StatisticsReductionBlockCount(NF));
		}

		GPU::Launch_ComputeVelocityStats(
			d_xVelF_.Data(),
			d_yVelF_.Data(),
			d_zVelF_.Data(),
			NF,
			stream_,
			d_velocityStatsPartials_.Data(),
			d_velocityStats_.Data());

		GPU::Copy_DeviceToHost(&h_velocityStats_, d_velocityStats_.Data(), 1, stream_);
		GPU::StreamSync(stream_);

		Set(record, StatMetric::CFL_Number, h_velocityStats_.velMagMax * simParams.timeStep / simParams.particleSizeF);
		Set(record, StatMetric::VelMag_Avg, h_velocityStats_.velMagSum * invNF);
		Set(record, StatMetric::VelMag_Min, h_velocityStats_.velMagMin);
		Set(record, StatMetric::VelMag_Max, h_velocityStats_.velMagMax);
	}

	benchmarksTracker_.StopBench(Bench::BenchOp::Statistics_Computation);
}

void SimBackend_GPU::UpdateRigidParticles()
{
	GPU::Launch_UpdateRigidParticles(
		d_bodyIdR_.Data(), d_xRelPosR_.Data(), d_yRelPosR_.Data(), d_zRelPosR_.Data(),
		d_xCMRB_.Data(), d_yCMRB_.Data(), d_zCMRB_.Data(),
		d_xVelRB_.Data(), d_yVelRB_.Data(), d_zVelRB_.Data(),
		d_xOmegaRB_.Data(), d_yOmegaRB_.Data(), d_zOmegaRB_.Data(),
		d_a00RB_.Data(), d_a01RB_.Data(), d_a02RB_.Data(),
		d_a10RB_.Data(), d_a11RB_.Data(), d_a12RB_.Data(),
		d_a20RB_.Data(), d_a21RB_.Data(), d_a22RB_.Data(),
		N_R_, N_RB_, stream_,
		d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
		d_xVelR_.Data(), d_yVelR_.Data(), d_zVelR_.Data());
}

void SimBackend_GPU::ExecuteNeighborSearch(const SimParams& params, bool bench)
{
	if (N_F_ == 0 && N_R_ == 0)
	{
		N_totalFF_ = N_totalFB_ = N_totalFR_ = N_totalRF_ = N_totalRB_ = N_totalRR_ = 0;
		N_packedFF_ = N_packedFB_ = N_packedFR_ = 0;
		N_delta8FF_ = N_delta8FB_ = N_delta8FR_ = 0;
		N_delta16FF_ = N_delta16FB_ = N_delta16FR_ = 0;
		N_abs32FF_ = N_abs32FB_ = N_abs32FR_ = 0;
		return;
	}

	GPU::ReloadConstants(params, N_F_, N_B_, stream_);
	UpdateRigidParticles();

	if (bench)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
	}
	neighborSearch_->DataReOrder(params);
	if (bench)
	{
		benchmarksTracker_.StopBench(Bench::BenchOp::ParticleData_Reordering);
	}

	if (bench)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
	}
	neighborSearch_->Construct(params);
	if (bench)
	{
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);
	}

	if (bench)
	{
		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
	}
	neighborSearch_->Query_Count(params);
	FinalizeNeighborLists_Count();
	neighborSearch_->Query_Fill(params);
	if (bench)
	{
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);
	}
}

void SimBackend_GPU::ExecuteFullSimStep(const SimParams& params, uint32_t N_steps)
{
	if (N_F_ == 0) // empty sim
	{
		simStep_ += N_steps;
		simTime_ += N_steps * params.timeStep;

		return;
	}

	const SolverConstants solverConstants = BuildSolverConstants(params.particleSizeF, params.restDensity);

	for (uint64_t step = 0; step < N_steps; step++)
	{
		GPU::ReloadConstants(params, N_F_, N_B_, stream_);

		if (N_RB_ > 0)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_StateUpdate);
			UpdateRigidParticles();
			GPU::StreamSync(stream_);
			benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_StateUpdate);
		}

		// --- [ particle data re-ordering ] -----------------------------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::ParticleData_Reordering);
		neighborSearch_->DataReOrder_SolverStep(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::ParticleData_Reordering);

		// --- [ neighbor search ] ----------------------------------------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Construction);
		neighborSearch_->Construct(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Construction);

		benchmarksTracker_.StartBench(Bench::BenchOp::Neighbors_Query);
		neighborSearch_->Query_Count(params);
		FinalizeNeighborLists_Count();
		neighborSearch_->Query_Fill(params);
		benchmarksTracker_.StopBench(Bench::BenchOp::Neighbors_Query);

		// --- [ density computations ] -----------------------------------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Density_Computation);
		GPU::Launch_Density(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
			d_psiB_.Data(), d_psiR_.Data(),
			d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
			d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
			d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
			d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
			d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
			d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
			d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
			N_F_, N_B_, N_R_, stream_,
			d_densityF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::Density_Computation);

		// --- [ non-pressure accelerations ] -----------------------------------------------------------

		if (N_RB_ > 0)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
			GPU::Launch_ClearRigidForcesAndAddGravity(
				d_massRB_.Data(), d_massInvRB_.Data(),
				params.gravity.x, params.gravity.y, params.gravity.z,
				N_RB_, stream_,
				d_xForceRB_.Data(), d_yForceRB_.Data(), d_zForceRB_.Data(),
				d_xTorqueRB_.Data(), d_yTorqueRB_.Data(), d_zTorqueRB_.Data());
			GPU::StreamSync(stream_);
			benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
		}

		benchmarksTracker_.StartBench(Bench::BenchOp::NonPressure_Acceleration);
		GPU::Launch_NonPressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data(),
			d_xVelR_.Data(), d_yVelR_.Data(), d_zVelR_.Data(),
			d_densityF_.Data(),
			d_volB_.Data(), d_volR_.Data(),
			d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
			d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
			d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
			d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
			d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
			d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
			d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
			N_F_, N_B_, N_R_, stream_,
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::NonPressure_Acceleration);

		if (N_RB_ > 0)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
			GPU::Launch_AccumulateRigidParticleNonPressureForces(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
				d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data(),
				d_xVelR_.Data(), d_yVelR_.Data(), d_zVelR_.Data(),
				d_volR_.Data(),
				d_nbrCountRF_.Data(), d_nbrOffsetsRF_.Data(), d_nbrListRF_.Data(),
				solverConstants.sizeInvF, solverConstants.sizeSquF,
				params.kernelRadiusSqu, solverConstants.cubicSplineAlpha,
				params.massF, params.viscosityB, N_R_, stream_,
				d_xForceR_.Data(), d_yForceR_.Data(), d_zForceR_.Data());
			GPU::StreamSync(stream_);
			benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
		}

		// --- [ source-term (density invariance) and diagonal elements ] -------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_SourceAndDiagonal);
		GPU::Launch_SourceTermAndDiagonalElementAndPressureJacobiInit(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_xVelR_.Data(), d_yVelR_.Data(), d_zVelR_.Data(),
			d_densityF_.Data(),
			d_psiB_.Data(), d_psiR_.Data(),
			d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
			d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
			d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
			d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
			d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
			d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
			d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
			N_F_, N_B_, N_R_, stream_,
			d_sourceTermF_.Data(), d_diagElemF_.Data(), d_pressureF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_SourceAndDiagonal);

		// --- [ relaxed jacobi solver for pressure ] ---------------------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_JacobiSolver);

		// rest solver-related variables
		N_solverIters_ = 1;

		do
		{
			// [ loop 1: pressure acceleration ] --------------------------------------------------------
			GPU::Launch_PressureAcceleration(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
				d_pressureF_.Data(),
				d_psiB_.Data(), d_psiR_.Data(),
				d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
				d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
				d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
				d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
				d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
				d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
				d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
				N_F_, N_B_, N_R_, stream_,
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

			// [ loop 2: relaxed jacobi solver ] --------------------------------------------------------
			GPU::Launch_PressureRelaxedJacobiSolver(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
				d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
				d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
				d_sourceTermF_.Data(), d_diagElemF_.Data(),
				d_psiB_.Data(), d_psiR_.Data(),
				d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
				d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
				d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
				d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
				d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
				d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
				d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
				N_F_, N_B_, N_R_, stream_,
				d_pressureF_.Data());

			N_solverIters_++;
		} while (!TerminationConditionSatisfied(params));
		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_JacobiSolver);

		// --- [ finalize pressure acceleration for integration ] ---------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Pressure_Acceleration);
		GPU::Launch_PressureAcceleration(
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xPosB_.Data(), d_yPosB_.Data(), d_zPosB_.Data(),
			d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
			d_pressureF_.Data(),
			d_psiB_.Data(), d_psiR_.Data(),
			d_nbrCountFF_.Data(), d_nbrCountFB_.Data(), d_nbrCountFR_.Data(),
			d_nbrChunkOffsetsFF_.Data(), d_nbrChunkOffsetsFB_.Data(), d_nbrChunkOffsetsFR_.Data(),
			d_nbrChunkBaseIdxFF_.Data(), d_nbrChunkBaseIdxFB_.Data(), d_nbrChunkBaseIdxFR_.Data(),
			d_nbrChunkFormatFF_.Data(), d_nbrChunkFormatFB_.Data(), d_nbrChunkFormatFR_.Data(),
			d_nbrListDelta8FF_.Data(), d_nbrListDelta8FB_.Data(), d_nbrListDelta8FR_.Data(),
			d_nbrListDelta16FF_.Data(), d_nbrListDelta16FB_.Data(), d_nbrListDelta16FR_.Data(),
			d_nbrListAbs32FF_.Data(), d_nbrListAbs32FB_.Data(), d_nbrListAbs32FR_.Data(),
			N_F_, N_B_, N_R_, stream_,
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::Pressure_Acceleration);

		if (N_RB_ > 0)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_ForceAccumulation);
			GPU::Launch_AccumulateRigidParticlePressureForces(
				d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
				d_xPosR_.Data(), d_yPosR_.Data(), d_zPosR_.Data(),
				d_pressureF_.Data(),
				d_psiR_.Data(),
				d_nbrCountRF_.Data(), d_nbrOffsetsRF_.Data(), d_nbrListRF_.Data(),
				solverConstants.sizeInvF, params.kernelRadiusSqu, solverConstants.cubicSplineAlpha,
				params.restDensity, params.massF, params.pressGamma, N_R_, stream_,
				d_xForceR_.Data(), d_yForceR_.Data(), d_zForceR_.Data());

			// reduce rigid particle forces into body force and torque
			GPU::Launch_ReduceRigidParticleForces(
				d_rigidParticleBeginRB_.Data(),
				d_rigidParticleCountRB_.Data(),
				d_xRelPosR_.Data(),
				d_yRelPosR_.Data(),
				d_zRelPosR_.Data(),
				d_xForceR_.Data(),
				d_yForceR_.Data(),
				d_zForceR_.Data(),
				d_a00RB_.Data(), d_a01RB_.Data(), d_a02RB_.Data(),
				d_a10RB_.Data(), d_a11RB_.Data(), d_a12RB_.Data(),
				d_a20RB_.Data(), d_a21RB_.Data(), d_a22RB_.Data(),
				N_RB_, N_R_, stream_,
				d_xForceRB_.Data(), d_yForceRB_.Data(), d_zForceRB_.Data(),
				d_xTorqueRB_.Data(), d_yTorqueRB_.Data(), d_zTorqueRB_.Data());
			GPU::StreamSync(stream_);
			benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_ForceAccumulation);
		}

		// --- [ velocity and position updates ] --------------------------------------------------------

		benchmarksTracker_.StartBench(Bench::BenchOp::Particle_Integration);
		GPU::Launch_Integration(
			d_xVelPredF_.Data(), d_yVelPredF_.Data(), d_zVelPredF_.Data(),
			d_xAccPressF_.Data(), d_yAccPressF_.Data(), d_zAccPressF_.Data(),
			N_F_, N_B_, stream_,
			d_xPosF_.Data(), d_yPosF_.Data(), d_zPosF_.Data(),
			d_xVelF_.Data(), d_yVelF_.Data(), d_zVelF_.Data());

		GPU::StreamSync(stream_);
		benchmarksTracker_.StopBench(Bench::BenchOp::Particle_Integration);

		if (N_RB_ > 0)
		{
			benchmarksTracker_.StartBench(Bench::BenchOp::Rigid_StateUpdate);
			GPU::Launch_IntegrateRigidBodies(
				params.timeStep, N_RB_, stream_,
				d_xCMRB_.Data(), d_yCMRB_.Data(), d_zCMRB_.Data(),
				d_xVelRB_.Data(), d_yVelRB_.Data(), d_zVelRB_.Data(),
				d_xOmegaRB_.Data(), d_yOmegaRB_.Data(), d_zOmegaRB_.Data(),
				d_xAngMomRB_.Data(), d_yAngMomRB_.Data(), d_zAngMomRB_.Data(),
				d_xForceRB_.Data(), d_yForceRB_.Data(), d_zForceRB_.Data(),
				d_xTorqueRB_.Data(), d_yTorqueRB_.Data(), d_zTorqueRB_.Data(),
				d_a00RB_.Data(), d_a01RB_.Data(), d_a02RB_.Data(),
				d_a10RB_.Data(), d_a11RB_.Data(), d_a12RB_.Data(),
				d_a20RB_.Data(), d_a21RB_.Data(), d_a22RB_.Data(),
				d_massInvRB_.Data(),
				d_iBodyInv00RB_.Data(), d_iBodyInv01RB_.Data(), d_iBodyInv02RB_.Data(),
				d_iBodyInv10RB_.Data(), d_iBodyInv11RB_.Data(), d_iBodyInv12RB_.Data(),
				d_iBodyInv20RB_.Data(), d_iBodyInv21RB_.Data(), d_iBodyInv22RB_.Data());
			UpdateRigidParticles();
			GPU::StreamSync(stream_);
			benchmarksTracker_.StopBench(Bench::BenchOp::Rigid_StateUpdate);
		}
	}

	simStep_ += N_steps;
	simTime_ += N_steps * params.timeStep;
}

bool SimBackend_GPU::TerminationConditionSatisfied(const SimParams& params) const
{
	// check min/max iteration condition for potential early-out
	if (N_solverIters_ < params.minSolverIters) { return false; }
	if (N_solverIters_ >= params.maxSolverIters) { return true; }

	bool satisfied = false;

	// check actual termination condition
	switch (params.solverTerminationCond)
	{
	case Base::SolverTerminationCond::FixedIters:
		if (N_solverIters_ >= params.fixedIters)
		{
			satisfied = true;
		}
		break;

	case Base::SolverTerminationCond::RelDensityError_Raw:
	case Base::SolverTerminationCond::RelDensityError_Pos:
	case Base::SolverTerminationCond::RelDensityError_Abs:
	{
		// TODO: implement
		static bool warnedNotImplemented = false;
		if (!warnedNotImplemented)
		{
			SPDLOG_WARN("GPU solver termination condition '{}' is currently not implemented. "
				"Falling back to maxSolverIters.",
				Base::ToCString(params.solverTerminationCond));
			warnedNotImplemented = true;
		}
	}
		break;

	default:
		// unknown enum value
		static bool warnedUnknown = false;
		if (!warnedUnknown)
		{
			SPDLOG_WARN("GPU solver termination condition '<unknown>' is currently not implemented. "
				"Falling back to maxSolverIters.");
			warnedUnknown = true;
		}
		break;
	}

	return satisfied;
}
