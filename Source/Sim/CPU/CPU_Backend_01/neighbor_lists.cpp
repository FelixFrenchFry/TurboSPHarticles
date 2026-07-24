// neighbor_lists.cpp

// external
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

// internal
#include "neighbor_lists.h"



using namespace Sim;

namespace
{
	uint64_t ExpectedPackedCSRCapacity(
		const uint32_t N_F,
		const uint32_t expctdAvgNbrCountHardMax)
	{
		return static_cast<uint64_t>(N_F) * static_cast<uint64_t>(expctdAvgNbrCountHardMax);
	}

	template<typename T>
	void EnsurePackedCSRStorageCapacity(
		std::vector<T>& nbrList,
		const uint64_t count)
	{
		if (nbrList.capacity() < count)
		{
			nbrList.reserve(count);
		}
	}

	template<typename T>
	void EnsurePackedCSRStorageCapacity(
		std::vector<T>& nbrList,
		const uint64_t count,
		const uint32_t extraAllocCapacityPct)
	{
		if (nbrList.capacity() < count)
		{
			const uint64_t capTarget = count + Base::ExtraPercent(count, extraAllocCapacityPct);
			nbrList.reserve(capTarget);
		}
	}

	SELLNeighborCount NarrowNeighborCount_Debug(
		const uint32_t count,
		const char* overflowMsg)
	{
		#if defined(_DEBUG)
		if (count > std::numeric_limits<SELLNeighborCount>::max()) [[unlikely]]
		{
			throw std::runtime_error(overflowMsg);
		}
		#else
		(void)overflowMsg;
		#endif

		return static_cast<SELLNeighborCount>(count);
	}

	void FinalizeCSRNeighborStorage(
		const std::vector<SELLNeighborCount>& nbrCount,
		std::vector<uint64_t>& nbrOffsets,
		std::vector<uint32_t>& nbrList)
	{
		nbrOffsets.resize(static_cast<uint64_t>(nbrCount.size()) + 1ull);
		nbrOffsets[0] = 0;
		for (uint32_t i = 0; i < nbrCount.size(); i++)
		{
			nbrOffsets[i + 1] = nbrOffsets[i] + nbrCount[i];
		}
		nbrList.resize(nbrOffsets.back());
	}


} // anonymous

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL neighbor list, explicit ] ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL_Explicit_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
	const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;
	N_R_ = N_R;

	const uint32_t numChunks = ChunkNumTotal(N_F_);
	const uint32_t paddedCount = numChunks * chunkWidth;

	// count arrays are padded, tail chunk still has a full SIMD width
	nbrCountFF_.assign(paddedCount, 0);
	nbrCountFB_.assign(paddedCount, 0);
	nbrCountFR_.assign(paddedCount, 0);
	nbrChunkMaxCountFF_.assign(numChunks, 0);
	nbrChunkMaxCountFB_.assign(numChunks, 0);
	nbrChunkMaxCountFR_.assign(numChunks, 0);
	nbrChunkOffsetsFF_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFB_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFR_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrListFF_.clear();
	nbrListFB_.clear();
	nbrListFR_.clear();

	// rigid source particles use CSR because they are not SIMD-reordered
	// TODO: use chunked SELL too?
	nbrCountRF_.assign(N_R_, 0);
	nbrCountRB_.assign(N_R_, 0);
	nbrCountRR_.assign(N_R_, 0);
	nbrOffsetsRF_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrOffsetsRB_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrOffsetsRR_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrListRF_.clear();
	nbrListRB_.clear();
	nbrListRR_.clear();

	// reserve packed storage up front based on the expected neighbor count
	EnsurePackedCSRStorageCapacity(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
	EnsurePackedCSRStorageCapacity(nbrListFR_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
	EnsurePackedCSRStorageCapacity(nbrListRF_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListRB_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FB));
	EnsurePackedCSRStorageCapacity(nbrListRR_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF));
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::Finalize_Count()
{
	BuildChunkMaxCount(nbrCountFF_, nbrChunkMaxCountFF_);
	BuildChunkMaxCount(nbrCountFB_, nbrChunkMaxCountFB_);
	BuildChunkMaxCount(nbrCountFR_, nbrChunkMaxCountFR_);

	// chunk pitches are fixed after the count-query and shared by all lanes
	FinalizeChunkOffsets(nbrChunkMaxCountFF_, nbrChunkOffsetsFF_, nbrListFF_);
	FinalizeChunkOffsets(nbrChunkMaxCountFB_, nbrChunkOffsetsFB_, nbrListFB_);
	FinalizeChunkOffsets(nbrChunkMaxCountFR_, nbrChunkOffsetsFR_, nbrListFR_);
	FinalizeCSRNeighborStorage(nbrCountRF_, nbrOffsetsRF_, nbrListRF_);
	FinalizeCSRNeighborStorage(nbrCountRB_, nbrOffsetsRB_, nbrListRB_);
	FinalizeCSRNeighborStorage(nbrCountRR_, nbrOffsetsRR_, nbrListRR_);
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrCountFF_[i] = NarrowNeighborCount_Debug(
		count,
		"Chunked SELL FF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrCountFB_[i] = NarrowNeighborCount_Debug(
		count,
		"Chunked SELL FB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountFR(
	const uint32_t i, const uint32_t count)
{
	nbrCountFR_[i] = NarrowNeighborCount_Debug(
		count,
		"Chunked SELL FR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountRF(
	const uint32_t i, const uint32_t count)
{
	nbrCountRF_[i] = NarrowNeighborCount_Debug(
		count,
		"RF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountRB(
	const uint32_t i, const uint32_t count)
{
	nbrCountRB_[i] = NarrowNeighborCount_Debug(
		count,
		"RB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrCountRR(
	const uint32_t i, const uint32_t count)
{
	nbrCountRR_[i] = NarrowNeighborCount_Debug(
		count,
		"RR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFF_[chunk];
	if (count > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FF neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = count; k < pitch; k++)
	{
		nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)] = i;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFB_[chunk];
	if (count > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FB neighbor count exceeds chunk pitch.");
	}

	const uint32_t safeBoundaryIdx = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = count; k < pitch; k++)
	{
		nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)] = safeBoundaryIdx;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListFR(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFR_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FR neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFR_[chunk];
	if (count > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FR neighbor count exceeds chunk pitch.");
	}

	const uint32_t safeRigidIdx = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = count; k < pitch; k++)
	{
		nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)] = safeRigidIdx;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListRF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RF neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRF_.data() + nbrOffsetsRF_[i]);
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListRB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RB neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRB_.data() + nbrOffsetsRB_[i]);
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::SetNbrListRR(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRR_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RR neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRR_.data() + nbrOffsetsRR_[i]);
	}
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(count);

	// reconstruct fluid neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)];
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(count);

	// reconstruct boundary neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)];
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::FR(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFR_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFR_.local();
	out.resize(count);

	// reconstruct rigid neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)];
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::RF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRF_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRF_.data() + nbrOffsetsRF_[i], count };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::RB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRB_.data() + nbrOffsetsRB_[i], count };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU01::RR(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRR_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRR_.data() + nbrOffsetsRR_[i], count };
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountFR(
	const uint32_t i) const
{
	return nbrCountFR_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountRF(
	const uint32_t i) const
{
	return nbrCountRF_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountRB(
	const uint32_t i) const
{
	return nbrCountRB_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::CountRR(
	const uint32_t i) const
{
	return nbrCountRR_[i];
}

Mem::ParticleClassMemUsage
NeighborLists_ChunkedSELL_Explicit_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};
	Mem::MemUsage usageFR{};

	// FF
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMaxCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMaxCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListFB_));

	// FR
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMaxCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkOffsetsFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListFR_));

	// RF
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRF_));

	// RB
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRB_));

	// RR
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRR_));

	return { usageFF, usageFB, usageFR };
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU01::FFChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		SELLChunkFormat::Explicit32
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU01::FBChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		SELLChunkFormat::Explicit32
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU01::FRChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		nbrListFR_.empty() ? nullptr : nbrListFR_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFR_[chunk],
		SELLChunkFormat::Explicit32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU01::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		SELLChunkFormat::Explicit32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU01::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		SELLChunkFormat::Explicit32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU01::FRChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		nbrListFR_.empty() ? nullptr : nbrListFR_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFR_[chunk],
		SELLChunkFormat::Explicit32
	};
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::FinishFFChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t pitch = nbrChunkMaxCountFF_[chunk];
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFF_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FF neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = count; k < pitch; k++)
		{
			nbrListFF_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = i;
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < pitch; k++)
		{
			nbrListFF_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = 0;
		}
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::FinishFBChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t pitch = nbrChunkMaxCountFB_[chunk];
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const uint32_t safeBoundaryIdx = 0;

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFB_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FB neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = count; k < pitch; k++)
		{
			nbrListFB_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = safeBoundaryIdx;
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < pitch; k++)
		{
			nbrListFB_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = safeBoundaryIdx;
		}
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::FinishFRChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t pitch = nbrChunkMaxCountFR_[chunk];
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];
	const uint32_t safeRigidIdx = 0;

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFR_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FR neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = count; k < pitch; k++)
		{
			nbrListFR_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = safeRigidIdx;
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < pitch; k++)
		{
			nbrListFR_[offset + static_cast<uint64_t>(k) * chunkWidth + lane] = safeRigidIdx;
		}
	}
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU01::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

uint64_t NeighborLists_ChunkedSELL_Explicit_CPU01::ParticleStorageIdx(
	const uint32_t i,
	const uint32_t k,
	const std::vector<uint64_t>& nbrChunkOffsets) noexcept
{
	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);

	// storage is neighbor-major inside each chunk: base + k * SIMD-width + lane
	return nbrChunkOffsets[chunk]
		+ static_cast<uint64_t>(k) * chunkWidth
		+ static_cast<uint64_t>(lane);
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::BuildChunkMaxCount(
	const std::vector<SELLNeighborCount>& nbrCounts,
	std::vector<SELLNeighborCount>& nbrChunkMaxCount) const
{
	const uint32_t numChunks = ChunkNumTotal(N_F_);
	nbrChunkMaxCount.assign(numChunks, 0);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const uint32_t begin = ChunkBegin(chunk);
				const uint32_t end = ChunkEnd(chunk, N_F_);
				SELLNeighborCount maxCount = 0;

				// one pitch per chunk keeps all SIMD lanes aligned for traversal
				for (uint32_t i = begin; i < end; i++)
				{
					maxCount = std::max(maxCount, nbrCounts[i]);
				}

				nbrChunkMaxCount[chunk] = maxCount;
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_ChunkedSELL_Explicit_CPU01::FinalizeChunkOffsets(
	const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
	std::vector<uint64_t>& nbrChunkOffsets,
	std::vector<uint32_t>& nbrList) const
{
	const uint32_t numChunks = ChunkNumTotal(N_F_);
	nbrChunkOffsets.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);

	for (uint32_t chunk = 0; chunk < numChunks; chunk++)
	{
		const uint64_t chunkStorage =
			static_cast<uint64_t>(nbrChunkMaxCount[chunk]) * chunkWidth;

		// offsets are an exclusive prefix sum over chunk pitch * SIMD-width
		nbrChunkOffsets[static_cast<uint64_t>(chunk) + 1ull]
			= nbrChunkOffsets[chunk] + chunkStorage;
	}

	const uint64_t N_total = nbrChunkOffsets[numChunks];
	EnsurePackedCSRStorageCapacity(nbrList, N_total, extraAllocCapacityPct_);
	nbrList.resize(N_total);
}

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL neighbor list, adaptive compression ] ---------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
	const SimParams& params)
{
	(void)params;

	N_F_ = N_F;
	N_B_ = N_B;
	N_R_ = N_R;

	const uint32_t numChunks = ChunkNumTotal(N_F_);
	const uint32_t paddedCount = numChunks * chunkWidth;
	const uint32_t invalidRef = std::numeric_limits<uint32_t>::max();

	// count arrays are padded, tail chunk still has a full SIMD width
	nbrCountFF_.assign(paddedCount, 0);
	nbrCountFB_.assign(paddedCount, 0);
	nbrCountFR_.assign(paddedCount, 0);
	nbrChunkMaxCountFF_.assign(numChunks, 0);
	nbrChunkMaxCountFB_.assign(numChunks, 0);
	nbrChunkMaxCountFR_.assign(numChunks, 0);
	nbrChunkMinRefFF_.assign(numChunks, invalidRef);
	nbrChunkMinRefFB_.assign(numChunks, invalidRef);
	nbrChunkMinRefFR_.assign(numChunks, invalidRef);
	nbrChunkMaxRefFF_.assign(numChunks, 0);
	nbrChunkMaxRefFB_.assign(numChunks, 0);
	nbrChunkMaxRefFR_.assign(numChunks, 0);
	nbrChunkBaseIdxFF_.assign(numChunks, 0);
	nbrChunkBaseIdxFB_.assign(numChunks, 0);
	nbrChunkBaseIdxFR_.assign(numChunks, 0);
	nbrChunkFormatFF_.assign(numChunks, SELLChunkFormat::Abs32);
	nbrChunkFormatFB_.assign(numChunks, SELLChunkFormat::Abs32);
	nbrChunkFormatFR_.assign(numChunks, SELLChunkFormat::Abs32);
	nbrChunkOffsetsFF_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFB_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFR_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);

	nbrListDelta8FF_.clear();
	nbrListDelta8FB_.clear();
	nbrListDelta8FR_.clear();
	nbrListDelta16FF_.clear();
	nbrListDelta16FB_.clear();
	nbrListDelta16FR_.clear();
	nbrListAbs32FF_.clear();
	nbrListAbs32FB_.clear();
	nbrListAbs32FR_.clear();

	// rigid source particles use CSR because they are not SIMD-reordered
	// TODO: use chunked SELL too?
	nbrCountRF_.assign(N_R_, 0);
	nbrCountRB_.assign(N_R_, 0);
	nbrCountRR_.assign(N_R_, 0);
	nbrOffsetsRF_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrOffsetsRB_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrOffsetsRR_.assign(static_cast<uint64_t>(N_R_) + 1ull, 0);
	nbrListRF_.clear();
	nbrListRB_.clear();
	nbrListRR_.clear();
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::Finalize_Count()
{
	BuildChunkMaxCount(nbrCountFF_, nbrChunkMaxCountFF_);
	BuildChunkMaxCount(nbrCountFB_, nbrChunkMaxCountFB_);
	BuildChunkMaxCount(nbrCountFR_, nbrChunkMaxCountFR_);

	// chunk formats and pitches are fixed after the count-query
	FinalizeChunkStorage(
		nbrChunkMaxCountFF_, nbrChunkMinRefFF_, nbrChunkMaxRefFF_,
		nbrChunkBaseIdxFF_, nbrChunkFormatFF_, nbrChunkOffsetsFF_,
		nbrListDelta8FF_, nbrListDelta16FF_, nbrListAbs32FF_);
	FinalizeChunkStorage(
		nbrChunkMaxCountFB_, nbrChunkMinRefFB_, nbrChunkMaxRefFB_,
		nbrChunkBaseIdxFB_, nbrChunkFormatFB_, nbrChunkOffsetsFB_,
		nbrListDelta8FB_, nbrListDelta16FB_, nbrListAbs32FB_);
	FinalizeChunkStorage(
		nbrChunkMaxCountFR_, nbrChunkMinRefFR_, nbrChunkMaxRefFR_,
		nbrChunkBaseIdxFR_, nbrChunkFormatFR_, nbrChunkOffsetsFR_,
		nbrListDelta8FR_, nbrListDelta16FR_, nbrListAbs32FR_);
	FinalizeCSRNeighborStorage(nbrCountRF_, nbrOffsetsRF_, nbrListRF_);
	FinalizeCSRNeighborStorage(nbrCountRB_, nbrOffsetsRB_, nbrListRB_);
	FinalizeCSRNeighborStorage(nbrCountRR_, nbrOffsetsRR_, nbrListRR_);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrCountFF_[i] = NarrowNeighborCount_Debug(
		count,
		"Compressed chunked SELL FF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrCountFB_[i] = NarrowNeighborCount_Debug(
		count,
		"Compressed chunked SELL FB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountFR(
	const uint32_t i, const uint32_t count)
{
	nbrCountFR_[i] = NarrowNeighborCount_Debug(
		count,
		"Compressed chunked SELL FR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountRF(
	const uint32_t i, const uint32_t count)
{
	nbrCountRF_[i] = NarrowNeighborCount_Debug(
		count,
		"RF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountRB(
	const uint32_t i, const uint32_t count)
{
	nbrCountRB_[i] = NarrowNeighborCount_Debug(
		count,
		"RB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrCountRR(
	const uint32_t i, const uint32_t count)
{
	nbrCountRR_[i] = NarrowNeighborCount_Debug(
		count,
		"RR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	ChunkedSELLNeighborSpan_Write chunkFF = FFChunkWrite(chunk);
	if (count > chunkFF.pitch) [[unlikely]]
	{
		throw std::runtime_error("Compressed chunked SELL FF neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		chunkFF.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safeFluidIdx =
		(chunkFF.format == SELLChunkFormat::Delta8 ||
		 chunkFF.format == SELLChunkFormat::Delta16)
		? chunkFF.baseIdx : i;
	for (uint32_t k = count; k < chunkFF.pitch; k++)
	{
		chunkFF.StoreNeighbor(lane, k, safeFluidIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const uint32_t safeBoundaryIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFB = FBChunkWrite(chunk);
	if (count > chunkFB.pitch) [[unlikely]]
	{
		throw std::runtime_error("Compressed chunked SELL FB neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		chunkFB.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingBoundaryIdx =
		(chunkFB.format == SELLChunkFormat::Delta8 ||
		 chunkFB.format == SELLChunkFormat::Delta16)
		? chunkFB.baseIdx : safeBoundaryIdx;
	for (uint32_t k = count; k < chunkFB.pitch; k++)
	{
		chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListFR(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFR_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FR neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const uint32_t safeRigidIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFR = FRChunkWrite(chunk);
	if (count > chunkFR.pitch) [[unlikely]]
	{
		throw std::runtime_error("Compressed chunked SELL FR neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		chunkFR.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingRigidIdx =
		(chunkFR.format == SELLChunkFormat::Delta8 ||
		 chunkFR.format == SELLChunkFormat::Delta16)
		? chunkFR.baseIdx : safeRigidIdx;
	for (uint32_t k = count; k < chunkFR.pitch; k++)
	{
		chunkFR.StoreNeighbor(lane, k, safePaddingRigidIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListRF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RF neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRF_.data() + nbrOffsetsRF_[i]);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListRB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RB neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRB_.data() + nbrOffsetsRB_[i]);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetNbrListRR(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountRR_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RR neighbor storage.");
	}
	#endif

	if (count > 0)
	{
		std::copy_n(js, count, nbrListRR_.data() + nbrOffsetsRR_[i]);
	}
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(count);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFF = FFChunk(chunk);

	// reconstruct fluid neighbor list of one particle from compressed SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = chunkFF.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(count);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFB = FBChunk(chunk);

	// reconstruct boundary neighbor list of one particle from compressed SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = chunkFB.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FR(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFR_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFR_.local();
	out.resize(count);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFR = FRChunk(chunk);

	// reconstruct rigid neighbor list of one particle from compressed SELL storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = chunkFR.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::RF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRF_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRF_.data() + nbrOffsetsRF_[i], count };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::RB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRB_.data() + nbrOffsetsRB_[i], count };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU01::RR(
	const uint32_t i) const
{
	const uint32_t count = nbrCountRR_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRR_.data() + nbrOffsetsRR_[i], count };
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountFR(
	const uint32_t i) const
{
	return nbrCountFR_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountRF(
	const uint32_t i) const
{
	return nbrCountRF_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountRB(
	const uint32_t i) const
{
	return nbrCountRB_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::CountRR(
	const uint32_t i) const
{
	return nbrCountRR_[i];
}

Mem::ParticleClassMemUsage
NeighborLists_ChunkedSELL_AdaptCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};
	Mem::MemUsage usageFR{};

	// FF
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMaxCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMinRefFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMaxRefFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkFormatFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListDelta8FF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListDelta16FF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListAbs32FF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMaxCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMinRefFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMaxRefFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkFormatFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListDelta8FB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListDelta16FB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListAbs32FB_));

	// FR
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMaxCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMinRefFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMaxRefFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkFormatFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkOffsetsFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListDelta8FR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListDelta16FR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListAbs32FR_));

	// RF
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRF_));

	// RB
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRB_));

	// RR
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRR_));

	return { usageFF, usageFB, usageFR };
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FFChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFF(chunk);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FBChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFB(chunk);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FRChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFR(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFF(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFB(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FRChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFR(chunk);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FinishFFChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	ChunkedSELLNeighborSpan_Write chunkFF = FFChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFF_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FF neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safeFluidIdx =
			(chunkFF.format == SELLChunkFormat::Delta8 ||
			 chunkFF.format == SELLChunkFormat::Delta16)
			? chunkFF.baseIdx : i;
		for (uint32_t k = count; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safeFluidIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailFluidIdx =
		(chunkFF.format == SELLChunkFormat::Delta8 ||
		 chunkFF.format == SELLChunkFormat::Delta16)
		? chunkFF.baseIdx : 0;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safeTailFluidIdx);
		}
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FinishFBChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t safeBoundaryIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFB = FBChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFB_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FB neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingBoundaryIdx =
			(chunkFB.format == SELLChunkFormat::Delta8 ||
			 chunkFB.format == SELLChunkFormat::Delta16)
			? chunkFB.baseIdx : safeBoundaryIdx;
		for (uint32_t k = count; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailBoundaryIdx =
		(chunkFB.format == SELLChunkFormat::Delta8 ||
		 chunkFB.format == SELLChunkFormat::Delta16)
		? chunkFB.baseIdx : safeBoundaryIdx;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safeTailBoundaryIdx);
		}
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FinishFRChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t safeRigidIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFR = FRChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFR_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FR neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingRigidIdx =
			(chunkFR.format == SELLChunkFormat::Delta8 ||
			 chunkFR.format == SELLChunkFormat::Delta16)
			? chunkFR.baseIdx : safeRigidIdx;
		for (uint32_t k = count; k < chunkFR.pitch; k++)
		{
			chunkFR.StoreNeighbor(lane, k, safePaddingRigidIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailRigidIdx =
		(chunkFR.format == SELLChunkFormat::Delta8 ||
		 chunkFR.format == SELLChunkFormat::Delta16)
		? chunkFR.baseIdx : safeRigidIdx;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFR.pitch; k++)
		{
			chunkFR.StoreNeighbor(lane, k, safeTailRigidIdx);
		}
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetChunkNeighborIdxRangeFF(
	const uint32_t chunk, const uint32_t minRef, const uint32_t maxRef)
{
	nbrChunkMinRefFF_[chunk] = minRef;
	nbrChunkMaxRefFF_[chunk] = maxRef;
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetChunkNeighborIdxRangeFB(
	const uint32_t chunk, const uint32_t minRef, const uint32_t maxRef)
{
	nbrChunkMinRefFB_[chunk] = minRef;
	nbrChunkMaxRefFB_[chunk] = maxRef;
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::SetChunkNeighborIdxRangeFR(
	const uint32_t chunk, const uint32_t minRef, const uint32_t maxRef)
{
	nbrChunkMinRefFR_[chunk] = minRef;
	nbrChunkMaxRefFR_[chunk] = maxRef;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

SELLChunkFormat NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChooseChunkFormat(
	const uint32_t pitch, const uint32_t minRef, const uint32_t maxRef) noexcept
{
	if (pitch == 0u || minRef == std::numeric_limits<uint32_t>::max() || maxRef < minRef)
	{
		return SELLChunkFormat::Abs32;
	}

	const uint32_t span = maxRef - minRef;
	if (span <= 0xffu) { return SELLChunkFormat::Delta8; }
	if (span <= 0xffffu) { return SELLChunkFormat::Delta16; }
	return SELLChunkFormat::Abs32;
}

uint64_t NeighborLists_ChunkedSELL_AdaptCompress_CPU01::ChunkStorageSize(
	const uint32_t pitch) noexcept
{
	return static_cast<uint64_t>(pitch) * chunkWidth;
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeReadSpanFF(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFF_[chunk];

	return ChunkedSELLNeighborSpan_Read
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FF_.empty()
			? nbrListAbs32FF_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FF_.empty()
			? nbrListDelta16FF_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FF_.empty()
			? nbrListDelta8FF_.data() + offset : nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		nbrChunkBaseIdxFF_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		format
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeReadSpanFB(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELLNeighborSpan_Read
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FB_.empty()
			? nbrListAbs32FB_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FB_.empty()
			? nbrListDelta16FB_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FB_.empty()
			? nbrListDelta8FB_.data() + offset : nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		nbrChunkBaseIdxFB_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		format
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeReadSpanFR(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFR_[chunk];

	return ChunkedSELLNeighborSpan_Read
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FR_.empty()
			? nbrListAbs32FR_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FR_.empty()
			? nbrListDelta16FR_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FR_.empty()
			? nbrListDelta8FR_.data() + offset : nullptr,
		nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		nbrChunkBaseIdxFR_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFR_[chunk],
		format
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeWriteSpanFF(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFF_[chunk];

	return ChunkedSELLNeighborSpan_Write
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FF_.empty()
			? nbrListAbs32FF_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FF_.empty()
			? nbrListDelta16FF_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FF_.empty()
			? nbrListDelta8FF_.data() + offset : nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		nbrChunkBaseIdxFF_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		format
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeWriteSpanFB(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELLNeighborSpan_Write
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FB_.empty()
			? nbrListAbs32FB_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FB_.empty()
			? nbrListDelta16FB_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FB_.empty()
			? nbrListDelta8FB_.data() + offset : nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		nbrChunkBaseIdxFB_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		format
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU01::MakeWriteSpanFR(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFR_[chunk];

	return ChunkedSELLNeighborSpan_Write
	{
		format == SELLChunkFormat::Abs32 && !nbrListAbs32FR_.empty()
			? nbrListAbs32FR_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta16 && !nbrListDelta16FR_.empty()
			? nbrListDelta16FR_.data() + offset : nullptr,
		format == SELLChunkFormat::Delta8 && !nbrListDelta8FR_.empty()
			? nbrListDelta8FR_.data() + offset : nullptr,
		nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		nbrChunkBaseIdxFR_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFR_[chunk],
		format
	};
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::BuildChunkMaxCount(
	const std::vector<SELLNeighborCount>& nbrCounts,
	std::vector<SELLNeighborCount>& nbrChunkMaxCount) const
{
	const uint32_t numChunks = ChunkNumTotal(N_F_);
	nbrChunkMaxCount.assign(numChunks, 0);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numChunks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
			{
				const uint32_t begin = ChunkBegin(chunk);
				const uint32_t end = ChunkEnd(chunk, N_F_);
				SELLNeighborCount maxCount = 0;

				// one pitch per chunk keeps all SIMD lanes aligned for traversal
				for (uint32_t i = begin; i < end; i++)
				{
					maxCount = std::max(maxCount, nbrCounts[i]);
				}

				nbrChunkMaxCount[chunk] = maxCount;
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU01::FinalizeChunkStorage(
	const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
	const std::vector<uint32_t>& nbrChunkMinRef,
	const std::vector<uint32_t>& nbrChunkMaxRef,
	std::vector<uint32_t>& nbrChunkBaseIdx,
	std::vector<SELLChunkFormat>& nbrChunkFormat,
	std::vector<uint64_t>& nbrChunkOffsets,
	std::vector<uint8_t>& nbrListDelta8,
	std::vector<uint16_t>& nbrListDelta16,
	std::vector<uint32_t>& nbrListAbs32) const
{
	const uint32_t numChunks = ChunkNumTotal(N_F_);
	nbrChunkOffsets.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);

	uint64_t N_delta8 = 0;
	uint64_t N_delta16 = 0;
	uint64_t N_abs32 = 0;

	for (uint32_t chunk = 0; chunk < numChunks; chunk++)
	{
		const uint32_t pitch = nbrChunkMaxCount[chunk];
		const uint64_t chunkStorage = ChunkStorageSize(pitch);
		const SELLChunkFormat format = ChooseChunkFormat(
			pitch, nbrChunkMinRef[chunk], nbrChunkMaxRef[chunk]);

		nbrChunkFormat[chunk] = format;
		nbrChunkBaseIdx[chunk] = (format == SELLChunkFormat::Abs32)
			? 0u : nbrChunkMinRef[chunk];

		if (format == SELLChunkFormat::Delta8)
		{
			nbrChunkOffsets[chunk] = N_delta8;
			N_delta8 += chunkStorage;
		}
		else if (format == SELLChunkFormat::Delta16)
		{
			nbrChunkOffsets[chunk] = N_delta16;
			N_delta16 += chunkStorage;
		}
		else
		{
			nbrChunkOffsets[chunk] = N_abs32;
			N_abs32 += chunkStorage;
		}
	}

	EnsurePackedCSRStorageCapacity(nbrListDelta8, N_delta8, extraAllocCapacityPct_);
	EnsurePackedCSRStorageCapacity(nbrListDelta16, N_delta16, extraAllocCapacityPct_);
	EnsurePackedCSRStorageCapacity(nbrListAbs32, N_abs32, extraAllocCapacityPct_);
	nbrListDelta8.resize(N_delta8);
	nbrListDelta16.resize(N_delta16);
	nbrListAbs32.resize(N_abs32);
}
