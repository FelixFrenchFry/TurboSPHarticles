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

} // anonymous

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL8 neighbor list, explicit ] ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL8_Explicit_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	const uint32_t numChunks = ChunkNumTotal(N_F_);
	const uint32_t paddedCount = numChunks * chunkWidth;

	// count arrays are padded, tail chunk still has 8 lanes
	nbrCountFF_.assign(paddedCount, 0);
	nbrCountFB_.assign(paddedCount, 0);
	nbrChunkMaxCountFF_.assign(numChunks, 0);
	nbrChunkMaxCountFB_.assign(numChunks, 0);
	nbrChunkOffsetsFF_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFB_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrListFF_.clear();
	nbrListFB_.clear();

	// reserve packed chunk storage up front based on the expected neighbor count
	EnsurePackedCSRStorageCapacity(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::Finalize_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrChunkOffsetsFF_.assign(1, 0);
		nbrChunkOffsetsFB_.assign(1, 0);
		nbrListFF_.clear();
		nbrListFB_.clear();
		return;
	}

	BuildChunkMaxCount(nbrCountFF_, nbrChunkMaxCountFF_);
	BuildChunkMaxCount(nbrCountFB_, nbrChunkMaxCountFB_);

	// chunk pitches are fixed after the count-query and shared by all lanes
	FinalizeChunkOffsets(nbrChunkMaxCountFF_, nbrChunkOffsetsFF_, nbrListFF_);
	FinalizeChunkOffsets(nbrChunkMaxCountFB_, nbrChunkOffsetsFB_, nbrListFB_);
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrCountFF_[i] = count;
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrCountFB_[i] = count;
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL8 FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFF_[chunk];
	if (count > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL8 FF neighbor count exceeds chunk pitch.");
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

void NeighborLists_ChunkedSELL8_Explicit_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL8 FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFB_[chunk];
	if (count > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL8 FB neighbor count exceeds chunk pitch.");
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

NeighborSpan NeighborLists_ChunkedSELL8_Explicit_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(count);

	// reconstruct fluid neighbor list of one particle from SELL8 storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)];
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL8_Explicit_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(count);

	// reconstruct boundary neighbor list of one particle from SELL8 storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)];
	}

	return NeighborSpan{ out.data(), count };
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_ChunkedSELL8_Explicit_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

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

	return { usageFF, usageFB };
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_Explicit_CPU01::FFChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELL8NeighborSpan_Read
	{
		nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		SELL8ChunkFormat::Explicit32
	};
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_Explicit_CPU01::FBChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELL8NeighborSpan_Read
	{
		nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		SELL8ChunkFormat::Explicit32
	};
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_Explicit_CPU01::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// query-fill writes neighbors directly into base + k * 8 + lane
	return ChunkedSELL8NeighborSpan_Write
	{
		nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		SELL8ChunkFormat::Explicit32
	};
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_Explicit_CPU01::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// query-fill writes neighbors directly into base + k * 8 + lane
	return ChunkedSELL8NeighborSpan_Write
	{
		nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		nullptr,
		nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		0,
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		SELL8ChunkFormat::Explicit32
	};
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::FinishFFChunkWrite(
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
				"Count and fill query mismatch in direct chunked SELL8 FF neighbor storage.");
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

void NeighborLists_ChunkedSELL8_Explicit_CPU01::FinishFBChunkWrite(
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
				"Count and fill query mismatch in direct chunked SELL8 FB neighbor storage.");
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

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

uint64_t NeighborLists_ChunkedSELL8_Explicit_CPU01::ParticleStorageIdx(
	const uint32_t i,
	const uint32_t k,
	const std::vector<uint64_t>& nbrChunkOffsets) noexcept
{
	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);

	// storage is neighbor-major inside each chunk: base + k * 8 + lane
	return nbrChunkOffsets[chunk]
		+ static_cast<uint64_t>(k) * chunkWidth
		+ static_cast<uint64_t>(lane);
}

void NeighborLists_ChunkedSELL8_Explicit_CPU01::BuildChunkMaxCount(
	const std::vector<uint32_t>& nbrCounts,
	std::vector<uint32_t>& nbrChunkMaxCount) const
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
				uint32_t maxCount = 0;

				// one pitch per chunk keeps all 8 lanes aligned for SIMD traversal
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

void NeighborLists_ChunkedSELL8_Explicit_CPU01::FinalizeChunkOffsets(
	const std::vector<uint32_t>& nbrChunkMaxCount,
	std::vector<uint64_t>& nbrChunkOffsets,
	std::vector<uint32_t>& nbrList) const
{
	const uint32_t numChunks = ChunkNumTotal(N_F_);
	nbrChunkOffsets.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);

	for (uint32_t chunk = 0; chunk < numChunks; chunk++)
	{
		const uint64_t chunkStorage =
			static_cast<uint64_t>(nbrChunkMaxCount[chunk]) * chunkWidth;

		// offsets are an exclusive prefix sum over chunk pitch * 8
		nbrChunkOffsets[static_cast<uint64_t>(chunk) + 1ull]
			= nbrChunkOffsets[chunk] + chunkStorage;
	}

	const uint64_t N_total = nbrChunkOffsets[numChunks];
	EnsurePackedCSRStorageCapacity(nbrList, N_total, extraAllocCapacityPct_);
	nbrList.resize(N_total);
}

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL8 neighbor list, adaptive compression ] ---------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	(void)params;

	N_F_ = N_F;
	N_B_ = N_B;

	const uint32_t numChunks = ChunkNumTotal(N_F_);
	const uint32_t paddedCount = numChunks * chunkWidth;
	const uint32_t invalidRef = std::numeric_limits<uint32_t>::max();

	// count arrays are padded, tail chunk still has 8 lanes
	nbrCountFF_.assign(paddedCount, 0);
	nbrCountFB_.assign(paddedCount, 0);
	nbrChunkMaxCountFF_.assign(numChunks, 0);
	nbrChunkMaxCountFB_.assign(numChunks, 0);
	nbrChunkMinRefFF_.assign(numChunks, invalidRef);
	nbrChunkMinRefFB_.assign(numChunks, invalidRef);
	nbrChunkMaxRefFF_.assign(numChunks, 0);
	nbrChunkMaxRefFB_.assign(numChunks, 0);
	nbrChunkBaseIdxFF_.assign(numChunks, 0);
	nbrChunkBaseIdxFB_.assign(numChunks, 0);
	nbrChunkFormatFF_.assign(numChunks, SELL8ChunkFormat::Abs32);
	nbrChunkFormatFB_.assign(numChunks, SELL8ChunkFormat::Abs32);
	nbrChunkOffsetsFF_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);
	nbrChunkOffsetsFB_.assign(static_cast<uint64_t>(numChunks) + 1ull, 0);

	nbrListDelta8FF_.clear();
	nbrListDelta8FB_.clear();
	nbrListDelta16FF_.clear();
	nbrListDelta16FB_.clear();
	nbrListAbs32FF_.clear();
	nbrListAbs32FB_.clear();
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::Finalize_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrChunkOffsetsFF_.assign(1, 0);
		nbrChunkOffsetsFB_.assign(1, 0);
		nbrListDelta8FF_.clear();
		nbrListDelta8FB_.clear();
		nbrListDelta16FF_.clear();
		nbrListDelta16FB_.clear();
		nbrListAbs32FF_.clear();
		nbrListAbs32FB_.clear();
		return;
	}

	BuildChunkMaxCount(nbrCountFF_, nbrChunkMaxCountFF_);
	BuildChunkMaxCount(nbrCountFB_, nbrChunkMaxCountFB_);

	// chunk formats and pitches are fixed after the count-query
	FinalizeChunkStorage(
		nbrChunkMaxCountFF_, nbrChunkMinRefFF_, nbrChunkMaxRefFF_,
		nbrChunkBaseIdxFF_, nbrChunkFormatFF_, nbrChunkOffsetsFF_,
		nbrListDelta8FF_, nbrListDelta16FF_, nbrListAbs32FF_);
	FinalizeChunkStorage(
		nbrChunkMaxCountFB_, nbrChunkMinRefFB_, nbrChunkMaxRefFB_,
		nbrChunkBaseIdxFB_, nbrChunkFormatFB_, nbrChunkOffsetsFB_,
		nbrListDelta8FB_, nbrListDelta16FB_, nbrListAbs32FB_);
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrCountFF_[i] = count;
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrCountFB_[i] = count;
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL8 FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	ChunkedSELL8NeighborSpan_Write chunkFF = FFChunkWrite(chunk);
	if (count > chunkFF.pitch) [[unlikely]]
	{
		throw std::runtime_error("Compressed chunked SELL8 FF neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		chunkFF.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safeFluidIdx =
		(chunkFF.format == SELL8ChunkFormat::Delta8 ||
		 chunkFF.format == SELL8ChunkFormat::Delta16)
		? chunkFF.baseIdx : i;
	for (uint32_t k = count; k < chunkFF.pitch; k++)
	{
		chunkFF.StoreNeighbor(lane, k, safeFluidIdx);
	}
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL8 FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const uint32_t safeBoundaryIdx = 0;
	ChunkedSELL8NeighborSpan_Write chunkFB = FBChunkWrite(chunk);
	if (count > chunkFB.pitch) [[unlikely]]
	{
		throw std::runtime_error("Compressed chunked SELL8 FB neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < count; k++)
	{
		chunkFB.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingBoundaryIdx =
		(chunkFB.format == SELL8ChunkFormat::Delta8 ||
		 chunkFB.format == SELL8ChunkFormat::Delta16)
		? chunkFB.baseIdx : safeBoundaryIdx;
	for (uint32_t k = count; k < chunkFB.pitch; k++)
	{
		chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
	}
}

NeighborSpan NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(count);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELL8NeighborSpan_Read chunkFF = FFChunk(chunk);

	// reconstruct fluid neighbor list of one particle from compressed SELL8 storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = chunkFF.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(count);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELL8NeighborSpan_Read chunkFB = FBChunk(chunk);

	// reconstruct boundary neighbor list of one particle from compressed SELL8 storage
	for (uint32_t k = 0; k < count; k++)
	{
		out[k] = chunkFB.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), count };
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

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

	return { usageFF, usageFB };
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FFChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFF(chunk);
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FBChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFB(chunk);
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFF(chunk);
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFB(chunk);
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FinishFFChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	ChunkedSELL8NeighborSpan_Write chunkFF = FFChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFF_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL8 FF neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safeFluidIdx =
			(chunkFF.format == SELL8ChunkFormat::Delta8 ||
			 chunkFF.format == SELL8ChunkFormat::Delta16)
			? chunkFF.baseIdx : i;
		for (uint32_t k = count; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safeFluidIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailFluidIdx =
		(chunkFF.format == SELL8ChunkFormat::Delta8 ||
		 chunkFF.format == SELL8ChunkFormat::Delta16)
		? chunkFF.baseIdx : 0;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safeTailFluidIdx);
		}
	}
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FinishFBChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint32_t safeBoundaryIdx = 0;
	ChunkedSELL8NeighborSpan_Write chunkFB = FBChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t count = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFB_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL8 FB neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingBoundaryIdx =
			(chunkFB.format == SELL8ChunkFormat::Delta8 ||
			 chunkFB.format == SELL8ChunkFormat::Delta16)
			? chunkFB.baseIdx : safeBoundaryIdx;
		for (uint32_t k = count; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailBoundaryIdx =
		(chunkFB.format == SELL8ChunkFormat::Delta8 ||
		 chunkFB.format == SELL8ChunkFormat::Delta16)
		? chunkFB.baseIdx : safeBoundaryIdx;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safeTailBoundaryIdx);
		}
	}
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetChunkNeighborIdxRangeFF(
	const uint32_t chunk, const uint32_t minRef, const uint32_t maxRef)
{
	nbrChunkMinRefFF_[chunk] = minRef;
	nbrChunkMaxRefFF_[chunk] = maxRef;
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::SetChunkNeighborIdxRangeFB(
	const uint32_t chunk, const uint32_t minRef, const uint32_t maxRef)
{
	nbrChunkMinRefFB_[chunk] = minRef;
	nbrChunkMaxRefFB_[chunk] = maxRef;
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

SELL8ChunkFormat NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChooseChunkFormat(
	const uint32_t pitch, const uint32_t minRef, const uint32_t maxRef) noexcept
{
	if (pitch == 0u || minRef == std::numeric_limits<uint32_t>::max() || maxRef < minRef)
	{
		return SELL8ChunkFormat::Abs32;
	}

	const uint32_t span = maxRef - minRef;
	if (span <= 0xffu) { return SELL8ChunkFormat::Delta8; }
	if (span <= 0xffffu) { return SELL8ChunkFormat::Delta16; }
	return SELL8ChunkFormat::Abs32;
}

uint64_t NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::ChunkStorageSize(
	const uint32_t pitch) noexcept
{
	return static_cast<uint64_t>(pitch) * chunkWidth;
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::MakeReadSpanFF(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELL8ChunkFormat format = nbrChunkFormatFF_[chunk];

	return ChunkedSELL8NeighborSpan_Read
	{
		format == SELL8ChunkFormat::Abs32 && !nbrListAbs32FF_.empty()
			? nbrListAbs32FF_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta16 && !nbrListDelta16FF_.empty()
			? nbrListDelta16FF_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta8 && !nbrListDelta8FF_.empty()
			? nbrListDelta8FF_.data() + offset : nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		nbrChunkBaseIdxFF_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		format
	};
}

ChunkedSELL8NeighborSpan_Read NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::MakeReadSpanFB(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELL8ChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELL8NeighborSpan_Read
	{
		format == SELL8ChunkFormat::Abs32 && !nbrListAbs32FB_.empty()
			? nbrListAbs32FB_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta16 && !nbrListDelta16FB_.empty()
			? nbrListDelta16FB_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta8 && !nbrListDelta8FB_.empty()
			? nbrListDelta8FB_.data() + offset : nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		nbrChunkBaseIdxFB_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		format
	};
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::MakeWriteSpanFF(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELL8ChunkFormat format = nbrChunkFormatFF_[chunk];

	return ChunkedSELL8NeighborSpan_Write
	{
		format == SELL8ChunkFormat::Abs32 && !nbrListAbs32FF_.empty()
			? nbrListAbs32FF_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta16 && !nbrListDelta16FF_.empty()
			? nbrListDelta16FF_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta8 && !nbrListDelta8FF_.empty()
			? nbrListDelta8FF_.data() + offset : nullptr,
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		nbrChunkBaseIdxFF_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk],
		format
	};
}

ChunkedSELL8NeighborSpan_Write NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::MakeWriteSpanFB(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELL8ChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELL8NeighborSpan_Write
	{
		format == SELL8ChunkFormat::Abs32 && !nbrListAbs32FB_.empty()
			? nbrListAbs32FB_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta16 && !nbrListDelta16FB_.empty()
			? nbrListDelta16FB_.data() + offset : nullptr,
		format == SELL8ChunkFormat::Delta8 && !nbrListDelta8FB_.empty()
			? nbrListDelta8FB_.data() + offset : nullptr,
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		nbrChunkBaseIdxFB_[chunk],
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk],
		format
	};
}

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::BuildChunkMaxCount(
	const std::vector<uint32_t>& nbrCounts,
	std::vector<uint32_t>& nbrChunkMaxCount) const
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
				uint32_t maxCount = 0;

				// one pitch per chunk keeps all 8 lanes aligned for SIMD traversal
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

void NeighborLists_ChunkedSELL8_AdaptCompress_CPU01::FinalizeChunkStorage(
	const std::vector<uint32_t>& nbrChunkMaxCount,
	const std::vector<uint32_t>& nbrChunkMinRef,
	const std::vector<uint32_t>& nbrChunkMaxRef,
	std::vector<uint32_t>& nbrChunkBaseIdx,
	std::vector<SELL8ChunkFormat>& nbrChunkFormat,
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
		const SELL8ChunkFormat format = ChooseChunkFormat(
			pitch, nbrChunkMinRef[chunk], nbrChunkMaxRef[chunk]);

		nbrChunkFormat[chunk] = format;
		nbrChunkBaseIdx[chunk] = (format == SELL8ChunkFormat::Abs32)
			? 0u : nbrChunkMinRef[chunk];

		if (format == SELL8ChunkFormat::Delta8)
		{
			nbrChunkOffsets[chunk] = N_delta8;
			N_delta8 += chunkStorage;
		}
		else if (format == SELL8ChunkFormat::Delta16)
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
