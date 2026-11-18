// neighbor_lists.cpp

// external
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>

// internal
#include "neighbor_lists.h"



using namespace Sim;

namespace
{
	uint64_t ExpectedPackedCSRCapacity(
		const uint32_t N_F,
		const uint32_t N_expectedAvgNbrsHardMax)
	{
		return static_cast<uint64_t>(N_F) *
			static_cast<uint64_t>(N_expectedAvgNbrsHardMax);
	}

	SELLNeighborCount NarrowNeighborCount_Debug(
		const uint32_t N_nbrs,
		const char* overflowMsg)
	{
		#if defined(_DEBUG)
		if (N_nbrs > std::numeric_limits<SELLNeighborCount>::max()) [[unlikely]]
		{
			throw std::runtime_error(overflowMsg);
		}
		#else
		(void)overflowMsg;
		#endif

		return static_cast<SELLNeighborCount>(N_nbrs);
	}

	SELLNeighborCount CommitChunkNeighborCounts(
		const uint32_t begin,
		const uint32_t* counts,
		std::vector<SELLNeighborCount>& nbrCounts,
		const char* overflowMsg)
	{
		SELLNeighborCount N_maxNbrs = 0;

		// query scratch includes zeroed inactive and tail lanes
		for (uint32_t lane = 0; lane != CPU::N_SIMD_WIDTH; lane++)
		{
			const SELLNeighborCount N_nbrs = NarrowNeighborCount_Debug(
				counts[lane], overflowMsg);
			nbrCounts[begin + lane] = N_nbrs;
			N_maxNbrs = std::max(N_maxNbrs, N_nbrs);
		}

		return N_maxNbrs;
	}

	void ClearFluidChunkCounts_Parallel(
		std::vector<SELLNeighborCount>& nbrCountFF,
		std::vector<SELLNeighborCount>& nbrCountFB,
		std::vector<SELLNeighborCount>& nbrCountFR,
		std::vector<SELLNeighborCount>& nbrChunkMaxCountFF,
		std::vector<SELLNeighborCount>& nbrChunkMaxCountFB,
		std::vector<SELLNeighborCount>& nbrChunkMaxCountFR)
	{
		const uint32_t N_chunks = static_cast<uint32_t>(nbrChunkMaxCountFF.size());
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
				{
					const uint32_t begin = chunk * CPU::N_SIMD_WIDTH;
					for (uint32_t lane = 0; lane != CPU::N_SIMD_WIDTH; lane++)
					{
						nbrCountFF[begin + lane] = 0;
						nbrCountFB[begin + lane] = 0;
						nbrCountFR[begin + lane] = 0;
					}
					nbrChunkMaxCountFF[chunk] = 0;
					nbrChunkMaxCountFB[chunk] = 0;
					nbrChunkMaxCountFR[chunk] = 0;
				}
			},
			tbb::auto_partitioner{}
		);
	}

	void ClearFluidChunkFormats_Parallel(
		std::vector<SELLChunkFormat>& nbrChunkFormatFF,
		std::vector<SELLChunkFormat>& nbrChunkFormatFB,
		std::vector<SELLChunkFormat>& nbrChunkFormatFR,
		std::vector<uint32_t>& nbrChunkBaseIdxFF,
		std::vector<uint32_t>& nbrChunkBaseIdxFB,
		std::vector<uint32_t>& nbrChunkBaseIdxFR,
		std::vector<uint16_t>& nbrChunkDictSizeFF)
	{
		const uint32_t N_chunks = static_cast<uint32_t>(nbrChunkFormatFF.size());
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
				{
					nbrChunkFormatFF[chunk] = SELLChunkFormat::Abs32;
					nbrChunkFormatFB[chunk] = SELLChunkFormat::Abs32;
					nbrChunkFormatFR[chunk] = SELLChunkFormat::Abs32;
					nbrChunkBaseIdxFF[chunk] = 0;
					nbrChunkBaseIdxFB[chunk] = 0;
					nbrChunkBaseIdxFR[chunk] = 0;
					nbrChunkDictSizeFF[chunk] = 0;
				}
			},
			tbb::auto_partitioner{}
		);
	}

	template<typename ValueAtIdx>
	uint64_t BuildExclusivePrefixSum_Parallel(
		const uint32_t N_values,
		const ValueAtIdx& valueAtIdx,
		std::vector<uint64_t>& offsets)
	{
		offsets.resize(static_cast<uint64_t>(N_values) + 1ull);
		offsets[0] = 0;
		if (N_values == 0u) { return 0; }

		return tbb::parallel_scan(
			tbb::blocked_range<uint32_t>(0, N_values),
			uint64_t{ 0 },
			[&](const tbb::blocked_range<uint32_t>& r,
				const uint64_t prefix,
				const bool isFinalScan)
			{
				uint64_t sum = prefix;
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					sum += valueAtIdx(i);
					if (isFinalScan)
					{
						offsets[static_cast<uint64_t>(i) + 1ull] = sum;
					}
				}
				return sum;
			},
			[](const uint64_t lhs, const uint64_t rhs)
			{
				return lhs + rhs;
			},
			tbb::auto_partitioner{}
		);
	}

	struct ChunkStorageTotals
	{
		uint64_t N_entries16 = 0;
		uint64_t N_entries32 = 0;
	};

	void FinalizeCSRNeighborStorage(
		const std::vector<SELLNeighborCount>& nbrCount,
		std::vector<uint64_t>& nbrOffsets,
		std::vector<uint32_t>& nbrList,
		const float extraAllocCapacityPct)
	{
		const uint64_t N_total = BuildExclusivePrefixSum_Parallel(
			static_cast<uint32_t>(nbrCount.size()),
			[&](const uint32_t i)
			{
				return static_cast<uint64_t>(nbrCount[i]);
			},
			nbrOffsets);
		Base::ResizeScratchVector(nbrList, N_total, extraAllocCapacityPct);
	}

} // namespace anonymous

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL neighbor list, explicit ] ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL_Explicit_CPU::Resize(
	const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
	const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;
	N_R_ = N_R;

	const uint32_t N_chunks = ChunkNumTotal(N_F_);
	const uint32_t N_padded = Sim::CheckedPaddedLocalCount(
		N_F_, chunkWidth, "Explicit SELL particle count");

	// count arrays are padded, tail chunk still has a full SIMD width
	Base::ResizeScratchVector(nbrCountFF_, N_padded);
	Base::ResizeScratchVector(nbrCountFB_, N_padded);
	Base::ResizeScratchVector(nbrCountFR_, N_padded);
	Base::ResizeScratchVector(nbrChunkMaxCountFF_, N_chunks);
	Base::ResizeScratchVector(nbrChunkMaxCountFB_, N_chunks);
	Base::ResizeScratchVector(nbrChunkMaxCountFR_, N_chunks);
	nbrChunkOffsetsFF_.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkOffsetsFB_.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkOffsetsFR_.resize(static_cast<uint64_t>(N_chunks) + 1ull);

	// rigid source particles use CSR because they are not SIMD-reordered
	// TODO: use chunked SELL too?
	// (the query writes every rigid count and the prefix scan every offset)
	Base::ResizeScratchVector(nbrCountRF_, N_R_);
	Base::ResizeScratchVector(nbrCountRB_, N_R_);
	Base::ResizeScratchVector(nbrCountRR_, N_R_);
	Base::ResizeScratchVector(nbrOffsetsRF_, static_cast<uint64_t>(N_R_) + 1ull);
	Base::ResizeScratchVector(nbrOffsetsRB_, static_cast<uint64_t>(N_R_) + 1ull);
	Base::ResizeScratchVector(nbrOffsetsRR_, static_cast<uint64_t>(N_R_) + 1ull);

	// reserve packed storage up front based on the expected neighbor count
	Base::ReserveScratchVector(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF), 0.0f);
	Base::ReserveScratchVector(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB), 0.0f);
	Base::ReserveScratchVector(nbrListFR_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB), 0.0f);
	Base::ReserveScratchVector(nbrListRF_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF), 0.0f);
	Base::ReserveScratchVector(nbrListRB_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FB), 0.0f);
	Base::ReserveScratchVector(nbrListRR_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF), 0.0f);
}

void NeighborLists_ChunkedSELL_Explicit_CPU::Finalize_Count()
{
	// chunk pitches are fixed after the count-query and shared by all lanes
	FinalizeChunkOffsets(nbrChunkMaxCountFF_, nbrChunkOffsetsFF_, nbrListFF_);
	FinalizeChunkOffsets(nbrChunkMaxCountFB_, nbrChunkOffsetsFB_, nbrListFB_);
	FinalizeChunkOffsets(nbrChunkMaxCountFR_, nbrChunkOffsetsFR_, nbrListFR_);
	FinalizeCSRNeighborStorage(nbrCountRF_, nbrOffsetsRF_, nbrListRF_, extraAllocCapacityPct_);
	FinalizeCSRNeighborStorage(nbrCountRB_, nbrOffsetsRB_, nbrListRB_, extraAllocCapacityPct_);
	FinalizeCSRNeighborStorage(nbrCountRR_, nbrOffsetsRR_, nbrListRR_, extraAllocCapacityPct_);
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountFF(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFF_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Chunked SELL FF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountFB(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFB_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Chunked SELL FB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountFR(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFR_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Chunked SELL FR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountRF(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRF_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountRB(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRB_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrCountRR(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRR_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::PrepareFluidCountWrite(
	const bool allChunksWillBeWritten)
{
	if (allChunksWillBeWritten) { return; }

	ClearFluidChunkCounts_Parallel(
		nbrCountFF_, nbrCountFB_, nbrCountFR_,
		nbrChunkMaxCountFF_, nbrChunkMaxCountFB_, nbrChunkMaxCountFR_);
}

void NeighborLists_ChunkedSELL_Explicit_CPU::CommitFluidChunkCounts(
	const uint32_t chunk,
	const ChunkedSELLNeighborCountResult& countsFF,
	const ChunkedSELLNeighborCountResult& countsFB,
	const ChunkedSELLNeighborCountResult& countsFR)
{
	const uint32_t begin = ChunkBegin(chunk);
	nbrChunkMaxCountFF_[chunk] = CommitChunkNeighborCounts(
		begin, countsFF.counts, nbrCountFF_,
		"Chunked SELL FF neighbor count exceeds uint16 count storage.");
	nbrChunkMaxCountFB_[chunk] = CommitChunkNeighborCounts(
		begin, countsFB.counts, nbrCountFB_,
		"Chunked SELL FB neighbor count exceeds uint16 count storage.");
	nbrChunkMaxCountFR_[chunk] = CommitChunkNeighborCounts(
		begin, countsFR.counts, nbrCountFR_,
		"Chunked SELL FR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFF_[chunk];
	if (N_nbrs > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FF neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = N_nbrs; k < pitch; k++)
	{
		nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)] = i;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFB_[chunk];
	if (N_nbrs > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FB neighbor count exceeds chunk pitch.");
	}

	const uint32_t safeBoundaryIdx = 0;
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = N_nbrs; k < pitch; k++)
	{
		nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)] = safeBoundaryIdx;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListFR(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFR_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit chunked SELL FR neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t pitch = nbrChunkMaxCountFR_[chunk];
	if (N_nbrs > pitch) [[unlikely]]
	{
		throw std::runtime_error("Chunked SELL FR neighbor count exceeds chunk pitch.");
	}

	const uint32_t safeRigidIdx = 0;
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)] = js[k];
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	for (uint32_t k = N_nbrs; k < pitch; k++)
	{
		nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)] = safeRigidIdx;
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListRF(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRF_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RF neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRF_.data() + nbrOffsetsRF_[i]);
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListRB(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRB_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RB neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRB_.data() + nbrOffsetsRB_[i]);
	}
}

void NeighborLists_ChunkedSELL_Explicit_CPU::SetNbrListRR(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRR_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RR neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRR_.data() + nbrOffsetsRR_[i]);
	}
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::FF(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFF_[i];
	if (N_nbrs == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(N_nbrs, boost::container::default_init);

	// reconstruct fluid neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = nbrListFF_[ParticleStorageIdx(i, k, nbrChunkOffsetsFF_)];
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::FB(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFB_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(N_nbrs, boost::container::default_init);

	// reconstruct boundary neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = nbrListFB_[ParticleStorageIdx(i, k, nbrChunkOffsetsFB_)];
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::FR(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFR_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFR_.local();
	out.resize(N_nbrs, boost::container::default_init);

	// reconstruct rigid neighbor list of one particle from SELL storage
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = nbrListFR_[ParticleStorageIdx(i, k, nbrChunkOffsetsFR_)];
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::RF(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRF_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRF_.data() + nbrOffsetsRF_[i], N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::RB(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRB_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRB_.data() + nbrOffsetsRB_[i], N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_Explicit_CPU::RR(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRR_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRR_.data() + nbrOffsetsRR_[i], N_nbrs };
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountFR(
	const uint32_t i) const
{
	return nbrCountFR_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountRF(
	const uint32_t i) const
{
	return nbrCountRF_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountRB(
	const uint32_t i) const
{
	return nbrCountRB_[i];
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::CountRR(
	const uint32_t i) const
{
	return nbrCountRR_[i];
}

Mem::ParticleClassMemUsage
NeighborLists_ChunkedSELL_Explicit_CPU::GetMemUsage() const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};
	Mem::MemUsage usageFR{};

	// FF neighbor list memory usage
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMaxCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	// FB neighbor list memory usage
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMaxCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListFB_));

	// FR neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMaxCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkOffsetsFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListFR_));

	// RF neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRF_));

	// RB neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRB_));

	// RR neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRR_));

	return { usageFF, usageFB, usageFR };
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU::FFChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		.counts = nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFF_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU::FBChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		.counts = nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFB_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_Explicit_CPU::FRChunk(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];

	// chunk span to expose the layout used directly by SIMD kernels
	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = nbrListFR_.empty() ? nullptr : nbrListFR_.data() + offset,
		.counts = nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFR_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = nbrListFF_.empty() ? nullptr : nbrListFF_.data() + offset,
		.counts = nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFF_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = nbrListFB_.empty() ? nullptr : nbrListFB_.data() + offset,
		.counts = nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFB_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_Explicit_CPU::FRChunkWrite(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];

	// query-fill writes neighbors directly into base + k * SIMD-width + lane
	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = nbrListFR_.empty() ? nullptr : nbrListFR_.data() + offset,
		.counts = nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFR_[chunk],
		.format = SELLChunkFormat::Abs32
	};
}

void NeighborLists_ChunkedSELL_Explicit_CPU::FinishFFChunkWrite(
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
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFF_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FF neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = N_written; k < pitch; k++)
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

void NeighborLists_ChunkedSELL_Explicit_CPU::FinishFBChunkWrite(
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
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFB_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FB neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = N_written; k < pitch; k++)
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

void NeighborLists_ChunkedSELL_Explicit_CPU::FinishFRChunkWrite(
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
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFR_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct chunked SELL FR neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		for (uint32_t k = N_written; k < pitch; k++)
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

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL_Explicit_CPU::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

uint64_t NeighborLists_ChunkedSELL_Explicit_CPU::ParticleStorageIdx(
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

void NeighborLists_ChunkedSELL_Explicit_CPU::FinalizeChunkOffsets(
	const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
	std::vector<uint64_t>& nbrChunkOffsets,
	std::vector<uint32_t>& nbrList) const
{
	const uint32_t N_chunks = ChunkNumTotal(N_F_);
	const uint64_t N_total = BuildExclusivePrefixSum_Parallel(
		N_chunks,
		[&](const uint32_t chunk)
		{
			// offsets are an exclusive prefix sum over chunk pitch * SIMD-width
			return static_cast<uint64_t>(nbrChunkMaxCount[chunk]) * chunkWidth;
		},
		nbrChunkOffsets);
	Base::ResizeScratchVector(nbrList, N_total, extraAllocCapacityPct_);
}

// -------------------------------------------------------------------------------------------------
// --- [ chunked SELL neighbor list, adaptive compression ] ---------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::Resize(
	const uint32_t N_F, const uint32_t N_B, const uint32_t N_R,
	const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;
	N_R_ = N_R;

	const uint32_t N_chunks = ChunkNumTotal(N_F_);
	const uint32_t N_padded = Sim::CheckedPaddedLocalCount(
		N_F_, chunkWidth, "Compressed SELL particle count");

	// count arrays are padded, tail chunk still has a full SIMD width
	Base::ResizeScratchVector(nbrCountFF_, N_padded);
	Base::ResizeScratchVector(nbrCountFB_, N_padded);
	Base::ResizeScratchVector(nbrCountFR_, N_padded);
	Base::ResizeScratchVector(nbrChunkMaxCountFF_, N_chunks);
	Base::ResizeScratchVector(nbrChunkMaxCountFB_, N_chunks);
	Base::ResizeScratchVector(nbrChunkMaxCountFR_, N_chunks);
	nbrChunkBaseIdxFF_.resize(N_chunks);
	nbrChunkBaseIdxFB_.resize(N_chunks);
	nbrChunkBaseIdxFR_.resize(N_chunks);
	nbrChunkFormatFF_.resize(N_chunks);
	nbrChunkFormatFB_.resize(N_chunks);
	nbrChunkFormatFR_.resize(N_chunks);
	nbrChunkOffsetsFF_.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkOffsetsFB_.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkOffsetsFR_.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkDictSizeFF_.resize(N_chunks);
	nbrChunkDictFormatFF_.resize(N_chunks);
	nbrChunkDictLowFF_.resize(N_chunks);
	nbrChunkBaseIdx2FF_.resize(N_chunks);
	nbrChunkDictOffsetsFF_.resize(static_cast<uint64_t>(N_chunks) + 1ull);

	// rigid source particles use CSR because they are not SIMD-reordered
	// TODO: use chunked SELL too?
	// (the query writes every rigid count and the prefix scan every offset)
	Base::ResizeScratchVector(nbrCountRF_, N_R_);
	Base::ResizeScratchVector(nbrCountRB_, N_R_);
	Base::ResizeScratchVector(nbrCountRR_, N_R_);
	Base::ResizeScratchVector(nbrOffsetsRF_, static_cast<uint64_t>(N_R_) + 1ull);
	Base::ResizeScratchVector(nbrOffsetsRB_, static_cast<uint64_t>(N_R_) + 1ull);
	Base::ResizeScratchVector(nbrOffsetsRR_, static_cast<uint64_t>(N_R_) + 1ull);

	// reserve packed rigid storage up front based on the expected neighbor count
	// (the compressed fluid payloads are left to the growth headroom in Finalize_Count)
	Base::ReserveScratchVector(nbrListRF_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF), 0.0f);
	Base::ReserveScratchVector(nbrListRB_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FB), 0.0f);
	Base::ReserveScratchVector(nbrListRR_,
		ExpectedPackedCSRCapacity(N_R_, params.expctdAvgNbrCountHardMax_FF), 0.0f);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::Finalize_Count()
{
	// chunk formats and pitches are fixed by the count-query
	FinalizeChunkStorage(
		nbrChunkMaxCountFF_, nbrChunkFormatFF_, &nbrChunkDictSizeFF_, nbrChunkOffsetsFF_,
		nbrList16FF_, nbrList32FF_);

	// dictionary offsets are an exclusive prefix sum over the dictionary byte sizes (0 for normal chunks)
	const uint64_t N_dictBytesFF = BuildExclusivePrefixSum_Parallel(
		ChunkNumTotal(N_F_),
		[&](const uint32_t chunk)
		{
			return static_cast<uint64_t>(
				SELLDictBytes(nbrChunkDictFormatFF_[chunk], nbrChunkDictSizeFF_[chunk]));
		},
		nbrChunkDictOffsetsFF_);
	Base::ResizeScratchVector(nbrChunkDictFF_, N_dictBytesFF, extraAllocCapacityPct_);

	FinalizeChunkStorage(
		nbrChunkMaxCountFB_, nbrChunkFormatFB_, nullptr, nbrChunkOffsetsFB_,
		nbrList16FB_, nbrList32FB_);
	FinalizeChunkStorage(
		nbrChunkMaxCountFR_, nbrChunkFormatFR_, nullptr, nbrChunkOffsetsFR_,
		nbrList16FR_, nbrList32FR_);
	FinalizeCSRNeighborStorage(nbrCountRF_, nbrOffsetsRF_, nbrListRF_, extraAllocCapacityPct_);
	FinalizeCSRNeighborStorage(nbrCountRB_, nbrOffsetsRB_, nbrListRB_, extraAllocCapacityPct_);
	FinalizeCSRNeighborStorage(nbrCountRR_, nbrOffsetsRR_, nbrListRR_, extraAllocCapacityPct_);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountFF(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFF_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Compressed chunked SELL FF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountFB(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFB_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Compressed chunked SELL FB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountFR(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountFR_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"Compressed chunked SELL FR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountRF(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRF_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RF neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountRB(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRB_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RB neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrCountRR(
	const uint32_t i, const uint32_t N_nbrs)
{
	nbrCountRR_[i] = NarrowNeighborCount_Debug(
		N_nbrs,
		"RR neighbor count exceeds uint16 count storage.");
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::PrepareFluidCountWrite(
	const bool allChunksWillBeWritten)
{
	if (allChunksWillBeWritten) { return; }
	ClearFluidChunkCounts_Parallel(
		nbrCountFF_, nbrCountFB_, nbrCountFR_,
		nbrChunkMaxCountFF_, nbrChunkMaxCountFB_, nbrChunkMaxCountFR_);
	// the count-query only writes the formats of the chunks it visits
	ClearFluidChunkFormats_Parallel(
		nbrChunkFormatFF_, nbrChunkFormatFB_, nbrChunkFormatFR_,
		nbrChunkBaseIdxFF_, nbrChunkBaseIdxFB_, nbrChunkBaseIdxFR_,
		nbrChunkDictSizeFF_);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::CommitFluidChunkCounts(
	const uint32_t chunk,
	const ChunkedSELLNeighborCountResult& countsFF,
	const ChunkedSELLNeighborCountResult& countsFB,
	const ChunkedSELLNeighborCountResult& countsFR)
{
	const uint32_t begin = ChunkBegin(chunk);
	nbrChunkMaxCountFF_[chunk] = CommitChunkNeighborCounts(
		begin, countsFF.counts, nbrCountFF_,
		"Compressed chunked SELL FF neighbor count exceeds uint16 count storage.");
	nbrChunkMaxCountFB_[chunk] = CommitChunkNeighborCounts(
		begin, countsFB.counts, nbrCountFB_,
		"Compressed chunked SELL FB neighbor count exceeds uint16 count storage.");
	nbrChunkMaxCountFR_[chunk] = CommitChunkNeighborCounts(
		begin, countsFR.counts, nbrCountFR_,
		"Compressed chunked SELL FR neighbor count exceeds uint16 count storage.");

	// the chunk pitch and neighbor index range are complete here
	// (FF chunks with few distinct neighbors store them once in a dictionary)
	const bool isDictFF = countsFF.N_distinct > 0u && countsFF.N_distinct <= N_SELL_DICT_SIZE_MAX;
	uint32_t N_dictLowFF = 0;
	uint32_t baseIdx2FF = 0;
	const bool isDictDelta16x2FF = isDictFF && SplitDictDelta16x2(countsFF, N_dictLowFF, baseIdx2FF);
	const SELLChunkFormat formatFF = ChooseChunkFormat(
		nbrChunkMaxCountFF_[chunk], countsFF.minRef, countsFF.maxRef);
	const SELLChunkFormat formatFB = ChooseChunkFormat(
		nbrChunkMaxCountFB_[chunk], countsFB.minRef, countsFB.maxRef);
	const SELLChunkFormat formatFR = ChooseChunkFormat(
		nbrChunkMaxCountFR_[chunk], countsFR.minRef, countsFR.maxRef);
	nbrChunkFormatFF_[chunk] = formatFF;
	nbrChunkFormatFB_[chunk] = formatFB;
	nbrChunkFormatFR_[chunk] = formatFR;
	nbrChunkBaseIdxFF_[chunk] = (formatFF == SELLChunkFormat::Abs32 && !isDictDelta16x2FF) ? 0u : countsFF.minRef;
	nbrChunkBaseIdxFB_[chunk] = (formatFB == SELLChunkFormat::Abs32) ? 0u : countsFB.minRef;
	nbrChunkBaseIdxFR_[chunk] = (formatFR == SELLChunkFormat::Abs32) ? 0u : countsFR.minRef;
	nbrChunkDictSizeFF_[chunk] = isDictFF ? static_cast<uint16_t>(countsFF.N_distinct) : 0u;
	nbrChunkDictFormatFF_[chunk] = isDictDelta16x2FF
		? SELLDictChunkFormat::DictDelta16x2
		: SELLDictChunkFormat::DictAbs32;
	nbrChunkDictLowFF_[chunk] = static_cast<uint16_t>(N_dictLowFF);
	nbrChunkBaseIdx2FF_[chunk] = baseIdx2FF;
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFF_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FF neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	ChunkedSELLNeighborSpan_Write chunkFF = FFChunkWrite(chunk);
	if (chunkFF.dict) [[unlikely]]
	{
		throw std::runtime_error(
			"Compressed chunked SELL FF per-particle fill does not support dict chunks.");
	}
	if (N_nbrs > chunkFF.pitch) [[unlikely]]
	{
		throw std::runtime_error(
			"Compressed chunked SELL FF neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		chunkFF.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingFluidIdx = (chunkFF.format == SELLChunkFormat::Delta16)
		? chunkFF.baseIdx
		: i;
	for (uint32_t k = N_nbrs; k < chunkFF.pitch; k++)
	{
		chunkFF.StoreNeighbor(lane, k, safePaddingFluidIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFB_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FB neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const uint32_t safeBoundaryIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFB = FBChunkWrite(chunk);
	if (N_nbrs > chunkFB.pitch) [[unlikely]]
	{
		throw std::runtime_error(
			"Compressed chunked SELL FB neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		chunkFB.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingBoundaryIdx = (chunkFB.format == SELLChunkFormat::Delta16)
		? chunkFB.baseIdx
		: safeBoundaryIdx;
	for (uint32_t k = N_nbrs; k < chunkFB.pitch; k++)
	{
		chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListFR(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountFR_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in compressed chunked SELL FR neighbor storage.");
	}
	#endif

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const uint32_t safeRigidIdx = 0;
	ChunkedSELLNeighborSpan_Write chunkFR = FRChunkWrite(chunk);
	if (N_nbrs > chunkFR.pitch) [[unlikely]]
	{
		throw std::runtime_error(
			"Compressed chunked SELL FR neighbor count exceeds chunk pitch.");
	}

	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		chunkFR.StoreNeighbor(lane, k, js[k]);
	}

	// padding entries are valid indices so SIMD gathers never touch invalid memory
	const uint32_t safePaddingRigidIdx = (chunkFR.format == SELLChunkFormat::Delta16)
		? chunkFR.baseIdx
		: safeRigidIdx;
	for (uint32_t k = N_nbrs; k < chunkFR.pitch; k++)
	{
		chunkFR.StoreNeighbor(lane, k, safePaddingRigidIdx);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListRF(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRF_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RF neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRF_.data() + nbrOffsetsRF_[i]);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListRB(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRB_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RB neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRB_.data() + nbrOffsetsRB_[i]);
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::SetNbrListRR(
	const uint32_t i, const uint32_t* js, const uint32_t N_nbrs)
{
	#if defined(_DEBUG)
	if (nbrCountRR_[i] != N_nbrs) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in RR neighbor storage.");
	}
	#endif

	if (N_nbrs > 0)
	{
		std::copy_n(js, N_nbrs, nbrListRR_.data() + nbrOffsetsRR_[i]);
	}
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::FF(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFF_[i];
	if (N_nbrs == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFF_.local();
	out.resize(N_nbrs, boost::container::default_init);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFF = FFChunk(chunk);

	// reconstruct fluid neighbor list of one particle from compressed SELL storage
	// (dict chunks list the dict entries whose lane mask has the particle's lane, in dictionary order)
	if (chunkFF.dict)
	{
		uint32_t k = 0;
		for (uint32_t dictIdx = 0; dictIdx < chunkFF.N_dict && k < N_nbrs; dictIdx++)
		{
			if ((chunkFF.ReadDictLaneMask(dictIdx) >> lane) & 1u)
			{
				out[k++] = chunkFF.ReadDictNeighborIdx(dictIdx);
			}
		}
		assert(k == N_nbrs);
		return NeighborSpan{ out.data(), N_nbrs };
	}

	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = chunkFF.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::FB(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFB_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFB_.local();
	out.resize(N_nbrs, boost::container::default_init);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFB = FBChunk(chunk);

	// reconstruct boundary neighbor list of one particle from compressed SELL storage
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = chunkFB.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::FR(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountFR_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	auto& out = nbrScratchFR_.local();
	out.resize(N_nbrs, boost::container::default_init);

	const uint32_t chunk = ChunkOfParticle(i);
	const uint32_t lane = LaneOfParticle(i);
	const ChunkedSELLNeighborSpan_Read chunkFR = FRChunk(chunk);

	// reconstruct rigid neighbor list of one particle from compressed SELL storage
	for (uint32_t k = 0; k < N_nbrs; k++)
	{
		out[k] = chunkFR.ReadNeighborIdx(lane, k);
	}

	return NeighborSpan{ out.data(), N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::RF(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRF_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRF_.data() + nbrOffsetsRF_[i], N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::RB(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRB_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRB_.data() + nbrOffsetsRB_[i], N_nbrs };
}

NeighborSpan NeighborLists_ChunkedSELL_AdaptCompress_CPU::RR(
	const uint32_t i) const
{
	const uint32_t N_nbrs = nbrCountRR_[i];
	if (N_nbrs == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	return NeighborSpan{ nbrListRR_.data() + nbrOffsetsRR_[i], N_nbrs };
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountFR(
	const uint32_t i) const
{
	return nbrCountFR_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountRF(
	const uint32_t i) const
{
	return nbrCountRF_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountRB(
	const uint32_t i) const
{
	return nbrCountRB_[i];
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::CountRR(
	const uint32_t i) const
{
	return nbrCountRR_[i];
}

Mem::ParticleClassMemUsage
NeighborLists_ChunkedSELL_AdaptCompress_CPU::GetMemUsage() const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};
	Mem::MemUsage usageFR{};

	// FF neighbor list memory usage
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkMaxCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkFormatFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkDictSizeFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkDictFormatFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkDictLowFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkBaseIdx2FF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkDictOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrChunkDictFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrList16FF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrList32FF_));

	// FB neighbor list memory usage
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkMaxCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkFormatFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrChunkOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrList16FB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrList32FB_));

	// FR neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkMaxCountFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkBaseIdxFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkFormatFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrChunkOffsetsFR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrList16FR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrList32FR_));

	// RF neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRF_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRF_));

	// RB neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRB_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRB_));

	// RR neighbor list memory usage
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrCountRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrOffsetsRR_));
	Mem::Add(usageFR, Mem::MemUsageOf_Vector(nbrListRR_));

	return { usageFF, usageFB, usageFR };
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkNumTotal() const noexcept
{
	return ChunkNumTotal(N_F_);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::FFChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFF(chunk);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::FBChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFB(chunk);
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::FRChunk(
	const uint32_t chunk) const noexcept
{
	return MakeReadSpanFR(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::FFChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFF(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::FBChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFB(chunk);
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::FRChunkWrite(
	const uint32_t chunk) noexcept
{
	return MakeWriteSpanFR(chunk);
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::FinishFFChunkWrite(
	const uint32_t chunk, const uint32_t* writtenCounts)
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	ChunkedSELLNeighborSpan_Write chunkFF = FFChunkWrite(chunk);

	// direct chunk fill already wrote the real neighbors
	for (uint32_t i = begin; i < end; i++)
	{
		const uint32_t lane = LaneOfParticle(i);
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFF_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FF neighbor storage.");
		}
		#endif

		// dict chunks have no per-lane payload to pad
		if (chunkFF.dict) { continue; }

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingFluidIdx = (chunkFF.format == SELLChunkFormat::Delta16)
			? chunkFF.baseIdx
			: i;
		for (uint32_t k = N_written; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safePaddingFluidIdx);
		}
	}

	if (chunkFF.dict) { return; }

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailFluidIdx = (chunkFF.format == SELLChunkFormat::Delta16)
		? chunkFF.baseIdx
		: 0;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFF.pitch; k++)
		{
			chunkFF.StoreNeighbor(lane, k, safeTailFluidIdx);
		}
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::FinishFBChunkWrite(
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
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFB_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FB neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingBoundaryIdx = (chunkFB.format == SELLChunkFormat::Delta16)
			? chunkFB.baseIdx
			: safeBoundaryIdx;
		for (uint32_t k = N_written; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safePaddingBoundaryIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailBoundaryIdx = (chunkFB.format == SELLChunkFormat::Delta16)
		? chunkFB.baseIdx
		: safeBoundaryIdx;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFB.pitch; k++)
		{
			chunkFB.StoreNeighbor(lane, k, safeTailBoundaryIdx);
		}
	}
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::FinishFRChunkWrite(
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
		const uint32_t N_written = writtenCounts[lane];

		#if defined(_DEBUG)
		if (nbrCountFR_[i] != N_written) [[unlikely]]
		{
			throw std::runtime_error(
				"Count and fill query mismatch in direct compressed SELL FR neighbor storage.");
		}
		#endif

		// padding entries are valid indices so SIMD gathers never touch invalid memory
		const uint32_t safePaddingRigidIdx = (chunkFR.format == SELLChunkFormat::Delta16)
			? chunkFR.baseIdx
			: safeRigidIdx;
		for (uint32_t k = N_written; k < chunkFR.pitch; k++)
		{
			chunkFR.StoreNeighbor(lane, k, safePaddingRigidIdx);
		}
	}

	// tail lanes of the final partial chunk still get loaded by SIMD gathers
	const uint32_t safeTailRigidIdx = (chunkFR.format == SELLChunkFormat::Delta16)
		? chunkFR.baseIdx
		: safeRigidIdx;
	for (uint32_t lane = end - begin; lane < chunkWidth; lane++)
	{
		for (uint32_t k = 0; k < chunkFR.pitch; k++)
		{
			chunkFR.StoreNeighbor(lane, k, safeTailRigidIdx);
		}
	}
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkOfParticle(
	const uint32_t i) noexcept
{
	return i / chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::LaneOfParticle(
	const uint32_t i) noexcept
{
	return i % chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkBegin(
	const uint32_t chunk) noexcept
{
	return chunk * chunkWidth;
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkEnd(
	const uint32_t chunk, const uint32_t N_F) noexcept
{
	return std::min(ChunkBegin(chunk) + chunkWidth, N_F);
}

uint32_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / chunkWidth) + 1u;
}

SELLChunkFormat NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChooseChunkFormat(
	const uint32_t pitch, const uint32_t minRef, const uint32_t maxRef) noexcept
{
	if (pitch == 0u || minRef == std::numeric_limits<uint32_t>::max() || maxRef < minRef)
	{
		return SELLChunkFormat::Abs32;
	}

	const uint32_t span = maxRef - minRef;
	if (span <= 0xffffu) { return SELLChunkFormat::Delta16; }
	return SELLChunkFormat::Abs32;
}

bool NeighborLists_ChunkedSELL_AdaptCompress_CPU::SplitDictDelta16x2(
	const ChunkedSELLNeighborCountResult& counts, uint32_t& N_dictLow, uint32_t& baseIdx2) noexcept
{
	// all neighbors within uint16 reach of the lowest one form the low group
	N_dictLow = counts.N_distinct;
	baseIdx2 = counts.minRef;
	if (counts.maxRef - counts.minRef <= 0xffffu) { return true; }
	if (!counts.distinct) { return false; }

	// the rest form the high group, which needs to be within uint16 reach of its own lowest one
	N_dictLow = 0;
	baseIdx2 = std::numeric_limits<uint32_t>::max();
	for (uint32_t n = 0; n < counts.N_distinct; n++)
	{
		const uint32_t j = counts.distinct[n];
		if (j - counts.minRef <= 0xffffu) { N_dictLow++; }
		else { baseIdx2 = std::min(baseIdx2, j); }
	}

	return counts.maxRef - baseIdx2 <= 0xffffu;
}

uint64_t NeighborLists_ChunkedSELL_AdaptCompress_CPU::ChunkStorageSize(
	const uint32_t pitch) noexcept
{
	return static_cast<uint64_t>(pitch) * chunkWidth;
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeReadSpanFF(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFF_[chunk];
	const bool isDict = nbrChunkDictSizeFF_[chunk] > 0u;

	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = !isDict && format == SELLChunkFormat::Abs32 && !nbrList32FF_.empty()
			? nbrList32FF_.data() + offset : nullptr,
		.data16 = !isDict && format == SELLChunkFormat::Delta16 && !nbrList16FF_.empty()
			? nbrList16FF_.data() + offset : nullptr,
		.counts = nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		.baseIdx = nbrChunkBaseIdxFF_[chunk],
		.baseIdx2 = nbrChunkBaseIdx2FF_[chunk],
		.N_dictLow = nbrChunkDictLowFF_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFF_[chunk],
		.format = format,
		.dict = isDict ? nbrChunkDictFF_.data() + nbrChunkDictOffsetsFF_[chunk] : nullptr,
		.dictFormat = nbrChunkDictFormatFF_[chunk],
		.N_dict = nbrChunkDictSizeFF_[chunk],
		.dictLaneMasks = isDict
			? nbrChunkDictFF_.data() + nbrChunkDictOffsetsFF_[chunk]
				+ SELLDictEntryBytes(nbrChunkDictFormatFF_[chunk]) * nbrChunkDictSizeFF_[chunk]
			: nullptr
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeReadSpanFB(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = format == SELLChunkFormat::Abs32 && !nbrList32FB_.empty()
			? nbrList32FB_.data() + offset : nullptr,
		.data16 = format == SELLChunkFormat::Delta16 && !nbrList16FB_.empty()
			? nbrList16FB_.data() + offset : nullptr,
		.counts = nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		.baseIdx = nbrChunkBaseIdxFB_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFB_[chunk],
		.format = format
	};
}

ChunkedSELLNeighborSpan_Read NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeReadSpanFR(
	const uint32_t chunk) const noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFR_[chunk];

	return ChunkedSELLNeighborSpan_Read
	{
		.data32 = format == SELLChunkFormat::Abs32 && !nbrList32FR_.empty()
			? nbrList32FR_.data() + offset : nullptr,
		.data16 = format == SELLChunkFormat::Delta16 && !nbrList16FR_.empty()
			? nbrList16FR_.data() + offset : nullptr,
		.counts = nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		.baseIdx = nbrChunkBaseIdxFR_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFR_[chunk],
		.format = format
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeWriteSpanFF(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFF_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFF_[chunk];
	const bool isDict = nbrChunkDictSizeFF_[chunk] > 0u;

	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = !isDict && format == SELLChunkFormat::Abs32 && !nbrList32FF_.empty()
			? nbrList32FF_.data() + offset : nullptr,
		.data16 = !isDict && format == SELLChunkFormat::Delta16 && !nbrList16FF_.empty()
			? nbrList16FF_.data() + offset : nullptr,
		.counts = nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		.baseIdx = nbrChunkBaseIdxFF_[chunk],
		.baseIdx2 = nbrChunkBaseIdx2FF_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFF_[chunk],
		.format = format,
		.dict = isDict ? nbrChunkDictFF_.data() + nbrChunkDictOffsetsFF_[chunk] : nullptr,
		.dictFormat = nbrChunkDictFormatFF_[chunk],
		.N_dictMax = nbrChunkDictSizeFF_[chunk],
		.dictLaneMasks = isDict
			? nbrChunkDictFF_.data() + nbrChunkDictOffsetsFF_[chunk]
				+ SELLDictEntryBytes(nbrChunkDictFormatFF_[chunk]) * nbrChunkDictSizeFF_[chunk]
			: nullptr
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeWriteSpanFB(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFB_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFB_[chunk];

	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = format == SELLChunkFormat::Abs32 && !nbrList32FB_.empty()
			? nbrList32FB_.data() + offset : nullptr,
		.data16 = format == SELLChunkFormat::Delta16 && !nbrList16FB_.empty()
			? nbrList16FB_.data() + offset : nullptr,
		.counts = nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		.baseIdx = nbrChunkBaseIdxFB_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFB_[chunk],
		.format = format
	};
}

ChunkedSELLNeighborSpan_Write NeighborLists_ChunkedSELL_AdaptCompress_CPU::MakeWriteSpanFR(
	const uint32_t chunk) noexcept
{
	const uint32_t begin = ChunkBegin(chunk);
	const uint32_t end = ChunkEnd(chunk, N_F_);
	const uint64_t offset = nbrChunkOffsetsFR_[chunk];
	const SELLChunkFormat format = nbrChunkFormatFR_[chunk];

	return ChunkedSELLNeighborSpan_Write
	{
		.data32 = format == SELLChunkFormat::Abs32 && !nbrList32FR_.empty()
			? nbrList32FR_.data() + offset : nullptr,
		.data16 = format == SELLChunkFormat::Delta16 && !nbrList16FR_.empty()
			? nbrList16FR_.data() + offset : nullptr,
		.counts = nbrCountFR_.empty() ? nullptr : nbrCountFR_.data(),
		.baseIdx = nbrChunkBaseIdxFR_[chunk],
		.chunk = chunk, .begin = begin, .end = end,
		.pitch = nbrChunkMaxCountFR_[chunk],
		.format = format
	};
}

void NeighborLists_ChunkedSELL_AdaptCompress_CPU::FinalizeChunkStorage(
	const std::vector<SELLNeighborCount>& nbrChunkMaxCount,
	const std::vector<SELLChunkFormat>& nbrChunkFormat,
	const std::vector<uint16_t>* nbrChunkDictSize,
	std::vector<uint64_t>& nbrChunkOffsets,
	std::vector<uint16_t>& nbrList16,
	std::vector<uint32_t>& nbrList32) const
{
	const uint32_t N_chunks = ChunkNumTotal(N_F_);
	nbrChunkOffsets.resize(static_cast<uint64_t>(N_chunks) + 1ull);
	nbrChunkOffsets[N_chunks] = 0;

	ChunkStorageTotals totals{};
	if (N_chunks > 0u)
	{
		totals = tbb::parallel_scan(
			tbb::blocked_range<uint32_t>(0, N_chunks),
			ChunkStorageTotals{},
			[&](const tbb::blocked_range<uint32_t>& r,
				const ChunkStorageTotals& prefix,
				const bool isFinalScan)
			{
				ChunkStorageTotals running = prefix;
				for (uint32_t chunk = r.begin(); chunk != r.end(); chunk++)
				{
					const uint32_t pitch = nbrChunkMaxCount[chunk];
					const uint64_t chunkStorage = ChunkStorageSize(pitch);
					const SELLChunkFormat format = nbrChunkFormat[chunk];

					// (dict chunks keep their neighbors and lane masks in the dictionary, without per-lane storage)
					if (nbrChunkDictSize && (*nbrChunkDictSize)[chunk] > 0u)
					{
						if (isFinalScan)
						{
							nbrChunkOffsets[chunk] = 0;
						}
					}
					else if (format == SELLChunkFormat::Delta16)
					{
						if (isFinalScan)
						{
							nbrChunkOffsets[chunk] = running.N_entries16;
						}
						running.N_entries16 += chunkStorage;
					}
					else
					{
						if (isFinalScan)
						{
							nbrChunkOffsets[chunk] = running.N_entries32;
						}
						running.N_entries32 += chunkStorage;
					}
				}
				return running;
			},
			[](const ChunkStorageTotals& lhs, const ChunkStorageTotals& rhs)
			{
				return ChunkStorageTotals
				{
					lhs.N_entries16 + rhs.N_entries16,
					lhs.N_entries32 + rhs.N_entries32
				};
			},
			tbb::auto_partitioner{}
		);
	}

	Base::ResizeScratchVector(nbrList16, totals.N_entries16, extraAllocCapacityPct_);
	Base::ResizeScratchVector(nbrList32, totals.N_entries32, extraAllocCapacityPct_);
}
