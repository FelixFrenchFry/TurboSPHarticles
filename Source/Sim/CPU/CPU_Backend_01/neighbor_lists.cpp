// neighbor_lists.cpp

// external
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cassert>
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
// --- [ individual vectors neighbor lists, explicit ] ---------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_IndividualVectors_Explicit_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// individual neighbor lists
	nbrListFF_.resize(N_F_);
	nbrListFB_.resize(N_F_);

	// reserve expected capacities to reduce re-allocations
	// (single-threaded to minimize fragmentation and alloc contention)
	for (uint32_t i = 0; i < N_F_; i++)
	{
		nbrListFF_[i].reserve(params.expctdNbrCountSoftMax_FF);
		nbrListFB_[i].reserve(params.expctdNbrCountSoftMax_FB);

		nbrListFF_[i].clear();
		nbrListFB_[i].clear();
	}
}

void NeighborLists_IndividualVectors_Explicit_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) [[unlikely]] { nbrListFF_[i].clear(); return; }

	// overwrite and optional resize
	nbrListFF_[i].assign(js, js + count);
}

void NeighborLists_IndividualVectors_Explicit_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) [[likely]] { nbrListFB_[i].clear(); return; }

	// overwrite and optional resize
	nbrListFB_[i].assign(js, js + count);
}

NeighborSpan NeighborLists_IndividualVectors_Explicit_CPU01::FF(
	const uint32_t i) const
{
	const auto& nbrs = nbrListFF_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

NeighborSpan NeighborLists_IndividualVectors_Explicit_CPU01::FB(
	const uint32_t i) const
{
	const auto& nbrs = nbrListFB_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_IndividualVectors_Explicit_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// FF
	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(nbrListFF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(nbrListFB_));

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ single CSR vector neighbor list, explicit ] -----------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_SingleCSRVector_Explicit_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	const uint32_t numBlocks = OffsetBlockNumTotal(N_F_);
	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);
	nbrLocalOffsetsFF_.assign(N_F_, 0);
	nbrLocalOffsetsFB_.assign(N_F_, 0);
	nbrBlockBasesFF_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrBlockBasesFB_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrListFF_.clear();
	nbrListFB_.clear();

	// reserve packed CSR storage up front to avoid a late full re-allocation
	// of large final neighbor buffers during long or near-capacity runs
	EnsurePackedCSRStorageCapacity(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::Finalize_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrBlockBasesFF_.assign(1, 0);
		nbrBlockBasesFB_.assign(1, 0);
		nbrListFF_.clear();
		nbrListFB_.clear();
		return;
	}

	FinalizeBlockCSR(
		nbrCountFF_, nbrLocalOffsetsFF_, nbrBlockBasesFF_, nbrListFF_,
		"Block-local CSR offset overflow in explicit FF neighbor storage.");
	FinalizeBlockCSR(
		nbrCountFB_, nbrLocalOffsetsFB_, nbrBlockBasesFB_, nbrListFB_,
		"Block-local CSR offset overflow in explicit FB neighbor storage.");
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrCountFF_[i] = count;
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrCountFB_[i] = count;
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	// TODO: make this only active in debug compile mode?
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit CSR FF neighbor storage.");
	}

	if (count == 0) [[unlikely]] { return; }

	const uint32_t block = OffsetBlockOfParticle(i);
	const uint64_t start = nbrBlockBasesFF_[block]
						 + static_cast<uint64_t>(nbrLocalOffsetsFF_[i]);
	std::memcpy(nbrListFF_.data() + start, js, count * sizeof(uint32_t));
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	// TODO: make this only active in debug compile mode?
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit CSR FB neighbor storage.");
	}

	if (count == 0) [[likely]] { return; }

	const uint32_t block = OffsetBlockOfParticle(i);
	const uint64_t start = nbrBlockBasesFB_[block]
						 + static_cast<uint64_t>(nbrLocalOffsetsFB_[i]);
	std::memcpy(nbrListFB_.data() + start, js, count * sizeof(uint32_t));
}

NeighborSpan NeighborLists_SingleCSRVector_Explicit_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint32_t block = OffsetBlockOfParticle(i);
	const uint64_t start = nbrBlockBasesFF_[block]
						 + static_cast<uint64_t>(nbrLocalOffsetsFF_[i]);
	return NeighborSpan{ nbrListFF_.data() + start, count };
}

NeighborSpan NeighborLists_SingleCSRVector_Explicit_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint32_t block = OffsetBlockOfParticle(i);
	const uint64_t start = nbrBlockBasesFB_[block]
						 + static_cast<uint64_t>(nbrLocalOffsetsFB_[i]);
	return NeighborSpan{ nbrListFB_.data() + start, count };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_SingleCSRVector_Explicit_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// FF
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrLocalOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrBlockBasesFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrLocalOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrBlockBasesFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListFB_));

	return { usageFF, usageFB };
}

uint32_t NeighborLists_SingleCSRVector_Explicit_CPU01::OffsetBlockNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / offsetBlockParticleCount_) + 1u;
}

uint32_t NeighborLists_SingleCSRVector_Explicit_CPU01::OffsetBlockOfParticle(
	const uint32_t i) noexcept
{
	return i / offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_Explicit_CPU01::OffsetBlockBegin(
	const uint32_t block) noexcept
{
	return block * offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_Explicit_CPU01::OffsetBlockEnd(
	const uint32_t block, const uint32_t N_F) noexcept
{
	return std::min(OffsetBlockBegin(block) + offsetBlockParticleCount_, N_F);
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::FinalizeBlockCSR(
	const std::vector<uint32_t>& nbrCounts,
	std::vector<uint32_t>& nbrLocalOffsets,
	std::vector<uint64_t>& nbrBlockBases,
	std::vector<uint32_t>& nbrList,
	const char* offsetOverflowMsg) const
{
	const uint32_t numBlocks = OffsetBlockNumTotal(N_F_);
	std::vector<uint64_t> blockTotals(static_cast<uint64_t>(numBlocks), 0ull);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numBlocks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t block = r.begin(); block != r.end(); block++)
			{
				const uint32_t begin = OffsetBlockBegin(block);
				const uint32_t end = OffsetBlockEnd(block, N_F_);
				uint64_t sum = 0;

				for (uint32_t i = begin; i < end; i++)
				{
					if (sum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
					{
						throw std::runtime_error(offsetOverflowMsg);
					}

					nbrLocalOffsets[i] = static_cast<uint32_t>(sum);
					sum += nbrCounts[i];
				}

				blockTotals[block] = sum;
			}
		},
		tbb::auto_partitioner{}
	);

	// compute global block base indices from block totals
	nbrBlockBases.resize(static_cast<uint64_t>(numBlocks) + 1ull);
	nbrBlockBases[0] = 0;
	for (uint32_t block = 0; block < numBlocks; block++)
	{
		nbrBlockBases[static_cast<uint64_t>(block) + 1ull]
			= nbrBlockBases[block] + blockTotals[block];
	}

	const uint64_t N_total = nbrBlockBases[numBlocks];
	EnsurePackedCSRStorageCapacity(nbrList, N_total, extraAllocCapacityPct_);
	nbrList.resize(N_total);
}

// -------------------------------------------------------------------------------------------------
// --- [ single CSR vector neighbor list, adaptive compression ] -----------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	// TODO: implement
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::Finalize_Count()
{
	// TODO: implement
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	// TODO: implement
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	// TODO: implement
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// TODO: implement
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// TODO: implement
}

NeighborSpan NeighborLists_SingleCSRVector_AdaptCompress_CPU01::FF(
	const uint32_t i) const
{
	// TODO: implement

	return NeighborSpan{ nullptr, 0 };
}

NeighborSpan NeighborLists_SingleCSRVector_AdaptCompress_CPU01::FB(
	const uint32_t i) const
{
	// TODO: implement

	return NeighborSpan{ nullptr, 0 };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_SingleCSRVector_AdaptCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// FF
	// TODO: implement

	// FB
	// TODO: implement

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ single CSR vector neighbor lists, VByte compression ] -------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);
	nbrLocalOffsetsFF_.assign(N_F_, 0);
	nbrLocalOffsetsFB_.assign(N_F_, 0);
	nbrBlockBasesFF_.assign(static_cast<uint64_t>(OffsetBlockNumTotal(N_F_)) + 1ull, 0);
	nbrBlockBasesFB_.assign(static_cast<uint64_t>(OffsetBlockNumTotal(N_F_)) + 1ull, 0);
	nbrListCompressedFF_.clear();
	nbrListCompressedFB_.clear();
	fillPhase_ = FillPhase::Measure;

	// reserve packed CSR storage up front to avoid a late full re-allocation
	// of large final neighbor buffers during long or near-capacity runs
	// (using 1 expected byte per neighbor)
	EnsurePackedCSRStorageCapacity(
		nbrListCompressedFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF) * 1);
	EnsurePackedCSRStorageCapacity(
		nbrListCompressedFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB) * 1);
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::Finalize_Count()
{
	// reset to prepare for first fill-query (measure phase)
	const uint32_t numBlocks = OffsetBlockNumTotal(N_F_);
	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);
	nbrLocalOffsetsFF_.assign(N_F_, 0);
	nbrLocalOffsetsFB_.assign(N_F_, 0);
	nbrBlockBasesFF_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrBlockBasesFB_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrListCompressedFF_.clear();
	nbrListCompressedFB_.clear();
	fillPhase_ = FillPhase::Measure;
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::Finalize_Fill()
{
	if (fillPhase_ == FillPhase::Measure)
	{
		if (N_F_ == 0) // nothing to do for empty sim
		{
			nbrBlockBasesFF_.assign(1, 0);
			nbrBlockBasesFB_.assign(1, 0);
			nbrListCompressedFF_.clear();
			nbrListCompressedFB_.clear();
			fillPhase_ = FillPhase::Encode;
			return;
		}

		FinalizeBlockCSR(nbrLocalOffsetsFF_, nbrBlockBasesFF_, nbrListCompressedFF_);
		FinalizeBlockCSR(nbrLocalOffsetsFB_, nbrBlockBasesFB_, nbrListCompressedFB_);
		fillPhase_ = FillPhase::Encode;
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		fillPhase_ = FillPhase::Ready;
		return;
	}

	throw std::runtime_error(
		"Unexpected fill-phase transition in compressed single-CSR neighbor lists.");
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFF_.local();

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFF_[i] = count;
		const uint64_t encodedBytes = MeasureEncodedBytes(js, count, scratch);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrLocalOffsetsFF_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		if (nbrCountFF_[i] != count)
		{
			throw std::runtime_error(
				"Measured and encoded FF neighbor counts do not match in compressed single-CSR storage.");
		}

		const uint64_t s = ParticleStorageBegin(i, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		const uint64_t e = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		uint8_t* dst = nbrListCompressedFF_.empty()
					 ? nullptr
					 : nbrListCompressedFF_.data() + s;

		EncodeNeighborList(dst, e - s, js, count, scratch);
		return;
	}

	throw std::runtime_error(
		"Compressed single-CSR neighbor lists received FF fill data in an invalid state.");
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFB_.local();

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFB_[i] = count;
		const uint64_t encodedBytes = MeasureEncodedBytes(js, count, scratch);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrLocalOffsetsFB_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		if (nbrCountFB_[i] != count)
		{
			throw std::runtime_error(
				"Measured and encoded FB neighbor counts do not match in compressed single-CSR storage.");
		}

		const uint64_t s = ParticleStorageBegin(i, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		const uint64_t e = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		uint8_t* dst = nbrListCompressedFB_.empty()
					 ? nullptr
					 : nbrListCompressedFB_.data() + s;

		EncodeNeighborList(dst, e - s, js, count, scratch);
		return;
	}

	throw std::runtime_error(
		"Compressed single-CSR neighbor lists received FB fill data in an invalid state.");
}

NeighborSpan NeighborLists_SingleCSRVector_VByteCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
	const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
	const uint8_t* src = nbrListCompressedFF_.data() + start;

	// decode into per-thread scratch
	auto& out = nbrScratchFF_.local();
	DecodeNeighborList(src, end - start, count, out);

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_SingleCSRVector_VByteCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
	const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
	const uint8_t* src = nbrListCompressedFB_.data() + start;

	// decode into per-thread scratch
	auto& out = nbrScratchFB_.local();
	DecodeNeighborList(src, end - start, count, out);

	return NeighborSpan{ out.data(), count };
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_SingleCSRVector_VByteCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// FF
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrLocalOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrBlockBasesFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListCompressedFF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrLocalOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrBlockBasesFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListCompressedFB_));

	return { usageFF, usageFB };
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::OffsetBlockNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / offsetBlockParticleCount_) + 1u;
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::OffsetBlockOfParticle(
	const uint32_t i) noexcept
{
	return i / offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::OffsetBlockBegin(
	const uint32_t block) noexcept
{
	return block * offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::OffsetBlockEnd(
	const uint32_t block, const uint32_t N_F) noexcept
{
	return std::min(OffsetBlockBegin(block) + offsetBlockParticleCount_, N_F);
}

uint64_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::ParticleStorageBegin(
	const uint32_t i,
	const std::vector<uint32_t>& nbrLocalOffsets,
	const std::vector<uint64_t>& nbrBlockBases) noexcept
{
	const uint32_t block = OffsetBlockOfParticle(i);
	return nbrBlockBases[block] + static_cast<uint64_t>(nbrLocalOffsets[i]);
}

uint64_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::ParticleStorageEnd(
	const uint32_t i,
	const uint32_t N_F,
	const std::vector<uint32_t>& nbrLocalOffsets,
	const std::vector<uint64_t>& nbrBlockBases) noexcept
{
	const uint32_t block = OffsetBlockOfParticle(i);
	const uint32_t blockEnd = OffsetBlockEnd(block, N_F);

	if (i + 1u < blockEnd)
	{
		return nbrBlockBases[block] + static_cast<uint64_t>(nbrLocalOffsets[i + 1u]);
	}

	return nbrBlockBases[static_cast<uint64_t>(block) + 1ull];
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::FinalizeBlockCSR(
	std::vector<uint32_t>& nbrLocalOffsets,
	std::vector<uint64_t>& nbrBlockBases,
	std::vector<uint8_t>& nbrList) const
{
	const uint32_t numBlocks = OffsetBlockNumTotal(N_F_);
	std::vector<uint64_t> blockTotals(static_cast<uint64_t>(numBlocks), 0ull);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, numBlocks),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t block = r.begin(); block != r.end(); block++)
			{
				const uint32_t begin = OffsetBlockBegin(block);
				const uint32_t end = OffsetBlockEnd(block, N_F_);
				uint64_t sum = 0;

				for (uint32_t i = begin; i < end; i++)
				{
					const uint32_t countBytes = nbrLocalOffsets[i];
					nbrLocalOffsets[i] = static_cast<uint32_t>(sum);
					sum += countBytes;
				}

				blockTotals[block] = sum;
			}
		},
		tbb::auto_partitioner{}
	);

	// compute global block base indices from block totals
	nbrBlockBases.resize(static_cast<uint64_t>(numBlocks) + 1ull);
	nbrBlockBases[0] = 0;
	for (uint32_t block = 0; block < numBlocks; block++)
	{
		nbrBlockBases[static_cast<uint64_t>(block) + 1ull]
			= nbrBlockBases[block] + blockTotals[block];
	}

	const uint64_t N_total = nbrBlockBases[numBlocks];
	EnsurePackedCSRStorageCapacity(nbrList, N_total, extraAllocCapacityPct_);
	nbrList.resize(N_total);
}

uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::ControlBytesFor(
	const uint32_t count) noexcept
{
	// there are (count - 1) deltas, and 4 masks fit into one control byte
	return (count <= 1) ? 0u : ((count - 1u + 3u) / 4u);
}

uint8_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::DeltaMask(
	const uint32_t delta) noexcept
{
	if (delta == 0u) { return 0u; } // 00 (-> 0 data bytes encoded via mask)
	if (delta == 1u) { return 1u; } // 01 (-> 0 data bytes, encoded via mask)
	if (delta < 256u) { return 2u; } // 10 (-> 1 data byte)
	return 3u; // 11 (-> 4 data bytes)
}

uint64_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::MeasureEncodedBytes(
	const uint32_t* js, const uint32_t count, std::vector<uint32_t>& scratch)
{
	if (count == 0)
	{
		return 0;
	}

	// copy into thread-local scratch and sort to avoid negative deltas
	// TODO: guarantee sorted input from in the query to avoid sorting here?
	scratch.assign(js, js + count);
	std::sort(scratch.begin(), scratch.end());

	// determine size of encoded data
	const uint32_t ctrlBytes = ControlBytesFor(count);
	uint64_t dataBytes = 0;
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t gap = scratch[k] - scratch[k - 1];

		// gap must be >= 1 for delta = gap - 1 to be non-negative
	#if defined(_DEBUG)
		assert(gap >= 1u && "Neighbor list must be strictly increasing");
	#endif

		const uint32_t delta = gap - 1u;

		// map to size after encoding (0/1 => 0B, [2,256) => 1B, else 4B)
		if (delta >= 2u && delta < 256u) { dataBytes += 1ull; }
		else if (delta >= 256u) { dataBytes += 4ull; }
	}

	return 4ull + static_cast<uint64_t>(ctrlBytes) + dataBytes;
}

inline uint32_t NeighborLists_SingleCSRVector_VByteCompress_CPU01::ReadU32LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint32_t>(src[0])
		| (static_cast<uint32_t>(src[1]) << 8)
		| (static_cast<uint32_t>(src[2]) << 16)
		| (static_cast<uint32_t>(src[3]) << 24);
}

inline void NeighborLists_SingleCSRVector_VByteCompress_CPU01::WriteU32LE(
	uint8_t* dst, const uint32_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
	dst[2] = static_cast<uint8_t>(v >> 16);
	dst[3] = static_cast<uint8_t>(v >> 24);
}

// TODO: implement with SIMD optimizations?
void NeighborLists_SingleCSRVector_VByteCompress_CPU01::DecodeNeighborList(
	const uint8_t* src, const uint64_t srcBytes, const uint32_t count,
	std::vector<uint32_t>& out)
{
	if (count == 0)
	{
		out.clear();
		return;
	}

	// expected size check (debug only)
	#if defined(_DEBUG)
	{
		// must at least hold first neighbor
		assert(srcBytes >= 4 && "Compressed list too small");

		const uint32_t ctrlBytes = ControlBytesFor(count);
		assert(srcBytes >= 4ull + ctrlBytes && "Compressed list too small for control stream");

		uint64_t expectedDataBytes = 0;
		const uint8_t* ctrlDbg = src + 4;

		for (uint32_t idx = 0; idx < count - 1; idx++)
		{
			const uint8_t c = ctrlDbg[idx >> 2];
			const uint8_t mask = static_cast<uint8_t>((c >> ((idx & 3u) * 2u)) & 0x3u);

			if (mask == 2u) { expectedDataBytes += 1ull; }
			else if (mask == 3u) { expectedDataBytes += 4ull; }
			// mask 0/1 => 0 bytes
		}

		const uint64_t expectedBytes = 4ull + static_cast<uint64_t>(ctrlBytes) + expectedDataBytes;
		assert(srcBytes == expectedBytes && "Compressed list size mismatch");
	}
	#endif

	out.resize(count);

	// decode first element (uncompressed)
	uint32_t v = ReadU32LE(src);
	out[0] = v;
	if (count == 1) { return; }

	// control bytes after first element
	const uint32_t ctrlBytes = ControlBytesFor(count);
	const uint8_t* ctrl = src + 4;
	const uint8_t* data = ctrl + ctrlBytes;

	// decode (count - 1) deltas, then undo (gap = delta + 1, prefix sum)
	for (uint32_t idx = 0; idx < count - 1; idx++)
	{
		const uint8_t c = ctrl[idx >> 2];
		const uint8_t mask = static_cast<uint8_t>((c >> ((idx & 3u) * 2u)) & 0x3u);

		uint32_t delta = 0;
		if (mask == 0u)
		{
			delta = 0u;
		}
		else if (mask == 1u)
		{
			delta = 1u;
		}
		else if (mask == 2u)
		{
			delta = static_cast<uint32_t>(*data++);
		}
		else // mask == 3
		{
			delta = ReadU32LE(data);
			data += 4;
		}

		v += (delta + 1u);
		out[idx + 1] = v;
	}

	#if defined(_DEBUG)
	assert(data == src + srcBytes && "Decoded byte count mismatch");
	#endif
}

// TODO: implement with SIMD optimizations?
void NeighborLists_SingleCSRVector_VByteCompress_CPU01::EncodeNeighborList(
	uint8_t* dst, const uint64_t dstBytes, const uint32_t* js,
	const uint32_t count, std::vector<uint32_t>& scratch)
{
	if (count == 0)
	{
	#if defined(_DEBUG)
		assert(dstBytes == 0 && "Compressed list byte count mismatch for empty list");
	#endif
		return;
	}

	// copy into thread-local scratch and sort to avoid negative deltas
	// TODO: guarantee sorted input from in the query to avoid sorting here?
	scratch.assign(js, js + count);
	std::sort(scratch.begin(), scratch.end());

	// determine size of encoded data
	const uint32_t ctrlBytes = ControlBytesFor(count);
	uint64_t dataBytes = 0;
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t gap = scratch[k] - scratch[k - 1];

		// gap must be >= 1 for delta = gap - 1 to be non-negative
	#if defined(_DEBUG)
		assert(gap >= 1u && "Neighbor list must be strictly increasing");
	#endif

		const uint32_t delta = gap - 1u;

		// map to size after encoding (0/1 => 0B, [2,256) => 1B, else 4B)
		if (delta >= 2u && delta < 256u) { dataBytes += 1ull; }
		else if (delta >= 256u) { dataBytes += 4ull; }
	}
	const uint64_t requiredBytes = 4ull + static_cast<uint64_t>(ctrlBytes) + dataBytes;
	if (requiredBytes != dstBytes)
	{
		throw std::runtime_error(
			"Measured and encoded VByte-compressed neighbor list sizes do not match.");
	}

	// encode first element (uncompressed)
	WriteU32LE(dst, scratch[0]);

	// control bytes after first element
	uint8_t* ctrl = dst + 4;
	std::memset(ctrl, 0, ctrlBytes);

	// data bytes after control bytes
	uint8_t* data = ctrl + ctrlBytes;

	// encode (count - 1) deltas
	for (uint32_t idx = 0; idx < count - 1; idx++)
	{
		const uint32_t k = idx + 1;

		const uint32_t gap = scratch[k] - scratch[k - 1];
		const uint32_t delta = gap - 1u;

		const uint8_t mask = DeltaMask(delta);
		ctrl[idx >> 2] |= static_cast<uint8_t>(mask << ((idx & 3u) * 2u));

		// mask 0/1 -> 0 data bytes
		if (mask == 2u) // 1 data byte
		{
			*data++ = static_cast<uint8_t>(delta);
		}
		else if (mask == 3u) // 4 data bytes
		{
			WriteU32LE(data, delta);
			data += 4;
		}
	}

	#if defined(_DEBUG)
	assert(data == dst + dstBytes && "Encoded byte count mismatch");
	#endif
}
