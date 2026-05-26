// neighbor_lists.cpp

// external
#include <algorithm>
#include <array>
#include <cmath>
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
	struct VByteControlInfo
	{
		uint8_t payloadBytes = 0;
		std::array<uint8_t, 4> masks{};
	};

	inline std::array<VByteControlInfo, 256> BuildVByteControlLut() noexcept
	{
		std::array<VByteControlInfo, 256> lut{};

		for (uint32_t c = 0; c < 256u; c++)
		{
			VByteControlInfo info{};
			for (uint32_t lane = 0; lane < 4u; lane++)
			{
				const uint8_t mask = static_cast<uint8_t>((c >> (lane * 2u)) & 0x3u);
				info.masks[lane] = mask;

				if (mask == 2u) { info.payloadBytes += 1u; }
				else if (mask == 3u) { info.payloadBytes += 4u; }
			}

			lut[c] = info;
		}

		return lut;
	}

	inline const std::array<VByteControlInfo, 256>& VByteControlLut() noexcept
	{
		static const std::array<VByteControlInfo, 256> lut = BuildVByteControlLut();
		return lut;
	}

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

	inline void AssertStrictlyIncreasing(
		const uint32_t* js,
		const uint32_t count) noexcept
	{
	#if defined(_DEBUG)
		for (uint32_t k = 1; k < count; k++)
		{
			assert(js[k - 1] < js[k] && "Neighbor list input must be strictly increasing");
		}
	#else
		(void)js;
		(void)count;
	#endif
	}

	template<typename T_Scratch>
	inline const uint32_t* PrepareSortedNeighborList(
		const uint32_t* js,
		const uint32_t count,
		const NeighborListInputOrder inputOrder,
		T_Scratch& scratch)
	{
		if (count <= 1u) { return js; }

		if (inputOrder == NeighborListInputOrder::GuaranteedSorted)
		{
			AssertStrictlyIncreasing(js, count);
			return js;
		}

		uint32_t firstInversion = count;
		for (uint32_t k = 1; k < count; k++)
		{
			if (!(js[k - 1u] < js[k]))
			{
				firstInversion = k;
				break;
			}
		}

		// input is already sorted, no need for copying and sorting
		if (firstInversion == count) { return js; }

		if (scratch.capacity() < count)
		{
			scratch.reserve(count);
		}
		scratch.assign(js, js + firstInversion);

		for (uint32_t k = firstInversion; k < count; k++)
		{
			const uint32_t v = js[k];
			scratch.push_back(v);

			uint32_t pos = static_cast<uint32_t>(scratch.size()) - 1u;
			while (pos > 0u && v < scratch[pos - 1u])
			{
				scratch[pos] = scratch[pos - 1u];
				pos--;
			}

			scratch[pos] = v;
		}

		AssertStrictlyIncreasing(scratch.data(), static_cast<uint32_t>(scratch.size()));
		return scratch.data();
	}

	template<typename T_Out>
	inline void EnsureDecodeScratchSize(
		T_Out& out,
		const uint32_t count)
	{
		if (out.size() < count)
		{
			out.resize(count);
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

	// reserve packed CSR storage up front based on the expected bytes per neighbor list
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
	#if defined(_DEBUG)
	// (count-query and fill-query MUST find the same number of neighbors)
	if (nbrCountFF_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit CSR FF neighbor storage.");
	} 
	#endif

	if (count == 0) [[unlikely]] { return; }

	const uint32_t block = OffsetBlockOfParticle(i);
	const uint64_t start = nbrBlockBasesFF_[block]
						 + static_cast<uint64_t>(nbrLocalOffsetsFF_[i]);
	std::memcpy(nbrListFF_.data() + start, js, count * sizeof(uint32_t));
}

void NeighborLists_SingleCSRVector_Explicit_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	#if defined(_DEBUG)
	// (count-query and fill-query MUST find the same number of neighbors)
	if (nbrCountFB_[i] != count) [[unlikely]]
	{
		throw std::runtime_error(
			"Count and fill query mismatch in explicit CSR FB neighbor storage.");
	}
	#endif

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
	N_F_ = N_F;
	N_B_ = N_B;

	const uint32_t numBlocks = OffsetBlockNumTotal(N_F_);
	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);
	nbrLocalOffsetsFF_.assign(N_F_, 0);
	nbrLocalOffsetsFB_.assign(N_F_, 0);
	nbrBlockBasesFF_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrBlockBasesFB_.assign(static_cast<uint64_t>(numBlocks) + 1ull, 0);
	nbrListCompressedFF_.clear();
	nbrListCompressedFB_.clear();
	nbrWidthCodeFF_.assign(N_F_, 0u);
	nbrWidthCodeFB_.assign(N_F_, 0u);
	fillPhase_ = FillPhase::Measure;

	// reserve packed CSR storage up front based on the expected bytes per compressed list
	const uint64_t expctdBytesFF = static_cast<uint64_t>(std::ceil(
		static_cast<double>(N_F_) *
		(static_cast<double>(overheadBytesPerNeighborList)
		 + expctdBytePerCompressedNeighbor
		 * static_cast<double>(params.expctdAvgNbrCountHardMax_FF))));
	const uint64_t expctdBytesFB = static_cast<uint64_t>(std::ceil(
		static_cast<double>(N_F_) *
		(static_cast<double>(overheadBytesPerNeighborList)
		 + expctdBytePerCompressedNeighbor
		 * static_cast<double>(params.expctdAvgNbrCountHardMax_FB))));
	EnsurePackedCSRStorageCapacity(nbrListCompressedFF_, expctdBytesFF);
	EnsurePackedCSRStorageCapacity(nbrListCompressedFB_, expctdBytesFB);
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::Finalize_Count()
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
	nbrWidthCodeFF_.assign(N_F_, 0u);
	nbrWidthCodeFB_.assign(N_F_, 0u);
	fillPhase_ = FillPhase::Measure;
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::Finalize_Fill()
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
		"Unexpected fill-phase transition in adapt-compressed single-CSR neighbor lists.");
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFF_.local();
	const uint32_t* jsSorted = PrepareSortedNeighborList(js, count, inputOrder_, scratch);

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFF_[i] = count;
		uint8_t widthCode = 0u;
		const uint64_t encodedBytes = MeasureEncodedBytes(jsSorted, count, widthCode);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle adapt-compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrWidthCodeFF_[i] = widthCode;
		nbrLocalOffsetsFF_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		#if defined(_DEBUG)
		if (nbrCountFF_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Measured and encoded FF neighbor counts do not match in adapt-compressed single-CSR storage.");
		}
		#endif

		const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		uint8_t* dst = nbrListCompressedFF_.empty()
					 ? nullptr
					 : nbrListCompressedFF_.data() + start;

		EncodeNeighborList(dst, end - start, jsSorted, count, nbrWidthCodeFF_[i]);
		return;
	}

	throw std::runtime_error(
		"Adapt-compressed single-CSR neighbor lists received FF fill data in an invalid state.");
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFB_.local();
	const uint32_t* jsSorted = PrepareSortedNeighborList(js, count, inputOrder_, scratch);

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFB_[i] = count;
		uint8_t widthCode = 0u;
		const uint64_t encodedBytes = MeasureEncodedBytes(jsSorted, count, widthCode);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle adapt-compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrWidthCodeFB_[i] = widthCode;
		nbrLocalOffsetsFB_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		#if defined(_DEBUG)
		if (nbrCountFB_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Measured and encoded FB neighbor counts do not match in adapt-compressed single-CSR storage.");
		}
		#endif

		const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		uint8_t* dst = nbrListCompressedFB_.empty()
					 ? nullptr
					 : nbrListCompressedFB_.data() + start;

		EncodeNeighborList(dst, end - start, jsSorted, count, nbrWidthCodeFB_[i]);
		return;
	}

	throw std::runtime_error(
		"Adapt-compressed single-CSR neighbor lists received FB fill data in an invalid state.");
}

NeighborSpan NeighborLists_SingleCSRVector_AdaptCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) [[unlikely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
	const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
	const uint8_t* src = nbrListCompressedFF_.data() + start;

	auto& out = nbrScratchFF_.local();
	DecodeNeighborList(src, end - start, count, out);

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_SingleCSRVector_AdaptCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) [[likely]] { return NeighborSpan{ nullptr, 0 }; }

	const uint64_t start = ParticleStorageBegin(i, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
	const uint64_t end = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
	const uint8_t* src = nbrListCompressedFB_.data() + start;

	auto& out = nbrScratchFB_.local();
	DecodeNeighborList(src, end - start, count, out);

	return NeighborSpan{ out.data(), count };
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_SingleCSRVector_AdaptCompress_CPU01::GetMemUsage(
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
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrWidthCodeFF_));

	// FB
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrLocalOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrBlockBasesFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListCompressedFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrWidthCodeFB_));

	return { usageFF, usageFB };
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::OffsetBlockNumTotal(
	const uint32_t N_F) noexcept
{
	return (N_F == 0u) ? 0u : ((N_F - 1u) / offsetBlockParticleCount_) + 1u;
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::OffsetBlockOfParticle(
	const uint32_t i) noexcept
{
	return i / offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::OffsetBlockBegin(
	const uint32_t block) noexcept
{
	return block * offsetBlockParticleCount_;
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::OffsetBlockEnd(
	const uint32_t block, const uint32_t N_F) noexcept
{
	return std::min(OffsetBlockBegin(block) + offsetBlockParticleCount_, N_F);
}

uint64_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::ParticleStorageBegin(
	const uint32_t i,
	const std::vector<uint32_t>& nbrLocalOffsets,
	const std::vector<uint64_t>& nbrBlockBases) noexcept
{
	const uint32_t block = OffsetBlockOfParticle(i);
	return nbrBlockBases[block] + static_cast<uint64_t>(nbrLocalOffsets[i]);
}

uint64_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::ParticleStorageEnd(
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

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::FinalizeBlockCSR(
	std::vector<uint32_t>& nbrLocalOffsets,
	std::vector<uint64_t>& nbrBlockBases,
	std::vector<uint8_t>& nbrList) const
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrBlockBases.assign(1, 0);
		nbrList.clear();
		return;
	}

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

uint8_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::ChooseWidthCode(
	const uint32_t maxGap) noexcept
{
	if (maxGap <= 0xFFu) { return 0u; }
	if (maxGap <= 0xFFFFu) { return 1u; }
	return 2u;
}

uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::GetNumWidthBytes(
	const uint8_t widthCode) noexcept
{
	if (widthCode == 0u) { return 1u; }
	if (widthCode == 1u) { return 2u; }
	return 4u;
}

uint64_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::MeasureEncodedBytes(
	const uint32_t* js, const uint32_t count, uint8_t& widthCode)
{
	if (count == 0)
	{
		widthCode = 0u;
		return 0;
	}

	uint32_t maxGap = 0u;
	for (uint32_t k = 1; k < count; k++)
	{
		maxGap = std::max(maxGap, js[k] - js[k - 1]);
	}

	widthCode = ChooseWidthCode(maxGap);
	return 1ull + 4ull + static_cast<uint64_t>(count - 1u) * GetNumWidthBytes(widthCode);
}

inline uint16_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::ReadU16LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint16_t>(src[0])
		| (static_cast<uint16_t>(src[1]) << 8);
}

inline uint32_t NeighborLists_SingleCSRVector_AdaptCompress_CPU01::ReadU32LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint32_t>(src[0])
		| (static_cast<uint32_t>(src[1]) << 8)
		| (static_cast<uint32_t>(src[2]) << 16)
		| (static_cast<uint32_t>(src[3]) << 24);
}

inline void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::WriteU16LE(
	uint8_t* dst, const uint16_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
}

inline void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::WriteU32LE(
	uint8_t* dst, const uint32_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
	dst[2] = static_cast<uint8_t>(v >> 16);
	dst[3] = static_cast<uint8_t>(v >> 24);
}

template<typename T_Out>
void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::DecodeNeighborList(
	const uint8_t* src, const uint64_t srcBytes, const uint32_t count,
	T_Out& out)
{
	if (count == 0)
	{
		out.clear();
		return;
	}

	const uint8_t widthCode = src[0];

	#if defined(_DEBUG)
	const uint64_t expectedBytes =
		1ull + 4ull + static_cast<uint64_t>(count - 1u) * GetNumWidthBytes(widthCode);
	assert(srcBytes == expectedBytes && "Compressed list size mismatch");
	#endif

	EnsureDecodeScratchSize(out, count);
	uint32_t v = ReadU32LE(src + 1);
	out[0] = v;

	const uint8_t* p = src + 1 + 4;
	if (widthCode == 0u)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			v += static_cast<uint32_t>(*p++);
			out[k] = v;
		}
	}
	else if (widthCode == 1u)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			v += static_cast<uint32_t>(ReadU16LE(p));
			p += 2;
			out[k] = v;
		}
	}
	else
	{
		for (uint32_t k = 1; k < count; k++)
		{
			v += ReadU32LE(p);
			p += 4;
			out[k] = v;
		}
	}

	#if defined(_DEBUG)
	assert(p == src + srcBytes && "Decoded byte count mismatch");
	#endif
}

void NeighborLists_SingleCSRVector_AdaptCompress_CPU01::EncodeNeighborList(
	uint8_t* dst, const uint64_t dstBytes, const uint32_t* js,
	const uint32_t count, const uint8_t widthCode)
{
	if (count == 0)
	{
	#if defined(_DEBUG)
		assert(dstBytes == 0 && "Compressed list byte count mismatch for empty list");
	#endif
		return;
	}

	const uint32_t widthBytes = GetNumWidthBytes(widthCode);
	const uint64_t requiredBytes =
		1ull + 4ull + static_cast<uint64_t>(count - 1u) * widthBytes;
	if (requiredBytes != dstBytes)
	{
		throw std::runtime_error(
			"Measured and encoded adapt-compressed neighbor list sizes do not match.");
	}

	dst[0] = widthCode;
	WriteU32LE(dst + 1, js[0]);

	uint8_t* p = dst + 1 + 4;
	if (widthCode == 0u)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			*p++ = static_cast<uint8_t>(js[k] - js[k - 1]);
		}
	}
	else if (widthCode == 1u)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			WriteU16LE(p, static_cast<uint16_t>(js[k] - js[k - 1]));
			p += 2;
		}
	}
	else
	{
		for (uint32_t k = 1; k < count; k++)
		{
			WriteU32LE(p, js[k] - js[k - 1]);
			p += 4;
		}
	}

	#if defined(_DEBUG)
	assert(p == dst + dstBytes && "Encoded byte count mismatch");
	#endif
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

	// reserve packed CSR storage up front based on the expected bytes per compressed list
	const uint64_t expctdBytesFF = static_cast<uint64_t>(std::ceil(
		static_cast<double>(N_F_) *
		(static_cast<double>(overheadBytesPerNeighborList)
		 + expctdBytePerCompressedNeighbor
		 * static_cast<double>(params.expctdAvgNbrCountHardMax_FF))));
	const uint64_t expctdBytesFB = static_cast<uint64_t>(std::ceil(
		static_cast<double>(N_F_) *
		(static_cast<double>(overheadBytesPerNeighborList)
		 + expctdBytePerCompressedNeighbor
		 * static_cast<double>(params.expctdAvgNbrCountHardMax_FB))));
	EnsurePackedCSRStorageCapacity(nbrListCompressedFF_, expctdBytesFF);
	EnsurePackedCSRStorageCapacity(nbrListCompressedFB_, expctdBytesFB);
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
	const uint32_t* jsSorted = PrepareSortedNeighborList(js, count, inputOrder_, scratch);

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFF_[i] = count;
		const uint64_t encodedBytes = MeasureEncodedBytes(jsSorted, count);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrLocalOffsetsFF_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		#if defined(_DEBUG)
		if (nbrCountFF_[i] != count) [[unlikely]]
		{
			throw std::runtime_error(
				"Measured and encoded FF neighbor counts do not match in compressed single-CSR storage.");
		}
		#endif

		const uint64_t s = ParticleStorageBegin(i, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		const uint64_t e = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFF_, nbrBlockBasesFF_);
		uint8_t* dst = nbrListCompressedFF_.empty()
					 ? nullptr
					 : nbrListCompressedFF_.data() + s;

		EncodeNeighborList(dst, e - s, jsSorted, count);
		return;
	}

	throw std::runtime_error(
		"Compressed single-CSR neighbor lists received FF fill data in an invalid state.");
}

void NeighborLists_SingleCSRVector_VByteCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFB_.local();
	const uint32_t* jsSorted = PrepareSortedNeighborList(js, count, inputOrder_, scratch);

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFB_[i] = count;
		const uint64_t encodedBytes = MeasureEncodedBytes(jsSorted, count);
	#if defined(_DEBUG)
		assert(encodedBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
			&& "Per-particle compressed neighbor list exceeds uint32_t byte count");
	#endif
		nbrLocalOffsetsFB_[i] = static_cast<uint32_t>(encodedBytes);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		#if defined(_DEBUG)
		if (nbrCountFB_[i] != count)
		{
			throw std::runtime_error(
				"Measured and encoded FB neighbor counts do not match in compressed single-CSR storage.");
		}
		#endif

		const uint64_t s = ParticleStorageBegin(i, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		const uint64_t e = ParticleStorageEnd(i, N_F_, nbrLocalOffsetsFB_, nbrBlockBasesFB_);
		uint8_t* dst = nbrListCompressedFB_.empty()
					 ? nullptr
					 : nbrListCompressedFB_.data() + s;

		EncodeNeighborList(dst, e - s, jsSorted, count);
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
	const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		return 0;
	}

	const uint32_t ctrlBytes = ControlBytesFor(count);
	uint64_t dataBytes = 0ull;
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t gap = js[k] - js[k - 1];

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

template<typename T_Out>
void NeighborLists_SingleCSRVector_VByteCompress_CPU01::DecodeNeighborList(
	const uint8_t* src, const uint64_t srcBytes, const uint32_t count,
	T_Out& out)
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
		const auto& ctrlLutDbg = VByteControlLut();

		for (uint32_t idx = 0; idx < count - 1; idx++)
		{
			const uint8_t c = ctrlDbg[idx >> 2];
			expectedDataBytes += static_cast<uint64_t>(
				ctrlLutDbg[c].masks[idx & 3u] == 2u ? 1u
				: ctrlLutDbg[c].masks[idx & 3u] == 3u ? 4u
				: 0u);
		}

		const uint64_t expectedBytes = 4ull + static_cast<uint64_t>(ctrlBytes) + expectedDataBytes;
		assert(srcBytes == expectedBytes && "Compressed list size mismatch");
	}
	#endif

	EnsureDecodeScratchSize(out, count);

	// decode first element (uncompressed)
	uint32_t v = ReadU32LE(src);
	out[0] = v;
	if (count == 1) { return; }

	// control bytes after first element
	const uint32_t ctrlBytes = ControlBytesFor(count);
	const uint8_t* ctrl = src + 4;
	const uint8_t* data = ctrl + ctrlBytes;
	const auto& ctrlLut = VByteControlLut();

	// decode in groups of 4 deltas (one control byte), then undo with a local prefix sum
	uint32_t outIdx = 1u;
	uint32_t remaining = count - 1u;
	while (remaining > 0u)
	{
		const uint8_t c = *ctrl++;
		const auto& info = ctrlLut[c];
		const uint32_t laneCount = std::min(remaining, 4u);
		const uint8_t* groupData = data;
		data += info.payloadBytes;

		// shortcuts for common cases to avoid always decoding LUT info
		if (laneCount == 4u && c == 0x00u) // all 4 deltas are 0 -> control bits only
		{
			v += 1u; out[outIdx + 0u] = v;
			v += 1u; out[outIdx + 1u] = v;
			v += 1u; out[outIdx + 2u] = v;
			v += 1u; out[outIdx + 3u] = v;
		}
		else if (laneCount == 4u && c == 0x55u) // all 4 deltas are 1 -> control bits only
		{
			v += 2u; out[outIdx + 0u] = v;
			v += 2u; out[outIdx + 1u] = v;
			v += 2u; out[outIdx + 2u] = v;
			v += 2u; out[outIdx + 3u] = v;
		}
		else if (laneCount == 4u && c == 0xAAu) // all 4 deltas are [2,256) -> 1 data byte each
		{
			v += static_cast<uint32_t>(groupData[0]) + 1u; out[outIdx + 0u] = v;
			v += static_cast<uint32_t>(groupData[1]) + 1u; out[outIdx + 1u] = v;
			v += static_cast<uint32_t>(groupData[2]) + 1u; out[outIdx + 2u] = v;
			v += static_cast<uint32_t>(groupData[3]) + 1u; out[outIdx + 3u] = v;
		}
		else // general case with LUT info decoding and variable data byte counts
		{
			const uint8_t* p = groupData;

			for (uint32_t lane = 0; lane < laneCount; lane++)
			{
				uint32_t delta = 0u;
				switch (info.masks[lane])
				{
					case 0u: delta = 0u; break;
					case 1u: delta = 1u; break;
					case 2u: delta = static_cast<uint32_t>(*p++); break;
					default:
						delta = ReadU32LE(p);
						p += 4;
						break;
				}

				v += (delta + 1u);
				out[outIdx + lane] = v;
			}
		}

		outIdx += laneCount;
		remaining -= laneCount;
	}

	#if defined(_DEBUG)
	assert(data == src + srcBytes && "Decoded byte count mismatch");
	#endif
}

// TODO: implement with SIMD optimizations?
void NeighborLists_SingleCSRVector_VByteCompress_CPU01::EncodeNeighborList(
	uint8_t* dst, const uint64_t dstBytes, const uint32_t* js,
	const uint32_t count)
{
	if (count == 0)
	{
	#if defined(_DEBUG)
		assert(dstBytes == 0 && "Compressed list byte count mismatch for empty list");
	#endif
		return;
	}

	const uint32_t ctrlBytes = ControlBytesFor(count);
	#if defined(_DEBUG)
	const uint64_t requiredBytes = MeasureEncodedBytes(js, count);
	if (requiredBytes != dstBytes)
	{
		throw std::runtime_error(
			"Measured and encoded VByte-compressed neighbor list sizes do not match.");
	}
	#endif

	// encode first element (uncompressed)
	WriteU32LE(dst, js[0]);

	// control bytes after first element
	uint8_t* ctrl = dst + 4;
	std::memset(ctrl, 0, ctrlBytes);

	// data bytes after control bytes
	uint8_t* data = ctrl + ctrlBytes;

	// encode (count - 1) deltas
	for (uint32_t idx = 0; idx < count - 1; idx++)
	{
		const uint32_t k = idx + 1;

		const uint32_t gap = js[k] - js[k - 1];
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
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk]
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
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk]
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
		nbrCountFF_.empty() ? nullptr : nbrCountFF_.data(),
		chunk, begin, end,
		nbrChunkMaxCountFF_[chunk]
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
		nbrCountFB_.empty() ? nullptr : nbrCountFB_.data(),
		chunk, begin, end,
		nbrChunkMaxCountFB_[chunk]
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
