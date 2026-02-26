// neighbor_lists.cpp

// external
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>
#include "spdlog/spdlog.h"

// internal
#include "neighbor_lists.h"



using namespace Sim;

// -------------------------------------------------------------------------------------------------
// --- [ on-demand neighbor lists ] ----------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_OnDemand_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	(void)params;

	N_F_ = N_F;
	N_B_ = N_B;
}

NeighborSpan NeighborLists_OnDemand_CPU01::FF(
	const uint32_t i) const
{
	assert(neighborSearchStrat_ != nullptr);
	assert(i < N_F_);
	return neighborSearchStrat_->QueryParticleFF(i);
}

NeighborSpan NeighborLists_OnDemand_CPU01::FB(
	const uint32_t i) const
{
	assert(neighborSearchStrat_ != nullptr);
	assert(i < N_F_);
	return neighborSearchStrat_->QueryParticleFB(i);
}

uint32_t NeighborLists_OnDemand_CPU01::CountFF(
	const uint32_t i) const
{
	return FF(i).count;
}

uint32_t NeighborLists_OnDemand_CPU01::CountFB(
	const uint32_t i) const
{
	return FB(i).count;
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_OnDemand_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ individual vectors neighbor lists ] -------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_IndividualVectors_CPU01::Resize(
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

void NeighborLists_IndividualVectors_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrListFF_[i].clear();
				nbrListFF_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_IndividualVectors_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { nbrListFF_[i].clear(); return; }

	// overwrite and resize combined
	nbrListFF_[i].assign(js, js + count);
}

void NeighborLists_IndividualVectors_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { nbrListFB_[i].clear(); return; }

	// overwrite and resize combined
	nbrListFB_[i].assign(js, js + count);
}

NeighborSpan NeighborLists_IndividualVectors_CPU01::FF(
	const uint32_t i) const
{
	const auto& nbrs = nbrListFF_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

NeighborSpan NeighborLists_IndividualVectors_CPU01::FB(
	const uint32_t i) const
{
	const auto& nbrs = nbrListFB_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_IndividualVectors_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(nbrListFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(nbrListFB_));

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ CSR vector neighbor lists, final concat ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_CSRVector_FinalConcat_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// individual neighbor lists scratch
	nbrTempFF_.resize(N_F_);
	nbrTempFB_.resize(N_F_);

	// reserve expected capacities to reduce re-allocations
	// (single-threaded to minimize fragmentation and alloc contention)
	for (uint32_t i = 0; i < N_F_; i++)
	{
		nbrTempFF_[i].reserve(params.expctdNbrCountSoftMax_FF);
		nbrTempFB_[i].reserve(params.expctdNbrCountSoftMax_FB);

		nbrTempFF_[i].clear();
		nbrTempFB_[i].clear();
	}

	// CSR vector structure
	nbrOffsetsFF_.resize(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrOffsetsFB_.resize(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrListFF_.clear();
	nbrListFB_.clear();
}

void NeighborLists_CSRVector_FinalConcat_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrTempFF_[i].clear();
				nbrTempFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);

	// TODO: .assign() instead of only .resize() to make sure values are overwritten?
	nbrOffsetsFF_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrOffsetsFB_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrListFF_.clear();
	nbrListFB_.clear();
}

void NeighborLists_CSRVector_FinalConcat_CPU01::Finalize_Fill()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrOffsetsFF_.assign(1, 0);
		nbrOffsetsFB_.assign(1, 0);
		nbrListFF_.clear();
		nbrListFB_.clear();
		return;
	}

	// --- [ build CSR neighbor storage for for FF ] -----------------------------------------------

	nbrOffsetsFF_.resize(static_cast<uint64_t>(N_F_) + 1);

	// compute total size via parallel prefix sum
	const uint64_t N_totalFF = tbb::parallel_scan(
		tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (isFinal)
				{
					nbrOffsetsFF_[i] = sum;
				}
				sum += static_cast<uint64_t>(nbrTempFF_[i].size());
			}
			return sum;
		},
		[](const uint64_t a, const uint64_t b) { return a + b; }
	);

	// finalize total fluid neighbor count
	nbrOffsetsFF_[N_F_] = N_totalFF;
	if (nbrListFF_.capacity() < N_totalFF)
	{
		const uint64_t capTarget = N_totalFF
			+ Base::ExtraPercent(N_totalFF, extraAllocCapacityPct_);
		nbrListFF_.reserve(capTarget);
	}
	nbrListFF_.resize(N_totalFF);

	// concatenate individual lists into final CSR neighbor storage
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const auto& src = nbrTempFF_[i];
				const uint64_t base = nbrOffsetsFF_[i];
				const uint32_t count = static_cast<uint32_t>(src.size());

				if (count)
				{
					std::memcpy(nbrListFF_.data() + base, src.data(),
						count * sizeof(uint32_t));
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build CSR neighbor storage for for FB ] -----------------------------------------------

	nbrOffsetsFB_.resize(static_cast<uint64_t>(N_F_) + 1);

	// compute total size via parallel prefix sum
	const uint64_t N_totalFB = tbb::parallel_scan(
		tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (isFinal)
				{
					nbrOffsetsFB_[i] = sum;
				}
				sum += static_cast<uint64_t>(nbrTempFB_[i].size());
			}
			return sum;
		},
		[](const uint64_t a, const uint64_t b) { return a + b; }
	);

	// finalize total boundary neighbor count
	nbrOffsetsFB_[N_F_] = N_totalFB;
	if (nbrListFB_.capacity() < N_totalFB)
	{
		const uint64_t capTarget = N_totalFB
			+ Base::ExtraPercent(N_totalFB, extraAllocCapacityPct_);
		nbrListFB_.reserve(capTarget);
	}
	nbrListFB_.resize(N_totalFB);

	// concatenate individual lists into final CSR neighbor storage
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const auto& src = nbrTempFB_[i];
				const uint64_t base = nbrOffsetsFB_[i];
				const uint32_t count = static_cast<uint32_t>(src.size());

				if (count)
				{
					std::memcpy(nbrListFB_.data() + base, src.data(),
						count * sizeof(uint32_t));
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CSRVector_FinalConcat_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { nbrTempFF_[i].clear(); return; }

	// overwrite and resize combined
	nbrTempFF_[i].assign(js, js + count);
}

void NeighborLists_CSRVector_FinalConcat_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0) { nbrTempFB_[i].clear(); return; }

	// overwrite and resize combined
	nbrTempFB_[i].assign(js, js + count);
}

NeighborSpan NeighborLists_CSRVector_FinalConcat_CPU01::FF(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for fluid particle i
	const uint64_t s = nbrOffsetsFF_[i];
	const uint64_t e = nbrOffsetsFF_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ nbrListFF_.data() + s, count };
}

NeighborSpan NeighborLists_CSRVector_FinalConcat_CPU01::FB(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for boundary particle i
	const uint64_t s = nbrOffsetsFB_[i];
	const uint64_t e = nbrOffsetsFB_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ nbrListFB_.data() + s, count };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_CSRVector_FinalConcat_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrTempFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrTempFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListFB_));

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ CSR vector neighbor lists, double query ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_CSRVector_DoubleQuery_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrOffsetsFF_.assign(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrOffsetsFB_.assign(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrListFF_.clear();
	nbrListFB_.clear();
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::ClearAll()
{
	// TODO: .assign() instead of only .resize() to make sure values are overwritten?
	nbrOffsetsFF_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrOffsetsFB_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrListFF_.clear();
	nbrListFB_.clear();
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::Finalize_Count()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		nbrOffsetsFF_.assign(1, 0);
		nbrOffsetsFB_.assign(1, 0);
		nbrListFF_.clear();
		nbrListFB_.clear();
		return;
	}

	// turn temporary counts into actual offsets for FF
	const uint64_t N_totalFF = tbb::parallel_scan(
		tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// neighbor counts stored by count-query
				const uint64_t count = nbrOffsetsFF_[i];

				// now become CSR offset
				if (isFinal) { nbrOffsetsFF_[i] = sum; }

				sum += count;
			}
			return sum;
		},
		[](const uint64_t a, const uint64_t b) { return a + b; }
	);

	// finalize total fluid neighbor count
	nbrOffsetsFF_[N_F_] = N_totalFF;
	if (nbrListFF_.capacity() < N_totalFF)
	{
		const uint64_t capTarget = N_totalFF
			+ Base::ExtraPercent(N_totalFF, extraAllocCapacityPct_);
		nbrListFF_.reserve(capTarget);
	}
	nbrListFF_.resize(N_totalFF);

	// turn temporary counts into actual offsets for FB
	const uint64_t N_totalFB = tbb::parallel_scan(
		tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
		[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// neighbor counts stored by count-query
				const uint64_t count = nbrOffsetsFB_[i];

				// now become CSR offset
				if (isFinal) { nbrOffsetsFB_[i] = sum; }

				sum += count;
			}
			return sum;
		},
		[](const uint64_t a, const uint64_t b) { return a + b; }
	);

	// finalize total boundary neighbor count
	nbrOffsetsFB_[N_F_] = N_totalFB;
	if (nbrListFB_.capacity() < N_totalFB)
	{
		const uint64_t capTarget = N_totalFB
			+ Base::ExtraPercent(N_totalFB, extraAllocCapacityPct_);
		nbrListFB_.reserve(capTarget);
	}
	nbrListFB_.resize(N_totalFB);
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::SetNbrCountFF(
	const uint32_t i, const uint32_t count)
{
	nbrOffsetsFF_[i] = static_cast<uint64_t>(count);
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::SetNbrCountFB(
	const uint32_t i, const uint32_t count)
{
	nbrOffsetsFB_[i] = static_cast<uint64_t>(count);
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	const uint64_t s = nbrOffsetsFF_[i];

	if (count)
	{
		std::memcpy(nbrListFF_.data() + s, js, 
			count * sizeof(uint32_t));
	}
}

void NeighborLists_CSRVector_DoubleQuery_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	// (count-query and fill-query MUST find the same number of neighbors)
	const uint64_t s = nbrOffsetsFB_[i];

	if (count)
	{
		std::memcpy(nbrListFB_.data() + s, js,
			count * sizeof(uint32_t));
	}
}

NeighborSpan NeighborLists_CSRVector_DoubleQuery_CPU01::FF(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for fluid particle i
	const uint64_t s = nbrOffsetsFF_[i];
	const uint64_t e = nbrOffsetsFF_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ nbrListFF_.data() + s, count };
}

NeighborSpan NeighborLists_CSRVector_DoubleQuery_CPU01::FB(
	const uint32_t i) const
{
	// start and end offsets of the neighbor list for boundary particle i
	const uint64_t s = nbrOffsetsFB_[i];
	const uint64_t e = nbrOffsetsFB_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ nbrListFB_.data() + s, count };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_CSRVector_DoubleQuery_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListFB_));

	return { usageFF, usageFB };
}

// -------------------------------------------------------------------------------------------------
// --- [ individual vectors neighbor lists, adaptive compression ] ---------------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_IndividualVectors_AdaptCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	nbrListCompressedFF_.resize(N_F_);
	nbrListCompressedFB_.resize(N_F_);

	// reserve expected capacities to reduce re-allocations
	// (single-threaded to minimize fragmentation and alloc contention)
	for (uint32_t i = 0; i < N_F_; i++)
	{
		nbrListCompressedFF_[i].reserve(
			1 + 4 + params.expctdNbrCountSoftMax_FF * expctBytesPerNbrCeil_);
		nbrListCompressedFB_[i].reserve(
			1 + 4 + params.expctdNbrCountSoftMax_FB * expctBytesPerNbrCeil_);

		nbrListCompressedFF_[i].clear();
		nbrListCompressedFB_[i].clear();
	}
}

void NeighborLists_IndividualVectors_AdaptCompress_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;

				nbrListCompressedFF_[i].clear();
				nbrListCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_IndividualVectors_AdaptCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFF_[i] = count;

	auto& dst = nbrListCompressedFF_[i];
	auto& scratch = sortScratchFF_.local();

	EncodeNeighborList(dst, js, count, scratch);
}

void NeighborLists_IndividualVectors_AdaptCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFB_[i] = count;

	auto& dst = nbrListCompressedFB_[i];
	auto& scratch = sortScratchFB_.local();

	EncodeNeighborList(dst, js, count, scratch);
}

NeighborSpan NeighborLists_IndividualVectors_AdaptCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const auto& src = nbrListCompressedFF_[i];

	// decode into per-thread scratch
	auto& out = nbrScratchFF_.local();
	DecodeNeighborList(src, count, out);

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_IndividualVectors_AdaptCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const auto& src = nbrListCompressedFB_[i];

	// decode into per-thread scratch
	auto& out = nbrScratchFB_.local();
	DecodeNeighborList(src, count, out);

	return NeighborSpan{ out.data(), count };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_IndividualVectors_AdaptCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(nbrListCompressedFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(nbrListCompressedFB_));

	return { usageFF, usageFB };
}

uint8_t NeighborLists_IndividualVectors_AdaptCompress_CPU01::ChooseWidthCode(
	const uint32_t maxDelta) noexcept
{
	if (maxDelta <= 0xFFu) { return 0; }
	if (maxDelta <= 0xFFFFu) { return 1; }
	return 2;
}

uint32_t NeighborLists_IndividualVectors_AdaptCompress_CPU01::GetNumWidthBytes(
	const uint8_t widthCode) noexcept
{
	if (widthCode == 0) { return 1u; }
	if (widthCode == 1) { return 2u; }
	return 4u;
}

inline uint16_t NeighborLists_IndividualVectors_AdaptCompress_CPU01::ReadU16LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint16_t>(src[0])
		| (static_cast<uint16_t>(src[1]) << 8);
}

inline uint32_t NeighborLists_IndividualVectors_AdaptCompress_CPU01::ReadU32LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint32_t>(src[0])
		| (static_cast<uint32_t>(src[1]) << 8)
		| (static_cast<uint32_t>(src[2]) << 16)
		| (static_cast<uint32_t>(src[3]) << 24);
}

inline void NeighborLists_IndividualVectors_AdaptCompress_CPU01::WriteU16LE(
	uint8_t* dst, const uint16_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
}

inline void NeighborLists_IndividualVectors_AdaptCompress_CPU01::WriteU32LE(
	uint8_t* dst, const uint32_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
	dst[2] = static_cast<uint8_t>(v >> 16);
	dst[3] = static_cast<uint8_t>(v >> 24);
}

// TODO: implement with SIMD optimizations?
void NeighborLists_IndividualVectors_AdaptCompress_CPU01::DecodeNeighborList(
	const std::vector<uint8_t, uint8_tbb_alloc>& src, const uint32_t count,
	std::vector<uint32_t>& out)
{
	if (count == 0)
	{
		out.clear();
		return;
	}

	const uint8_t widthCode = src[0];

	// expected size check (debug only)
	#if defined(_DEBUG)
	const uint64_t expectedBytes =
		1ull + 4ull + static_cast<uint64_t>(count - 1) * GetNumWidthBytes(widthCode);
	assert(src.size() == expectedBytes && "Compressed list size mismatch");
	#endif

	out.resize(count);
	const uint8_t* p = src.data();

	// decode width and first element
	uint32_t v = ReadU32LE(p + 1);
	out[0] = v;

	// decode deltas of remaining elements relative to each other
	const uint8_t* q = p + 1 + 4;
	if (widthCode == 0)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = static_cast<uint32_t>(*q++);
			v += d;
			out[k] = v;
		}
	}
	else if (widthCode == 1)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = static_cast<uint32_t>(ReadU16LE(q));
			q += 2;
			v += d;
			out[k] = v;
		}
	}
	else
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = ReadU32LE(q);
			q += 4;
			v += d;
			out[k] = v;
		}
	}
}

// TODO: implement with SIMD optimizations?
void NeighborLists_IndividualVectors_AdaptCompress_CPU01::EncodeNeighborList(
	std::vector<uint8_t, uint8_tbb_alloc>& dst, const uint32_t* js,
	const uint32_t count, std::vector<uint32_t>& scratch)
{
	if (count == 0)
	{
		dst.clear();
		return;
	}

	// copy into thread-local scratch and sort to avoid negative deltas
	// TODO: guarantee sorted input from in the query to avoid sorting here?
	scratch.assign(js, js + count);
	std::sort(scratch.begin(), scratch.end());

	// compute max delta (needed to choose encoding width)
	uint32_t maxDelta = 0;
	if (count > 1)
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = scratch[k] - scratch[k - 1];
			maxDelta = std::max(d, maxDelta);
		}
	}

	const uint8_t widthCode = ChooseWidthCode(maxDelta);
	const uint32_t widthBytes = GetNumWidthBytes(widthCode);

	// determine size of encoded data
	const uint64_t requiredBytes = 1ull + 4ull + static_cast<uint64_t>(count - 1) * widthBytes;
	dst.resize(requiredBytes);
	uint8_t* p = dst.data();

	// encode width and first element
	p[0] = widthCode;
	WriteU32LE(p + 1, scratch[0]);

	// encode deltas of remaining elements relative to each other
	uint8_t* q = p + 1 + 4;
	if (widthCode == 0) // 1 byte
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = scratch[k] - scratch[k - 1];
			*q++ = static_cast<uint8_t>(d);
		}
	}
	else if (widthCode == 1) // 2 bytes
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = scratch[k] - scratch[k - 1];
			WriteU16LE(q, static_cast<uint16_t>(d));
			q += 2;
		}
	}
	else // 4 bytes
	{
		for (uint32_t k = 1; k < count; k++)
		{
			const uint32_t d = scratch[k] - scratch[k - 1];
			WriteU32LE(q, d);
			q += 4;
		}
	}

	#if defined(_DEBUG)
	// make sure pointer arithmetic matches the expected size
	assert(q == dst.data() + dst.size() && "Encoded byte count mismatch");
	#endif
}

// -------------------------------------------------------------------------------------------------
// --- [ individual vectors neighbor lists, VByteDerived compression ] -----------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	nbrListCompressedFF_.resize(N_F_);
	nbrListCompressedFB_.resize(N_F_);

	// reserve expected capacities to reduce re-allocations
	// (single-threaded to minimize fragmentation and alloc contention)
	const uint32_t ctrlF = ControlBytesFor(params.expctdNbrCountSoftMax_FF);
	const uint32_t ctrlB = ControlBytesFor(params.expctdNbrCountSoftMax_FB);
	for (uint32_t i = 0; i < N_F_; i++)
	{
		nbrListCompressedFF_[i].reserve(
			4 + ctrlF + params.expctdNbrCountSoftMax_FF * expctBytesPerNbrCeil_);
		nbrListCompressedFB_[i].reserve(
			4 + ctrlB + params.expctdNbrCountSoftMax_FB * expctBytesPerNbrCeil_);

		nbrListCompressedFF_[i].clear();
		nbrListCompressedFB_[i].clear();
	}
}

void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;

				nbrListCompressedFF_[i].clear();
				nbrListCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFF_[i] = count;

	auto& dst = nbrListCompressedFF_[i];
	auto& scratch = sortScratchFF_.local();

	EncodeNeighborList(dst, js, count, scratch);
}

void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	nbrCountFB_[i] = count;

	auto& dst = nbrListCompressedFB_[i];
	auto& scratch = sortScratchFB_.local();

	EncodeNeighborList(dst, js, count, scratch);
}

NeighborSpan NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const auto& src = nbrListCompressedFF_[i];

	// decode into per-thread scratch
	auto& out = nbrScratchFF_.local();
	DecodeNeighborList(src, count, out);

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const auto& src = nbrListCompressedFB_[i];

	// decode into per-thread scratch
	auto& out = nbrScratchFB_.local();
	DecodeNeighborList(src, count, out);

	return NeighborSpan{ out.data(), count };
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(nbrListCompressedFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(nbrListCompressedFB_));

	return { usageFF, usageFB };
}

uint32_t NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::ControlBytesFor(
	const uint32_t count) noexcept
{
	// there are (count - 1) deltas, and 4 masks fit into one control byte
	return (count <= 1) ? 0u : ((count - 1u + 3u) / 4u);
}

uint8_t NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::DeltaMask(
	const uint32_t delta) noexcept
{
	if (delta == 0u) { return 0u; } // 00 (-> 0 data bytes encoded via mask)
	if (delta == 1u) { return 1u; } // 01 (-> 0 data bytes, encoded via mask)
	if (delta < 256u) { return 2u; } // 10 (-> 1 data byte)
	return 3u; // 11 (-> 4 data bytes)
}

inline uint32_t NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::ReadU32LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint32_t>(src[0])
		| (static_cast<uint32_t>(src[1]) << 8)
		| (static_cast<uint32_t>(src[2]) << 16)
		| (static_cast<uint32_t>(src[3]) << 24);
}

inline void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::WriteU32LE(
	uint8_t* dst, const uint32_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
	dst[2] = static_cast<uint8_t>(v >> 16);
	dst[3] = static_cast<uint8_t>(v >> 24);
}

// TODO: implement with SIMD optimizations?
void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::DecodeNeighborList(
	const std::vector<uint8_t, uint8_tbb_alloc>& src, const uint32_t count,
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
		assert(src.size() >= 4 && "Compressed list too small");

		const uint32_t ctrlBytes = ControlBytesFor(count);
		assert(src.size() >= 4ull + ctrlBytes && "Compressed list too small for control stream");

		uint64_t expectedDataBytes = 0;
		const uint8_t* ctrlDbg = src.data() + 4;

		for (uint32_t idx = 0; idx < count - 1; idx++)
		{
			const uint8_t c = ctrlDbg[idx >> 2];
			const uint8_t mask = static_cast<uint8_t>((c >> ((idx & 3u) * 2u)) & 0x3u);

			if (mask == 2u) { expectedDataBytes += 1ull; }
			else if (mask == 3u) { expectedDataBytes += 4ull; }
			// mask 0/1 => 0 bytes
		}

		const uint64_t expectedBytes = 4ull + static_cast<uint64_t>(ctrlBytes) + expectedDataBytes;
		assert(src.size() == expectedBytes && "Compressed list size mismatch");
	}
	#endif

	out.resize(count);
	const uint8_t* p = src.data();

	// decode first element (uncompressed)
	uint32_t v = ReadU32LE(p);
	out[0] = v;
	if (count == 1) { return; }

	// control bytes after first element
	const uint32_t ctrlBytes = ControlBytesFor(count);
	const uint8_t* ctrl = p + 4;
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
	assert(data == src.data() + src.size() && "Decoded byte count mismatch");
	#endif
}

// TODO: implement with SIMD optimizations?
void NeighborLists_IndividualVectors_VByteDerivedCompress_CPU01::EncodeNeighborList(
	std::vector<uint8_t, uint8_tbb_alloc>& dst, const uint32_t* js,
	const uint32_t count, std::vector<uint32_t>& scratch)
{
	if (count == 0)
	{
		dst.clear();
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
	dst.resize(requiredBytes);
	uint8_t* p = dst.data();

	// encode first element (uncompressed)
	WriteU32LE(p, scratch[0]);

	// control bytes after first element
	uint8_t* ctrl = p + 4;
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
	assert(data == dst.data() + dst.size() && "Encoded byte count mismatch");
	#endif
}

/*

// -------------------------------------------------------------------------------------------------
// --- [ compact vectors neighbor lists (ULEB128 compression of deltas) ] --------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// temporary 2D vector structure
	tmpCompressedFF_.resize(N_F_);
	tmpCompressedFB_.resize(N_F_);

	// reserve space for each temporary compressed neighbor list vector
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (tmpCompressedFF_[i].capacity() < N_COMPRESSED_NBR_SIZE_FF_)
				{
					tmpCompressedFF_[i].reserve(N_COMPRESSED_NBR_SIZE_FF_);
				}

				if (tmpCompressedFB_[i].capacity() < N_COMPRESSED_NBR_SIZE_FB_)
				{
					tmpCompressedFB_[i].reserve(N_COMPRESSED_NBR_SIZE_FB_);
				}

				// ensure lists are empty but keep capacity
				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	// compact vector structure (bytes)
	offsetsFF_.resize(N_F_ + 1, 0);
	offsetsFB_.resize(N_F_ + 1, 0);
	compressedNeighborsFF_.clear();
	compressedNeighborsFB_.clear();
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;

				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::Finalize()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		offsetsFF_.assign(1, 0);
		offsetsFB_.assign(1, 0);
		compressedNeighborsFF_.clear();
		compressedNeighborsFB_.clear();

		return;
	}

	// --- [ build compact vector for compressed FF ] ----------------------------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFF_[0] = 0;
	uint64_t runningFF = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFF_[i] = runningFF;
		runningFF += static_cast<uint64_t>(tmpCompressedFF_[i].size());
	}
	offsetsFF_[N_F_] = runningFF;

	// allocate flat neighbor storage
	compressedNeighborsFF_.resize(runningFF);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFF_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFF_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFF_.data() + base, 
						tmpCompressedFF_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build compact vector for compressed FB ] ----------------------------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFB_[0] = 0;
	uint64_t runningFB = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFB_[i] = runningFB;
		runningFB += static_cast<uint64_t>(tmpCompressedFB_[i].size());
	}
	offsetsFB_[N_F_] = runningFB;

	// allocate flat neighbor storage
	compressedNeighborsFB_.resize(runningFB);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFB_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFB_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFB_.data() + base, 
						tmpCompressedFB_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFF_[i] = 0;
		tmpCompressedFF_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFF_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFF_[i] = newCount;

	EncodeNeighborList_ULEB128(scratch.data(), newCount, tmpCompressedFF_[i]);
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFB_[i] = 0;
		tmpCompressedFB_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFB_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFB_[i] = newCount;

	EncodeNeighborList_ULEB128(scratch.data(), newCount, tmpCompressedFB_[i]);
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFF_.local();
	if (scratch.capacity() < N_expctMaxNbrCountF) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountF);
	}
	const uint8_t* data = compressedNeighborsFF_.data() + s;
	DecodeNeighborList_ULEB128(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

NeighborSpan NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFB_.local();
	if (scratch.capacity() < N_expctMaxNbrCountB) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountB);
	}
	const uint8_t* data = compressedNeighborsFB_.data() + s;
	DecodeNeighborList_ULEB128(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

std::pair<Mem::MemUsage, Mem::MemUsage> NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// tmp per-particle compressed vectors
	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(tmpCompressedFF_));
	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(tmpCompressedFB_));

	// compressed storage
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(offsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(compressedNeighborsFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(offsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(compressedNeighborsFB_));

	// thread-local decode scratch
	for (auto const& v : nbrScratchFF_)
	{
		Mem::Add(usageFF, Mem::MemUsageOf_Vector(v));
	}
	for (auto const& v : nbrScratchFB_)
	{
		Mem::Add(usageFB, Mem::MemUsageOf_Vector(v));
	}

	// thread-local sort scratch
	for (auto const& v : sortScratchFF_)
	{
		Mem::Add(usageFF, Mem::MemUsageOf_Vector(v));
	}
	for (auto const& v : sortScratchFB_)
	{
		Mem::Add(usageFB, Mem::MemUsageOf_Vector(v));
	}

	return { usageFF, usageFB };
}

inline void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::PutU32(
	uint32_t v, uint8_t*& dst) noexcept
{
	// write 7 bits per byte, most significant bit of 1 means more bytes follow
	// (LEB128 encoding)
	while (v >= 0x80u)
	{
		*dst++ = static_cast<uint8_t>((v & 0x7Fu) | 0x80u);
		v >>= 7;
	}
	*dst++ = static_cast<uint8_t>(v);
}

inline uint32_t NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::GetU32(
	const uint8_t*& src, const uint8_t* end) noexcept
{
	uint32_t result = 0;
	uint32_t shift = 0;

	// read until continuation bit is 0 (or max 5 bytes)
	for (int i = 0; i < 5 && src < end; i++)
	{
		uint32_t byte = *src++;

		result |= (byte & 0x7Fu) << shift;
		if ((byte & 0x80u) == 0) break;

		shift += 7;
	}

	return result;
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::EncodeNeighborList_ULEB128(
	const uint32_t* js, const uint32_t count, std::vector<uint8_t>& out)
{
	if (count == 0) { out.clear(); return; }

	// max number of bytes for LEB128 encoding of uint32_t is 5
	const uint32_t maxBytes = count * 5u;

	out.resize(maxBytes);
	uint8_t* dst = out.data();

	// encode deltas with LEB128 to compress indices
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t curr = js[k];
		const uint32_t delta = curr - prev;
		PutU32(delta, dst);
		prev = curr;
	}

	// shrink to actual bytes written
	out.resize(static_cast<size_t>(dst - out.data()));
}

void NeighborLists_CompactVectors_Compressed_ULEB128_CPU01::DecodeNeighborList_ULEB128(
	const uint8_t* data, const uint32_t count, uint32_t byteCount, std::vector<uint32_t>& out)
{
	out.resize(count);

	const uint8_t* src = data;
	const uint8_t* end = data + byteCount;

	// decode deltas with LEB128 to reconstruct original indices
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; k++)
	{
		const uint32_t delta = GetU32(src, end);
		const uint32_t curr = prev + delta;

		out[k] = curr;
		prev = curr;
	}
}

// -------------------------------------------------------------------------------------------------
// --- [ compact vectors neighbor lists (Stream VByte delta compression) ] -------------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// temporary 2D vector structure
	tmpCompressedFF_.resize(N_F_);
	tmpCompressedFB_.resize(N_F_);

	// reserve space for each temporary compressed neighbor list vector
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				if (tmpCompressedFF_[i].capacity() < N_COMPRESSED_NBR_SIZE_FF_)
				{
					tmpCompressedFF_[i].reserve(N_COMPRESSED_NBR_SIZE_FF_);
				}

				if (tmpCompressedFB_[i].capacity() < N_COMPRESSED_NBR_SIZE_FB_)
				{
					tmpCompressedFB_[i].reserve(N_COMPRESSED_NBR_SIZE_FB_);
				}

				// ensure lists are empty but keep capacity
				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	// compact vector structure (bytes)
	offsetsFF_.resize(N_F_ + 1, 0);
	offsetsFB_.resize(N_F_ + 1, 0);
	compressedNeighborsFF_.clear();
	compressedNeighborsFB_.clear();
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;

				tmpCompressedFF_[i].clear();
				tmpCompressedFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::Finalize()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		offsetsFF_.assign(1, 0);
		offsetsFB_.assign(1, 0);
		compressedNeighborsFF_.clear();
		compressedNeighborsFB_.clear();

		return;
	}

	// --- [ build compact vector for compressed FF ] ----------------------------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFF_[0] = 0;
	uint64_t runningFF = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFF_[i] = runningFF;
		runningFF += static_cast<uint64_t>(tmpCompressedFF_[i].size());
	}
	offsetsFF_[N_F_] = runningFF;

	// allocate flat neighbor storage
	compressedNeighborsFF_.resize(runningFF);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFF_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFF_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFF_.data() + base,
						tmpCompressedFF_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build compact vector for compressed FB ] ----------------------------------------------

	// build offsets via prefix sum of byte sizes
	offsetsFB_[0] = 0;
	uint64_t runningFB = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFB_[i] = runningFB;
		runningFB += static_cast<uint64_t>(tmpCompressedFB_[i].size());
	}
	offsetsFB_[N_F_] = runningFB;

	// allocate flat neighbor storage
	compressedNeighborsFB_.resize(runningFB);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t base = offsetsFB_[i];
				const uint32_t byteCount = static_cast<uint32_t>(tmpCompressedFB_[i].size());

				if (byteCount)
				{
					std::memcpy(compressedNeighborsFB_.data() + base,
						tmpCompressedFB_[i].data(), byteCount);
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFF_[i] = 0;
		tmpCompressedFF_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFF_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFF_[i] = newCount;

	EncodeNeighborList_StreamVByteDelta(scratch.data(), newCount, 
		tmpCompressedFF_[i]);
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	if (count == 0)
	{
		nbrCountFB_[i] = 0;
		tmpCompressedFB_[i].clear();
		return;
	}

	// copy to thread-local scratch and sort to avoid encoding of negative deltas
	auto& scratch = sortScratchFB_.local();
	if (scratch.capacity() < count) { scratch.reserve(count); }
	scratch.assign(js, js + count);

	std::sort(scratch.begin(), scratch.end());

	const uint32_t newCount = static_cast<uint32_t>(scratch.size());
	nbrCountFB_[i] = newCount;

	EncodeNeighborList_StreamVByteDelta(scratch.data(), newCount, 
		tmpCompressedFB_[i]);
}

NeighborSpan NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFF_.local();
	if (scratch.capacity() < N_expctMaxNbrCountF) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountF);
	}
	const uint8_t* data = compressedNeighborsFF_.data() + s;
	DecodeNeighborList_StreamVByteDelta(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

NeighborSpan NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0) { return NeighborSpan{ nullptr, 0 }; }

	// start and end offsets of the compressed neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t byteCount = static_cast<uint32_t>(e - s);

	// // decode into provided per-thread scratch vector
	auto& scratch = nbrScratchFB_.local();
	if (scratch.capacity() < N_expctMaxNbrCountB) // reserve "once" on first call
	{
		scratch.reserve(N_expctMaxNbrCountB);
	}
	const uint8_t* data = compressedNeighborsFB_.data() + s;
	DecodeNeighborList_StreamVByteDelta(data, count, byteCount, scratch);

	return NeighborSpan{ scratch.data(), count };
}

std::pair<Mem::MemUsage, Mem::MemUsage> NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	// tmp per-particle compressed vectors
	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(tmpCompressedFF_));
	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(tmpCompressedFB_));

	// compressed storage
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(offsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(compressedNeighborsFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(offsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(compressedNeighborsFB_));

	// thread-local decode scratch
	for (auto const& v : nbrScratchFF_)
	{
		Mem::Add(usageFF, Mem::MemUsageOf_Vector(v));
	}
	for (auto const& v : nbrScratchFB_)
	{
		Mem::Add(usageFB, Mem::MemUsageOf_Vector(v));
	}

	// thread-local sort scratch
	for (auto const& v : sortScratchFF_)
	{
		Mem::Add(usageFF, Mem::MemUsageOf_Vector(v));
	}
	for (auto const& v : sortScratchFB_)
	{
		Mem::Add(usageFB, Mem::MemUsageOf_Vector(v));
	}

	return { usageFF, usageFB };
}

inline void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::PutU32(
	uint32_t v, uint8_t*& dst) noexcept
{
	*dst++ = static_cast<uint8_t>(v);
	*dst++ = static_cast<uint8_t>(v >> 8);
	*dst++ = static_cast<uint8_t>(v >> 16);
	*dst++ = static_cast<uint8_t>(v >> 24);
}

inline uint32_t NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::GetU32(
	const uint8_t*& src, const uint8_t* end) noexcept
{
	if ((end - src) < 4)
	{
		src = end;
		return 0;
	}

	const uint32_t b0 = src[0];
	const uint32_t b1 = src[1];
	const uint32_t b2 = src[2];
	const uint32_t b3 = src[3];
	src += 4;

	return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static inline uint32_t ControlByteCount_ForNeighborCount(uint32_t count) noexcept
{
	// first neighbor is stored uncompressed, remaining (count-1) values use 2 control bits each
	if (count <= 1) return 0;
	const uint32_t deltas = count - 1;
	return (deltas + 3u) / 4u; // 4 deltas per control byte
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::EncodeNeighborList_StreamVByteDelta(
	const uint32_t* js, const uint32_t count, std::vector<uint8_t>& out)
{
	if (count == 0) { out.clear(); return; }

	const uint32_t ctrlBytes = ControlByteCount_ForNeighborCount(count);
	const uint32_t deltas = (count > 0) ? (count - 1) : 0;

	// worst case size with 4 bytes for n1, ctrlBytes for control bytes, 4 bytes per delta
	const uint32_t maxBytes = static_cast<uint32_t>(4ull + ctrlBytes + 4ull * deltas);

	out.resize(maxBytes);
	uint8_t* base = out.data();
	uint8_t* dst = base;

	// store first neighbor uncompressed
	PutU32(js[0], dst);

	// reserve and zero control byte stream
	uint8_t* ctrl = dst;
	if (ctrlBytes)
	{
		std::memset(ctrl, 0, ctrlBytes);
	}
	dst += ctrlBytes;

	// dst now points to the beginning of the data byte stream
	uint8_t* data = dst;

	// encode deltas as reduced deltas with control bits + data bytes
	uint32_t prev = js[0];
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t curr = js[k];

		// reduced delta mapping: rDelta = curr - prev - 1 (>= 0)
		const uint32_t diff = curr - prev;
		const uint32_t rDelta = diff - 1u;

		// choose control code + emit bytes
		uint8_t code = 0;
		if (rDelta == 0u)
		{
			code = 0; // 0b00
		}
		else if (rDelta == 1u)
		{
			code = 1; // 0b01
		}
		else if (rDelta < 256u)
		{
			code = 2; // 0b10
			*data++ = static_cast<uint8_t>(rDelta);
		}
		else
		{
			code = 3; // 0b11
			PutU32(rDelta, data);
		}

		const uint32_t d = k - 1; // rDelta index [0..count-2]
		const uint32_t cIdx = d >> 2; // 4 rDeltas per control byte
		const uint32_t shift = (d & 3u) * 2u; // 2 bits per rDelta
		ctrl[cIdx] = static_cast<uint8_t>(ctrl[cIdx] | (code << shift));

		prev = curr;
	}

	// shrink to actual bytes written
	out.resize(static_cast<size_t>(data - base));
}

void NeighborLists_CompactVectors_Compressed_StreamVByteDelta_CPU01::DecodeNeighborList_StreamVByteDelta(
	const uint8_t* data, const uint32_t count, uint32_t byteCount, std::vector<uint32_t>& out)
{
	out.resize(count);
	if (count == 0) { return; }

	const uint8_t* src = data;
	const uint8_t* end = data + byteCount;

	const uint32_t ctrlBytes = ControlByteCount_ForNeighborCount(count);

	// read first neighbor uncompressed
	const uint32_t n1 = GetU32(src, end);
	out[0] = n1;

	// src now points to the data byte stream
	const uint8_t* ctrl = src;
	src += ctrlBytes;

	uint32_t prev = n1;

	// decode reduced deltas from control stream + data stream
	for (uint32_t k = 1; k < count; k++)
	{
		const uint32_t d = k - 1; // rDelta index [0..count-2]
		const uint32_t cIdx = d >> 2; // 4 rDeltas per control byte
		const uint32_t shift = (d & 3u) * 2u; // 2 bits per rDelta
		const uint32_t code = (ctrlBytes ? ((ctrl[cIdx] >> shift) & 0x3u) : 0u);

		uint32_t rDelta = 0;
		if (code == 0u)
		{
			rDelta = 0u;
		}
		else if (code == 1u)
		{
			rDelta = 1u;
		}
		else if (code == 2u)
		{
			if (src >= end) { rDelta = 0u; }
			else { rDelta = static_cast<uint32_t>(*src++); }
		}
		else // code == 3
		{
			rDelta = GetU32(src, end);
		}

		// reconstruct particle index
		const uint32_t curr = prev + rDelta + 1u;

		out[k] = curr;
		prev = curr;
	}
}
*/
