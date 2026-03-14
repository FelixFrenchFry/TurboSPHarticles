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
#include "spdlog/spdlog.h"

// internal
#include "neighbor_lists.h"



using namespace Sim;

namespace
{
	size_t ToSizeT_Checked(const uint64_t value)
	{
		if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
		{
			throw std::runtime_error(
				"Neighbor list size exceeds addressable range on this platform.");
		}

		return static_cast<size_t>(value);
	}

	uint64_t ExpectedPackedCSRCapacity(
		const uint32_t N_F,
		const uint32_t expctdAvgNbrCountHardMax)
	{
		return static_cast<uint64_t>(N_F)
			* static_cast<uint64_t>(expctdAvgNbrCountHardMax);
	}

	template<typename T>
	void EnsurePackedCSRStorageCapacity(
		std::vector<T>& nbrList,
		const uint64_t count)
	{
		if (nbrList.capacity() < count)
		{
			nbrList.reserve(ToSizeT_Checked(count));
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
			const uint64_t capTarget = count
				+ Base::ExtraPercent(count, extraAllocCapacityPct);
			nbrList.reserve(ToSizeT_Checked(capTarget));
		}
	}
}

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

	// reserve the packed CSR storage up front to avoid a late full re-allocation
	// of the large final neighbor buffers during long or near-capacity runs
	EnsurePackedCSRStorageCapacity(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
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
	EnsurePackedCSRStorageCapacity(nbrListFF_, N_totalFF, extraAllocCapacityPct_);
	nbrListFF_.resize(ToSizeT_Checked(N_totalFF));

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
	EnsurePackedCSRStorageCapacity(nbrListFB_, N_totalFB, extraAllocCapacityPct_);
	nbrListFB_.resize(ToSizeT_Checked(N_totalFB));

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

	Mem::Add(usageFF, Mem::MemUsageOf_VectorOfVector(nbrTempFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_VectorOfVector(nbrTempFB_));
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

	// reserve packed CSR storage up front to avoid a late full re-allocation
	// of large final neighbor buffers during long or near-capacity runs
	EnsurePackedCSRStorageCapacity(nbrListFF_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FF));
	EnsurePackedCSRStorageCapacity(nbrListFB_,
		ExpectedPackedCSRCapacity(N_F_, params.expctdAvgNbrCountHardMax_FB));
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
	EnsurePackedCSRStorageCapacity(nbrListFF_, N_totalFF, extraAllocCapacityPct_);
	nbrListFF_.resize(ToSizeT_Checked(N_totalFF));

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
	EnsurePackedCSRStorageCapacity(nbrListFB_, N_totalFB, extraAllocCapacityPct_);
	nbrListFB_.resize(ToSizeT_Checked(N_totalFB));
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

// -------------------------------------------------------------------------------------------------
// --- [ CSR vector neighbor lists, double query, VByteDerived compression ] -----------------------
// -------------------------------------------------------------------------------------------------

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::Resize(
	const uint32_t N_F, const uint32_t N_B, const SimParams& params)
{
	(void)params;

	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);
	nbrOffsetsFF_.assign(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrOffsetsFB_.assign(static_cast<uint64_t>(N_F_) + 1, 0);
	nbrListCompressedFF_.clear();
	nbrListCompressedFB_.clear();
	fillPhase_ = FillPhase::Measure;
}

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::ClearAll()
{
	nbrCountFF_.resize(N_F_);
	nbrCountFB_.resize(N_F_);
	nbrOffsetsFF_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrOffsetsFB_.resize(static_cast<uint64_t>(N_F_) + 1);
	nbrListCompressedFF_.clear();
	nbrListCompressedFB_.clear();
	fillPhase_ = FillPhase::Measure;
}

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::Finalize_Count()
{
	ClearAll();
}

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::Finalize_Fill()
{
	if (fillPhase_ == FillPhase::Measure)
	{
		if (N_F_ == 0) // nothing to do for empty sim
		{
			nbrOffsetsFF_.assign(1, 0);
			nbrOffsetsFB_.assign(1, 0);
			nbrListCompressedFF_.clear();
			nbrListCompressedFB_.clear();
			fillPhase_ = FillPhase::Encode;
			return;
		}

		// turn temporary byte counts into actual offsets for FF
		const uint64_t N_totalFF = tbb::parallel_scan(
			tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
			[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compressed byte counts stored by first fill-query
					const uint64_t count = nbrOffsetsFF_[i];

					// now become CSR byte offsets
					if (isFinal) { nbrOffsetsFF_[i] = sum; }

					sum += count;
				}
				return sum;
			},
			[](const uint64_t a, const uint64_t b) { return a + b; }
		);

		// finalize total compressed byte count for fluid neighbors
		nbrOffsetsFF_[N_F_] = N_totalFF;
		EnsurePackedCSRStorageCapacity(nbrListCompressedFF_, N_totalFF,
			extraAllocCapacityPct_);
		nbrListCompressedFF_.resize(ToSizeT_Checked(N_totalFF));

		// turn temporary byte counts into actual offsets for FB
		const uint64_t N_totalFB = tbb::parallel_scan(
			tbb::blocked_range<uint32_t>(0, N_F_), uint64_t{ 0 },
			[&](const tbb::blocked_range<uint32_t>& r, uint64_t sum, const bool isFinal)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compressed byte counts stored by first fill-query
					const uint64_t count = nbrOffsetsFB_[i];

					// now become CSR byte offsets
					if (isFinal) { nbrOffsetsFB_[i] = sum; }

					sum += count;
				}
				return sum;
			},
			[](const uint64_t a, const uint64_t b) { return a + b; }
		);

		// finalize total compressed byte count for boundary neighbors
		nbrOffsetsFB_[N_F_] = N_totalFB;
		EnsurePackedCSRStorageCapacity(nbrListCompressedFB_, N_totalFB,
			extraAllocCapacityPct_);
		nbrListCompressedFB_.resize(ToSizeT_Checked(N_totalFB));

		fillPhase_ = FillPhase::Encode;
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		fillPhase_ = FillPhase::Ready;
		return;
	}

	throw std::runtime_error(
		"Unexpected fill-phase transition in compressed double-query neighbor lists.");
}

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::SetNbrListFF(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFF_.local();

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFF_[i] = count;
		nbrOffsetsFF_[i] = MeasureEncodedBytes(js, count, scratch);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		if (nbrCountFF_[i] != count)
		{
			throw std::runtime_error(
				"Measured and encoded FF neighbor counts do not match in compressed double-query storage.");
		}

		const uint64_t s = nbrOffsetsFF_[i];
		const uint64_t e = nbrOffsetsFF_[i + 1];
		uint8_t* dst = nbrListCompressedFF_.empty()
			? nullptr
			: nbrListCompressedFF_.data() + s;

		EncodeNeighborList(dst, e - s, js, count, scratch);
		return;
	}

	throw std::runtime_error(
		"Compressed double-query neighbor lists received FF fill data in an invalid state.");
}

void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::SetNbrListFB(
	const uint32_t i, const uint32_t* js, const uint32_t count)
{
	auto& scratch = sortScratchFB_.local();

	if (fillPhase_ == FillPhase::Measure)
	{
		nbrCountFB_[i] = count;
		nbrOffsetsFB_[i] = MeasureEncodedBytes(js, count, scratch);
		return;
	}

	if (fillPhase_ == FillPhase::Encode)
	{
		if (nbrCountFB_[i] != count)
		{
			throw std::runtime_error(
				"Measured and encoded FB neighbor counts do not match in compressed double-query storage.");
		}

		const uint64_t s = nbrOffsetsFB_[i];
		const uint64_t e = nbrOffsetsFB_[i + 1];
		uint8_t* dst = nbrListCompressedFB_.empty()
			? nullptr
			: nbrListCompressedFB_.data() + s;

		EncodeNeighborList(dst, e - s, js, count, scratch);
		return;
	}

	throw std::runtime_error(
		"Compressed double-query neighbor lists received FB fill data in an invalid state.");
}

NeighborSpan NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::FF(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFF_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const uint64_t s = nbrOffsetsFF_[i];
	const uint64_t e = nbrOffsetsFF_[i + 1];
	const uint8_t* src = nbrListCompressedFF_.data() + s;

	// decode into per-thread scratch
	auto& out = nbrScratchFF_.local();
	DecodeNeighborList(src, e - s, count, out);

	return NeighborSpan{ out.data(), count };
}

NeighborSpan NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::FB(
	const uint32_t i) const
{
	const uint32_t count = nbrCountFB_[i];
	if (count == 0)
	{
		return NeighborSpan{ nullptr, 0 };
	}

	const uint64_t s = nbrOffsetsFB_[i];
	const uint64_t e = nbrOffsetsFB_[i + 1];
	const uint8_t* src = nbrListCompressedFB_.data() + s;

	// decode into per-thread scratch
	auto& out = nbrScratchFB_.local();
	DecodeNeighborList(src, e - s, count, out);

	return NeighborSpan{ out.data(), count };
}

uint32_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::CountFF(
	const uint32_t i) const
{
	return nbrCountFF_[i];
}

uint32_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::CountFB(
	const uint32_t i) const
{
	return nbrCountFB_[i];
}

std::pair<Mem::MemUsage, Mem::MemUsage>
NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageFF{};
	Mem::MemUsage usageFB{};

	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrCountFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrOffsetsFF_));
	Mem::Add(usageFF, Mem::MemUsageOf_Vector(nbrListCompressedFF_));

	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrCountFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrOffsetsFB_));
	Mem::Add(usageFB, Mem::MemUsageOf_Vector(nbrListCompressedFB_));

	return { usageFF, usageFB };
}

uint32_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::ControlBytesFor(
	const uint32_t count) noexcept
{
	// there are (count - 1) deltas, and 4 masks fit into one control byte
	return (count <= 1) ? 0u : ((count - 1u + 3u) / 4u);
}

uint8_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::DeltaMask(
	const uint32_t delta) noexcept
{
	if (delta == 0u) { return 0u; } // 00 (-> 0 data bytes encoded via mask)
	if (delta == 1u) { return 1u; } // 01 (-> 0 data bytes, encoded via mask)
	if (delta < 256u) { return 2u; } // 10 (-> 1 data byte)
	return 3u; // 11 (-> 4 data bytes)
}

uint64_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::MeasureEncodedBytes(
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

inline uint32_t NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::ReadU32LE(
	const uint8_t* src) noexcept
{
	return static_cast<uint32_t>(src[0])
		| (static_cast<uint32_t>(src[1]) << 8)
		| (static_cast<uint32_t>(src[2]) << 16)
		| (static_cast<uint32_t>(src[3]) << 24);
}

inline void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::WriteU32LE(
	uint8_t* dst, const uint32_t v) noexcept
{
	dst[0] = static_cast<uint8_t>(v);
	dst[1] = static_cast<uint8_t>(v >> 8);
	dst[2] = static_cast<uint8_t>(v >> 16);
	dst[3] = static_cast<uint8_t>(v >> 24);
}

// TODO: implement with SIMD optimizations?
void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::DecodeNeighborList(
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
void NeighborLists_CSRVector_DoubleQuery_VByteDerivedCompress_CPU01::EncodeNeighborList(
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
			"Measured and encoded VByte-derived neighbor list sizes do not match.");
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
