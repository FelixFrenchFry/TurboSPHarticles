// neighbor_lists.cpp

// external
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

// internal
#include "neighbor_lists.h"



using namespace Sim;

// -----------------------------------------------------------------------------
// --- [ small vector neighbor lists implementation ] --------------------------
// -----------------------------------------------------------------------------

void NeighborLists_SmallVectors_CPU01::Resize(uint32_t N_F, uint32_t N_B)
{
    N_F_ = N_F;
    N_B_ = N_B;

    neighborsFF_.resize(N_F_);
    neighborsFB_.resize(N_F_);
}

void NeighborLists_SmallVectors_CPU01::ClearOne(uint32_t i)
{
    neighborsFF_[i].clear();
    neighborsFB_[i].clear();
}

void NeighborLists_SmallVectors_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
                neighborsFF_[i].clear();
                neighborsFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_SmallVectors_CPU01::Finalize()
{
	// nothing to do
}

void NeighborLists_SmallVectors_CPU01::AddNbrFF(uint32_t i, uint32_t j)
{
	neighborsFF_[i].push_back(j);
}

void NeighborLists_SmallVectors_CPU01::AddNbrFB(uint32_t i, uint32_t j)
{
	neighborsFB_[i].push_back(j);
}

NeighborSpan NeighborLists_SmallVectors_CPU01::FF(uint32_t i) const
{
	const auto& nbrs = neighborsFF_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

NeighborSpan NeighborLists_SmallVectors_CPU01::FB(uint32_t i) const
{
	const auto& nbrs = neighborsFB_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

// -----------------------------------------------------------------------------
// --- [ fixed array neighbor lists implementation ] ---------------------------
// -----------------------------------------------------------------------------

void NeighborLists_FixedArrays_CPU01::Resize(uint32_t N_F, uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	nbrCountFF_.assign(N_F_, 0);
	nbrCountFB_.assign(N_F_, 0);

	neighborsFF_.resize(static_cast<size_t>(N_F_) * N_MAX_FF_);
	neighborsFB_.resize(static_cast<size_t>(N_F_) * N_MAX_FB_);
}

void NeighborLists_FixedArrays_CPU01::ClearOne(uint32_t i)
{
	nbrCountFF_[i] = 0;
	nbrCountFB_[i] = 0;
}

void NeighborLists_FixedArrays_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				nbrCountFF_[i] = 0;
				nbrCountFB_[i] = 0;
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_FixedArrays_CPU01::Finalize()
{
	// nothing to do
}

void NeighborLists_FixedArrays_CPU01::AddNbrFF(uint32_t i, uint32_t j)
{
	uint32_t& count = nbrCountFF_[i];

	if (count >= N_MAX_FF_) // neighbor list full, ignore new neighbor
	{
		return;
	}

	const uint64_t base = static_cast<size_t>(i) * N_MAX_FF_;
	neighborsFF_[base + count] = j;
	count++;
}

void NeighborLists_FixedArrays_CPU01::AddNbrFB(uint32_t i, uint32_t j)
{
	uint32_t& count = nbrCountFB_[i];

	if (count >= N_MAX_FB_) // neighbor list full, ignore new neighbor
	{
		return;
	}

	const uint64_t base = static_cast<size_t>(i) * N_MAX_FB_;
	neighborsFB_[base + count] = j;
	count++;
}

NeighborSpan NeighborLists_FixedArrays_CPU01::FF(uint32_t i) const
{
	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FF_;
	const uint32_t counter = nbrCountFF_[i];

	// data pointer to the start of the neighbor list for particle i and its size
	return NeighborSpan{ neighborsFF_.data() + base, counter};
}

NeighborSpan NeighborLists_FixedArrays_CPU01::FB(uint32_t i) const
{
	const uint64_t base = static_cast<uint64_t>(i) * N_MAX_FB_;
	const uint32_t counter = nbrCountFB_[i];

	// data pointer to the start of the neighbor list for particle i and its size
	return NeighborSpan{ neighborsFB_.data() + base, counter };
}

// -----------------------------------------------------------------------------
// --- [ compact vector neighbor lists implementation ] ------------------------
// -----------------------------------------------------------------------------

void NeighborLists_CompactVector_CPU01::Resize(uint32_t N_F, uint32_t N_B)
{
	N_F_ = N_F;
	N_B_ = N_B;

	// temporary 2D vector structure
	tmpFF_.clear();
	tmpFB_.clear();
	tmpFF_.resize(N_F_);
	tmpFB_.resize(N_F_);

	// compact vector structure
	offsetsFF_.resize(N_F_ + 1, 0);
	offsetsFB_.resize(N_F_ + 1, 0);
	neighborsFF_.clear();
	neighborsFB_.clear();
}

void NeighborLists_CompactVector_CPU01::ClearOne(uint32_t i)
{
	tmpFF_[i].clear();
	tmpFB_[i].clear();
}

void NeighborLists_CompactVector_CPU01::ClearAll()
{
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				tmpFF_[i].clear();
				tmpFB_[i].clear();
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVector_CPU01::Finalize()
{
	if (N_F_ == 0) // nothing to do for empty sim
	{
		offsetsFF_.assign(1, 0);
		offsetsFB_.assign(1, 0);
		neighborsFF_.clear();
		neighborsFB_.clear();
		
		return;
	}

	// --- [ build compact vector for FF ] -------------------------------------

	// build offsets via prefix sum
	offsetsFF_[0] = 0;
	uint64_t runningFF = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFF_[i] = runningFF;
		runningFF += static_cast<uint64_t>(tmpFF_[i].size());
	}
	offsetsFF_[N_F_] = runningFF;

	// allocate flat neighbor storage
	neighborsFF_.resize(runningFF);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t s = offsetsFF_[i];
				const uint32_t count = static_cast<uint32_t>(tmpFF_[i].size());

				for (uint32_t k = 0; k < count; k++)
				{
					neighborsFF_[s + k] = tmpFF_[i][k];
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ build compact vector for FB ] -------------------------------------

	// build offsets via prefix sum
	offsetsFB_[0] = 0;
	uint64_t runningFB = 0;
	for (uint32_t i = 0; i < N_F_; i++)
	{
		offsetsFB_[i] = runningFB;
		runningFB += static_cast<uint64_t>(tmpFB_[i].size());
	}
	offsetsFB_[N_F_] = runningFB;

	// allocate flat neighbor storage
	neighborsFB_.resize(runningFB);

	// copy data from temp into final data structure
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const uint64_t s = offsetsFB_[i];
				const uint32_t count = static_cast<uint32_t>(tmpFB_[i].size());

				for (uint32_t k = 0; k < count; k++)
				{
					neighborsFB_[s + k] = tmpFB_[i][k];
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NeighborLists_CompactVector_CPU01::AddNbrFF(uint32_t i, uint32_t j)
{
	tmpFF_[i].push_back(j);
}

void NeighborLists_CompactVector_CPU01::AddNbrFB(uint32_t i, uint32_t j)
{
	tmpFB_[i].push_back(j);
}

NeighborSpan NeighborLists_CompactVector_CPU01::FF(uint32_t i) const
{
	// start and end offsets of the neighbor list for fluid particle i
	const uint64_t s = offsetsFF_[i];
	const uint64_t e = offsetsFF_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ neighborsFF_.data() + s, count };
}

NeighborSpan NeighborLists_CompactVector_CPU01::FB(uint32_t i) const
{
	// start and end offsets of the neighbor list for boundary particle i
	const uint64_t s = offsetsFB_[i];
	const uint64_t e = offsetsFB_[i + 1];
	const uint32_t count = static_cast<uint32_t>(e - s);

	return NeighborSpan{ neighborsFB_.data() + s, count };
}
