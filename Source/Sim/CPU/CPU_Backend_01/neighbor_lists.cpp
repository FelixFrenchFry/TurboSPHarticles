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

void NeighborLists_SmallVector_CPU01::Resize(uint32_t N_F, uint32_t N_B)
{
    N_F_ = N_F;
    N_B_ = N_B;

    neighborsFF_.resize(N_F_);
    neighborsFB_.resize(N_F_);
}

void NeighborLists_SmallVector_CPU01::ClearOne(uint32_t i)
{
    neighborsFF_[i].clear();
    neighborsFB_[i].clear();
}

void NeighborLists_SmallVector_CPU01::ClearAll()
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

void NeighborLists_SmallVector_CPU01::Finalize()
{
	// nothing to do
}

void NeighborLists_SmallVector_CPU01::AddNbrFF(uint32_t i, uint32_t j)
{
	neighborsFF_[i].push_back(j);
}

void NeighborLists_SmallVector_CPU01::AddNbrFB(uint32_t i, uint32_t j)
{
	neighborsFB_[i].push_back(j);
}

NeighborSpan NeighborLists_SmallVector_CPU01::FF(uint32_t i) const
{
	const auto& nbrs = neighborsFF_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}

NeighborSpan NeighborLists_SmallVector_CPU01::FB(uint32_t i) const
{
	const auto& nbrs = neighborsFB_[i];
	return NeighborSpan{ nbrs.data(), static_cast<uint32_t>(nbrs.size()) };
}
