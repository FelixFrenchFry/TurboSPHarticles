// neighbor_search.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <omp.h>
#include <vector>

// internal
#include "neighbor_search.h"
#include "cpu_backend_01.h"



using namespace Sim;

// -----------------------------------------------------------------------------
// --- [ brute force implementation ] ------------------------------------------
// -----------------------------------------------------------------------------

NS_BruteForce_CPU01::NS_BruteForce_CPU01(SimBackend_CPU01& backend)
	: be_{ backend }
{
	// nothing
}

void NS_BruteForce_CPU01::Init(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_CPU01::Construct(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_CPU01::Query(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	#pragma omp for schedule(static)
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		// neighbor lists for particle i
		// (delete contents, but keep capacity)
		std::vector<uint32_t>& ff = be_.neighborsFF_[i];
		std::vector<uint32_t>& fb = be_.neighborsFB_[i];
		ff.clear();
		fb.clear();

		const float xi = be_.xPosF_[i];
		const float yi = be_.yPosF_[i];
		const float zi = be_.zPosF_[i];

		// check all fluid particles j
		for (uint32_t j = 0; j < be_.N_F_; j++)
		{
			const float dx = xi - be_.xPosF_[j];
			const float dy = yi - be_.yPosF_[j];
			const float dz = zi - be_.zPosF_[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= kernelRadiusSqu) [[unlikely]]
			{
				// add actual neighbor to list
				ff.push_back(j);
			}
		}

		// check all boundary particles j
		for (uint32_t j = 0; j < be_.N_B_; j++)
		{
			const float dx = xi - be_.xPosB_[j];
			const float dy = yi - be_.yPosB_[j];
			const float dz = zi - be_.zPosB_[j];

			// squared distance between particles i and j
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= kernelRadiusSqu) [[unlikely]]
			{
				// add actual neighbor to list
				fb.push_back(j);
			}
		}
	}
}

// -----------------------------------------------------------------------------
// --- [ spatial hashing implementation ] --------------------------------------
// -----------------------------------------------------------------------------

NS_SpatialHashing_Basic_CPU01::NS_SpatialHashing_Basic_CPU01(SimBackend_CPU01& backend)
	: be_{ backend }
{
	// nothing
}

void NS_SpatialHashing_Basic_CPU01::Init(const SimParams& params)
{
	cellGridF_.clear();
	cellGridB_.clear();
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
}

void NS_SpatialHashing_Basic_CPU01::Construct(const SimParams& params)
{
	cellSize_ = params.kernelRadius;
	cellSizeInv_ = 1.0f / params.kernelRadius;

	// clear hash tables completely, including per-cell vectors
	cellGridF_.clear();
	cellGridB_.clear();

	// reserve memory for cells, assuming average of ~4 particles per cell
	if (be_.N_F_)
	{
		cellGridF_.reserve(std::max<uint32_t>(1024, be_.N_F_ / 4));
	}
	if (be_.N_B_)
	{
		cellGridB_.reserve(std::max<uint32_t>(1024, be_.N_B_ / 4));
	}

	// TODO: multi-thread this?
	// fill in fluid cell grid
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		// compute cell index
		const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
		const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
		const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

		// convert cell index into key and insert particle index
		const Key key = PackKey(cx, cy, cz);
		cellGridF_[key].push_back(i);
	}

	// TODO: multi-thread this?
	// fill in boundary cell grid
	for (uint32_t i = 0; i < be_.N_B_; i++)
	{
		// compute cell index
		const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
		const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
		const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

		// convert cell index into key and insert particle index
		const Key key = PackKey(cx, cy, cz);
		cellGridB_[key].push_back(i);
	}
}

void NS_SpatialHashing_Basic_CPU01::Query(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	#pragma omp for schedule(guided)
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		// neighbor lists for particle i
		// (delete contents, but keep capacity)
		std::vector<uint32_t>& ff = be_.neighborsFF_[i];
		std::vector<uint32_t>& fb = be_.neighborsFB_[i];
		ff.clear();
		fb.clear();

		const float xi = be_.xPosF_[i];
		const float yi = be_.yPosF_[i];
		const float zi = be_.zPosF_[i];

		const int cxi = FloorToInt(xi * cellSizeInv_);
		const int cyi = FloorToInt(yi * cellSizeInv_);
		const int czi = FloorToInt(zi * cellSizeInv_);

		// check all particles j in 27 neighboring grid cells
		for (int dz = -1; dz <= 1; dz++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				for (int dx = -1; dx <= 1; dx++)
				{
					// current cell key
					const uint64_t key = PackKey(cxi + dx,
												 cyi + dy,
												 czi + dz);

					// fluid particles
					if (auto bucket = cellGridF_.find(key);
						bucket != cellGridF_.end())
					{
						// check all neighbor candidates in the cell bucket
						for (const uint32_t j : bucket->second)
						{
							const float dx = xi - be_.xPosF_[j];
							const float dy = yi - be_.yPosF_[j];
							const float dz = zi - be_.zPosF_[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// add actual neighbor to list
								ff.push_back(j);
							}
						}
					}

					// boundary particles
					if (auto bucket = cellGridB_.find(key);
						bucket != cellGridB_.end())
					{
						// check all neighbor candidates in the cell bucket
						for (const uint32_t j : bucket->second)
						{
							const float dx = xi - be_.xPosB_[j];
							const float dy = yi - be_.yPosB_[j];
							const float dz = zi - be_.zPosB_[j];

							// squared distance between particles i and j
							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// add actual neighbor to list
								fb.push_back(j);
							}
						}
					}
				}
			}
		}
	}
}

inline int NS_SpatialHashing_Basic_CPU01::FloorToInt(float v) noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}

// bijective mapping of signed to unsigned integers
// (-1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...)
inline uint32_t NS_SpatialHashing_Basic_CPU01::ZigZag(int v) noexcept
{
	// shift left by 1 and XOR with sign bit
	std::uint32_t uv = static_cast<std::uint32_t>(v);
	std::uint32_t even = uv << 1;
	std::uint32_t sign = static_cast<std::uint32_t>(-(v < 0)); // 0 or 0xFFFFFFFF

	return even ^ sign;
}

inline uint64_t NS_SpatialHashing_Basic_CPU01::PackKey(int cx, int cy, int cz) noexcept
{
	// apply zig-zag encoding
	const uint64_t x = ZigZag(cx);
	const uint64_t y = ZigZag(cy);
	const uint64_t z = ZigZag(cz);

	// guard against overflow
	assert(x < (1u << 21) && y < (1u << 21) && z < (1u << 21));

	// pack into 64-bit key
	return (x) | (y << 21) | (z << 42);
}
