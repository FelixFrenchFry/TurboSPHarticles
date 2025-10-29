// neighbor_search.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <omp.h>
#include <vector>

// internal
#include "neighbor_search.h"



namespace Sim
{
	// -------------------------------------------------------------------------
	// --- [ brute force implementation ] --------------------------------------
	// -------------------------------------------------------------------------

	BruteForce::BruteForce(Bench::BenchmarksMgr& benchmarksMgr)
		: benchmarksMgr_{ benchmarksMgr }
	{
		// nothing to do here
	}

	void BruteForce::Init()
	{
		// nothing to do here
	}

	void BruteForce::Construct(const NeighborSearchContext& ctx)
	{
		// nothing to do here
	}

	void BruteForce::Query(const NeighborSearchContext& ctx, 
						   const NeighborSearchResult& out)
	{
		#pragma omp for schedule(static)
		for (uint32_t i = 0; i < ctx.N_F; i++)
		{
			// neighbor lists for particle i
			// (delete contents, but keep capacity)
			auto& ff = out.neighborsFF[i];
			auto& fb = out.neighborsFB[i];
			ff.clear();
			fb.clear();

			const float xi = ctx.xPosF[i];
			const float yi = ctx.yPosF[i];
			const float zi = ctx.zPosF[i];

			// check all fluid particles j
			for (uint32_t j = 0; j < ctx.N_F; j++)
			{
				const float dx = xi - ctx.xPosF[j];
				const float dy = yi - ctx.yPosF[j];
				const float dz = zi - ctx.zPosF[j];

				// squared distance between particles i and j
				const float distSqu = dx * dx + dy * dy + dz * dz;

				if (distSqu <= ctx.kernelRadiusSqu) [[unlikely]]
				{
					// add actual neighbor to list
					ff.push_back(j);
				}
			}

			// check all boundary particles j
			for (uint32_t j = 0; j < ctx.N_B; j++)
			{
				const float dx = xi - ctx.xPosB[j];
				const float dy = yi - ctx.yPosB[j];
				const float dz = zi - ctx.zPosB[j];

				// squared distance between particles i and j
				const float distSqu = dx * dx + dy * dy + dz * dz;

				if (distSqu <= ctx.kernelRadiusSqu) [[unlikely]]
				{
					// add actual neighbor to list
					fb.push_back(j);
				}
			}
		}
	}

	// -------------------------------------------------------------------------
	// --- [ spatial hashing implementation ] ----------------------------------
	// -------------------------------------------------------------------------

	SpatialHashing_Simple::SpatialHashing_Simple(Bench::BenchmarksMgr& benchmarksMgr)
		: benchmarksMgr_{ benchmarksMgr }
	{
		// nothing to do here
	}

	void SpatialHashing_Simple::Init()
	{
		cellGridF_.clear();
		cellGridB_.clear();
		cellSize_ = 0.0f;
		cellSizeInv_ = 0.0f;
	}

	void SpatialHashing_Simple::Construct(const NeighborSearchContext& ctx)
	{
		cellSize_ = ctx.kernelRadius;
		cellSizeInv_ = 1.0f / ctx.kernelRadius;

		// clear hash tables completely, including per-cell vectors
		cellGridF_.clear();
		cellGridB_.clear();

		// reserve memory for cells, assuming average of ~4 particles per cell
		if (ctx.N_F)
		{
			cellGridF_.reserve(std::max<uint32_t>(1024, ctx.N_F / 4));
		}
		if (ctx.N_B)
		{
			cellGridB_.reserve(std::max<uint32_t>(1024, ctx.N_B / 4));
		}

		// TODO: multi-thread this?
		// fill in fluid cell grid
		for (uint32_t i = 0; i < ctx.N_F; i++)
		{
			// compute cell index
			const int cx = FloorToInt(ctx.xPosF[i] * cellSizeInv_);
			const int cy = FloorToInt(ctx.yPosF[i] * cellSizeInv_);
			const int cz = FloorToInt(ctx.zPosF[i] * cellSizeInv_);

			// convert cell index into key and insert particle index
			const Key key = PackKey(cx, cy, cz);
			cellGridF_[key].push_back(i);
		}

		// TODO: multi-thread this?
		// fill in boundary cell grid
		for (uint32_t i = 0; i < ctx.N_B; i++)
		{
			// compute cell index
			const int cx = FloorToInt(ctx.xPosB[i] * cellSizeInv_);
			const int cy = FloorToInt(ctx.yPosB[i] * cellSizeInv_);
			const int cz = FloorToInt(ctx.zPosB[i] * cellSizeInv_);

			// convert cell index into key and insert particle index
			const Key key = PackKey(cx, cy, cz);
			cellGridB_[key].push_back(i);
		}
	}

	void SpatialHashing_Simple::Query(const NeighborSearchContext& ctx,
									  const NeighborSearchResult& out)
	{
		const float kernelRadiusSqu = ctx.kernelRadiusSqu;

		#pragma omp for schedule(guided)
		for (uint32_t i = 0; i < ctx.N_F; i++)
		{
			// neighbor lists for particle i
			// (delete contents, but keep capacity)
			auto& ff = out.neighborsFF[i];
			auto& fb = out.neighborsFB[i];
			ff.clear();
			fb.clear();

			const float xi = ctx.xPosF[i];
			const float yi = ctx.yPosF[i];
			const float zi = ctx.zPosF[i];

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
								const float dx = xi - ctx.xPosF[j];
								const float dy = yi - ctx.yPosF[j];
								const float dz = zi - ctx.zPosF[j];

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
								const float dx = xi - ctx.xPosB[j];
								const float dy = yi - ctx.yPosB[j];
								const float dz = zi - ctx.zPosB[j];

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

	inline int SpatialHashing_Simple::FloorToInt(float v) noexcept
	{
		int i = static_cast<int>(v);
		
		return (v < static_cast<float>(i)) ? (i - 1) : i;
	}

	// bijective mapping of signed to unsigned integers
	// (-1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...)
	inline uint32_t SpatialHashing_Simple::ZigZag(int v) noexcept
	{
		// shift left by 1 and XOR with sign bit
		std::uint32_t uv = static_cast<std::uint32_t>(v);
		std::uint32_t even = uv << 1;
		std::uint32_t sign = static_cast<std::uint32_t>(-(v < 0)); // 0 or 0xFFFFFFFF

		return even ^ sign;
	}

	inline uint64_t SpatialHashing_Simple::PackKey(int cx, int cy, int cz) noexcept
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

} // namespace Sim
