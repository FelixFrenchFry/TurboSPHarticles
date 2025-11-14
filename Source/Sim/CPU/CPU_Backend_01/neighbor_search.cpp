// neighbor_search.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <execution>
#include <numeric>
#include <omp.h>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/partitioner.h>
#include <vector>

// internal
#include "cpu_backend_01.h"
#include "glm/common.hpp"
#include "neighbor_search.h"



using namespace Sim;

// -----------------------------------------------------------------------------
// --- [ brute force implementation ] ------------------------------------------
// -----------------------------------------------------------------------------

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

		// cell ID of particle i
		const int cxi = FloorToInt(xi * cellSizeInv_);
		const int cyi = FloorToInt(yi * cellSizeInv_);
		const int czi = FloorToInt(zi * cellSizeInv_);

		// check all particles j in 27 neighboring grid cells
		for (int rz = -1; rz <= 1; rz++)
		{
			for (int ry = -1; ry <= 1; ry++)
			{
				for (int rx = -1; rx <= 1; rx++)
				{
					// current cell key
					const uint64_t key = PackKey(cxi + rx,
												 cyi + ry,
												 czi + rz);

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

inline int NS_SpatialHashing_Basic_CPU01::FloorToInt(float v) const noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}

// -----------------------------------------------------------------------------
// --- [ cell linked list basic implementation ] -------------------------------
// -----------------------------------------------------------------------------

void NS_CellLinkedList_Basic_CPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;

	xBoxOrig_ = 0.0f;
	yBoxOrig_ = 0.0f;
	zBoxOrig_ = 0.0f;

	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
	N_Cells_ = 0;

	cellStartIdxF_.clear();
	cellEndIdxF_.clear();
	cellIdF_.clear();
	sortedParticleIdxF_.clear();

	cellStartIdxB_.clear();
	cellEndIdxB_.clear();
	cellIdB_.clear();
	sortedParticleIdxB_.clear();
}

void NS_CellLinkedList_Basic_CPU01::Construct(const SimParams& params)
{
	if (be_.N_F_ == 0 && be_.N_B_ == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");

		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		N_Cells_ = 0;

		return;
	}

	cellSize_ = params.kernelRadius;
	cellSizeInv_ = 1.0f / params.kernelRadius;

	// --- [ combined AABB over fluid and boundary particles ] -----------------

	// accumulators
	float initMin = std::numeric_limits<float>::max();
	float initMax = std::numeric_limits<float>::lowest();
	float xMin = initMin, yMin = initMin, zMin = initMin;
	float xMax = initMax, yMax = initMax, zMax = initMax;

	// fluid
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		const float x = be_.xPosF_[i];
		const float y = be_.yPosF_[i];
		const float z = be_.zPosF_[i];

		xMin = x < xMin ? x : xMin;
		yMin = y < yMin ? y : yMin;
		zMin = z < zMin ? z : zMin;
		xMax = x > xMax ? x : xMax;
		yMax = y > yMax ? y : yMax;
		zMax = z > zMax ? z : zMax;
	}

	// boundary
	for (uint32_t i = 0; i < be_.N_B_; i++)
	{
		const float x = be_.xPosB_[i];
		const float y = be_.yPosB_[i];
		const float z = be_.zPosB_[i];

		xMin = x < xMin ? x : xMin;
		yMin = y < yMin ? y : yMin;
		zMin = z < zMin ? z : zMin;
		xMax = x > xMax ? x : xMax;
		yMax = y > yMax ? y : yMax;
		zMax = z > zMax ? z : zMax;
	}

	// padding to avoid edge clamp issues
	const float pad = 1e-6f;
	xMin -= pad; yMin -= pad; zMin -= pad;
	xMax += pad; yMax += pad; zMax += pad;

	xBoxOrig_ = xMin;
	yBoxOrig_ = yMin;
	zBoxOrig_ = zMin;

	const float xLen = std::max(0.0f, xMax - xMin);
	const float yLen = std::max(0.0f, yMax - yMin);
	const float zLen = std::max(0.0f, zMax - zMin);

	// compute grid dimensions
	N_DimX_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(xLen * cellSizeInv_)));
	N_DimY_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(yLen * cellSizeInv_)));
	N_DimZ_ = std::max<uint32_t>(1, 
		static_cast<uint32_t>(std::ceil(zLen * cellSizeInv_)));
	
	// overflow protection
	const uint64_t N_DimX = static_cast<uint64_t>(N_DimX_);
	const uint64_t N_DimY = static_cast<uint64_t>(N_DimY_);
	const uint64_t N_DimZ = static_cast<uint64_t>(N_DimZ_);
	const uint64_t N_Cells = N_DimX * N_DimY * N_DimZ;
	if (N_Cells > std::numeric_limits<uint32_t>::max())
	{
		SPDLOG_ERROR("Too many grid cells for uint32_t ({})", N_Cells);
	}
	else
	{
		N_Cells_ = static_cast<uint32_t>(N_Cells);
	}

	// --- [ preparation data structures ] -------------------------------------

	// fluid
	cellStartIdxF_.assign(N_Cells_, EMPTY);
	cellEndIdxF_.assign(N_Cells_, EMPTY);

	cellIdF_.resize(be_.N_F_);
	sortedParticleIdxF_.resize(be_.N_F_);

	// TODO: parallelize this?
	// #pragma omp parallel for schedule(static)
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		// compute cell coordinates
		int cx = FloorToInt((be_.xPosF_[i] - xBoxOrig_) * cellSizeInv_);
		int cy = FloorToInt((be_.yPosF_[i] - yBoxOrig_) * cellSizeInv_);
		int cz = FloorToInt((be_.zPosF_[i] - zBoxOrig_) * cellSizeInv_);

		// clamp to grid dimensions
		uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
		uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
		uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

		cellIdF_[i] = MapToCellId_Basic(ux, uy, uz);
		sortedParticleIdxF_[i] = i;
	}

	// boundary
	cellStartIdxB_.assign(N_Cells_, EMPTY);
	cellEndIdxB_.assign(N_Cells_, EMPTY);

	cellIdB_.resize(be_.N_B_);
	sortedParticleIdxB_.resize(be_.N_B_);

	// TODO: parallelize this?
	// #pragma omp parallel for schedule(static)
	for (uint32_t i = 0; i < be_.N_B_; i++)
	{
		// compute cell coordinates
		int cx = FloorToInt((be_.xPosB_[i] - xBoxOrig_) * cellSizeInv_);
		int cy = FloorToInt((be_.yPosB_[i] - yBoxOrig_) * cellSizeInv_);
		int cz = FloorToInt((be_.zPosB_[i] - zBoxOrig_) * cellSizeInv_);

		// clamp to grid dimensions
		uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
		uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
		uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

		cellIdB_[i] = MapToCellId_Basic(ux, uy, uz);
		sortedParticleIdxB_[i] = i;
	}

	// --- [ sort particle indices by cell IDs ] -------------------------------

	auto ByCellIdPredF = [&](uint32_t a, uint32_t b)
		{ return cellIdF_[a] < cellIdF_[b]; };

	auto ByCellIdPredB = [&](uint32_t a, uint32_t b)
		{ return cellIdB_[a] < cellIdB_[b]; };

	// single-threaded basic sorting version
	/*
	std::ranges::sort(sortedParticleIdxF_, ByCellIdPredF);

	std::ranges::sort(sortedParticleIdxB_, ByCellIdPredB);
	*/

	// multi-threaded parallel sorting version using C++ standard library
	/*
	std::sort(std::execution::par, sortedParticleIdxF_.begin(), 
		sortedParticleIdxF_.end(), ByCellIdPredF);

	std::sort(std::execution::par, sortedParticleIdxB_.begin(),
		sortedParticleIdxB_.end(), ByCellIdPredB);
	*/

	// multi-threaded parallel sorting version using TBB
	tbb::parallel_sort(sortedParticleIdxF_.begin(), sortedParticleIdxF_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdF_[a] < cellIdF_[b]; });

	tbb::parallel_sort(sortedParticleIdxB_.begin(), sortedParticleIdxB_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdB_[a] < cellIdB_[b]; });

	// --- [ fill in cell start/end indices ] ----------------------------------

	// fluid
	{
		uint32_t prevCell = EMPTY;
		for (uint32_t i = 0; i < be_.N_F_; i++)
		{
			const uint32_t pidx = sortedParticleIdxF_[i];
			const uint32_t cid = cellIdF_[pidx];

			if (cid != prevCell) // new cell encountered
			{
				if (prevCell != EMPTY)
				{
					// set end index of previous cell
					cellEndIdxF_[prevCell] = i;
				}

				if (cellStartIdxF_[cid] == EMPTY)
				{
					// set start index of current cell
					cellStartIdxF_[cid] = i;
				}

				prevCell = cid;
			}
		}

		if (prevCell != EMPTY)
		{
			// set end index of last cell to close it
			cellEndIdxF_[prevCell] = be_.N_F_;
		}
	}

	// boundary
	{
		uint32_t prevCell = EMPTY;
		for (uint32_t i = 0; i < be_.N_B_; i++)
		{
			const uint32_t pidx = sortedParticleIdxB_[i];
			const uint32_t cid = cellIdB_[pidx];

			if (cid != prevCell) // new cell encountered
			{
				if (prevCell != EMPTY)
				{
					// set end index of previous cell
					cellEndIdxB_[prevCell] = i;
				}

				if (cellStartIdxB_[cid] == EMPTY)
				{
					// set start index of current cell
					cellStartIdxB_[cid] = i;
				}

				prevCell = cid;
			}
		}

		if (prevCell != EMPTY)
		{
			// set end index of last cell to close it
			cellEndIdxB_[prevCell] = be_.N_B_;
		}
	}
}

void NS_CellLinkedList_Basic_CPU01::Query(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	#pragma omp for schedule (guided)
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

		// cell ID of particle i
		const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
		const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
		const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

		// check all particles j in 27 neighboring grid cells
		for (int rz = -1; rz <= 1; rz++)
		{
			int cz = czi + rz; // cell coordinate in z-direction
			if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

			for (int ry = -1; ry <= 1; ry++)
			{
				int cy = cyi + ry; // cell coordinate in y-direction
				if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

				for (int rx = -1; rx <= 1; rx++)
				{
					int cx = cxi + rx; // cell coordinate in x-direction
					if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

					// cell ID
					const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
														   static_cast<uint32_t>(cy),
														   static_cast<uint32_t>(cz));

					// fluid particles
					const uint32_t sF = cellStartIdxF_[cid];
					const uint32_t eF = cellEndIdxF_[cid];
					if (sF != EMPTY && eF != EMPTY)
					{
						// check all neighbor candidates in the cell
						for (uint32_t k = sF; k < eF; k++)
						{
							// index of neighbor candidate
							const uint32_t j = sortedParticleIdxF_[k];

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
					const uint32_t sB = cellStartIdxB_[cid];
					const uint32_t eB = cellEndIdxB_[cid];
					if (sB != EMPTY && eB != EMPTY)
					{
						// check all neighbor candidates in the cell
						for (uint32_t k = sB; k < eB; k++)
						{
							// index of neighbor candidate
							const uint32_t j = sortedParticleIdxB_[k];

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

inline uint32_t NS_CellLinkedList_Basic_CPU01::MapToCellId_Basic(
	uint32_t cx, uint32_t cy, uint32_t cz) const noexcept
{
	return cx + N_DimX_ * (cy + N_DimY_ * cz);
}

inline int NS_CellLinkedList_Basic_CPU01::FloorToInt(
	float v) const noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}

inline uint32_t NS_CellLinkedList_Basic_CPU01::ClampInt(
	int v, int low, int high) const noexcept
{
	return static_cast<uint32_t>(v < low ? low : (v > high ? high : v));
}
