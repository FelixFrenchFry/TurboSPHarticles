// neighbor_search.cpp

// external
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <execution>
#include <numeric>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>
#include <vector>

// internal
#include "cpu_backend_02.h"
#include "neighbor_search.h"
#include "Source/Base/helpers.h"

#if defined(_MSC_VER)
	#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
	#define RESTRICT __restrict__
#else
	#define RESTRICT
#endif



using namespace Sim;

// -----------------------------------------------------------------------------
// --- [ brute force implementation ] ------------------------------------------
// -----------------------------------------------------------------------------

void NS_BruteForce_CPU02::Init(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_CPU02::Construct(const SimParams& params)
{
	// nothing
}

void NS_BruteForce_CPU02::Query(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, be_.N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// neighbor lists for particle i
				// (delete contents, but keep capacity)
				std::vector<uint32_t>& ff = be_.neighborsFF_[i];
				std::vector<uint32_t>& fb = be_.neighborsFB_[i];
				ff.clear();
				fb.clear();

				const Eigen::Vector3f posi = be_.posF_[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < be_.N_F_; j++)
				{
					const Eigen::Vector3f dposij = posi - be_.posF_[j];

					// squared distance between particles i and j
					const float distSqu = dposij.squaredNorm();

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// add actual neighbor to list
						ff.push_back(j);
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < be_.N_B_; j++)
				{
					const Eigen::Vector3f dposij = posi - be_.posB_[j];

					// squared distance between particles i and j
					const float distSqu = dposij.squaredNorm();

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// add actual neighbor to list
						fb.push_back(j);
					}
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU02::DataReOrder(const SimParams& params)
{
	// TODO: implement
}

// -----------------------------------------------------------------------------
// --- [ spatial hashing implementation ] --------------------------------------
// -----------------------------------------------------------------------------

void NS_SpatialHashing_Basic_CPU02::Init(const SimParams& params)
{
	cellGridF_.clear();
	cellGridB_.clear();
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
}

void NS_SpatialHashing_Basic_CPU02::Construct(const SimParams& params)
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

	// fill in fluid cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysF(be_.N_F_);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, be_.N_F_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const Eigen::Vector3f posi = be_.posF_[i];

					// compute cell index
					const int cx = FloorToInt(posi.x() * cellSizeInv_);
					const int cy = FloorToInt(posi.y() * cellSizeInv_);
					const int cz = FloorToInt(posi.z() * cellSizeInv_);

					// convert cell index into key
					cellKeysF[i] = PackKey(cx, cy, cz);
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < be_.N_F_; i++)
		{
			const Key key = cellKeysF[i];
			cellGridF_[key].push_back(i);
		}
	}

	// fill in boundary cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysB(be_.N_B_);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, be_.N_B_),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const Eigen::Vector3f posi = be_.posB_[i];

					// compute cell index
					const int cx = FloorToInt(posi.x() * cellSizeInv_);
					const int cy = FloorToInt(posi.y() * cellSizeInv_);
					const int cz = FloorToInt(posi.z() * cellSizeInv_);

					// convert cell index into key
					cellKeysB[i] = PackKey(cx, cy, cz);
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < be_.N_B_; i++)
		{
			const Key key = cellKeysB[i];
			cellGridB_[key].push_back(i);
		}
	}
}

void NS_SpatialHashing_Basic_CPU02::Query(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, be_.N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// neighbor lists for particle i
				// (delete contents, but keep capacity)
				std::vector<uint32_t>& ff = be_.neighborsFF_[i];
				std::vector<uint32_t>& fb = be_.neighborsFB_[i];
				ff.clear();
				fb.clear();

				const Eigen::Vector3f posi = be_.posF_[i];

				// cell ID of particle i
				const int cxi = FloorToInt(posi.x() * cellSizeInv_);
				const int cyi = FloorToInt(posi.y() * cellSizeInv_);
				const int czi = FloorToInt(posi.z() * cellSizeInv_);

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
									const Eigen::Vector3f dposij = posi - be_.posF_[j];

									// squared distance between particles i and j
									const float distSqu = dposij.squaredNorm();

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
									const Eigen::Vector3f dposij = posi - be_.posB_[j];

									// squared distance between particles i and j
									const float distSqu = dposij.squaredNorm();

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
		},
		tbb::auto_partitioner{}
	);
}

void NS_SpatialHashing_Basic_CPU02::DataReOrder(const SimParams& params)
{
	// TODO: implement
}

// bijective mapping of signed to unsigned integers
// (-1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...)
inline uint32_t NS_SpatialHashing_Basic_CPU02::ZigZag(int v) noexcept
{
	// shift left by 1 and XOR with sign bit
	std::uint32_t uv = static_cast<std::uint32_t>(v);
	std::uint32_t even = uv << 1;
	std::uint32_t sign = static_cast<std::uint32_t>(-(v < 0)); // 0 or 0xFFFFFFFF

	return even ^ sign;
}

inline uint64_t NS_SpatialHashing_Basic_CPU02::PackKey(int cx, int cy, int cz) noexcept
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

inline int NS_SpatialHashing_Basic_CPU02::FloorToInt(float v) noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}

// -----------------------------------------------------------------------------
// --- [ cell linked list basic implementation ] -------------------------------
// -----------------------------------------------------------------------------

void NS_CellLinkedList_Basic_CPU02::Init(const SimParams& params)
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

void NS_CellLinkedList_Basic_CPU02::Construct(const SimParams& params)
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
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
				 ? 0.0f : (1.0f / params.kernelRadius);

	// --- [ combined AABB over fluid and boundary particles ] -----------------

	// accumulators
	float initMin = std::numeric_limits<float>::max();
	float initMax = std::numeric_limits<float>::lowest();
	float xMin = initMin, yMin = initMin, zMin = initMin;
	float xMax = initMax, yMax = initMax, zMax = initMax;

	// fluid
	for (uint32_t i = 0; i < be_.N_F_; i++)
	{
		const Eigen::Vector3f posi = be_.posF_[i];

		const float x = posi.x();
		const float y = posi.y();
		const float z = posi.z();

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
		const Eigen::Vector3f posi = be_.posB_[i];

		const float x = posi.x();
		const float y = posi.y();
		const float z = posi.z();

		xMin = x < xMin ? x : xMin;
		yMin = y < yMin ? y : yMin;
		zMin = z < zMin ? z : zMin;
		xMax = x > xMax ? x : xMax;
		yMax = y > yMax ? y : yMax;
		zMax = z > zMax ? z : zMax;
	}

	// padding to avoid edge clamp issues
	constexpr float pad = 1e-6f;
	xMin -= pad; yMin -= pad; zMin -= pad;
	xMax += pad; yMax += pad; zMax += pad;

	xBoxOrig_ = xMin;
	yBoxOrig_ = yMin;
	zBoxOrig_ = zMin;

	const float xLen = fmaxf(0.0f, xMax - xMin);
	const float yLen = fmaxf(0.0f, yMax - yMin);
	const float zLen = fmaxf(0.0f, zMax - zMin);

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

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, be_.N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const Eigen::Vector3f posi = be_.posF_[i];

				// compute cell coordinates
				int cx = FloorToInt((posi.x() - xBoxOrig_) * cellSizeInv_);
				int cy = FloorToInt((posi.y() - yBoxOrig_) * cellSizeInv_);
				int cz = FloorToInt((posi.z() - zBoxOrig_) * cellSizeInv_);

				// clamp to grid dimensions
				uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
				uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
				uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

				cellIdF_[i] = MapToCellId_Basic(ux, uy, uz);
				sortedParticleIdxF_[i] = i; // initial identity permutation (unsorted)
			}
		},
		tbb::auto_partitioner{}
	);

	// boundary
	cellStartIdxB_.assign(N_Cells_, EMPTY);
	cellEndIdxB_.assign(N_Cells_, EMPTY);

	cellIdB_.resize(be_.N_B_);
	sortedParticleIdxB_.resize(be_.N_B_);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, be_.N_B_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				const Eigen::Vector3f posi = be_.posB_[i];

				// compute cell coordinates
				int cx = FloorToInt((posi.x() - xBoxOrig_) * cellSizeInv_);
				int cy = FloorToInt((posi.y() - yBoxOrig_) * cellSizeInv_);
				int cz = FloorToInt((posi.z() - zBoxOrig_) * cellSizeInv_);

				// clamp to grid dimensions
				uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
				uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
				uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

				cellIdB_[i] = MapToCellId_Basic(ux, uy, uz);
				sortedParticleIdxB_[i] = i; // initial identity permutation (unsorted)
			}
		},
		tbb::auto_partitioner{}
	);

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
		uint32_t cid_prev = EMPTY;
		for (uint32_t i = 0; i < be_.N_F_; i++)
		{
			const uint32_t pidx = sortedParticleIdxF_[i];
			const uint32_t cid = cellIdF_[pidx];

			if (cid != cid_prev) // new cell encountered
			{
				if (cid_prev != EMPTY)
				{
					// set end index of previous cell
					cellEndIdxF_[cid_prev] = i;
				}

				if (cellStartIdxF_[cid] == EMPTY)
				{
					// set start index of current cell
					cellStartIdxF_[cid] = i;
				}

				cid_prev = cid;
			}
		}

		if (cid_prev != EMPTY)
		{
			// set end index of last cell to close it
			cellEndIdxF_[cid_prev] = be_.N_F_;
		}
	}

	// boundary
	{
		uint32_t cid_prev = EMPTY;
		for (uint32_t i = 0; i < be_.N_B_; i++)
		{
			const uint32_t pidx = sortedParticleIdxB_[i];
			const uint32_t cid = cellIdB_[pidx];

			if (cid != cid_prev) // new cell encountered
			{
				if (cid_prev != EMPTY)
				{
					// set end index of previous cell
					cellEndIdxB_[cid_prev] = i;
				}

				if (cellStartIdxB_[cid] == EMPTY)
				{
					// set start index of current cell
					cellStartIdxB_[cid] = i;
				}

				cid_prev = cid;
			}
		}

		if (cid_prev != EMPTY)
		{
			// set end index of last cell to close it
			cellEndIdxB_[cid_prev] = be_.N_B_;
		}
	}
}

void NS_CellLinkedList_Basic_CPU02::Query(const SimParams& params)
{
	// TODO: instead of iterating over particles in the outermost loop,
	// TODO: iterate over cells and their particles, because for all particles
	// TODO: in a cell, the neighboring cells and neighbor candidates are the same?

	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, be_.N_F_),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// neighbor lists for particle i
				// (delete contents, but keep capacity)
				std::vector<uint32_t>& ff = be_.neighborsFF_[i];
				std::vector<uint32_t>& fb = be_.neighborsFB_[i];
				ff.clear();
				fb.clear();

				const Eigen::Vector3f posi = be_.posF_[i];
				const float xi = posi.x();
				const float yi = posi.y();
				const float zi = posi.z();

				// cell coordinates of particle i
				const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
				const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
				const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

				// check all particles j in 27 neighboring grid cells
				for (int rz = -1; rz <= 1; rz++)
				{
					int cz = czi + rz; // target cell coordinate in z-direction
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -1; ry <= 1; ry++)
					{
						int cy = cyi + ry; // target cell coordinate in y-direction
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -1; rx <= 1; rx++)
						{
							int cx = cxi + rx; // target cell coordinate in x-direction
							if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

							// cell ID
							const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
																   static_cast<uint32_t>(cy),
																   static_cast<uint32_t>(cz));

							// check fluid candidates
							const uint32_t sF = cellStartIdxF_[cid];
							const uint32_t eF = cellEndIdxF_[cid];
							if (sF != EMPTY && eF != EMPTY)
							{
								// check all neighbor candidates in the cell
								for (uint32_t k = sF; k < eF; k++)
								{
									// index of neighbor candidate
									const uint32_t j = sortedParticleIdxF_[k];

									// position difference
									const Eigen::Vector3f dposij = posi - be_.posF_[j];

									// squared distance between particles i and j
									const float distSqu = dposij.squaredNorm();

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// add actual neighbor to list
										ff.push_back(j);
									}
								}
							}

							// check boundary candidates
							const uint32_t sB = cellStartIdxB_[cid];
							const uint32_t eB = cellEndIdxB_[cid];
							if (sB != EMPTY && eB != EMPTY)
							{
								// check all neighbor candidates in the cell
								for (uint32_t k = sB; k < eB; k++)
								{
									// index of neighbor candidate
									const uint32_t j = sortedParticleIdxB_[k];

									// position difference
									const Eigen::Vector3f dposij = posi - be_.posB_[j];

									// squared distance between particles i and j
									const float distSqu = dposij.squaredNorm();

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
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Basic_CPU02::DataReOrder(const SimParams& params)
{
	// TODO: implement
}

inline uint32_t NS_CellLinkedList_Basic_CPU02::MapToCellId_Basic(
	uint32_t cx, uint32_t cy, uint32_t cz) const noexcept
{
	return cx + N_DimX_ * (cy + N_DimY_ * cz);
}

inline int NS_CellLinkedList_Basic_CPU02::FloorToInt(float v) noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}

inline uint32_t NS_CellLinkedList_Basic_CPU02::ClampInt(
	int v, int low, int high) noexcept
{
	const int t = (v < low) ? low : v;
	return static_cast<uint32_t>((t > high) ? high : t);
}

// -----------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -----------------------------
// -----------------------------------------------------------------------------

void NS_CellLinkedList_Compact_CPU02::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;

	uniqueCellIdF_.clear();
	cellStartIdxF_.clear();
	cellIdF_.clear();
	sortedParticleIdxF_.clear();

	uniqueCellIdB_.clear();
	cellStartIdxB_.clear();
	cellIdB_.clear();
	sortedParticleIdxB_.clear();
}

void NS_CellLinkedList_Compact_CPU02::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");

		uniqueCellIdF_.clear();
		cellStartIdxF_.clear();
		cellIdF_.clear();
		sortedParticleIdxF_.clear();

		uniqueCellIdB_.clear();
		cellStartIdxB_.clear();
		cellIdB_.clear();
		sortedParticleIdxB_.clear();

		return;
	}

	cellSize_ = params.kernelRadius;
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
				 ? 0.0f : (1.0f / params.kernelRadius);

	// --- [ cell coordinates and indices for fluid and boundary particles ] ---

	// fluid
	cellIdF_.resize(N_F);
	sortedParticleIdxF_.resize(N_F);

	if (N_F > 0)
	{
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const Eigen::Vector3f posi = be_.posF_[i];

					// compute cell coordinates
					const int cx = FloorToInt(posi.x() * cellSizeInv_);
					const int cy = FloorToInt(posi.y() * cellSizeInv_);
					const int cz = FloorToInt(posi.z() * cellSizeInv_);

					cellIdF_[i] = MapToCellId_Morton(cx, cy, cz);
					sortedParticleIdxF_[i] = i; // initial identity permutation (unsorted)
				}
			},
			tbb::auto_partitioner{}
		);
	}

	// boundary
	cellIdB_.resize(N_B);
	sortedParticleIdxB_.resize(N_B);

	if (N_B > 0)
	{
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const Eigen::Vector3f posi = be_.posB_[i];

					// compute cell coordinates
					const int cx = FloorToInt(posi.x() * cellSizeInv_);
					const int cy = FloorToInt(posi.y() * cellSizeInv_);
					const int cz = FloorToInt(posi.z() * cellSizeInv_);

					cellIdB_[i] = MapToCellId_Morton(cx, cy, cz);
					sortedParticleIdxB_[i] = i; // initial identity permutation (unsorted)
				}
			},
			tbb::auto_partitioner{}
		);
	}

	// --- [ sort particle indices by cell IDs ] -------------------------------

	if (N_F > 0)
	{
		tbb::parallel_sort(sortedParticleIdxF_.begin(), sortedParticleIdxF_.end(),
			[&](uint32_t a, uint32_t b)
			{
				return cellIdF_[a] < cellIdF_[b];
			});
	}

	if (N_B > 0)
	{
		tbb::parallel_sort(sortedParticleIdxB_.begin(), sortedParticleIdxB_.end(),
			[&](uint32_t a, uint32_t b)
			{
				return cellIdB_[a] < cellIdB_[b];
			});
	}

	// --- [ build compact cell list and fill in cell start indices ] ----------

	uniqueCellIdF_.clear();
	cellStartIdxF_.clear();

	// fluid
	// TODO: replace this with a parallel marker and scan version?
	if (N_F > 0)
	{
		// worst case, all particles are in unique cells (highly unlikely)
		// best case (realistically), each cell has 8 particles
		// TODO: find a better heuristic
		uniqueCellIdF_.reserve(N_F / 2);
		cellStartIdxF_.reserve(N_F + 1); // +1 for sentinel

		uint64_t cid_prev = EMPTY_64;
		for (uint32_t i = 0; i < N_F; i++)
		{
			const uint32_t pidx = sortedParticleIdxF_[i];
			const uint64_t cid = cellIdF_[pidx];

			if (cid != cid_prev) // new cell encountered
			{
				uniqueCellIdF_.push_back(cid);
				cellStartIdxF_.push_back(i);

				cid_prev = cid;
			}
		}

		// end of the last cell's range as sentinel
		cellStartIdxF_.push_back(N_F);
	}

	uniqueCellIdB_.clear();
	cellStartIdxB_.clear();

	// boundary
	// TODO: replace this with a parallel marker and scan version?
	if (N_B > 0)
	{
		// worst case, all particles are in unique cells (highly unlikely)
		// best case (realistically), each cell has 8 particles
		// TODO: find a better heuristic
		uniqueCellIdB_.reserve(N_B / 2);
		cellStartIdxB_.reserve(N_B + 1); // +1 for sentinel

		uint64_t cid_prev = EMPTY_64;
		for (uint32_t i = 0; i < N_B; i++)
		{
			const uint32_t pidx = sortedParticleIdxB_[i];
			const uint64_t cid = cellIdB_[pidx];

			if (cid != cid_prev) // new cell encountered
			{
				uniqueCellIdB_.push_back(cid);
				cellStartIdxB_.push_back(i);

				cid_prev = cid;
			}
		}

		// end of the last cell's range as sentinel
		cellStartIdxB_.push_back(N_B);
	}
}

void NS_CellLinkedList_Compact_CPU02::Query(const SimParams& params)
{
	Query_Smart(params);
}

void NS_CellLinkedList_Compact_CPU02::Query_Basic(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// clear neighbor lists of all particles in this cell
				for (uint32_t k = sF; k < eF; k++)
				{
					const uint32_t i = sortedParticleIdxF_[k];

					be_.neighborsFF_[i].clear();
					be_.neighborsFB_[i].clear();
				}

				// determine a representative particle i0
				const uint32_t i0 = sortedParticleIdxF_[sF];

				const Eigen::Vector3f pos0 = be_.posF_[i0];
				const float xi0 = pos0.x();
				const float yi0 = pos0.y();
				const float zi0 = pos0.z();

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// check all particles j in 27 (3x3x3) neighboring grid cells
				for (int rz = -1; rz <= 1; rz++)
				{
					int nz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -1; ry <= 1; ry++)
					{
						int ny = cy + ry; // target cell coordinate in y-direction

						for (int rx = -1; rx <= 1; rx++)
						{
							int nx = cx + rx; // target cell coordinate in x-direction

							// cell ID of the neighbor cell
							const uint64_t cid_nbr = MapToCellId_Morton(nx, ny, nz);

							// find neighbor fluid cell in compact list (binary search)
							// TODO: use ternary search and BigMin-LitMax instead?
							auto itF = std::ranges::lower_bound(uniqueCellIdF_, cid_nbr);

							// check fluid candidates in this neighbor cell
							if (itF != uniqueCellIdF_.end() && *itF == cid_nbr)
							{
								const uint32_t cF_nbr = static_cast<uint32_t>(itF - uniqueCellIdF_.begin());

								// range of all fluid particles in the neighbor cell
								const uint32_t sF_nbr = cellStartIdxF_[cF_nbr]; // start
								const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1]; // end
								if (sF_nbr >= eF_nbr) { continue; }

								// outer loop over particles in this cell (c)
								// inner loop over all particles in the neighbor cell (cF_nbr)
								for (uint32_t k = sF; k < eF; k++)
								{
									const uint32_t i = sortedParticleIdxF_[k];

									const Eigen::Vector3f posi = be_.posF_[i];

									for (uint32_t l = sF_nbr; l < eF_nbr; l++)
									{
										// index of fluid neighbor candidate
										const uint32_t j = sortedParticleIdxF_[l];

										// position difference
										const Eigen::Vector3f dposij = posi - be_.posF_[j];

										// squared distance between particles i and j
										const float distSqu = dposij.squaredNorm();

										if (distSqu <= kernelRadiusSqu) [[unlikely]]
										{
											// add actual neighbor to list
											be_.neighborsFF_[i].push_back(j);
										}
									}
								}
							}

							// find neighbor boundary cell in compact list (binary search)
							// TODO: use ternary search and BigMin-LitMax instead?
							auto itB = std::ranges::lower_bound(uniqueCellIdB_, cid_nbr);

							// check fluid candidates in this neighbor cell
							if (itB != uniqueCellIdB_.end() && *itB == cid_nbr)
							{
								const uint32_t cB_nbr = static_cast<uint32_t>(itB - uniqueCellIdB_.begin());

								// range of all boundary particles in the neighbor cell
								const uint32_t sB_nbr = cellStartIdxB_[cB_nbr]; // start
								const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1]; // end
								if (sB_nbr >= eB_nbr) { continue; }

								// outer loop over particles in this cell (c)
								// inner loop over all particles in the neighbor cell (cB_nbr)
								for (uint32_t k = sF; k < eF; k++)
								{
									const uint32_t i = sortedParticleIdxF_[k];

									const Eigen::Vector3f posi = be_.posF_[i];

									for (uint32_t l = sB_nbr; l < eB_nbr; l++)
									{
										// index of boundary neighbor candidate
										const uint32_t j = sortedParticleIdxB_[l];

										// position difference
										const Eigen::Vector3f dposij = posi - be_.posB_[j];

										// squared distance between particles i and j
										const float distSqu = dposij.squaredNorm();

										if (distSqu <= kernelRadiusSqu) [[unlikely]]
										{
											// add actual neighbor to list
											be_.neighborsFB_[i].push_back(j);
										}
									}
								}
							}
						}
					}
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Compact_CPU02::Query_Smart(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// clear neighbor lists of all particles in this cell
				for (uint32_t k = sF; k < eF; k++)
				{
					const uint32_t i = sortedParticleIdxF_[k];

					be_.neighborsFF_[i].clear();
					be_.neighborsFB_[i].clear();
				}

				// determine a representative particle i0
				const uint32_t i0 = sortedParticleIdxF_[sF];

				const Eigen::Vector3f pos0 = be_.posF_[i0];
				const float xi0 = pos0.x();
				const float yi0 = pos0.y();
				const float zi0 = pos0.z();

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// pre-compute neighbor cell IDs
				std::array<uint8_t, N_NbrCells_> hasFluidCell{};
				std::array<uint32_t, N_NbrCells_> sNbrF{};
				std::array<uint32_t, N_NbrCells_> eNbrF{};

				std::array<uint8_t, N_NbrCells_> hasBoundaryCell{};
				std::array<uint32_t, N_NbrCells_> sNbrB{};
				std::array<uint32_t, N_NbrCells_> eNbrB{};

				// check existence and location of all 27 (3x3x3) neighboring grid cells
				for (int rz = -1; rz <= 1; rz++)
				{
					const int oz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -1; ry <= 1; ry++)
					{
						const int oy = cy + ry; // target cell coordinate in y-direction

						for (int rx = -1; rx <= 1; rx++)
						{
							const int ox = cx + rx; // target cell coordinate in x-direction

							// index of this neighbor cell in the 3x3x3 neighborhood
							const int idx_nbr = NeighborOffset(rx, ry, rz);

							// cell ID of this neighbor cell
							const uint64_t cid_nbr = MapToCellId_Morton(ox, oy, oz);

							// fluid scope
							{
								// find out if neighbor cell exists in compact cell list
								auto iterF = std::ranges::lower_bound(uniqueCellIdF_, cid_nbr);
								if (iterF != uniqueCellIdF_.end() && *iterF == cid_nbr)
								{
									// get range of all fluid particles in the neighbor cell
									const uint32_t cF_nbr = static_cast<uint32_t>(iterF - uniqueCellIdF_.begin());
									const uint32_t sF_nbr = cellStartIdxF_[cF_nbr]; // start
									const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1]; // end

									hasFluidCell[idx_nbr] = (sF_nbr < eF_nbr);
									sNbrF[idx_nbr] = sF_nbr;
									eNbrF[idx_nbr] = eF_nbr;
								}
								else
								{
									hasFluidCell[idx_nbr] = false;
								}
							}

							// boundary scope
							{
								// find out if neighbor cell exists in compact cell list
								auto iterB = std::ranges::lower_bound(uniqueCellIdB_, cid_nbr);
								if (iterB != uniqueCellIdB_.end() && *iterB == cid_nbr)
								{
									// get range of all boundary particles in the neighbor cell
									const uint32_t cB_nbr = static_cast<uint32_t>(iterB - uniqueCellIdB_.begin());
									const uint32_t sB_nbr = cellStartIdxB_[cB_nbr]; // start
									const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1]; // end

									hasBoundaryCell[idx_nbr] = (sB_nbr < eB_nbr);
									sNbrB[idx_nbr] = sB_nbr;
									eNbrB[idx_nbr] = eB_nbr;
								}
								else
								{
									hasBoundaryCell[idx_nbr] = false;
								}
							}
						}
					}
				}

				// check all particles j in 27 (3x3x3) neighboring grid cells
				for (uint32_t k = sF; k < eF; k++)
				{
					const uint32_t i = sortedParticleIdxF_[k];

					const Eigen::Vector3f posi = be_.posF_[i];

					// fluid neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells_; idx_nbr++)
					{
						if (!hasFluidCell[idx_nbr]) { continue; }

						const uint32_t sF_nbr = sNbrF[idx_nbr];
						const uint32_t eF_nbr = eNbrF[idx_nbr];

						for (uint32_t l = sF_nbr; l < eF_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxF_[l];

							// position difference
							const Eigen::Vector3f dposij = posi - be_.posF_[j];

							const float distSqu = dposij.squaredNorm();

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								be_.neighborsFF_[i].push_back(j);
							}
						}
					}

					// boundary neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells_; idx_nbr++)
					{
						if (!hasBoundaryCell[idx_nbr]) { continue; }

						const uint32_t sB_nbr = sNbrB[idx_nbr];
						const uint32_t eB_nbr = eNbrB[idx_nbr];

						for (uint32_t l = sB_nbr; l < eB_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxB_[l];

							// position difference
							const Eigen::Vector3f dposij = posi - be_.posB_[j];

							const float distSqu = dposij.squaredNorm();

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								be_.neighborsFB_[i].push_back(j);
							}
						}
					}
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Compact_CPU02::DataReOrder(const SimParams& params)
{
	// TODO: implement
}

inline uint64_t NS_CellLinkedList_Compact_CPU02::MapToCellId_XYZ(
	int cx, int cy, int cz) noexcept
{
	// zig-zag encoding of coordinates to handle negative values
	const uint64_t ux = static_cast<uint64_t>(ZigZag(cx));
	const uint64_t uy = static_cast<uint64_t>(ZigZag(cy));
	const uint64_t uz = static_cast<uint64_t>(ZigZag(cz));

	// each axis must fit into 21 bits
	assert(ux < (1u << 21));
	assert(uy < (1u << 21));
	assert(uz < (1u << 21));

	// packed XYZ encoding
	return (ux) | (uy << 21) | (uz << 42);
}

inline uint64_t NS_CellLinkedList_Compact_CPU02::MapToCellId_Morton(
	int cx, int cy, int cz) noexcept
{
	// zig-zag encoding of coordinates to handle negative values
	const uint32_t ux = ZigZag(cx);
	const uint32_t uy = ZigZag(cy);
	const uint32_t uz = ZigZag(cz);

	const uint64_t xx = SpreadBitsWithGaps_21to63(ux); // bits at pos 0,3,6,...
	const uint64_t yy = SpreadBitsWithGaps_21to63(uy) << 1; // bits at pos 1,4,7,...
	const uint64_t zz = SpreadBitsWithGaps_21to63(uz) << 2; // bits at pos 2,5,8,...

	return xx | yy | zz; // interleaved 3D Morton/Z-order code
}

// spread out lower 21 bits of v into lower 63 bits and fill with gaps zeros like 000x00x00x00x00x...
inline uint64_t NS_CellLinkedList_Compact_CPU02::SpreadBitsWithGaps_21to63(uint32_t v) noexcept
{
	uint64_t x = static_cast<uint64_t>(v & 0x1fffffU); // lower 21 bits

	x = (x | (x << 32)) & 0x1f00000000ffffULL;
	x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
	x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
	x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
	x = (x | (x << 2)) & 0x1249249249249249ULL;

	return x;
}

// bijective mapping of signed to unsigned integers
// before: [0  -1   1  -2   2  -3   3 ...]
// after:  [0   1   2   3   4   5   6 ...]
inline uint32_t NS_CellLinkedList_Compact_CPU02::ZigZag(int v) noexcept
{
	// shift left by 1 and XOR with sign bit
	std::uint32_t uv = static_cast<std::uint32_t>(v);
	std::uint32_t even = uv << 1;
	std::uint32_t sign = static_cast<std::uint32_t>(-(v < 0)); // 0 or 0xFFFFFFFF

	return even ^ sign;
}

inline int NS_CellLinkedList_Compact_CPU02::NeighborOffset(int rx, int ry, int rz) noexcept
{
	return (rz + 1) * 9 + (ry + 1) * 3 + (rx + 1);
}

inline int NS_CellLinkedList_Compact_CPU02::FloorToInt(float v) noexcept
{
	int i = static_cast<int>(v);

	return (v < static_cast<float>(i)) ? (i - 1) : i;
}
