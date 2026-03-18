// neighbor_search.cpp

// external
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <execution>
#include <numeric>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// internal
#include "cpu_backend_01.h"
#include "neighbor_lists.h"
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

namespace
{
	void ComputeNegativeNeighborCandidatePercentages(
		const uint64_t candCountFF,
		const uint64_t posCountFF,
		const uint64_t candCountFB,
		const uint64_t posCountFB,
		float& negNbrCandPctFF,
		float& negNbrCandPctFB)
	{
		negNbrCandPctFF = candCountFF > 0
			? static_cast<float>(candCountFF - posCountFF)
			/ static_cast<float>(candCountFF)
			: 0.0f;

		negNbrCandPctFB = candCountFB > 0
			? static_cast<float>(candCountFB - posCountFB)
			/ static_cast<float>(candCountFB)
			: 0.0f;
	}

	void ApplyPermutation(
		const std::vector<uint64_t>& sortedKeys,
		std::vector<float>& data,
		std::vector<float>& scratch,
		bool preserveDataCapacity = false)
	{
		// (capped at uint32_t to correspond to max particle count)
		const uint32_t N = static_cast<uint32_t>(sortedKeys.size());
		if (N == 0) { return; }

		assert(data.size() == N && "Permutation and data vector size mismatch");
		scratch.resize(N);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const uint32_t srcIdx = static_cast<uint32_t>(sortedKeys[i]);
					scratch[i] = data[srcIdx];
				}
			},
			tbb::auto_partitioner{}
		);

		if (preserveDataCapacity)
		{
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						data[i] = scratch[i];
					}
				},
				tbb::auto_partitioner{}
			);
		}
		else
		{
			data.swap(scratch);
		}
	}

	inline int FloorToInt(float v) noexcept
	{
		int i = static_cast<int>(v);

		return (v < static_cast<float>(i)) ? (i - 1) : i;
	}

	inline uint32_t ClampInt(int v, int low, int high) noexcept
	{
		const int t = (v < low) ? low : v;
		return static_cast<uint32_t>((t > high) ? high : t);
	}

	struct FluidParticleSoARefs
	{
		std::vector<float>& xPos;
		std::vector<float>& yPos;
		std::vector<float>& zPos;
		std::vector<float>& xVel;
		std::vector<float>& yVel;
		std::vector<float>& zVel;
		std::vector<float>& xVelPred;
		std::vector<float>& yVelPred;
		std::vector<float>& zVelPred;
		std::vector<float>& xAccPress;
		std::vector<float>& yAccPress;
		std::vector<float>& zAccPress;
		std::vector<float>& sourceTerm;
		std::vector<float>& diagElem;
		std::vector<float>& pressure;
		std::vector<float>& density;
	};

	struct BoundaryParticleSoARefs
	{
		std::vector<float>& xPos;
		std::vector<float>& yPos;
		std::vector<float>& zPos;
	};

	struct SceneAABB
	{
		float minX = std::numeric_limits<float>::max();
		float minY = std::numeric_limits<float>::max();
		float minZ = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float maxY = std::numeric_limits<float>::lowest();
		float maxZ = std::numeric_limits<float>::lowest();
	};

	struct CellCoordFrame
	{
		int cellOrigX = 0;
		int cellOrigY = 0;
		int cellOrigZ = 0;
		uint32_t N_DimX = 0;
		uint32_t N_DimY = 0;
		uint32_t N_DimZ = 0;
	};

	// TODO: parallelize?
	inline SceneAABB ComputeSceneAABB_Combined(
		const float* xPosF, const float* yPosF, const float* zPosF, uint32_t N_F,
		const float* xPosB, const float* yPosB, const float* zPosB, uint32_t N_B)
	{
		SceneAABB aabb{};

		for (uint32_t i = 0; i < N_F; i++)
		{
			const float x = xPosF[i];
			const float y = yPosF[i];
			const float z = zPosF[i];

			aabb.minX = std::min(aabb.minX, x);
			aabb.minY = std::min(aabb.minY, y);
			aabb.minZ = std::min(aabb.minZ, z);
			aabb.maxX = std::max(aabb.maxX, x);
			aabb.maxY = std::max(aabb.maxY, y);
			aabb.maxZ = std::max(aabb.maxZ, z);
		}

		for (uint32_t i = 0; i < N_B; i++)
		{
			const float x = xPosB[i];
			const float y = yPosB[i];
			const float z = zPosB[i];

			aabb.minX = std::min(aabb.minX, x);
			aabb.minY = std::min(aabb.minY, y);
			aabb.minZ = std::min(aabb.minZ, z);
			aabb.maxX = std::max(aabb.maxX, x);
			aabb.maxY = std::max(aabb.maxY, y);
			aabb.maxZ = std::max(aabb.maxZ, z);
		}

		return aabb;
	}

	inline void ExpandSceneAABB(
		SceneAABB& aabb,
		float pad) noexcept
	{
		aabb.minX -= pad;
		aabb.minY -= pad;
		aabb.minZ -= pad;
		aabb.maxX += pad;
		aabb.maxY += pad;
		aabb.maxZ += pad;
	}

	inline CellCoordFrame ComputeCellCoordFrame_FromAABB(
		const SceneAABB& aabb,
		float cellSizeInv)
	{
		CellCoordFrame frame{};

		const int cxMin = FloorToInt(aabb.minX * cellSizeInv);
		const int cyMin = FloorToInt(aabb.minY * cellSizeInv);
		const int czMin = FloorToInt(aabb.minZ * cellSizeInv);
		const int cxMax = FloorToInt(aabb.maxX * cellSizeInv);
		const int cyMax = FloorToInt(aabb.maxY * cellSizeInv);
		const int czMax = FloorToInt(aabb.maxZ * cellSizeInv);

		frame.cellOrigX = cxMin;
		frame.cellOrigY = cyMin;
		frame.cellOrigZ = czMin;
		frame.N_DimX = static_cast<uint32_t>(cxMax - cxMin + 1);
		frame.N_DimY = static_cast<uint32_t>(cyMax - cyMin + 1);
		frame.N_DimZ = static_cast<uint32_t>(czMax - czMin + 1);

		return frame;
	}

	// expand lower 10 bits into every-third-bit positions: 0b...xyz -> 0b...00x00y00z
	inline uint32_t ExpandBits10_30bit(uint32_t v) noexcept
	{
		v = (v | (v << 16)) & 0x030000FFu;
		v = (v | (v <<  8)) & 0x0300F00Fu;
		v = (v | (v <<  4)) & 0x030C30C3u;
		v = (v | (v <<  2)) & 0x09249249u;
		return v;
	}

	// compute a 30-bit Morton code for a point in the unit cube [0, 1]^3
	inline uint32_t Morton3D_30bit_Normalized(float nx, float ny, float nz) noexcept
	{
		// clamp to [0, 1] and quantize to 10-bit integer (0..1023)
		const auto Quantize = [](float v) -> uint32_t
			{
				return std::min(1023u, static_cast<uint32_t>(
					std::max(0.0f, std::min(1.0f, v)) * 1023.0f));
			};

		const uint32_t ix = Quantize(nx);
		const uint32_t iy = Quantize(ny);
		const uint32_t iz = Quantize(nz);

		return (ExpandBits10_30bit(ix) << 2) | (ExpandBits10_30bit(iy) << 1)
			| ExpandBits10_30bit(iz);
	}

	inline SceneAABB ComputeSceneAABB(
		const std::vector<float>& xPos,
		const std::vector<float>& yPos,
		const std::vector<float>& zPos)
	{
		const uint32_t N = static_cast<uint32_t>(xPos.size());
		assert(yPos.size() == N && zPos.size() == N && "Position vector size mismatch");

		SceneAABB init{};
		if (N == 0) { return init; }

		return tbb::parallel_reduce(
			tbb::blocked_range<uint32_t>(0, N),
			init,
			[&](const tbb::blocked_range<uint32_t>& r, SceneAABB val) -> SceneAABB
			{
				float lMinX = val.minX, lMinY = val.minY, lMinZ = val.minZ;
				float lMaxX = val.maxX, lMaxY = val.maxY, lMaxZ = val.maxZ;

				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					lMinX = std::min(lMinX, xPos[i]);
					lMinY = std::min(lMinY, yPos[i]);
					lMinZ = std::min(lMinZ, zPos[i]);
					lMaxX = std::max(lMaxX, xPos[i]);
					lMaxY = std::max(lMaxY, yPos[i]);
					lMaxZ = std::max(lMaxZ, zPos[i]);
				}

				val.minX = lMinX; val.minY = lMinY; val.minZ = lMinZ;
				val.maxX = lMaxX; val.maxY = lMaxY; val.maxZ = lMaxZ;
				return val;
			},
			[](SceneAABB a, const SceneAABB& b) -> SceneAABB
			{
				a.minX = std::min(a.minX, b.minX);
				a.minY = std::min(a.minY, b.minY);
				a.minZ = std::min(a.minZ, b.minZ);
				a.maxX = std::max(a.maxX, b.maxX);
				a.maxY = std::max(a.maxY, b.maxY);
				a.maxZ = std::max(a.maxZ, b.maxZ);
				return a;
			},
			tbb::auto_partitioner{}
		);
	}

	void BuildMortonOrderKeys_30bitNormalized(
		const std::vector<float>& xPos,
		const std::vector<float>& yPos,
		const std::vector<float>& zPos,
		std::vector<uint64_t>& outSortedKeys)
	{
		const uint32_t N = static_cast<uint32_t>(xPos.size());
		assert(yPos.size() == N && zPos.size() == N && "Position vector size mismatch");

		outSortedKeys.resize(N);
		if (N == 0) { return; }

		const SceneAABB sceneAABB = ComputeSceneAABB(xPos, yPos, zPos);

		constexpr float eps = 1e-6f;
		const float invExtX = 1.0f / std::max(eps, sceneAABB.maxX - sceneAABB.minX);
		const float invExtY = 1.0f / std::max(eps, sceneAABB.maxY - sceneAABB.minY);
		const float invExtZ = 1.0f / std::max(eps, sceneAABB.maxZ - sceneAABB.minZ);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					const float nx = (xPos[i] - sceneAABB.minX) * invExtX;
					const float ny = (yPos[i] - sceneAABB.minY) * invExtY;
					const float nz = (zPos[i] - sceneAABB.minZ) * invExtZ;

					const uint64_t morton = static_cast<uint64_t>(
						Morton3D_30bit_Normalized(nx, ny, nz));

					// key layout [morton:30][padding:2][index:32]
					// tie-break by index to keep deterministic ordering
					outSortedKeys[i] = (morton << 32) | static_cast<uint64_t>(i);
				}
			},
			tbb::auto_partitioner{}
		);

		tbb::parallel_sort(outSortedKeys.begin(), outSortedKeys.end());
	}

	void ApplyPermutationFluidSoA(
		const std::vector<uint64_t>& sortedKeys,
		FluidParticleSoARefs fluid,
		std::vector<float>& scratch)
	{
		ApplyPermutation(sortedKeys, fluid.xPos, scratch);
		ApplyPermutation(sortedKeys, fluid.yPos, scratch);
		ApplyPermutation(sortedKeys, fluid.zPos, scratch);

		ApplyPermutation(sortedKeys, fluid.xVel, scratch);
		ApplyPermutation(sortedKeys, fluid.yVel, scratch);
		ApplyPermutation(sortedKeys, fluid.zVel, scratch);

		ApplyPermutation(sortedKeys, fluid.xVelPred, scratch);
		ApplyPermutation(sortedKeys, fluid.yVelPred, scratch);
		ApplyPermutation(sortedKeys, fluid.zVelPred, scratch);

		ApplyPermutation(sortedKeys, fluid.xAccPress, scratch);
		ApplyPermutation(sortedKeys, fluid.yAccPress, scratch);
		ApplyPermutation(sortedKeys, fluid.zAccPress, scratch);

		ApplyPermutation(sortedKeys, fluid.sourceTerm, scratch);
		ApplyPermutation(sortedKeys, fluid.diagElem, scratch);
		ApplyPermutation(sortedKeys, fluid.pressure, scratch);
		ApplyPermutation(sortedKeys, fluid.density, scratch);
	}

	void ApplyPermutationBoundarySoA(
		const std::vector<uint64_t>& sortedKeys,
		BoundaryParticleSoARefs boundary,
		std::vector<float>& scratch)
	{
		ApplyPermutation(sortedKeys, boundary.xPos, scratch);
		ApplyPermutation(sortedKeys, boundary.yPos, scratch);
		ApplyPermutation(sortedKeys, boundary.zPos, scratch);
	}

	void DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs fluid,
		BoundaryParticleSoARefs boundary,
		uint32_t N_F, uint32_t N_B,
		std::vector<uint64_t>& sortKeyBuffer,
		std::vector<float>& scratchF,
		std::vector<float>& scratchB)
	{
		if (N_F > 0)
		{
			assert(fluid.xPos.size() == N_F && fluid.yPos.size() == N_F
				&& fluid.zPos.size() == N_F && "Fluid position vector size mismatch");

			BuildMortonOrderKeys_30bitNormalized(
				fluid.xPos, fluid.yPos, fluid.zPos, sortKeyBuffer);

			ApplyPermutationFluidSoA(sortKeyBuffer, fluid, scratchF);
		}

		if (N_B > 0)
		{
			assert(boundary.xPos.size() == N_B && boundary.yPos.size() == N_B
				&& boundary.zPos.size() == N_B && "Boundary position vector size mismatch");

			BuildMortonOrderKeys_30bitNormalized(
				boundary.xPos, boundary.yPos, boundary.zPos, sortKeyBuffer);

			ApplyPermutationBoundarySoA(sortKeyBuffer, boundary, scratchB);
		}
	}

} // unnamed

// -------------------------------------------------------------------------------------------------
// --- [ brute force implementation ] --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BruteForce_CPU01::Init(const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Construct(const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BruteForce_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						cntFF++;
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						cntFB++;
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BruteForce_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// reset neighbor list for particle i
				// (delete contents, but keep capacity)
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// check all fluid particles j
				for (uint32_t j = 0; j < N_F; j++)
				{
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// accumulate actual neighbors in scratch list
						scratchFF.push_back(j);
					}
				}

				// check all boundary particles j
				for (uint32_t j = 0; j < N_B; j++)
				{
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];

					// squared distance between particles i and j
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						// accumulate actual neighbors in scratch list
						scratchFB.push_back(j);
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

NeighborSpan NS_BruteForce_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	for (uint32_t j = 0; j < N_F; j++)
	{
		const float dx = xi - xPosF[j];
		const float dy = yi - yPosF[j];
		const float dz = zi - zPosF[j];
		const float distSqu = dx * dx + dy * dy + dz * dz;

		if (distSqu <= kernelRadiusSqu_) [[unlikely]]
		{
			scratchFF.push_back(j);
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_BruteForce_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	for (uint32_t j = 0; j < N_B; j++)
	{
		const float dx = xi - xPosB[j];
		const float dy = yi - yPosB[j];
		const float dz = zi - zPosB[j];
		const float distSqu = dx * dx + dy * dy + dz * dz;

		if (distSqu <= kernelRadiusSqu_) [[unlikely]]
		{
			scratchFB.push_back(j);
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_BruteForce_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BruteForce_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	return { usageF, usageB };
}

// -------------------------------------------------------------------------------------------------
// --- [ spatial hashing basic implementation ] ----------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_SpatialHashing_Basic_CPU01::Init(const SimParams& params)
{
	cellGridF_.clear();
	cellGridB_.clear();
	cellGridF_.max_load_factor(hashTableMaxLoadFactor_);
	cellGridB_.max_load_factor(hashTableMaxLoadFactor_);
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	kernelRadiusSqu_ = 0.0f;
	cellQueryReach_ = 1;
	cellOrigX_ = 0;
	cellOrigY_ = 0;
	cellOrigZ_ = 0;
	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
}

void NS_SpatialHashing_Basic_CPU01::Construct(const SimParams& params)
{
	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	const float querySpanCells = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize);

	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellSize_ = cellSize;
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = Base::ApproxEqu(cellSize, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");
		cellGridF_.clear();
		cellGridB_.clear();
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;

		return;
	}

	// compute local cell frame for mapping signed -> positive cell coordinates
	const SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B);
	const CellCoordFrame cellFrame = ComputeCellCoordFrame_FromAABB(sceneAABB, cellSizeInv_);

	cellOrigX_ = cellFrame.cellOrigX;
	cellOrigY_ = cellFrame.cellOrigY;
	cellOrigZ_ = cellFrame.cellOrigZ;
	N_DimX_ = cellFrame.N_DimX;
	N_DimY_ = cellFrame.N_DimY;
	N_DimZ_ = cellFrame.N_DimZ;

	static constexpr uint32_t MaxAxisCells = (1u << 21);
	if (N_DimX_ > MaxAxisCells || N_DimY_ > MaxAxisCells || N_DimZ_ > MaxAxisCells)
	{
		SPDLOG_ERROR("Spatial hashing basic frame exceeds 21-bit axis limit [{}, {}, {}]",
			N_DimX_, N_DimY_, N_DimZ_);
		assert(false);
		cellGridF_.clear();
		cellGridB_.clear();
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;

		return;
	}

	// clear hash tables completely, including per-cell vectors
	cellGridF_.clear();
	cellGridB_.clear();

	// reserve memory for cells, assuming average of ~4 particles per cell
	if (N_F)
	{
		// TODO: fine-tune heuristic?
		cellGridF_.reserve(std::max<uint32_t>(1024, N_F / 4));
	}
	if (N_B)
	{
		// TODO: fine-tune heuristic?
		cellGridB_.reserve(std::max<uint32_t>(1024, N_B / 4));
	}

	// fill in fluid cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysF(N_F);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute local cell coordinates
					const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_) - cellOrigX_;
					const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_) - cellOrigY_;
					const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_) - cellOrigZ_;
					assert(cx >= 0 && cx < static_cast<int>(N_DimX_));
					assert(cy >= 0 && cy < static_cast<int>(N_DimY_));
					assert(cz >= 0 && cz < static_cast<int>(N_DimZ_));

					// convert cell index into key
					cellKeysF[i] = PackKey(static_cast<uint32_t>(cx),
										   static_cast<uint32_t>(cy),
										   static_cast<uint32_t>(cz));
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < N_F; i++)
		{
			const Key key = cellKeysF[i];
			cellGridF_[key].push_back(i);
		}
	}

	// fill in boundary cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysB(N_B);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute local cell coordinates
					const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_) - cellOrigX_;
					const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_) - cellOrigY_;
					const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_) - cellOrigZ_;
					assert(cx >= 0 && cx < static_cast<int>(N_DimX_));
					assert(cy >= 0 && cy < static_cast<int>(N_DimY_));
					assert(cz >= 0 && cz < static_cast<int>(N_DimZ_));

					// convert cell index into key
					cellKeysB[i] = PackKey(static_cast<uint32_t>(cx),
										   static_cast<uint32_t>(cy),
										   static_cast<uint32_t>(cz));
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < N_B; i++)
		{
			const Key key = cellKeysB[i];
			cellGridB_[key].push_back(i);
		}
	}
}

void NS_SpatialHashing_Basic_CPU01::Query_Count(const SimParams& params)
{
	(void)params;
	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// local cell coordinates of particle i
				const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
				const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
				const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

				// check all particles j in neighboring grid cells
				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int cz = czi + rz;
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int cy = cyi + ry;
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int cx = cxi + rx;
							if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

							// current cell key
							const uint64_t key = PackKey(static_cast<uint32_t>(cx),
								static_cast<uint32_t>(cy),
								static_cast<uint32_t>(cz));

							// fluid particles
							if (auto bucket = cellGridF_.find(key);
								bucket != cellGridF_.end())
							{
								// check all neighbor candidates in the cell bucket
								for (const uint32_t j : bucket->second)
								{
									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFF++;
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
									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFB++;
									}
								}
							}
						}
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);

	// candidate negativity statistics are only determined during fill queries
	be_.negNbrCandPctFF_ = 0.0f;
	be_.negNbrCandPctFB_ = 0.0f;
}

void NS_SpatialHashing_Basic_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();
	std::atomic<uint64_t> candCountFF{ 0 };
	std::atomic<uint64_t> posCountFF{ 0 };
	std::atomic<uint64_t> candCountFB{ 0 };
	std::atomic<uint64_t> posCountFB{ 0 };

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			uint64_t candCountFF_local = 0;
			uint64_t posCountFF_local = 0;
			uint64_t candCountFB_local = 0;
			uint64_t posCountFB_local = 0;

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// local cell coordinates of particle i
				const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
				const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
				const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

				// check all particles j in neighboring grid cells
				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int cz = czi + rz;
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int cy = cyi + ry;
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int cx = cxi + rx;
							if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

							// current cell key
							const uint64_t key = PackKey(static_cast<uint32_t>(cx),
								static_cast<uint32_t>(cy),
								static_cast<uint32_t>(cz));

							// fluid particles
							if (auto bucket = cellGridF_.find(key);
								bucket != cellGridF_.end())
							{
								// check all neighbor candidates in the cell bucket
								for (const uint32_t j : bucket->second)
								{
									candCountFF_local++;

									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										posCountFF_local++;
										scratchFF.push_back(j);
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
									candCountFB_local++;

									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										posCountFB_local++;
										scratchFB.push_back(j);
									}
								}
							}
						}
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}

			candCountFF.fetch_add(candCountFF_local, std::memory_order_relaxed);
			posCountFF.fetch_add(posCountFF_local, std::memory_order_relaxed);
			candCountFB.fetch_add(candCountFB_local, std::memory_order_relaxed);
			posCountFB.fetch_add(posCountFB_local, std::memory_order_relaxed);
		},
		tbb::auto_partitioner{}
	);

	ComputeNegativeNeighborCandidatePercentages(
		candCountFF.load(std::memory_order_relaxed),
		posCountFF.load(std::memory_order_relaxed),
		candCountFB.load(std::memory_order_relaxed),
		posCountFB.load(std::memory_order_relaxed),
		be_.negNbrCandPctFF_,
		be_.negNbrCandPctFB_
	);
}

NeighborSpan NS_SpatialHashing_Basic_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
	const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
	const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint64_t key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				if (auto bucket = cellGridF_.find(key);
					bucket != cellGridF_.end())
				{
					for (const uint32_t j : bucket->second)
					{
						const float dx = xi - xPosF[j];
						const float dy = yi - yPosF[j];
						const float dz = zi - zPosF[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							scratchFF.push_back(j);
						}
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_SpatialHashing_Basic_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
	const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
	const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint64_t key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				if (auto bucket = cellGridB_.find(key);
					bucket != cellGridB_.end())
				{
					for (const uint32_t j : bucket->second)
					{
						const float dx = xi - xPosB[j];
						const float dy = yi - yPosB[j];
						const float dz = zi - zPosB[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							scratchFB.push_back(j);
						}
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_SpatialHashing_Basic_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_SpatialHashing_Basic_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	//Mem::Add(usageF, Mem::MemUsageOf_UnorderedDenseMap(cellGridF_));
	//Mem::Add(usageB, Mem::MemUsageOf_UnorderedDenseMap(cellGridB_));

	return { usageF, usageB };
}

inline uint64_t NS_SpatialHashing_Basic_CPU01::PackKey(
	uint32_t cx, uint32_t cy, uint32_t cz) noexcept
{
	// guard against overflow
	assert(cx < (1u << 21) && cy < (1u << 21) && cz < (1u << 21));

	// pack into 64-bit key
	return static_cast<uint64_t>(cx)
		| (static_cast<uint64_t>(cy) << 21)
		| (static_cast<uint64_t>(cz) << 42);
}

// -------------------------------------------------------------------------------------------------
// --- [ spatial hashing compact implementation ] --------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_SpatialHashing_Compact_CPU01::Init(const SimParams& params)
{
	cellGridF_.clear();
	cellGridB_.clear();
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	cellOrigX_ = 0;
	cellOrigY_ = 0;
	cellOrigZ_ = 0;
	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
}

void NS_SpatialHashing_Compact_CPU01::Construct(const SimParams& params)
{
	cellSize_ = params.kernelRadius;
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
		? 0.0f : (1.0f / params.kernelRadius);

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");
		cellGridF_.clear();
		cellGridB_.clear();
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;

		return;
	}

	// compute local cell frame for mapping signed -> positive cell coordinates
	const SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B);
	const CellCoordFrame cellFrame = ComputeCellCoordFrame_FromAABB(sceneAABB, cellSizeInv_);

	cellOrigX_ = cellFrame.cellOrigX;
	cellOrigY_ = cellFrame.cellOrigY;
	cellOrigZ_ = cellFrame.cellOrigZ;
	N_DimX_ = cellFrame.N_DimX;
	N_DimY_ = cellFrame.N_DimY;
	N_DimZ_ = cellFrame.N_DimZ;

	static constexpr uint32_t MaxAxisCells = (1u << 21);
	if (N_DimX_ > MaxAxisCells || N_DimY_ > MaxAxisCells || N_DimZ_ > MaxAxisCells)
	{
		SPDLOG_ERROR("Spatial hashing compact frame exceeds 21-bit axis limit [{}, {}, {}]",
			N_DimX_, N_DimY_, N_DimZ_);
		assert(false);
		cellGridF_.clear();
		cellGridB_.clear();
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;

		return;
	}

	// clear hash tables completely, including per-cell vectors
	cellGridF_.clear();
	cellGridB_.clear();

	// reserve memory for cells, assuming average of ~4 particles per cell
	if (N_F)
	{
		// TODO: fine-tune heuristic?
		cellGridF_.reserve(std::max<uint32_t>(1024, N_F / 4));
	}
	if (N_B)
	{
		// TODO: fine-tune heuristic?
		cellGridB_.reserve(std::max<uint32_t>(1024, N_B / 4));
	}

	// fill in fluid cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysF(N_F);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute local cell coordinates
					const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_) - cellOrigX_;
					const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_) - cellOrigY_;
					const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_) - cellOrigZ_;
					assert(cx >= 0 && cx < static_cast<int>(N_DimX_));
					assert(cy >= 0 && cy < static_cast<int>(N_DimY_));
					assert(cz >= 0 && cz < static_cast<int>(N_DimZ_));

					// convert cell index into key
					cellKeysF[i] = PackKey(static_cast<uint32_t>(cx),
										   static_cast<uint32_t>(cy),
										   static_cast<uint32_t>(cz));
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < N_F; i++)
		{
			const Key key = cellKeysF[i];
			cellGridF_[key].push_back(i);
		}
	}

	// fill in boundary cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysB(N_B);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute local cell coordinates
					const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_) - cellOrigX_;
					const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_) - cellOrigY_;
					const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_) - cellOrigZ_;
					assert(cx >= 0 && cx < static_cast<int>(N_DimX_));
					assert(cy >= 0 && cy < static_cast<int>(N_DimY_));
					assert(cz >= 0 && cz < static_cast<int>(N_DimZ_));

					// convert cell index into key
					cellKeysB[i] = PackKey(static_cast<uint32_t>(cx),
										   static_cast<uint32_t>(cy),
										   static_cast<uint32_t>(cz));
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the hash table single-threaded
		for (uint32_t i = 0; i < N_B; i++)
		{
			const Key key = cellKeysB[i];
			cellGridB_[key].push_back(i);
		}
	}
}

void NS_SpatialHashing_Compact_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// local cell coordinates of particle i
				const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
				const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
				const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

				// check all particles j in 27 neighboring grid cells
				for (int rz = -1; rz <= 1; rz++)
				{
					const int cz = czi + rz;
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -1; ry <= 1; ry++)
					{
						const int cy = cyi + ry;
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -1; rx <= 1; rx++)
						{
							const int cx = cxi + rx;
							if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

							// current cell key
							const uint64_t key = PackKey(static_cast<uint32_t>(cx),
														 static_cast<uint32_t>(cy),
														 static_cast<uint32_t>(cz));

							// fluid particles
							if (auto bucket = cellGridF_.find(key);
								bucket != cellGridF_.end())
							{
								// check all neighbor candidates in the cell bucket
								for (const uint32_t j : bucket->second)
								{
									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
														+ dy * dy
														+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFF++;
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
									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
														+ dy * dy
														+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFB++;
									}
								}
							}
						}
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_SpatialHashing_Compact_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// local cell coordinates of particle i
				const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
				const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
				const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

				// check all particles j in 27 neighboring grid cells
				for (int rz = -1; rz <= 1; rz++)
				{
					const int cz = czi + rz;
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -1; ry <= 1; ry++)
					{
						const int cy = cyi + ry;
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -1; rx <= 1; rx++)
						{
							const int cx = cxi + rx;
							if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

							// current cell key
							const uint64_t key = PackKey(static_cast<uint32_t>(cx),
														 static_cast<uint32_t>(cy),
														 static_cast<uint32_t>(cz));

							// fluid particles
							if (auto bucket = cellGridF_.find(key);
								bucket != cellGridF_.end())
							{
								// check all neighbor candidates in the cell bucket
								for (const uint32_t j : bucket->second)
								{
									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
														+ dy * dy
														+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										scratchFF.push_back(j);
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
									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
														+ dy * dy
														+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										scratchFB.push_back(j);
									}
								}
							}
						}
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

NeighborSpan NS_SpatialHashing_Compact_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const float kernelRadiusSqu = cellSize_ * cellSize_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
	const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
	const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

	for (int rz = -1; rz <= 1; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -1; ry <= 1; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -1; rx <= 1; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint64_t key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				if (auto bucket = cellGridF_.find(key);
					bucket != cellGridF_.end())
				{
					for (const uint32_t j : bucket->second)
					{
						const float dx = xi - xPosF[j];
						const float dy = yi - yPosF[j];
						const float dz = zi - zPosF[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							scratchFF.push_back(j);
						}
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_SpatialHashing_Compact_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const float kernelRadiusSqu = cellSize_ * cellSize_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt(xi * cellSizeInv_) - cellOrigX_;
	const int cyi = FloorToInt(yi * cellSizeInv_) - cellOrigY_;
	const int czi = FloorToInt(zi * cellSizeInv_) - cellOrigZ_;

	for (int rz = -1; rz <= 1; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -1; ry <= 1; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -1; rx <= 1; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint64_t key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				if (auto bucket = cellGridB_.find(key);
					bucket != cellGridB_.end())
				{
					for (const uint32_t j : bucket->second)
					{
						const float dx = xi - xPosB[j];
						const float dy = yi - yPosB[j];
						const float dz = zi - zPosB[j];
						const float distSqu = dx * dx + dy * dy + dz * dz;

						if (distSqu <= kernelRadiusSqu) [[unlikely]]
						{
							scratchFB.push_back(j);
						}
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_SpatialHashing_Compact_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_SpatialHashing_Compact_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_UnorderedDenseMap(cellGridF_));
	Mem::Add(usageB, Mem::MemUsageOf_UnorderedDenseMap(cellGridB_));

	return { usageF, usageB };
}

inline uint64_t NS_SpatialHashing_Compact_CPU01::PackKey(
	uint32_t cx, uint32_t cy, uint32_t cz) noexcept
{
	// guard against overflow
	assert(cx < (1u << 21) && cy < (1u << 21) && cz < (1u << 21));

	// pack into 64-bit key
	return static_cast<uint64_t>(cx)
		| (static_cast<uint64_t>(cy) << 21)
		| (static_cast<uint64_t>(cz) << 42);
}

// -------------------------------------------------------------------------------------------------
// --- [ spatial hashing explicit implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_SpatialHashing_Explicit_CPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;

	xBoxOrig_ = 0.0f;
	yBoxOrig_ = 0.0f;
	zBoxOrig_ = 0.0f;

	xBoxMax_ = 0.0f;
	yBoxMax_ = 0.0f;
	zBoxMax_ = 0.0f;

	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
	N_Cells_ = 0;

	cellGridF_.clear();
	cellGridB_.clear();
}

void NS_SpatialHashing_Explicit_CPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");

		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		N_Cells_ = 0;

		cellGridF_.clear();
		cellGridB_.clear();

		return;
	}

	const bool cellSizeChanged = !Base::ApproxEqu(cellSize_, params.kernelRadius);
	cellSize_ = params.kernelRadius;
	cellSizeInv_ = Base::ApproxEqu(params.kernelRadius, 0.0f)
		? 0.0f : (1.0f / params.kernelRadius);

	// --- [ combined AABB over fluid and boundary particles ] ----------------------------------------
	SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B);
	ExpandSceneAABB(sceneAABB, AABB_pad_);

	const float xMin = sceneAABB.minX;
	const float yMin = sceneAABB.minY;
	const float zMin = sceneAABB.minZ;
	const float xMax = sceneAABB.maxX;
	const float yMax = sceneAABB.maxY;
	const float zMax = sceneAABB.maxZ;

	// determine whether to rebuild the grid dimensions
	bool rebuildGrid = cellSizeChanged
		|| N_Cells_ == 0
		|| cellGridF_.size() != static_cast<size_t>(N_Cells_)
		|| cellGridB_.size() != static_cast<size_t>(N_Cells_);

	if (!rebuildGrid)
	{
		const bool outsidePrevAABB =
			(xMin < xBoxOrig_) || (yMin < yBoxOrig_) || (zMin < zBoxOrig_) ||
			(xMax > xBoxMax_) || (yMax > yBoxMax_) || (zMax > zBoxMax_);
		rebuildGrid = outsidePrevAABB;
	}

	// --- [ initialize / refresh explicit grid ] -------------------------------------------------

	if (rebuildGrid)
	{
		// expand AABB by a margin (in cells) to avoid rebuilding every step
		const float margin = static_cast<float>(N_gridMarginCells_) * cellSize_;

		xBoxOrig_ = xMin - margin;
		yBoxOrig_ = yMin - margin;
		zBoxOrig_ = zMin - margin;

		xBoxMax_ = xMax + margin;
		yBoxMax_ = yMax + margin;
		zBoxMax_ = zMax + margin;

		const float xLen = fmaxf(0.0f, xBoxMax_ - xBoxOrig_);
		const float yLen = fmaxf(0.0f, yBoxMax_ - yBoxOrig_);
		const float zLen = fmaxf(0.0f, zBoxMax_ - zBoxOrig_);

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
		if (N_Cells > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
		{
			SPDLOG_ERROR("Too many grid cells for explicit vector storage ({})", N_Cells);

			N_DimX_ = 0;
			N_DimY_ = 0;
			N_DimZ_ = 0;
			N_Cells_ = 0;

			cellGridF_.clear();
			cellGridB_.clear();

			return;
		}
		N_Cells_ = N_Cells;

		cellGridF_.clear();
		cellGridB_.clear();
		cellGridF_.resize(static_cast<size_t>(N_Cells_));
		cellGridB_.resize(static_cast<size_t>(N_Cells_));
	}
	else
	{
		// clear all per-cell buckets, but keep capacity
		tbb::parallel_for(
			tbb::blocked_range<uint64_t>(0, N_Cells_),
			[&](const tbb::blocked_range<uint64_t>& r)
			{
				for (uint64_t c = r.begin(); c != r.end(); c++)
				{
					cellGridF_[static_cast<size_t>(c)].clear();
					cellGridB_[static_cast<size_t>(c)].clear();
				}
			},
			tbb::auto_partitioner{}
		);
	}

	// fill in fluid cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysF(N_F);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute cell coordinates
					int cx = FloorToInt((be_.xPosF_[i] - xBoxOrig_) * cellSizeInv_);
					int cy = FloorToInt((be_.yPosF_[i] - yBoxOrig_) * cellSizeInv_);
					int cz = FloorToInt((be_.zPosF_[i] - zBoxOrig_) * cellSizeInv_);

					// clamp to grid dimensions
					uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
					uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
					uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

					cellKeysF[i] = PackKey(ux, uy, uz);
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the cell grid single-threaded
		for (uint32_t i = 0; i < N_F; i++)
		{
			const Key key = cellKeysF[i];
			cellGridF_[static_cast<size_t>(key)].push_back(i);
		}
	}

	// fill in boundary cell grid
	{
		// compute keys in parallel
		std::vector<Key> cellKeysB(N_B);

		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute cell coordinates
					int cx = FloorToInt((be_.xPosB_[i] - xBoxOrig_) * cellSizeInv_);
					int cy = FloorToInt((be_.yPosB_[i] - yBoxOrig_) * cellSizeInv_);
					int cz = FloorToInt((be_.zPosB_[i] - zBoxOrig_) * cellSizeInv_);

					// clamp to grid dimensions
					uint32_t ux = ClampInt(cx, 0, static_cast<int>(N_DimX_ - 1));
					uint32_t uy = ClampInt(cy, 0, static_cast<int>(N_DimY_ - 1));
					uint32_t uz = ClampInt(cz, 0, static_cast<int>(N_DimZ_ - 1));

					cellKeysB[i] = PackKey(ux, uy, uz);
				}
			},
			tbb::auto_partitioner{}
		);

		// then insert into the cell grid single-threaded
		for (uint32_t i = 0; i < N_B; i++)
		{
			const Key key = cellKeysB[i];
			cellGridB_[static_cast<size_t>(key)].push_back(i);
		}
	}
}

void NS_SpatialHashing_Explicit_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

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

							// current cell key
							const Key key = PackKey(static_cast<uint32_t>(cx),
													static_cast<uint32_t>(cy),
													static_cast<uint32_t>(cz));

							// fluid particles
							const Bucket& bucketF = cellGridF_[static_cast<size_t>(key)];
							for (const uint32_t j : bucketF)
							{
								const float dx = xi - xPosF[j];
								const float dy = yi - yPosF[j];
								const float dz = zi - zPosF[j];

								// squared distance between particles i and j
								const float distSqu = dx * dx
													+ dy * dy
													+ dz * dz;

								if (distSqu <= kernelRadiusSqu) [[unlikely]]
								{
									cntFF++;
								}
							}

							// boundary particles
							const Bucket& bucketB = cellGridB_[static_cast<size_t>(key)];
							for (const uint32_t j : bucketB)
							{
								const float dx = xi - xPosB[j];
								const float dy = yi - yPosB[j];
								const float dz = zi - zPosB[j];

								// squared distance between particles i and j
								const float distSqu = dx * dx
													+ dy * dy
													+ dz * dz;

								if (distSqu <= kernelRadiusSqu) [[unlikely]]
								{
									cntFB++;
								}
							}
						}
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_SpatialHashing_Explicit_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadius * params.kernelRadius;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

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

							// current cell key
							const Key key = PackKey(static_cast<uint32_t>(cx),
													static_cast<uint32_t>(cy),
													static_cast<uint32_t>(cz));

							// fluid particles
							const Bucket& bucketF = cellGridF_[static_cast<size_t>(key)];
							for (const uint32_t j : bucketF)
							{
								const float dx = xi - xPosF[j];
								const float dy = yi - yPosF[j];
								const float dz = zi - zPosF[j];

								// squared distance between particles i and j
								const float distSqu = dx * dx
													+ dy * dy
													+ dz * dz;

								if (distSqu <= kernelRadiusSqu) [[unlikely]]
								{
									// accumulate actual neighbors in scratch list
									scratchFF.push_back(j);
								}
							}

							// boundary particles
							const Bucket& bucketB = cellGridB_[static_cast<size_t>(key)];
							for (const uint32_t j : bucketB)
							{
								const float dx = xi - xPosB[j];
								const float dy = yi - yPosB[j];
								const float dz = zi - zPosB[j];

								// squared distance between particles i and j
								const float distSqu = dx * dx
													+ dy * dy
													+ dz * dz;

								if (distSqu <= kernelRadiusSqu) [[unlikely]]
								{
									// accumulate actual neighbors in scratch list
									scratchFB.push_back(j);
								}
							}
						}
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

NeighborSpan NS_SpatialHashing_Explicit_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const float kernelRadiusSqu = cellSize_ * cellSize_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
	const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
	const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

	for (int rz = -1; rz <= 1; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -1; ry <= 1; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -1; rx <= 1; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const Key key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));
				const Bucket& bucketF = cellGridF_[static_cast<size_t>(key)];

				for (const uint32_t j : bucketF)
				{
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						scratchFF.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_SpatialHashing_Explicit_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const float kernelRadiusSqu = cellSize_ * cellSize_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
	const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
	const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

	for (int rz = -1; rz <= 1; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -1; ry <= 1; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -1; rx <= 1; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const Key key = PackKey(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));
				const Bucket& bucketB = cellGridB_[static_cast<size_t>(key)];

				for (const uint32_t j : bucketB)
				{
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						scratchFB.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_SpatialHashing_Explicit_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_SpatialHashing_Explicit_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_VectorOfVector(cellGridF_));
	Mem::Add(usageB, Mem::MemUsageOf_VectorOfVector(cellGridB_));

	return { usageF, usageB };
}

inline NS_SpatialHashing_Explicit_CPU01::Key NS_SpatialHashing_Explicit_CPU01::PackKey(
	uint32_t cx, uint32_t cy, uint32_t cz) const noexcept
{
	return static_cast<Key>(cx) + static_cast<Key>(N_DimX_) * (static_cast<Key>(cy)
								+ static_cast<Key>(N_DimY_) * static_cast<Key>(cz));
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list basic implementation ] ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Basic_CPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	kernelRadiusSqu_ = 0.0f;
	cellQueryReach_ = 1;

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

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	const float querySpanCells = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize);

	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellSize_ = cellSize;
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = Base::ApproxEqu(cellSize, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	// --- [ combined AABB over fluid and boundary particles ] ----------------------------------------
	SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), be_.N_F_,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), be_.N_B_);

	// padding to avoid edge clamp issues
	constexpr float pad = 1e-7f;
	ExpandSceneAABB(sceneAABB, pad);

	const float xMin = sceneAABB.minX;
	const float yMin = sceneAABB.minY;
	const float zMin = sceneAABB.minZ;
	const float xMax = sceneAABB.maxX;
	const float yMax = sceneAABB.maxY;
	const float zMax = sceneAABB.maxZ;

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

	// --- [ preparation data structures ] ------------------------------------------------------------

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
				// compute cell coordinates
				int cx = FloorToInt((be_.xPosF_[i] - xBoxOrig_) * cellSizeInv_);
				int cy = FloorToInt((be_.yPosF_[i] - yBoxOrig_) * cellSizeInv_);
				int cz = FloorToInt((be_.zPosF_[i] - zBoxOrig_) * cellSizeInv_);

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
				// compute cell coordinates
				int cx = FloorToInt((be_.xPosB_[i] - xBoxOrig_) * cellSizeInv_);
				int cy = FloorToInt((be_.yPosB_[i] - yBoxOrig_) * cellSizeInv_);
				int cz = FloorToInt((be_.zPosB_[i] - zBoxOrig_) * cellSizeInv_);

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

	// --- [ sort particle indices by cell IDs ] ------------------------------------------------------

	// single-threaded basic sorting version
	/*
	std::ranges::sort(sortedParticleIdxF_,
		[&](uint32_t a, uint32_t b) { return cellIdF_[a] < cellIdF_[b]; });

	std::ranges::sort(sortedParticleIdxB_,
		[&](uint32_t a, uint32_t b) { return cellIdB_[a] < cellIdB_[b]; });
	*/

	// multi-threaded parallel sorting version using C++ standard library
	/*
	std::sort(std::execution::par, sortedParticleIdxF_.begin(), sortedParticleIdxF_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdF_[a] < cellIdF_[b]; });

	std::sort(std::execution::par, sortedParticleIdxB_.begin(), sortedParticleIdxB_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdB_[a] < cellIdB_[b]; });
	*/

	// multi-threaded parallel sorting version using TBB
	tbb::parallel_sort(sortedParticleIdxF_.begin(), sortedParticleIdxF_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdF_[a] < cellIdF_[b]; });

	tbb::parallel_sort(sortedParticleIdxB_.begin(), sortedParticleIdxB_.end(),
		[&](uint32_t a, uint32_t b) { return cellIdB_[a] < cellIdB_[b]; });

	// --- [ fill in cell start/end indices ] ---------------------------------------------------------

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

void NS_CellLinkedList_Basic_CPU01::Query_Count(const SimParams& params)
{
	// TODO: instead of iterating over particles in the outermost loop,
	// TODO: iterate over cells and their particles, because for all particles
	// TODO: in a cell, the neighboring cells and neighbor candidates are the same?

	(void)params;
	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// cell coordinates of particle i
				const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
				const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
				const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

				// check all particles j in neighboring grid cells
				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					int cz = czi + rz; // target cell coordinate in z-direction
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						int cy = cyi + ry; // target cell coordinate in y-direction
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
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
									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFF++;
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
									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										cntFB++;
									}
								}
							}
						}
					}
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_CellLinkedList_Basic_CPU01::Query_Fill(const SimParams& params)
{
	// TODO: instead of iterating over particles in the outermost loop,
	// TODO: iterate over cells and their particles, because for all particles
	// TODO: in a cell, the neighboring cells and neighbor candidates are the same?

	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// cell coordinates of particle i
				const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
				const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
				const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

				// check all particles j in neighboring grid cells
				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					int cz = czi + rz; // target cell coordinate in z-direction
					if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						int cy = cyi + ry; // target cell coordinate in y-direction
						if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
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
									const float dx = xi - xPosF[j];
									const float dy = yi - yPosF[j];
									const float dz = zi - zPosF[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										scratchFF.push_back(j);
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
									const float dx = xi - xPosB[j];
									const float dy = yi - yPosB[j];
									const float dz = zi - zPosB[j];

									// squared distance between particles i and j
									const float distSqu = dx * dx
										+ dy * dy
										+ dz * dz;

									if (distSqu <= kernelRadiusSqu) [[unlikely]]
									{
										// accumulate actual neighbors in scratch list
										scratchFB.push_back(j);
									}
								}
							}
						}
					}
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

NeighborSpan NS_CellLinkedList_Basic_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
	const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
	const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				const uint32_t sF = cellStartIdxF_[cid];
				const uint32_t eF = cellEndIdxF_[cid];
				if (sF == EMPTY || eF == EMPTY) { continue; }

				for (uint32_t k = sF; k < eF; k++)
				{
					const uint32_t j = sortedParticleIdxF_[k];
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						scratchFF.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_CellLinkedList_Basic_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const float kernelRadiusSqu = kernelRadiusSqu_;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cxi = FloorToInt((xi - xBoxOrig_) * cellSizeInv_);
	const int cyi = FloorToInt((yi - yBoxOrig_) * cellSizeInv_);
	const int czi = FloorToInt((zi - zBoxOrig_) * cellSizeInv_);

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		const int cz = czi + rz;
		if (cz < 0 || cz >= static_cast<int>(N_DimZ_)) { continue; }

		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			const int cy = cyi + ry;
			if (cy < 0 || cy >= static_cast<int>(N_DimY_)) { continue; }

			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const int cx = cxi + rx;
				if (cx < 0 || cx >= static_cast<int>(N_DimX_)) { continue; }

				const uint32_t cid = MapToCellId_Basic(static_cast<uint32_t>(cx),
					static_cast<uint32_t>(cy),
					static_cast<uint32_t>(cz));

				const uint32_t sB = cellStartIdxB_[cid];
				const uint32_t eB = cellEndIdxB_[cid];
				if (sB == EMPTY || eB == EMPTY) { continue; }

				for (uint32_t k = sB; k < eB; k++)
				{
					const uint32_t j = sortedParticleIdxB_[k];
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu) [[unlikely]]
					{
						scratchFB.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_CellLinkedList_Basic_CPU01::DataReOrder(
	const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_CellLinkedList_Basic_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellStartIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellEndIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(sortedParticleIdxF_));

	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellEndIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(sortedParticleIdxB_));

	return { usageF, usageB };
}

inline uint32_t NS_CellLinkedList_Basic_CPU01::MapToCellId_Basic(
	uint32_t cx, uint32_t cy, uint32_t cz) const noexcept
{
	return cx + N_DimX_ * (cy + N_DimY_ * cz);
}

// -------------------------------------------------------------------------------------------------
// --- [ cell linked list compact implementation ] -------------------------------------------------
// -------------------------------------------------------------------------------------------------

void NS_CellLinkedList_Compact_CPU01::Init(const SimParams& params)
{
	cellSize_ = 0.0f;
	cellSizeInv_ = 0.0f;
	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellQueryReach_ = 1;
	cellOrigX_ = 0;
	cellOrigY_ = 0;
	cellOrigZ_ = 0;
	N_DimX_ = 0;
	N_DimY_ = 0;
	N_DimZ_ = 0;
	cellFrameValid_ = false;

	uniqueCellIdF_.clear();
	cellStartIdxF_.clear();
	cellIdF_.clear();
	sortedParticleIdxF_.clear();

	uniqueCellIdB_.clear();
	cellStartIdxB_.clear();
	cellIdB_.clear();
	sortedParticleIdxB_.clear();
}

void NS_CellLinkedList_Compact_CPU01::Construct(const SimParams& params)
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		cellFrameValid_ = false;

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

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	const float querySpanCells = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (params.kernelRadius / cellSize);
	const uint32_t cellQueryReach = Base::ApproxEqu(cellSize, 0.0f)
		? 1u : std::max<uint32_t>(1u,
			static_cast<uint32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));

	const bool cellSizeChanged = !Base::ApproxEqu(cellSize_, cellSize);
	const bool cellQueryReachChanged = (cellQueryReach_ != cellQueryReach);

	kernelRadiusSqu_ = params.kernelRadiusSqu;
	cellSize_ = cellSize;
	cellSizeInv_ = Base::ApproxEqu(cellSize, 0.0f)
		? 0.0f : (1.0f / cellSize);
	cellQueryReach_ = cellQueryReach;

	if (cellSizeChanged || cellQueryReachChanged) { cellFrameValid_ = false; }
	UpdateCellFrame();

	// --- [ cell coordinates and indices for fluid and boundary particles ] --------------------------

	// fluid
	cellIdF_.resize(N_F);
	sortedParticleIdxF_.resize(N_F);

	if (N_F > 0)
	{
		// TODO: should sortedParticleIdxF_ be initialized to identity in every step?
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_F),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute cell coordinates
					const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
					const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
					const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

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
		// TODO: should sortedParticleIdxB_ be initialized to identity in every step?
		tbb::parallel_for(tbb::blocked_range<uint32_t>(0, N_B),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t i = r.begin(); i != r.end(); i++)
				{
					// compute cell coordinates
					const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
					const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
					const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

					cellIdB_[i] = MapToCellId_Morton(cx, cy, cz);
					sortedParticleIdxB_[i] = i; // initial identity permutation (unsorted)
				}
			},
			tbb::auto_partitioner{}
		);
	}

	// --- [ sort particle indices by cell IDs ] ------------------------------------------------------

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

	// --- [ build compact cell list and fill in cell start indices ] ---------------------------------

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

void NS_CellLinkedList_Compact_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const uint32_t N_NbrCells = NeighborCellCount();

	// local references for fewer pointer indirections
	float* RESTRICT xPosF = be_.xPosF_.data();
	float* RESTRICT yPosF = be_.yPosF_.data();
	float* RESTRICT zPosF = be_.zPosF_.data();
	float* RESTRICT xPosB = be_.xPosB_.data();
	float* RESTRICT yPosB = be_.yPosB_.data();
	float* RESTRICT zPosB = be_.zPosB_.data();

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			static constexpr uint32_t LookupCacheSize = 256;
			static_assert((LookupCacheSize & (LookupCacheSize - 1)) == 0);

			std::array<uint64_t, LookupCacheSize> lookupCacheKeyF{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValF{};
			std::array<uint64_t, LookupCacheSize> lookupCacheKeyB{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValB{};
			lookupCacheKeyF.fill(EMPTY_64);
			lookupCacheKeyB.fill(EMPTY_64);
			lookupCacheValF.fill(EMPTY_32);
			lookupCacheValB.fill(EMPTY_32);

			std::vector<uint8_t> hasFluidCell(N_NbrCells, 0);
			std::vector<uint32_t> sNbrF(N_NbrCells, 0);
			std::vector<uint32_t> eNbrF(N_NbrCells, 0);

			std::vector<uint8_t> hasBoundaryCell(N_NbrCells, 0);
			std::vector<uint32_t> sNbrB(N_NbrCells, 0);
			std::vector<uint32_t> eNbrB(N_NbrCells, 0);

			std::vector<uint64_t> nbrCellId(N_NbrCells, 0);

			const auto findCellIdxTernary = [](const std::vector<uint64_t>& sortedCellIds,
				uint64_t key,
				uint32_t lo,
				uint32_t hi) noexcept
				{
					while (hi > lo + 24u)
					{
						const uint32_t span = hi - lo;
						const uint32_t third = span / 3u;
						const uint32_t m1 = lo + third;
						const uint32_t m2 = hi - third - 1u;

						const uint64_t v1 = sortedCellIds[m1];
						const uint64_t v2 = sortedCellIds[m2];

						if (key < v1)
						{
							hi = m1;
							continue;
						}
						if (key > v2)
						{
							lo = m2 + 1u;
							continue;
						}
						if (key == v1) { return m1; }
						if (key == v2) { return m2; }

						lo = m1 + 1u;
						hi = m2;
					}

					while (lo < hi)
					{
						const uint32_t mid = lo + ((hi - lo) >> 1u);
						if (sortedCellIds[mid] < key)
						{
							lo = mid + 1u;
						}
						else
						{
							hi = mid;
						}
					}

					if (lo < sortedCellIds.size() && sortedCellIds[lo] == key)
					{
						return lo;
					}

					return EMPTY_32;
				};

			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// determine a representative particle i0
				const uint32_t i0 = sortedParticleIdxF_[sF];

				const float xi0 = xPosF[i0];
				const float yi0 = yPosF[i0];
				const float zi0 = zPosF[i0];

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// pre-compute neighbor cell IDs
				std::fill(hasFluidCell.begin(), hasFluidCell.end(), 0);
				std::fill(hasBoundaryCell.begin(), hasBoundaryCell.end(), 0);

				// BigMin/LitMax range of the queried neighborhood in Morton space
				uint64_t litMaxCellId = std::numeric_limits<uint64_t>::max();
				uint64_t bigMinCellId = 0;

				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int oz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int oy = cy + ry; // target cell coordinate in y-direction

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int ox = cx + rx; // target cell coordinate in x-direction

							// index of this neighbor cell in the queried neighborhood
							const int idx_nbr = NeighborOffset(rx, ry, rz);

							// cell ID of this neighbor cell
							const uint64_t cid_nbr = MapToCellId_Morton(ox, oy, oz);
							nbrCellId[idx_nbr] = cid_nbr;
							litMaxCellId = std::min(litMaxCellId, cid_nbr);
							bigMinCellId = std::max(bigMinCellId, cid_nbr);
						}
					}
				}

				// fluid lower/upper bounds over the full neighborhood range
				uint32_t loF = 0;
				uint32_t hiF = 0;
				if (!uniqueCellIdF_.empty())
				{
					loF = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdF_, litMaxCellId) - uniqueCellIdF_.begin());
					hiF = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdF_, bigMinCellId) - uniqueCellIdF_.begin());
					if (hiF < loF) { hiF = loF; }
				}

				// boundary lower/upper bounds over the full neighborhood range
				uint32_t loB = 0;
				uint32_t hiB = 0;
				if (!uniqueCellIdB_.empty())
				{
					loB = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdB_, litMaxCellId) - uniqueCellIdB_.begin());
					hiB = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdB_, bigMinCellId) - uniqueCellIdB_.begin());
					if (hiB < loB) { hiB = loB; }
				}

				// check existence and location of all queried grid cells
				for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
				{
					const uint64_t cid_nbr = nbrCellId[idx_nbr];

					// fluid scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cF_nbr = EMPTY_32;

						if (lookupCacheKeyF[slot] == cid_nbr)
						{
							cF_nbr = lookupCacheValF[slot];
						}
						else
						{
							if (loF < hiF)
							{
								cF_nbr = findCellIdxTernary(uniqueCellIdF_, cid_nbr, loF, hiF);
							}
							lookupCacheKeyF[slot] = cid_nbr;
							lookupCacheValF[slot] = cF_nbr;
						}

						if (cF_nbr != EMPTY_32)
						{
							const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
							const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];
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
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cB_nbr = EMPTY_32;

						if (lookupCacheKeyB[slot] == cid_nbr)
						{
							cB_nbr = lookupCacheValB[slot];
						}
						else
						{
							if (loB < hiB)
							{
								cB_nbr = findCellIdxTernary(uniqueCellIdB_, cid_nbr, loB, hiB);
							}
							lookupCacheKeyB[slot] = cid_nbr;
							lookupCacheValB[slot] = cB_nbr;
						}

						if (cB_nbr != EMPTY_32)
						{
							const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
							const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];
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

				// check all particles j in the queried neighboring grid cells
				for (uint32_t k = sF; k < eF; k++)
				{
					uint32_t cntFF = 0;
					uint32_t cntFB = 0;

					const uint32_t i = sortedParticleIdxF_[k];

					const float xi = xPosF[i];
					const float yi = yPosF[i];
					const float zi = zPosF[i];

					// fluid neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
					{
						if (!hasFluidCell[idx_nbr]) { continue; }

						const uint32_t sF_nbr = sNbrF[idx_nbr];
						const uint32_t eF_nbr = eNbrF[idx_nbr];

						#pragma omp simd
						for (uint32_t l = sF_nbr; l < eF_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxF_[l];

							// position difference
							const float dx = xi - xPosF[j];
							const float dy = yi - yPosF[j];
							const float dz = zi - zPosF[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								cntFF++;
							}
						}
					}

					// boundary neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
					{
						if (!hasBoundaryCell[idx_nbr]) { continue; }

						const uint32_t sB_nbr = sNbrB[idx_nbr];
						const uint32_t eB_nbr = eNbrB[idx_nbr];

						#pragma omp simd
						for (uint32_t l = sB_nbr; l < eB_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxB_[l];

							// position difference
							const float dx = xi - xPosB[j];
							const float dy = yi - yPosB[j];
							const float dz = zi - zPosB[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								cntFB++;
							}
						}
					}

					// commit final neighbor counts to backend storage
					be_.neighborLists_->SetNbrCountFF(i, cntFF);
					be_.neighborLists_->SetNbrCountFB(i, cntFB);
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// candidate negativity statistics are only determined during fill queries
	be_.negNbrCandPctFF_ = 0.0f;
	be_.negNbrCandPctFB_ = 0.0f;
}

void NS_CellLinkedList_Compact_CPU01::Query_Fill(const SimParams& params)
{
	Query_Fill_Smart(params);
}

void NS_CellLinkedList_Compact_CPU01::Query_Fill_Basic(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	// local references for fewer pointer indirections
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	std::atomic<uint64_t> candCountFF{ 0 };
	std::atomic<uint64_t> posCountFF{ 0 };
	std::atomic<uint64_t> candCountFB{ 0 };
	std::atomic<uint64_t> posCountFB{ 0 };
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			uint64_t candCountFF_local = 0;
			uint64_t posCountFF_local = 0;
			uint64_t candCountFB_local = 0;
			uint64_t posCountFB_local = 0;

			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// determine a representative particle i0
				const uint32_t i0 = sortedParticleIdxF_[sF];

				const float xi0 = xPosF[i0];
				const float yi0 = yPosF[i0];
				const float zi0 = zPosF[i0];

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// loop over all particles i in this cell
				for (uint32_t k = sF; k < eF; k++)
				{
					const uint32_t i = sortedParticleIdxF_[k];

					scratchFF.clear();
					scratchFB.clear();

					const float xi = xPosF[i];
					const float yi = yPosF[i];
					const float zi = zPosF[i];

					// check all particles j in the queried neighboring grid cells
					for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
					{
						int nz = cz + rz; // target cell coordinate in z-direction

						for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
						{
							int ny = cy + ry; // target cell coordinate in y-direction

							for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
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

									#pragma omp simd
									for (uint32_t l = sF_nbr; l < eF_nbr; l++)
									{
										// index of fluid neighbor candidate
										const uint32_t j = sortedParticleIdxF_[l];
										candCountFF_local++;

										// position difference
										const float dx = xi - xPosF[j];
										const float dy = yi - yPosF[j];
										const float dz = zi - zPosF[j];

										// squared distance between particles i and j
										const float distSqu = dx * dx
															+ dy * dy
															+ dz * dz;

										if (distSqu <= kernelRadiusSqu) [[unlikely]]
										{
											// accumulate actual neighbors in scratch list
											posCountFF_local++;
											scratchFF.push_back(j);
										}
									}
								}

								// find neighbor boundary cell in compact list (binary search)
								// TODO: use ternary search and BigMin-LitMax instead?
								auto itB = std::ranges::lower_bound(uniqueCellIdB_, cid_nbr);

								// check boundary candidates in this neighbor cell
								if (itB != uniqueCellIdB_.end() && *itB == cid_nbr)
								{
									const uint32_t cB_nbr = static_cast<uint32_t>(itB - uniqueCellIdB_.begin());

									// range of all boundary particles in the neighbor cell
									const uint32_t sB_nbr = cellStartIdxB_[cB_nbr]; // start
									const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1]; // end

									#pragma omp simd
									for (uint32_t l = sB_nbr; l < eB_nbr; l++)
									{
										// index of boundary neighbor candidate
										const uint32_t j = sortedParticleIdxB_[l];
										candCountFB_local++;

										// position difference
										const float dx = xi - xPosB[j];
										const float dy = yi - yPosB[j];
										const float dz = zi - zPosB[j];

										// squared distance between particles i and j
										const float distSqu = dx * dx
															+ dy * dy
															+ dz * dz;

										if (distSqu <= kernelRadiusSqu) [[unlikely]]
										{
											// accumulate actual neighbors in scratch list
											posCountFB_local++;
											scratchFB.push_back(j);
										}
									}
								}
							}
						}
					}

					// commit final neighbor lists from scratch to backend storage
					be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
						static_cast<uint32_t>(scratchFF.size()));
					be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
						static_cast<uint32_t>(scratchFB.size()));
				}
			}

			candCountFF.fetch_add(candCountFF_local, std::memory_order_relaxed);
			posCountFF.fetch_add(posCountFF_local, std::memory_order_relaxed);
			candCountFB.fetch_add(candCountFB_local, std::memory_order_relaxed);
			posCountFB.fetch_add(posCountFB_local, std::memory_order_relaxed);
		},
		tbb::auto_partitioner{}
	);

	ComputeNegativeNeighborCandidatePercentages(
		candCountFF.load(std::memory_order_relaxed),
		posCountFF.load(std::memory_order_relaxed),
		candCountFB.load(std::memory_order_relaxed),
		posCountFB.load(std::memory_order_relaxed),
		be_.negNbrCandPctFF_,
		be_.negNbrCandPctFB_
	);
}

void NS_CellLinkedList_Compact_CPU01::Query_Fill_Smart(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const uint32_t N_NbrCells = NeighborCellCount();

	// local references for fewer pointer indirections
	float* RESTRICT xPosF = be_.xPosF_.data();
	float* RESTRICT yPosF = be_.yPosF_.data();
	float* RESTRICT zPosF = be_.zPosF_.data();
	float* RESTRICT xPosB = be_.xPosB_.data();
	float* RESTRICT yPosB = be_.yPosB_.data();
	float* RESTRICT zPosB = be_.zPosB_.data();

	// loop over all non-empty grid cells
	const uint64_t N_Cells = static_cast<uint64_t>(uniqueCellIdF_.size());
	std::atomic<uint64_t> candCountFF{ 0 };
	std::atomic<uint64_t> posCountFF{ 0 };
	std::atomic<uint64_t> candCountFB{ 0 };
	std::atomic<uint64_t> posCountFB{ 0 };
	tbb::parallel_for(
		tbb::blocked_range<uint64_t>(0, N_Cells),
		[&](const tbb::blocked_range<uint64_t>& range)
		{
			static constexpr uint32_t LookupCacheSize = 256;
			static_assert((LookupCacheSize & (LookupCacheSize - 1)) == 0);

			std::array<uint64_t, LookupCacheSize> lookupCacheKeyF{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValF{};
			std::array<uint64_t, LookupCacheSize> lookupCacheKeyB{};
			std::array<uint32_t, LookupCacheSize> lookupCacheValB{};
			lookupCacheKeyF.fill(EMPTY_64);
			lookupCacheKeyB.fill(EMPTY_64);
			lookupCacheValF.fill(EMPTY_32);
			lookupCacheValB.fill(EMPTY_32);

			std::vector<uint8_t> hasFluidCell(N_NbrCells, 0);
			std::vector<uint32_t> sNbrF(N_NbrCells, 0);
			std::vector<uint32_t> eNbrF(N_NbrCells, 0);

			std::vector<uint8_t> hasBoundaryCell(N_NbrCells, 0);
			std::vector<uint32_t> sNbrB(N_NbrCells, 0);
			std::vector<uint32_t> eNbrB(N_NbrCells, 0);

			std::vector<uint64_t> nbrCellId(N_NbrCells, 0);
			uint64_t candCountFF_local = 0;
			uint64_t posCountFF_local = 0;
			uint64_t candCountFB_local = 0;
			uint64_t posCountFB_local = 0;

			const auto findCellIdxTernary = [](const std::vector<uint64_t>& sortedCellIds,
				uint64_t key,
				uint32_t lo,
				uint32_t hi) noexcept
				{
					while (hi > lo + 24u)
					{
						const uint32_t span = hi - lo;
						const uint32_t third = span / 3u;
						const uint32_t m1 = lo + third;
						const uint32_t m2 = hi - third - 1u;

						const uint64_t v1 = sortedCellIds[m1];
						const uint64_t v2 = sortedCellIds[m2];

						if (key < v1)
						{
							hi = m1;
							continue;
						}
						if (key > v2)
						{
							lo = m2 + 1u;
							continue;
						}
						if (key == v1) { return m1; }
						if (key == v2) { return m2; }

						lo = m1 + 1u;
						hi = m2;
					}

					while (lo < hi)
					{
						const uint32_t mid = lo + ((hi - lo) >> 1u);
						if (sortedCellIds[mid] < key)
						{
							lo = mid + 1u;
						}
						else
						{
							hi = mid;
						}
					}

					if (lo < sortedCellIds.size() && sortedCellIds[lo] == key)
					{
						return lo;
					}

					return EMPTY_32;
				};

			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint64_t c = range.begin(); c != range.end(); c++)
			{
				// range of all fluid particles in this cell
				const uint32_t sF = cellStartIdxF_[c]; // start
				const uint32_t eF = cellStartIdxF_[c + 1]; // end
				if (sF >= eF) { continue; }

				// determine a representative particle i0
				const uint32_t i0 = sortedParticleIdxF_[sF];

				const float xi0 = xPosF[i0];
				const float yi0 = yPosF[i0];
				const float zi0 = zPosF[i0];

				// cell coordinates of all particles in this cell
				const int cx = FloorToInt(xi0 * cellSizeInv_);
				const int cy = FloorToInt(yi0 * cellSizeInv_);
				const int cz = FloorToInt(zi0 * cellSizeInv_);

				// pre-compute neighbor cell IDs
				std::fill(hasFluidCell.begin(), hasFluidCell.end(), 0);
				std::fill(hasBoundaryCell.begin(), hasBoundaryCell.end(), 0);

				// BigMin/LitMax range of the queried neighborhood in Morton space
				uint64_t litMaxCellId = std::numeric_limits<uint64_t>::max();
				uint64_t bigMinCellId = 0;

				for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
				{
					const int oz = cz + rz; // target cell coordinate in z-direction

					for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
					{
						const int oy = cy + ry; // target cell coordinate in y-direction

						for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
						{
							const int ox = cx + rx; // target cell coordinate in x-direction

							// index of this neighbor cell in the queried neighborhood
							const int idx_nbr = NeighborOffset(rx, ry, rz);

							// cell ID of this neighbor cell
							const uint64_t cid_nbr = MapToCellId_Morton(ox, oy, oz);
							nbrCellId[idx_nbr] = cid_nbr;
							litMaxCellId = std::min(litMaxCellId, cid_nbr);
							bigMinCellId = std::max(bigMinCellId, cid_nbr);
						}
					}
				}

				// fluid lower/upper bounds over the full neighborhood range
				uint32_t loF = 0;
				uint32_t hiF = 0;
				if (!uniqueCellIdF_.empty())
				{
					loF = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdF_, litMaxCellId) - uniqueCellIdF_.begin());
					hiF = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdF_, bigMinCellId) - uniqueCellIdF_.begin());
					if (hiF < loF) { hiF = loF; }
				}

				// boundary lower/upper bounds over the full neighborhood range
				uint32_t loB = 0;
				uint32_t hiB = 0;
				if (!uniqueCellIdB_.empty())
				{
					loB = static_cast<uint32_t>(
						std::ranges::lower_bound(uniqueCellIdB_, litMaxCellId) - uniqueCellIdB_.begin());
					hiB = static_cast<uint32_t>(
						std::ranges::upper_bound(uniqueCellIdB_, bigMinCellId) - uniqueCellIdB_.begin());
					if (hiB < loB) { hiB = loB; }
				}

				// check existence and location of all queried grid cells
				for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
				{
					const uint64_t cid_nbr = nbrCellId[idx_nbr];

					// fluid scope
					{
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cF_nbr = EMPTY_32;

						if (lookupCacheKeyF[slot] == cid_nbr)
						{
							cF_nbr = lookupCacheValF[slot];
						}
						else
						{
							if (loF < hiF)
							{
								cF_nbr = findCellIdxTernary(uniqueCellIdF_, cid_nbr, loF, hiF);
							}
							lookupCacheKeyF[slot] = cid_nbr;
							lookupCacheValF[slot] = cF_nbr;
						}

						if (cF_nbr != EMPTY_32)
						{
							const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
							const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];
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
						const uint32_t slot = static_cast<uint32_t>(cid_nbr) & (LookupCacheSize - 1u);
						uint32_t cB_nbr = EMPTY_32;

						if (lookupCacheKeyB[slot] == cid_nbr)
						{
							cB_nbr = lookupCacheValB[slot];
						}
						else
						{
							if (loB < hiB)
							{
								cB_nbr = findCellIdxTernary(uniqueCellIdB_, cid_nbr, loB, hiB);
							}
							lookupCacheKeyB[slot] = cid_nbr;
							lookupCacheValB[slot] = cB_nbr;
						}

						if (cB_nbr != EMPTY_32)
						{
							const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
							const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];
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

				// check all particles j in the queried neighboring grid cells
				for (uint32_t k = sF; k < eF; k++)
				{
					// reset scratch neighbor lists
					scratchFF.clear();
					scratchFB.clear();

					const uint32_t i = sortedParticleIdxF_[k];

					const float xi = xPosF[i];
					const float yi = yPosF[i];
					const float zi = zPosF[i];

					// fluid neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
					{
						if (!hasFluidCell[idx_nbr]) { continue; }

						const uint32_t sF_nbr = sNbrF[idx_nbr];
						const uint32_t eF_nbr = eNbrF[idx_nbr];

						#pragma omp simd
						for (uint32_t l = sF_nbr; l < eF_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxF_[l];
							candCountFF_local++;

							// position difference
							const float dx = xi - xPosF[j];
							const float dy = yi - yPosF[j];
							const float dz = zi - zPosF[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// accumulate actual neighbors in scratch list
								posCountFF_local++;
								scratchFF.push_back(j);
							}
						}
					}

					// boundary neighbors
					for (uint32_t idx_nbr = 0; idx_nbr < N_NbrCells; idx_nbr++)
					{
						if (!hasBoundaryCell[idx_nbr]) { continue; }

						const uint32_t sB_nbr = sNbrB[idx_nbr];
						const uint32_t eB_nbr = eNbrB[idx_nbr];

						#pragma omp simd
						for (uint32_t l = sB_nbr; l < eB_nbr; l++)
						{
							const uint32_t j = sortedParticleIdxB_[l];
							candCountFB_local++;

							// position difference
							const float dx = xi - xPosB[j];
							const float dy = yi - yPosB[j];
							const float dz = zi - zPosB[j];

							const float distSqu = dx * dx
												+ dy * dy
												+ dz * dz;

							if (distSqu <= kernelRadiusSqu) [[unlikely]]
							{
								// accumulate actual neighbors in scratch list
								posCountFB_local++;
								scratchFB.push_back(j);
							}
						}
					}

					// commit final neighbor lists from scratch to backend storage
					be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
						static_cast<uint32_t>(scratchFF.size()));
					be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
						static_cast<uint32_t>(scratchFB.size()));
				}
			}

			candCountFF.fetch_add(candCountFF_local, std::memory_order_relaxed);
			posCountFF.fetch_add(posCountFF_local, std::memory_order_relaxed);
			candCountFB.fetch_add(candCountFB_local, std::memory_order_relaxed);
			posCountFB.fetch_add(posCountFB_local, std::memory_order_relaxed);
		},
		tbb::auto_partitioner{}
	);

	ComputeNegativeNeighborCandidatePercentages(
		candCountFF.load(std::memory_order_relaxed),
		posCountFF.load(std::memory_order_relaxed),
		candCountFB.load(std::memory_order_relaxed),
		posCountFB.load(std::memory_order_relaxed),
		be_.negNbrCandPctFF_,
		be_.negNbrCandPctFB_
	);
}

NeighborSpan NS_CellLinkedList_Compact_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cx = FloorToInt(xi * cellSizeInv_);
	const int cy = FloorToInt(yi * cellSizeInv_);
	const int cz = FloorToInt(zi * cellSizeInv_);
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const uint64_t cid_nbr = MapToCellId_Morton(cx + rx, cy + ry, cz + rz);
				auto itF = std::ranges::lower_bound(uniqueCellIdF_, cid_nbr);

				if (itF == uniqueCellIdF_.end() || *itF != cid_nbr) { continue; }

				const uint32_t cF_nbr = static_cast<uint32_t>(itF - uniqueCellIdF_.begin());
				const uint32_t sF_nbr = cellStartIdxF_[cF_nbr];
				const uint32_t eF_nbr = cellStartIdxF_[cF_nbr + 1];

				for (uint32_t l = sF_nbr; l < eF_nbr; l++)
				{
					const uint32_t j = sortedParticleIdxF_[l];
					const float dx = xi - xPosF[j];
					const float dy = yi - yPosF[j];
					const float dz = zi - zPosF[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu_) [[unlikely]]
					{
						scratchFF.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_CellLinkedList_Compact_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	const int cx = FloorToInt(xi * cellSizeInv_);
	const int cy = FloorToInt(yi * cellSizeInv_);
	const int cz = FloorToInt(zi * cellSizeInv_);
	const int cellQueryReach = static_cast<int>(cellQueryReach_);

	for (int rz = -cellQueryReach; rz <= cellQueryReach; rz++)
	{
		for (int ry = -cellQueryReach; ry <= cellQueryReach; ry++)
		{
			for (int rx = -cellQueryReach; rx <= cellQueryReach; rx++)
			{
				const uint64_t cid_nbr = MapToCellId_Morton(cx + rx, cy + ry, cz + rz);
				auto itB = std::ranges::lower_bound(uniqueCellIdB_, cid_nbr);

				if (itB == uniqueCellIdB_.end() || *itB != cid_nbr) { continue; }

				const uint32_t cB_nbr = static_cast<uint32_t>(itB - uniqueCellIdB_.begin());
				const uint32_t sB_nbr = cellStartIdxB_[cB_nbr];
				const uint32_t eB_nbr = cellStartIdxB_[cB_nbr + 1];

				for (uint32_t l = sB_nbr; l < eB_nbr; l++)
				{
					const uint32_t j = sortedParticleIdxB_[l];
					const float dx = xi - xPosB[j];
					const float dy = yi - yPosB[j];
					const float dz = zi - zPosB[j];
					const float distSqu = dx * dx + dy * dy + dz * dz;

					if (distSqu <= kernelRadiusSqu_) [[unlikely]]
					{
						scratchFB.push_back(j);
					}
				}
			}
		}
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_CellLinkedList_Compact_CPU01::DataReOrder(const SimParams& params)
{
	(void)params;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_CellLinkedList_Compact_CPU01::GetMemUsage(
	const SimParams& params) const
{
	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_Vector(uniqueCellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellStartIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(cellIdF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(sortedParticleIdxF_));

	Mem::Add(usageB, Mem::MemUsageOf_Vector(uniqueCellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellStartIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(cellIdB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(sortedParticleIdxB_));

	return { usageF, usageB };
}

void NS_CellLinkedList_Compact_CPU01::UpdateCellFrame()
{
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0)
	{
		cellOrigX_ = 0;
		cellOrigY_ = 0;
		cellOrigZ_ = 0;
		N_DimX_ = 0;
		N_DimY_ = 0;
		N_DimZ_ = 0;
		cellFrameValid_ = false;

		return;
	}

	bool rebuildFrame = !cellFrameValid_;

	// check whether the current occupied cell box still fits into the
	// reusable frame with the current query pad on all sides
	if (!rebuildFrame)
	{
		const int margin = static_cast<int>(cellQueryReach_);
		const int xMaxFrame = cellOrigX_ + static_cast<int>(N_DimX_) - 1;
		const int yMaxFrame = cellOrigY_ + static_cast<int>(N_DimY_) - 1;
		const int zMaxFrame = cellOrigZ_ + static_cast<int>(N_DimZ_) - 1;

		const int xMinInner = cellOrigX_ + margin;
		const int yMinInner = cellOrigY_ + margin;
		const int zMinInner = cellOrigZ_ + margin;
		const int xMaxInner = xMaxFrame - margin;
		const int yMaxInner = yMaxFrame - margin;
		const int zMaxInner = zMaxFrame - margin;

		if (xMinInner > xMaxInner || yMinInner > yMaxInner || zMinInner > zMaxInner)
		{
			rebuildFrame = true;
		}
		else
		{
			const bool fluidExceeded = tbb::parallel_reduce(
				tbb::blocked_range<uint32_t>(0, N_F),
				false,
				[&](const tbb::blocked_range<uint32_t>& r, bool localExceeded) -> bool
				{
					if (localExceeded) { return true; }

					for (uint32_t i = r.begin(); i != r.end(); i++)
					{
						const int cx = FloorToInt(be_.xPosF_[i] * cellSizeInv_);
						const int cy = FloorToInt(be_.yPosF_[i] * cellSizeInv_);
						const int cz = FloorToInt(be_.zPosF_[i] * cellSizeInv_);

						if (cx < xMinInner || cx > xMaxInner ||
							cy < yMinInner || cy > yMaxInner ||
							cz < zMinInner || cz > zMaxInner)
						{
							localExceeded = true;
							break;
						}
					}

					return localExceeded;
				},
				[](bool a, bool b) -> bool { return a || b; },
				tbb::auto_partitioner{}
			);

			if (fluidExceeded)
			{
				rebuildFrame = true;
			}
			else
			{
				const bool boundaryExceeded = tbb::parallel_reduce(
					tbb::blocked_range<uint32_t>(0, N_B),
					false,
					[&](const tbb::blocked_range<uint32_t>& r, bool localExceeded) -> bool
					{
						if (localExceeded) { return true; }

						for (uint32_t i = r.begin(); i != r.end(); i++)
						{
							const int cx = FloorToInt(be_.xPosB_[i] * cellSizeInv_);
							const int cy = FloorToInt(be_.yPosB_[i] * cellSizeInv_);
							const int cz = FloorToInt(be_.zPosB_[i] * cellSizeInv_);

							if (cx < xMinInner || cx > xMaxInner ||
								cy < yMinInner || cy > yMaxInner ||
								cz < zMinInner || cz > zMaxInner)
							{
								localExceeded = true;
								break;
							}
						}

						return localExceeded;
					},
					[](bool a, bool b) -> bool { return a || b; },
					tbb::auto_partitioner{}
				);

				rebuildFrame = boundaryExceeded;
			}
		}
	}

	if (!rebuildFrame) { return; }

	// rebuild cell frame from current occupied cell AABB
	const SceneAABB sceneAABB = ComputeSceneAABB_Combined(
		be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(), N_F,
		be_.xPosB_.data(), be_.yPosB_.data(), be_.zPosB_.data(), N_B);
	const CellCoordFrame cellFrame = ComputeCellCoordFrame_FromAABB(sceneAABB, cellSizeInv_);

	const int margin = static_cast<int>(cellQueryReach_ + N_CellFrameRebuildMargin_);
	cellOrigX_ = cellFrame.cellOrigX - margin;
	cellOrigY_ = cellFrame.cellOrigY - margin;
	cellOrigZ_ = cellFrame.cellOrigZ - margin;

	N_DimX_ = cellFrame.N_DimX + static_cast<uint32_t>(2 * margin);
	N_DimY_ = cellFrame.N_DimY + static_cast<uint32_t>(2 * margin);
	N_DimZ_ = cellFrame.N_DimZ + static_cast<uint32_t>(2 * margin);

	static constexpr uint32_t MaxAxisCells = (1u << 21);
	if (N_DimX_ > MaxAxisCells || N_DimY_ > MaxAxisCells || N_DimZ_ > MaxAxisCells)
	{
		SPDLOG_ERROR("Compact CLL cell frame exceeds Morton 21-bit axis limit [{}, {}, {}]",
			N_DimX_, N_DimY_, N_DimZ_);
		assert(false);
		cellFrameValid_ = false;

		return;
	}

	cellFrameValid_ = true;
}

inline uint32_t NS_CellLinkedList_Compact_CPU01::NeighborCellCount() const noexcept
{
	const uint32_t N_NbrDim = 2u * cellQueryReach_ + 1u;
	return N_NbrDim * N_NbrDim * N_NbrDim;
}

inline uint64_t NS_CellLinkedList_Compact_CPU01::MapToCellId_XYZ(
	int cx, int cy, int cz) const noexcept
{
	// shift to local cell frame to avoid zig-zag preprocessing
	const int lx = cx - cellOrigX_;
	const int ly = cy - cellOrigY_;
	const int lz = cz - cellOrigZ_;
	assert(lx >= 0 && lx < static_cast<int>(N_DimX_));
	assert(ly >= 0 && ly < static_cast<int>(N_DimY_));
	assert(lz >= 0 && lz < static_cast<int>(N_DimZ_));

	const uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(lx));
	const uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(ly));
	const uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(lz));

	// each axis must fit into 21 bits
	assert(ux < (1u << 21));
	assert(uy < (1u << 21));
	assert(uz < (1u << 21));

	// packed XYZ encoding
	return (ux) | (uy << 21) | (uz << 42);
}

inline uint64_t NS_CellLinkedList_Compact_CPU01::MapToCellId_Morton(
	int cx, int cy, int cz) const noexcept
{
	// shift to local cell frame to avoid zig-zag preprocessing
	const int lx = cx - cellOrigX_;
	const int ly = cy - cellOrigY_;
	const int lz = cz - cellOrigZ_;
	assert(lx >= 0 && lx < static_cast<int>(N_DimX_));
	assert(ly >= 0 && ly < static_cast<int>(N_DimY_));
	assert(lz >= 0 && lz < static_cast<int>(N_DimZ_));

	const uint32_t ux = static_cast<uint32_t>(lx);
	const uint32_t uy = static_cast<uint32_t>(ly);
	const uint32_t uz = static_cast<uint32_t>(lz);
	assert(ux < (1u << 21));
	assert(uy < (1u << 21));
	assert(uz < (1u << 21));

	const uint64_t xx = SpreadBitsWithGaps_21to63(ux); // bits at pos 0,3,6,...
	const uint64_t yy = SpreadBitsWithGaps_21to63(uy) << 1; // bits at pos 1,4,7,...
	const uint64_t zz = SpreadBitsWithGaps_21to63(uz) << 2; // bits at pos 2,5,8,...

	return xx | yy | zz; // interleaved 3D Morton/Z-order code
}

// spread out lower 21 bits of v into lower 63 bits and fill with gaps zeros like 000x00x00x00x00x...
inline uint64_t NS_CellLinkedList_Compact_CPU01::SpreadBitsWithGaps_21to63(uint32_t v) noexcept
{
	uint64_t x = static_cast<uint64_t>(v & 0x1fffffU); // lower 21 bits

	x = (x | (x << 32)) & 0x1f00000000ffffULL;
	x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
	x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
	x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
	x = (x | (x << 2)) & 0x1249249249249249ULL;

	return x;
}

inline int NS_CellLinkedList_Compact_CPU01::NeighborOffset(int rx, int ry, int rz) const noexcept
{
	const int cellQueryReach = static_cast<int>(cellQueryReach_);
	const int N_NbrDim = 2 * cellQueryReach + 1;
	return (rz + cellQueryReach) * N_NbrDim * N_NbrDim
		+ (ry + cellQueryReach) * N_NbrDim
		+ (rx + cellQueryReach);
}

// -------------------------------------------------------------------------------------------------
// --- [ bounding volume hierarchy linear (LBVH) implementation ] ----------------------------------
// -------------------------------------------------------------------------------------------------

void NS_BoundVolHier_Linear_LBVH_CPU01::Init(const SimParams& params)
{
	mortonF_.clear();
	sortedIdxF_.clear();
	nodesF_.clear();

	mortonB_.clear();
	sortedIdxB_.clear();
	nodesB_.clear();

	boundaryBVHValid_ = false;
	cachedBoundaryCount_ = 0;
	cachedBoundaryXPosPtr_ = nullptr;
	cachedBoundaryYPosPtr_ = nullptr;
	cachedBoundaryZPosPtr_ = nullptr;
	kernelRadiusSqu_ = params.kernelRadiusSqu;
}

void NS_BoundVolHier_Linear_LBVH_CPU01::Construct(const SimParams& params)
{
	kernelRadiusSqu_ = params.kernelRadiusSqu;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;

	if (N_F == 0 && N_B == 0)
	{
		SPDLOG_WARN("No construction possible (empty simulation)");

		mortonF_.clear();
		sortedIdxF_.clear();
		nodesF_.clear();

		mortonB_.clear();
		sortedIdxB_.clear();
		nodesB_.clear();

		boundaryBVHValid_ = false;
		cachedBoundaryCount_ = 0;
		cachedBoundaryXPosPtr_ = nullptr;
		cachedBoundaryYPosPtr_ = nullptr;
		cachedBoundaryZPosPtr_ = nullptr;

		return;
	}

	// build fluid BVH
	if (N_F > 0)
	{
		BuildBVH(be_.xPosF_.data(), be_.yPosF_.data(), be_.zPosF_.data(),
			N_F, mortonF_, sortedIdxF_, nodesF_);
	}
	else
	{
		mortonF_.clear();
		sortedIdxF_.clear();
		nodesF_.clear();
	}

	// build boundary BVH
	if (N_B > 0)
	{
		const float* xPosB = be_.xPosB_.data();
		const float* yPosB = be_.yPosB_.data();
		const float* zPosB = be_.zPosB_.data();

		// only rebuild boundary BVH if storage/layout changed
		const bool boundaryLayoutChanged =
			(cachedBoundaryCount_ != N_B) ||
			(cachedBoundaryXPosPtr_ != xPosB) ||
			(cachedBoundaryYPosPtr_ != yPosB) ||
			(cachedBoundaryZPosPtr_ != zPosB);

		if (!boundaryBVHValid_ || boundaryLayoutChanged)
		{
			BuildBVH(xPosB, yPosB, zPosB, N_B,
				mortonB_, sortedIdxB_, nodesB_);

			boundaryBVHValid_ = true;
			cachedBoundaryCount_ = N_B;
			cachedBoundaryXPosPtr_ = xPosB;
			cachedBoundaryYPosPtr_ = yPosB;
			cachedBoundaryZPosPtr_ = zPosB;
		}
	}
	else
	{
		mortonB_.clear();
		sortedIdxB_.clear();
		nodesB_.clear();

		boundaryBVHValid_ = false;
		cachedBoundaryCount_ = 0;
		cachedBoundaryXPosPtr_ = nullptr;
		cachedBoundaryYPosPtr_ = nullptr;
		cachedBoundaryZPosPtr_ = nullptr;
	}
}

void NS_BoundVolHier_Linear_LBVH_CPU01::BuildBVH(
	const float* RESTRICT xPos, const float* RESTRICT yPos, const float* RESTRICT zPos,
	uint32_t N,
	std::vector<uint32_t>& morton, std::vector<uint32_t>& sortedIdx,
	std::vector<InternalNode>& nodes)
{
	// --- [ compute scene AABB for Morton code normalization ] ------------------------------------

	struct SceneExtents
	{
		float minX = std::numeric_limits<float>::max();
		float minY = std::numeric_limits<float>::max();
		float minZ = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float maxY = std::numeric_limits<float>::lowest();
		float maxZ = std::numeric_limits<float>::lowest();
	};

	SceneExtents sceneAABB = tbb::parallel_reduce(
		tbb::blocked_range<uint32_t>(0, N),
		SceneExtents{},
		[xPos, yPos, zPos](const tbb::blocked_range<uint32_t>& r, SceneExtents val) -> SceneExtents
		{
			float lMinX = val.minX, lMinY = val.minY, lMinZ = val.minZ;
			float lMaxX = val.maxX, lMaxY = val.maxY, lMaxZ = val.maxZ;

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				lMinX = std::min(lMinX, xPos[i]);
				lMinY = std::min(lMinY, yPos[i]);
				lMinZ = std::min(lMinZ, zPos[i]);
				lMaxX = std::max(lMaxX, xPos[i]);
				lMaxY = std::max(lMaxY, yPos[i]);
				lMaxZ = std::max(lMaxZ, zPos[i]);
			}

			val.minX = lMinX; val.minY = lMinY; val.minZ = lMinZ;
			val.maxX = lMaxX; val.maxY = lMaxY; val.maxZ = lMaxZ;
			return val;
		},
		[](SceneExtents a, const SceneExtents& b) -> SceneExtents
		{
			a.minX = std::min(a.minX, b.minX);
			a.minY = std::min(a.minY, b.minY);
			a.minZ = std::min(a.minZ, b.minZ);
			a.maxX = std::max(a.maxX, b.maxX);
			a.maxY = std::max(a.maxY, b.maxY);
			a.maxZ = std::max(a.maxZ, b.maxZ);
			return a;
		}
	);

	// add small epsilon to avoid division by zero on degenerate axes
	constexpr float eps = 1e-6f;
	const float extX = std::max(eps, sceneAABB.maxX - sceneAABB.minX);
	const float extY = std::max(eps, sceneAABB.maxY - sceneAABB.minY);
	const float extZ = std::max(eps, sceneAABB.maxZ - sceneAABB.minZ);
	const float invExtX = 1.0f / extX;
	const float invExtY = 1.0f / extY;
	const float invExtZ = 1.0f / extZ;

	// --- [ compute Morton codes and build leaf AABBs ] -------------------------------------------

	morton.resize(N);
	sortedIdx.resize(N);
	leafBoxScratch_.resize(N);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// normalize position to [0, 1] within the scene AABB
				const float nx = (xPos[i] - sceneAABB.minX) * invExtX;
				const float ny = (yPos[i] - sceneAABB.minY) * invExtY;
				const float nz = (zPos[i] - sceneAABB.minZ) * invExtZ;

				morton[i] = Morton3D_30bit(nx, ny, nz);
				sortedIdx[i] = i;

				// tighter leaves improve query pruning, exact radius check is done at leaf hits
				leafBoxScratch_[i] = MakeParticleAABB(xPos[i], yPos[i], zPos[i], 0.0f);
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ sort particles by Morton code ] -------------------------------------------------------

	tbb::parallel_sort(sortedIdx.begin(), sortedIdx.end(),
		[&](uint32_t a, uint32_t b)
		{
			// tie-break on original index for deterministic ordering of duplicate codes
			const uint32_t ma = morton[a];
			const uint32_t mb = morton[b];
			return (ma != mb) ? (ma < mb) : (a < b);
		}
	);

	// build sorted Morton code array (reorder Morton codes to sorted order)
	sortedMortonScratch_.resize(N);
	std::vector<uint32_t>& sortedMorton = sortedMortonScratch_;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				sortedMorton[i] = morton[sortedIdx[i]];
			}
		},
		tbb::auto_partitioner{}
	);

	// reorder leaf boxes to sorted order
	sortedLeafBoxScratch_.resize(N);
	std::vector<AABB>& sortedLeafBox = sortedLeafBoxScratch_;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				sortedLeafBox[i] = leafBoxScratch_[sortedIdx[i]];
			}
		},
		tbb::auto_partitioner{}
	);
	const std::vector<AABB>& leafBox = sortedLeafBox;

	// handle trivial case: single particle
	if (N == 1)
	{
		nodes.resize(1);
		nodes[0].box = leafBox[0];
		nodes[0].leftChild = 0;
		nodes[0].rightChild = 0;
		nodes[0].leftIsLeaf = true;
		nodes[0].rightIsLeaf = true;
		return;
	}

	// --- [ build radix tree (Karras 2012) ] ------------------------------------------------------

	const uint32_t N_internal = N - 1;
	nodes.resize(N_internal);

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_internal),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// determine direction of the range (d = +1 or -1)
				const int deltaRight = Delta(sortedMorton, N, static_cast<int>(i),
					static_cast<int>(i) + 1);
				const int deltaLeft = Delta(sortedMorton, N, static_cast<int>(i),
					static_cast<int>(i) - 1);
				const int d = (deltaRight >= deltaLeft) ? 1 : -1;

				// compute upper bound for the length of the range
				const int deltaMin = Delta(sortedMorton, N, static_cast<int>(i),
					static_cast<int>(i) - d);
				uint32_t lMax = 2;
				while (Delta(sortedMorton, N, static_cast<int>(i),
					static_cast<int>(i) + static_cast<int>(lMax) * d) > deltaMin)
				{
					lMax *= 2;
				}

				// find the other end using binary search
				uint32_t l = 0;
				for (uint32_t t = lMax >> 1; t >= 1; t >>= 1)
				{
					if (Delta(sortedMorton, N, static_cast<int>(i),
						static_cast<int>(i) + (static_cast<int>(l + t)) * d) > deltaMin)
					{
						l += t;
					}
				}
				const int j = static_cast<int>(i) + static_cast<int>(l) * d;

				// find the split position using binary search
				const int deltaNode = Delta(sortedMorton, N, static_cast<int>(i), j);
				uint32_t s = 0;
				// start t at the next power of two >= l (ceiling)
				uint32_t t = 1;
				while (t < l) { t *= 2; }
				for (; t >= 1; t = (t > 1) ? (t >> 1) : 0)
				{
					if (Delta(sortedMorton, N, static_cast<int>(i),
						static_cast<int>(i) + (static_cast<int>(s + t)) * d) > deltaNode)
					{
						s += t;
					}
					if (t == 1) { break; }
				}
				const int gamma = static_cast<int>(i) + static_cast<int>(s) * d
					+ std::min(d, 0);

				// determine children
				const int iMin = std::min(static_cast<int>(i), j);
				const int iMax = std::max(static_cast<int>(i), j);

				// left child
				if (iMin == gamma)
				{
					// left child is a leaf
					nodes[i].leftChild = static_cast<uint32_t>(gamma);
					nodes[i].leftIsLeaf = true;
				}
				else
				{
					// left child is an internal node
					nodes[i].leftChild = static_cast<uint32_t>(gamma);
					nodes[i].leftIsLeaf = false;
				}

				// right child
				if (iMax == gamma + 1)
				{
					// right child is a leaf
					nodes[i].rightChild = static_cast<uint32_t>(gamma + 1);
					nodes[i].rightIsLeaf = true;
				}
				else
				{
					// right child is an internal node
					nodes[i].rightChild = static_cast<uint32_t>(gamma + 1);
					nodes[i].rightIsLeaf = false;
				}
			}
		},
		tbb::auto_partitioner{}
	);

	// --- [ bottom-up AABB propagation ] ----------------------------------------------------------

	// parent lookup: for each node/leaf, which internal node is its parent?
	parentOfScratch_.assign(N_internal + N, std::numeric_limits<uint32_t>::max());
	std::vector<uint32_t>& parentOf = parentOfScratch_;
	for (uint32_t i = 0; i < N_internal; i++)
	{
		if (nodes[i].leftIsLeaf)
		{
			parentOf[N_internal + nodes[i].leftChild] = i;
		}
		else
		{
			parentOf[nodes[i].leftChild] = i;
		}

		if (nodes[i].rightIsLeaf)
		{
			parentOf[N_internal + nodes[i].rightChild] = i;
		}
		else
		{
			parentOf[nodes[i].rightChild] = i;
		}
	}

	// atomic visit counters for internal nodes
	std::vector<std::atomic<uint32_t>> visitCount(N_internal);
	for (auto& v : visitCount) { v.store(0, std::memory_order_relaxed); }

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t leafIdx = r.begin(); leafIdx != r.end(); leafIdx++)
			{
				uint32_t current = parentOf[N_internal + leafIdx];

				while (current != std::numeric_limits<uint32_t>::max())
				{
					// atomically increment the visit counter
					const uint32_t prev = visitCount[current].fetch_add(
						1, std::memory_order_acq_rel);

					if (prev == 0)
					{
						// first thread to arrive: the other child is not ready yet
						break;
					}

					// second thread: both children are ready, merge AABBs
					const auto& node = nodes[current];

					const AABB leftBox = node.leftIsLeaf
						? leafBox[node.leftChild]
						: nodes[node.leftChild].box;

					const AABB rightBox = node.rightIsLeaf
						? leafBox[node.rightChild]
						: nodes[node.rightChild].box;

					nodes[current].box = MergeAABB(leftBox, rightBox);

					// walk upward
					current = parentOf[current];
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BoundVolHier_Linear_LBVH_CPU01::Query_Count(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				uint32_t cntFF = 0;
				uint32_t cntFB = 0;

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// query fluid BVH
				if (N_F > 0)
				{
					QueryBVH_Count(xi, yi, zi, kernelRadiusSqu,
						xPosF, yPosF, zPosF, sortedIdxF_,
						nodesF_, N_F, cntFF);
				}

				// query boundary BVH
				if (N_B > 0)
				{
					QueryBVH_Count(xi, yi, zi, kernelRadiusSqu,
						xPosB, yPosB, zPosB, sortedIdxB_,
						nodesB_, N_B, cntFB);
				}

				// commit final neighbor counts to backend storage
				be_.neighborLists_->SetNbrCountFF(i, cntFF);
				be_.neighborLists_->SetNbrCountFB(i, cntFB);
			}
		},
		tbb::auto_partitioner{}
	);
}

void NS_BoundVolHier_Linear_LBVH_CPU01::Query_Fill(const SimParams& params)
{
	const float kernelRadiusSqu = params.kernelRadiusSqu;

	// local references for fewer pointer indirections
	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_F),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			// per-thread neighbor vectors for local accumulation and then bulk add
			// (reserve preparation in case of inline capacity not being enough)
			auto& scratchFF = scratchFF_.local();
			auto& scratchFB = scratchFB_.local();
			if (scratchFF.capacity() < params.expctdNbrCountHardMax_FF)
			{
				scratchFF.reserve(params.expctdNbrCountHardMax_FF);
			}
			if (scratchFB.capacity() < params.expctdNbrCountHardMax_FB)
			{
				scratchFB.reserve(params.expctdNbrCountHardMax_FB);
			}

			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				// reset neighbor list for particle i
				// (delete contents, but keep capacity)
				scratchFF.clear();
				scratchFB.clear();

				const float xi = xPosF[i];
				const float yi = yPosF[i];
				const float zi = zPosF[i];

				// query fluid BVH
				if (N_F > 0)
				{
					QueryBVH_Fill(xi, yi, zi, kernelRadiusSqu,
						xPosF, yPosF, zPosF, sortedIdxF_,
						nodesF_, N_F, scratchFF);
				}

				// query boundary BVH
				if (N_B > 0)
				{
					QueryBVH_Fill(xi, yi, zi, kernelRadiusSqu,
						xPosB, yPosB, zPosB, sortedIdxB_,
						nodesB_, N_B, scratchFB);
				}

				// commit final neighbor lists from scratch to backend storage
				be_.neighborLists_->SetNbrListFF(i, scratchFF.data(),
					static_cast<uint32_t>(scratchFF.size()));
				be_.neighborLists_->SetNbrListFB(i, scratchFB.data(),
					static_cast<uint32_t>(scratchFB.size()));
			}
		},
		tbb::auto_partitioner{}
	);
}

NeighborSpan NS_BoundVolHier_Linear_LBVH_CPU01::QueryParticleFF(
	const uint32_t i) const
{
	auto& scratchFF = scratchFF_.local();
	scratchFF.clear();

	const uint32_t N_F = be_.N_F_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	if (N_F > 0)
	{
		QueryBVH_Fill(xi, yi, zi, kernelRadiusSqu_,
			xPosF, yPosF, zPosF, sortedIdxF_,
			nodesF_, N_F, scratchFF);
	}

	return NeighborSpan{ scratchFF.data(), static_cast<uint32_t>(scratchFF.size()) };
}

NeighborSpan NS_BoundVolHier_Linear_LBVH_CPU01::QueryParticleFB(
	const uint32_t i) const
{
	auto& scratchFB = scratchFB_.local();
	scratchFB.clear();

	const uint32_t N_B = be_.N_B_;
	const float* RESTRICT xPosF = be_.xPosF_.data();
	const float* RESTRICT yPosF = be_.yPosF_.data();
	const float* RESTRICT zPosF = be_.zPosF_.data();
	const float* RESTRICT xPosB = be_.xPosB_.data();
	const float* RESTRICT yPosB = be_.yPosB_.data();
	const float* RESTRICT zPosB = be_.zPosB_.data();

	const float xi = xPosF[i];
	const float yi = yPosF[i];
	const float zi = zPosF[i];

	if (N_B > 0)
	{
		QueryBVH_Fill(xi, yi, zi, kernelRadiusSqu_,
			xPosB, yPosB, zPosB, sortedIdxB_,
			nodesB_, N_B, scratchFB);
	}

	return NeighborSpan{ scratchFB.data(), static_cast<uint32_t>(scratchFB.size()) };
}

void NS_BoundVolHier_Linear_LBVH_CPU01::QueryBVH_Count(
	float xi, float yi, float zi,
	float kernelRadiusSqu,
	const float* RESTRICT xPos, const float* RESTRICT yPos, const float* RESTRICT zPos,
	const std::vector<uint32_t>& sortedIdx,
	const std::vector<InternalNode>& nodes, uint32_t N, uint32_t& count) const
{
	// single leaf BVH: test once and return
	if (N == 1)
	{
		const uint32_t j = sortedIdx[0];
		const float dx = xi - xPos[j];
		const float dy = yi - yPos[j];
		const float dz = zi - zPos[j];
		const float distSqu = dx * dx + dy * dy + dz * dz;

		if (distSqu <= kernelRadiusSqu) [[unlikely]]
		{
			count++;
		}
		return;
	}

	// explicit traversal stack (avoids recursion overhead)
	std::array<uint32_t, MAX_STACK_DEPTH> stack{};
	// high bit encodes whether entry is a leaf (1) or internal node (0)
	constexpr uint32_t LEAF_FLAG = 0x80000000u;
	uint32_t stackTop = 0;

	// push root (internal node 0)
	stack[stackTop++] = 0;

	while (stackTop > 0)
	{
		const uint32_t entry = stack[--stackTop];

		if (entry & LEAF_FLAG)
		{
			// leaf node: check the actual particle
			const uint32_t leafIdx = entry & ~LEAF_FLAG;
			const uint32_t j = sortedIdx[leafIdx];

			const float dx = xi - xPos[j];
			const float dy = yi - yPos[j];
			const float dz = zi - zPos[j];
			const float distSqu = dx * dx + dy * dy + dz * dz;

			if (distSqu <= kernelRadiusSqu) [[unlikely]]
			{
				count++;
			}
		}
		else
		{
			// internal node: check AABB overlap with query sphere
			const auto& node = nodes[entry];

			if (!OverlapsAABB(node.box, xi, yi, zi, kernelRadiusSqu))
			{
				continue;
			}

			// push children (right first so left is processed first = depth-first)
			if (node.rightIsLeaf)
			{
				stack[stackTop++] = node.rightChild | LEAF_FLAG;
			}
			else
			{
				stack[stackTop++] = node.rightChild;
			}

			if (node.leftIsLeaf)
			{
				stack[stackTop++] = node.leftChild | LEAF_FLAG;
			}
			else
			{
				stack[stackTop++] = node.leftChild;
			}
		}
	}
}

void NS_BoundVolHier_Linear_LBVH_CPU01::DataReOrder(const SimParams& params)
{
	(void)params;

	// particle arrays are permuted; cached boundary tree/index mapping is no longer valid
	boundaryBVHValid_ = false;

	const uint32_t N_F = be_.N_F_;
	const uint32_t N_B = be_.N_B_;
	if (N_F == 0 && N_B == 0) { return; }

	DataReOrderParticles_Morton_30bitNormalized(
		FluidParticleSoARefs{
			be_.xPosF_, be_.yPosF_, be_.zPosF_,
			be_.xVelF_, be_.yVelF_, be_.zVelF_,
			be_.xVelPredF_, be_.yVelPredF_, be_.zVelPredF_,
			be_.xAccPressF_, be_.yAccPressF_, be_.zAccPressF_,
			be_.sourceTermF_, be_.diagElemF_, be_.pressureF_, be_.densityF_
		},
		BoundaryParticleSoARefs{
			be_.xPosB_, be_.yPosB_, be_.zPosB_
		},
		N_F, N_B,
		be_.sortKeyBuffer_,
		be_.workBufferF_,
		be_.workBufferB_
	);
}

std::pair<Mem::MemUsage, Mem::MemUsage> NS_BoundVolHier_Linear_LBVH_CPU01::GetMemUsage(
	const SimParams& params) const
{
	(void)params;

	Mem::MemUsage usageF{};
	Mem::MemUsage usageB{};

	Mem::Add(usageF, Mem::MemUsageOf_Vector(mortonF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(sortedIdxF_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(nodesF_));

	Mem::Add(usageB, Mem::MemUsageOf_Vector(mortonB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(sortedIdxB_));
	Mem::Add(usageB, Mem::MemUsageOf_Vector(nodesB_));

	// shared construction scratches
	Mem::Add(usageF, Mem::MemUsageOf_Vector(sortedMortonScratch_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(leafBoxScratch_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(sortedLeafBoxScratch_));
	Mem::Add(usageF, Mem::MemUsageOf_Vector(parentOfScratch_));

	return { usageF, usageB };
}

// --- [ BVH helper functions ] --------------------------------------------------------------------

inline NS_BoundVolHier_Linear_LBVH_CPU01::AABB NS_BoundVolHier_Linear_LBVH_CPU01::MakeParticleAABB(
	float x, float y, float z, float r) noexcept
{
	return { x - r, y - r, z - r, x + r, y + r, z + r };
}

inline NS_BoundVolHier_Linear_LBVH_CPU01::AABB NS_BoundVolHier_Linear_LBVH_CPU01::MergeAABB(
	const AABB& a, const AABB& b) noexcept
{
	return {
		std::min(a.minX, b.minX), std::min(a.minY, b.minY), std::min(a.minZ, b.minZ),
		std::max(a.maxX, b.maxX), std::max(a.maxY, b.maxY), std::max(a.maxZ, b.maxZ)
	};
}

inline bool NS_BoundVolHier_Linear_LBVH_CPU01::OverlapsAABB(
	const AABB& box, float x, float y, float z, float radiusSqu) noexcept
{
	// compute squared distance from query point to the closest point on the AABB
	float dSqu = 0.0f;

	if (x < box.minX) { const float d = box.minX - x; dSqu += d * d; }
	else if (x > box.maxX) { const float d = x - box.maxX; dSqu += d * d; }

	if (y < box.minY) { const float d = box.minY - y; dSqu += d * d; }
	else if (y > box.maxY) { const float d = y - box.maxY; dSqu += d * d; }

	if (z < box.minZ) { const float d = box.minZ - z; dSqu += d * d; }
	else if (z > box.maxZ) { const float d = z - box.maxZ; dSqu += d * d; }

	return dSqu <= radiusSqu;
}

// expand lower 10 bits into every-third-bit positions: 0b...xyz -> 0b...00x00y00z
inline uint32_t NS_BoundVolHier_Linear_LBVH_CPU01::ExpandBits10(uint32_t v) noexcept
{
	return ExpandBits10_30bit(v);
}

// compute a 30-bit Morton code for a point in the unit cube [0, 1]^3
inline uint32_t NS_BoundVolHier_Linear_LBVH_CPU01::Morton3D_30bit(float nx, float ny, float nz) noexcept
{
	return Morton3D_30bit_Normalized(nx, ny, nz);
}

inline int NS_BoundVolHier_Linear_LBVH_CPU01::CountLeadingZeros(uint32_t v) noexcept
{
	if (v == 0) { return 32; }

#if defined(_MSC_VER)
	unsigned long idx = 0;
	_BitScanReverse(&idx, v);
	return 31 - static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_clz(v);
#else
	// portable fallback
	int n = 0;
	if (v <= 0x0000FFFFu) { n += 16; v <<= 16; }
	if (v <= 0x00FFFFFFu) { n +=  8; v <<=  8; }
	if (v <= 0x0FFFFFFFu) { n +=  4; v <<=  4; }
	if (v <= 0x3FFFFFFFu) { n +=  2; v <<=  2; }
	if (v <= 0x7FFFFFFFu) { n +=  1; }
	return n;
#endif
}

// delta function for the Karras (2012) radix tree construction algorithm
// returns the length of the longest common prefix between sorted Morton codes at
// positions i and j, or -1 if j is out of range
inline int NS_BoundVolHier_Linear_LBVH_CPU01::Delta(
	const std::vector<uint32_t>& sortedMorton, uint32_t N, int i, int j) noexcept
{
	if (j < 0 || j >= static_cast<int>(N))
	{
		return -1;
	}

	const uint32_t mi = sortedMorton[i];
	const uint32_t mj = sortedMorton[j];

	if (mi != mj)
	{
		return CountLeadingZeros(mi ^ mj);
	}

	// Morton codes are identical: use index as tie-breaker (32 + clz of index XOR)
	return 32 + CountLeadingZeros(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(j));
}
