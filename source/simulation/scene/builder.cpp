// builder.cpp

// external
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "source/base/helpers.h"
#include "source/simulation/core/solver_constants.h"
#include "builder.h"



using namespace Scene;

namespace
{
	// uniform fallback when coefficient pass cannot compute a valid sum
	float DefaultBoundaryPsiB(const Sim::SimParams& params)
	{
		return params.massB;
	}

	float DefaultBoundaryVolB(const Sim::SimParams& params)
	{
		return params.restDensity > 0.0f
			? (DefaultBoundaryPsiB(params) / params.restDensity)
			: 0.0f;
	}

	// scalar kernel weight for boundary coefficient sums
	float KernelWeight_CubicSpline(
		const float dx, const float dy, const float dz,
		const float particleSizeInvF, const float kernelRadiusSqu,
		const float cubicSplineAlpha)
	{
		constexpr float ZERO_EQU_LIM_F = 1e-7f;

		const float distSqu = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));

		if (distSqu >= kernelRadiusSqu) [[unlikely]]
		{
			return 0.0f;
		}

		if (distSqu < ZERO_EQU_LIM_F) [[unlikely]]
		{
			return 4.0f * cubicSplineAlpha;
		}

		const float dist = std::sqrt(distSqu);
		const float distNorm = dist * particleSizeInvF;

		const float t1 = std::max(1.0f - distNorm, 0.0f);
		const float t2 = std::max(2.0f - distNorm, 0.0f);
		const float t1_2 = t1 * t1;
		const float t2_2 = t2 * t2;
		const float t1_3 = t1_2 * t1;
		const float t2_3 = t2_2 * t2;

		return cubicSplineAlpha * (t2_3 - 4.0f * t1_3);
	}

	float Dot(
		const glm::vec3& a,
		const glm::vec3& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	bool IsFinite(const glm::vec3& value)
	{
		return std::isfinite(value.x) &&
			std::isfinite(value.y) &&
			std::isfinite(value.z);
	}

	bool IsFinite(const glm::mat3& value)
	{
		for (uint32_t column = 0; column < 3; column++)
		{
			if (!IsFinite(value[column])) { return false; }
		}
		return true;
	}

	bool IsRotationMatrix(const glm::mat3& orientation)
	{
		constexpr float tolerance = 1e-4f;
		const glm::vec3 xAxis = orientation[0];
		const glm::vec3 yAxis = orientation[1];
		const glm::vec3 zAxis = orientation[2];

		return std::abs(glm::dot(xAxis, xAxis) - 1.0f) <= tolerance &&
			std::abs(glm::dot(yAxis, yAxis) - 1.0f) <= tolerance &&
			std::abs(glm::dot(zAxis, zAxis) - 1.0f) <= tolerance &&
			std::abs(glm::dot(xAxis, yAxis)) <= tolerance &&
			std::abs(glm::dot(xAxis, zAxis)) <= tolerance &&
			std::abs(glm::dot(yAxis, zAxis)) <= tolerance &&
			std::abs(glm::dot(glm::cross(xAxis, yAxis), zAxis) - 1.0f) <= tolerance;
	}

} // namespace anonymous

void ScenarioMgr::ComputeBoundaryCoefficients(
	Sync::BoundaryParticles& particlesB,
	const Sim::SimParams& params,
	const uint32_t beginB,
	const uint32_t endB)
{
	ComputeParticleCoefficients(
		particlesB.xPos, particlesB.yPos, particlesB.zPos,
		particlesB.volB, particlesB.psiB, particlesB.N_,
		params, beginB, endB);
}

void ScenarioMgr::ComputeRigidCoefficients(
	Sync::RigidParticles& particlesR,
	const Sim::SimParams& params,
	const uint32_t beginR,
	const uint32_t endR)
{
	ComputeParticleCoefficients(
		particlesR.xRelPos, particlesR.yRelPos, particlesR.zRelPos,
		particlesR.volR, particlesR.psiR, particlesR.N_,
		params, beginR, endR);
}

void ScenarioMgr::ComputeParticleCoefficients(
	const std::vector<float>& xPos,
	const std::vector<float>& yPos,
	const std::vector<float>& zPos,
	std::vector<float>& vol,
	std::vector<float>& psi,
	const uint32_t N,
	const Sim::SimParams& params,
	const uint32_t begin,
	const uint32_t end)
{
	// temp local cell buckets to avoid O(N^2) boundary neighbor search
	struct CellCoord
	{
		int32_t x = 0;
		int32_t y = 0;
		int32_t z = 0;

		bool operator==(const CellCoord&) const = default;
	};

	struct CellCoordHash
	{
		uint64_t operator()(const CellCoord& c) const noexcept
		{
			uint64_t h = static_cast<uint32_t>(c.x);
			h ^= static_cast<uint64_t>(static_cast<uint32_t>(c.y))
			   + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			h ^= static_cast<uint64_t>(static_cast<uint32_t>(c.z))
			   + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			return h;
		}
	};

	const auto makeCellCoord =
		[](const float x, const float y, const float z, const float cellSizeInv) -> CellCoord
		{
			return CellCoord
			{
				.x = static_cast<int32_t>(std::floor(x * cellSizeInv)),
				.y = static_cast<int32_t>(std::floor(y * cellSizeInv)),
				.z = static_cast<int32_t>(std::floor(z * cellSizeInv)),
			};
		};

	assert(begin <= end);
	assert(end <= N);
	if (begin == end) { return; }

	const float defaultVol = DefaultBoundaryVolB(params);
	const float defaultPsi = DefaultBoundaryPsiB(params);
	vol.resize(N);
	psi.resize(N);

	// keep coefficients valid if params are incomplete
	if (params.particleSizeF <= 0.0f ||
		params.particleSizeInvF <= 0.0f ||
		params.kernelRadius <= 0.0f ||
		params.kernelRadiusSqu <= 0.0f ||
		params.gridCellSizeRatio <= 0.0f ||
		params.restDensity <= 0.0f)
	{
		std::fill(vol.begin() + begin, vol.begin() + end, defaultVol);
		std::fill(psi.begin() + begin, psi.begin() + end, defaultPsi);
		return;
	}

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	if (cellSize <= 0.0f)
	{
		std::fill(vol.begin() + begin, vol.begin() + end, defaultVol);
		std::fill(psi.begin() + begin, psi.begin() + end, defaultPsi);
		return;
	}

	// same cell-reach rule as cell-linked list neighbor search
	const float querySpanCells = params.kernelRadius / cellSize;
	const int32_t cellQueryReach = std::max<int32_t>(
		1,
		static_cast<int32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));
	const float cellSizeInv = 1.0f / cellSize;
	const uint32_t N_compute = end - begin;
	const Sim::SolverConstants solverConstants = Sim::BuildSolverConstants(params.particleSizeF, params.restDensity);

	std::vector<CellCoord> cellCoords(N_compute);
	std::unordered_map<CellCoord, std::vector<uint32_t>, CellCoordHash> cellBuckets;
	cellBuckets.reserve(static_cast<uint64_t>(N_compute) * 2ull);

	// insert boundary particles into cell buckets
	for (uint32_t localIdx = 0; localIdx < N_compute; localIdx++)
	{
		const uint32_t i = begin + localIdx;
		cellCoords[localIdx] = makeCellCoord(
			xPos[i],
			yPos[i],
			zPos[i],
			cellSizeInv);
		cellBuckets[cellCoords[localIdx]].push_back(i);
	}

	// compute non-uniform boundary contribution coefficients for all boundary particles
	const auto& cellBucketsReadOnly = cellBuckets;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_compute),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t localIdx = r.begin(); localIdx != r.end(); localIdx++)
			{
				const uint32_t i = begin + localIdx;
				float sumW = 0.0f;
				const CellCoord baseCell = cellCoords[localIdx];

				for (int32_t dz = -cellQueryReach; dz <= cellQueryReach; dz++)
				{
					for (int32_t dy = -cellQueryReach; dy <= cellQueryReach; dy++)
					{
						for (int32_t dx = -cellQueryReach; dx <= cellQueryReach; dx++)
						{
							const CellCoord cell
							{
								.x = baseCell.x + dx,
								.y = baseCell.y + dy,
								.z = baseCell.z + dz,
							};
							const auto it = cellBucketsReadOnly.find(cell);
							if (it == cellBucketsReadOnly.end()) { continue; }

							for (const uint32_t j : it->second)
							{
								sumW += KernelWeight_CubicSpline(
									xPos[i] - xPos[j],
									yPos[i] - yPos[j],
									zPos[i] - zPos[j],
									solverConstants.sizeInvF, params.kernelRadiusSqu,
									solverConstants.cubicSplineAlpha);
							}
						}
					}
				}

				if (sumW > 0.0f && std::isfinite(sumW))
				{
					// boundary volume from inverse boundary sample density
					vol[i] = 1.0f / sumW;
					psi[i] = params.restDensity * vol[i];
				}
				else
				{
					vol[i] = defaultVol;
					psi[i] = defaultPsi;
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void ScenarioMgr::BeginBuildRegion(const glm::vec3& min, const glm::vec3& max)
{
	buildRegionMin_ = min;
	buildRegionMax_ = max;
	hasBuildRegion_ = true;

	N_builtGlobalF_ = 0;
	N_builtGlobalB_ = 0;
}

void ScenarioMgr::EndBuildRegion()
{
	hasBuildRegion_ = false;
}

bool ScenarioMgr::IsInsideBuildRegion(const glm::vec3& pos) const
{
	if (!hasBuildRegion_) { return true; }

	return pos.x >= buildRegionMin_.x && pos.x <= buildRegionMax_.x
		&& pos.y >= buildRegionMin_.y && pos.y <= buildRegionMax_.y
		&& pos.z >= buildRegionMin_.z && pos.z <= buildRegionMax_.z;
}

void ScenarioMgr::BuildBoundaryContainer_OBB(const OBB& box,
											 Sync::UniversalSimState& dst,
											 const Sim::SimParams& params)
{
	if (params.boundaryLayers <= 0) { return; }

	const float sizeB = params.particleSizeB;
	const float radiusB = params.particleSizeB * 0.5f;

	// max normal offset of the outermost layer
	const float pad = radiusB + (static_cast<float>(params.boundaryLayers) - 1.0f) * sizeB;

	// helper for extending a face in the tangential directions
	auto ExtendFaceUV = [](OBBFace f, float padU, float padV) -> OBBFace
		{
			f.origin -= f.uHat * padU + f.vHat * padV;
			f.uLen += 2.0f * padU;
			f.vLen += 2.0f * padV;
			return f;
		};

	auto faces = GetFacesFromOBB(box);

	// zMin (0), zMax (1) -> no extension
	// xMin (2), xMax (3) -> extended in BOTH tangential directions (u and v)
	faces[2] = ExtendFaceUV(faces[2], pad, pad);
	faces[3] = ExtendFaceUV(faces[3], pad, pad);

	// yMin (4), yMax (5) -> extended ONLY along face.vHat (z for these faces)
	faces[4] = ExtendFaceUV(faces[4], 0.0f, pad);
	faces[5] = ExtendFaceUV(faces[5], 0.0f, pad);

	for (const auto& face : faces)
	{
		BuildBoundaryLayersOnFace(face, dst, params);
	}

	ComputeBoundaryCoefficients(dst.particlesB, params, 0, dst.particlesB.N_);
}

void ScenarioMgr::BuildFluidBlock_OBB(const OBB& box,
									  Sync::UniversalSimState& dst,
									  const Sim::SimParams& params)
{
	const float size = params.particleSizeF;
	const float radius = params.particleSizeF * 0.5f;

	// inner lengths for centers (shrink by one radius per side)
	const float uLen_i = box.uLen - 2.0f * radius;
	const float vLen_i = box.vLen - 2.0f * radius;
	const float wLen_i = box.wLen - 2.0f * radius;
	if (uLen_i <= 0.0f || vLen_i <= 0.0f || wLen_i <= 0.0f) { return; }

	// number of particles in each direction (inside inner OBB)
	const uint32_t N_u = TilesCount(uLen_i, size);
	const uint32_t N_v = TilesCount(vLen_i, size);
	const uint32_t N_w = TilesCount(wLen_i, size);
	if (N_u <= 0 || N_v <= 0 || N_w <= 0) { return; }

	auto& F = dst.particlesF;

	// dense generation index of the first particle, so IDs stay the global scenario indices
	const uint32_t N_base = hasBuildRegion_ ? N_builtGlobalF_ : F.N_;
	const uint32_t N_add = N_u * N_v * N_w;
	N_builtGlobalF_ += N_add;

	// offsets from box planes
	const float uOff = radius + StartOffset(uLen_i, size, static_cast<int>(N_u));
	const float vOff = radius + StartOffset(vLen_i, size, static_cast<int>(N_v));
	const float wOff = radius + StartOffset(wLen_i, size, static_cast<int>(N_w));

	const float jitAmp = params.jitterFracF * params.particleSizeF;

	// world position of one generated particle, so the region passes below stay consistent
	const auto ParticlePosition = [&](const uint32_t iu, const uint32_t jv,
									  const uint32_t kw, const uint32_t idx) -> glm::vec3
	{
		// position components
		float u = uOff + static_cast<float>(iu) * size;
		float v = vOff + static_cast<float>(jv) * size;
		float w = wOff + static_cast<float>(kw) * size;

		// optionally apply jittering
		if (jitAmp > 0.0f)
		{
			// use (idx * 3 + {0,1,2}) as unique keys for x,y,z jitter
			const uint64_t baseSeed = static_cast<uint64_t>(params.jitterSeedF);

			u += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 0ull, jitAmp);
			v += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 1ull, jitAmp);
			w += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 2ull, jitAmp);
		}

		// clamp to make edge particles remain fully inside the box
		u = std::clamp(u, radius, box.uLen - radius);
		v = std::clamp(v, radius, box.vLen - radius);
		w = std::clamp(w, radius, box.wLen - radius);

		return box.origin
			 + box.uHat * u
			 + box.vHat * v
			 + box.wHat * w;
	};

	// writes one particle into its storage slot, tagged with its global scenario index
	const auto WriteParticle = [&](const uint32_t outIdx, const uint32_t idx, const glm::vec3& pos)
	{
		F.id[outIdx] = idx;

		// use pos for new particle
		F.xPos[outIdx] = pos.x;
		F.yPos[outIdx] = pos.y;
		F.zPos[outIdx] = pos.z;

		// no velocity by default
		F.xVel[outIdx] = 0.0f;
		F.yVel[outIdx] = 0.0f;
		F.zVel[outIdx] = 0.0f;
		F.xVelPred[outIdx] = 0.0f;
		F.yVelPred[outIdx] = 0.0f;
		F.zVelPred[outIdx] = 0.0f;

		// no acceleration by default
		F.xAccPress[outIdx] = 0.0f;
		F.yAccPress[outIdx] = 0.0f;
		F.zAccPress[outIdx] = 0.0f;

		// default pressure, density, etc...
		F.sourceTerm[outIdx] = 0.0f;
		F.diagElem[outIdx] = 0.0f;
		F.pressure[outIdx] = 0.0f;
		F.density[outIdx] = params.restDensity;
	};

	// keep only the generated particles inside the build region
	if (hasBuildRegion_)
	{
		// first pass to determine how many particles to keep
		uint32_t N_keep = 0;
		for (uint32_t kw = 0; kw < N_w; kw++) // depth
		{
			for (uint32_t jv = 0; jv < N_v; jv++) // height
			{
				for (uint32_t iu = 0; iu < N_u; iu++) // width
				{
					const uint32_t idx = N_base + (N_v * kw + jv) * N_u + iu;

					if (IsInsideBuildRegion(ParticlePosition(iu, jv, kw, idx))) { N_keep++; }
				}
			}
		}
		if (N_keep == 0) { return; }

		// resize fluid particle data structures according to the kept particle count
		const uint32_t N_localBase = F.N_;
		dst.ResizeFluid(N_localBase + N_keep);

		// second pass to actually add particles
		uint32_t outIdx = N_localBase;
		for (uint32_t kw = 0; kw < N_w; kw++) // depth
		{
			for (uint32_t jv = 0; jv < N_v; jv++) // height
			{
				for (uint32_t iu = 0; iu < N_u; iu++) // width
				{
					const uint32_t idx = N_base + (N_v * kw + jv) * N_u + iu;
					const glm::vec3 pos = ParticlePosition(iu, jv, kw, idx);

					// skip erroneous particles
					if (!IsInsideBuildRegion(pos)) { continue; }

					WriteParticle(outIdx, idx, pos);
					outIdx++;
				}
			}
		}

		return;
	}

	// resize fluid particle data structures according to the new total particle count
	dst.ResizeFluid(N_base + N_add);

	// add particles
	if (N_w > 0 && N_v > 0 && N_u > 0)
	{
		tbb::parallel_for(
			tbb::blocked_range3d<int>(0, N_w, 0, N_v, 0, N_u),
			[&](const tbb::blocked_range3d<int>& r)
			{
				for (int kw = r.pages().begin(); kw != r.pages().end(); kw++) // depth
				{
					for (int jv = r.rows().begin(); jv != r.rows().end(); jv++) // height
					{
						for (int iu = r.cols().begin(); iu != r.cols().end(); iu++) // width
						{
							// get global index of currently handled particle
							const uint32_t local = (N_v * kw + jv) * N_u + iu;
							const uint32_t idx = N_base + local;

							WriteParticle(idx, idx, ParticlePosition(
								static_cast<uint32_t>(iu),
								static_cast<uint32_t>(jv),
								static_cast<uint32_t>(kw),
								idx));
						}
					}
				}
			},
			tbb::auto_partitioner{}
		);
	}
}

void ScenarioMgr::BuildFluidBlock_OBB_Filtered(const OBB& box,
											   Sync::UniversalSimState& dst,
											   const Sim::SimParams& params,
											   const ParticleKeepPredicate& keep)
{
	const float size = params.particleSizeF;
	const float radius = params.particleSizeF * 0.5f;

	// inner lengths for centers (shrink by one radius per side)
	const float uLen_i = box.uLen - 2.0f * radius;
	const float vLen_i = box.vLen - 2.0f * radius;
	const float wLen_i = box.wLen - 2.0f * radius;
	if (uLen_i <= 0.0f || vLen_i <= 0.0f || wLen_i <= 0.0f) { return; }

	// number of particles in each direction (inside inner OBB)
	const uint32_t N_u = TilesCount(uLen_i, size);
	const uint32_t N_v = TilesCount(vLen_i, size);
	const uint32_t N_w = TilesCount(wLen_i, size);
	if (N_u <= 0 || N_v <= 0 || N_w <= 0) { return; }

	// offsets from box planes
	const float uOff = radius + StartOffset(uLen_i, size, static_cast<int>(N_u));
	const float vOff = radius + StartOffset(vLen_i, size, static_cast<int>(N_v));
	const float wOff = radius + StartOffset(wLen_i, size, static_cast<int>(N_w));

	auto& F = dst.particlesF;

	// dense generation index of the first particle, so IDs stay the global scenario indices
	const uint32_t N_base = hasBuildRegion_ ? N_builtGlobalF_ : F.N_;
	const float jitAmp = params.jitterFracF * params.particleSizeF;

	// world position of one generated particle, so the region passes below stay consistent
	const auto ParticlePosition = [&](const uint32_t iu, const uint32_t jv,
		const uint32_t kw, const uint32_t idx) -> glm::vec3
	{
		float u = uOff + static_cast<float>(iu) * size;
		float v = vOff + static_cast<float>(jv) * size;
		float w = wOff + static_cast<float>(kw) * size;

		if (jitAmp > 0.0f)
		{
			// use (idx * 3 + {0,1,2}) as unique keys for x,y,z jitter
			const uint64_t baseSeed = static_cast<uint64_t>(params.jitterSeedF);

			u += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 0ull, jitAmp);
			v += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 1ull, jitAmp);
			w += Jitter3D(baseSeed, static_cast<uint64_t>(idx) * 3ull + 2ull, jitAmp);
		}

		u = std::clamp(u, radius, box.uLen - radius);
		v = std::clamp(v, radius, box.vLen - radius);
		w = std::clamp(w, radius, box.wLen - radius);

		return box.origin
			+ box.uHat * u
			+ box.vHat * v
			+ box.wHat * w;
	};

	// first pass determines both the global filtered count and the local region count
	uint32_t N_keepGlobal = 0;
	uint32_t N_keepLocal = 0;
	for (uint32_t kw = 0; kw < N_w; kw++) // depth
	{
		for (uint32_t jv = 0; jv < N_v; jv++) // height
		{
			for (uint32_t iu = 0; iu < N_u; iu++) // width
			{
				// position components
				float u = uOff + static_cast<float>(iu) * size;
				float v = vOff + static_cast<float>(jv) * size;
				float w = wOff + static_cast<float>(kw) * size;

				// clamp to make edge particles remain fully inside the box
				u = std::clamp(u, radius, box.uLen - radius);
				v = std::clamp(v, radius, box.vLen - radius);
				w = std::clamp(w, radius, box.wLen - radius);

				// apply filter
				if (keep && !keep(u, v, w)) { continue; }

				const uint32_t globalIdx = N_base + N_keepGlobal;
				N_keepGlobal++;
				if (!hasBuildRegion_ || IsInsideBuildRegion(
					ParticlePosition(iu, jv, kw, globalIdx)))
				{
					N_keepLocal++;
				}
			}
		}
	}
	if (N_keepGlobal == 0) { return; }

	// advance the global count on every rank, even if this rank has no particles in the region
	N_builtGlobalF_ += N_keepGlobal;
	if (N_keepLocal == 0) { return; }

	// resize fluid particle data structures
	const uint32_t N_localBase = F.N_;
	dst.ResizeFluid(N_localBase + N_keepLocal);
	uint32_t outIdx = N_localBase;
	uint32_t idx = N_base;

	// second pass to actually add particles
	for (uint32_t kw = 0; kw < N_w; kw++) // depth
	{
		for (uint32_t jv = 0; jv < N_v; jv++) // height
		{
			for (uint32_t iu = 0; iu < N_u; iu++) // width
			{
				// position components
				float u = uOff + static_cast<float>(iu) * size;
				float v = vOff + static_cast<float>(jv) * size;
				float w = wOff + static_cast<float>(kw) * size;

				u = std::clamp(u, radius, box.uLen - radius);
				v = std::clamp(v, radius, box.vLen - radius);
				w = std::clamp(w, radius, box.wLen - radius);

				// apply filter
				if (keep && !keep(u, v, w)) { continue; }

				const glm::vec3 pos = ParticlePosition(iu, jv, kw, idx);
				if (hasBuildRegion_ && !IsInsideBuildRegion(pos))
				{
					idx++;
					continue;
				}

				F.id[outIdx] = idx;

				// use pos for new particle
				F.xPos[outIdx] = pos.x;
				F.yPos[outIdx] = pos.y;
				F.zPos[outIdx] = pos.z;

				// no velocity by default
				F.xVel[outIdx] = 0.0f;
				F.yVel[outIdx] = 0.0f;
				F.zVel[outIdx] = 0.0f;
				F.xVelPred[outIdx] = 0.0f;
				F.yVelPred[outIdx] = 0.0f;
				F.zVelPred[outIdx] = 0.0f;

				// no acceleration by default
				F.xAccPress[outIdx] = 0.0f;
				F.yAccPress[outIdx] = 0.0f;
				F.zAccPress[outIdx] = 0.0f;

				// default pressure, density, etc...
				F.sourceTerm[outIdx] = 0.0f;
				F.diagElem[outIdx] = 0.0f;
				F.pressure[outIdx] = 0.0f;
				F.density[outIdx] = params.restDensity;

				outIdx++;
				idx++;
			}
		}
	}
}

void ScenarioMgr::BuildBoundaryBlock_OBB_Hollow(const OBB& box,
											    Sync::UniversalSimState& dst,
											    const Sim::SimParams& params)
{
	const float size = params.particleSizeB;
	const float radius = params.particleSizeB * 0.5f;
	const uint32_t layers = params.boundaryLayers;

	// shrink OBB lengths by radius in all directions, resulting in "inner" lengths
	const float uLen_i = box.uLen - 2.0f * radius;
	const float vLen_i = box.vLen - 2.0f * radius;
	const float wLen_i = box.wLen - 2.0f * radius;
	if (uLen_i <= 0.0f || vLen_i <= 0.0f || wLen_i <= 0.0f) { return; }

	const glm::vec3 origin_i = box.origin
							 + box.uHat * radius
							 + box.vHat * radius
							 + box.wHat * radius;

	// number of particles in each direction
	// (along the shrunken "inner" OBB)
	const uint32_t N_u = TilesCount(uLen_i, size);
	const uint32_t N_v = TilesCount(vLen_i, size);
	const uint32_t N_w = TilesCount(wLen_i, size);

	// compute number of particles we keep to make the box hollow
	const uint32_t N_keep = ParticleCountHollowBox(N_u, N_v, N_w, layers);
	if (N_keep <= 0) { return; }

	auto& B = dst.particlesB;

	// resize boundary particle data structures
	const uint32_t N_base = B.N_;
	const uint32_t N_add = N_keep;
	dst.ResizeBoundary(N_base + N_add);

	// jittering params
	const float jitAmp = params.jitterFracB * params.particleSizeB;
	std::mt19937_64 rng(params.jitterSeedB);
	std::uniform_real_distribution<float> uniformDist(-jitAmp, +jitAmp);

	uint32_t idx = N_base;

	for (uint32_t kw = 0; kw < N_w; kw++) // depth
	{
		// interior particle beyond layers in w, and clamp to prevent drift
		const bool wInside = (kw >= layers) && (kw <= N_w - 1 - layers);
		float w = std::min(wLen_i, static_cast<float>(kw) * size);

		for (uint32_t jv = 0; jv < N_v; jv++) // height
		{
			// interior particle beyond layers in v, and clamp to prevent drift
			const bool vInside = (jv >= layers) && (jv <= N_v - 1 - layers);
			float v = std::min(vLen_i, static_cast<float>(jv) * size);

			for (uint32_t iu = 0; iu < N_u; iu++) // width
			{
				// interior particle beyond layers in u, and clamp to prevent drift
				const bool uInside = (iu >= layers) && (iu <= N_u - 1 - layers);
				float u = std::min(uLen_i, static_cast<float>(iu) * size);

				// inside all three axes -> interior hollow, skip
				if (uInside && vInside && wInside) { continue; }

				// optionally apply jittering
				if (jitAmp > 0.0f)
				{
					u += uniformDist(rng);
					v += uniformDist(rng);
					w += uniformDist(rng);
				}

				// clamp to make edge particles remain inside the box
				u = std::clamp(u, 0.0f, uLen_i);
				v = std::clamp(v, 0.0f, vLen_i);
				w = std::clamp(w, 0.0f, wLen_i);

				const glm::vec3 pos = origin_i
									+ box.uHat * u
									+ box.vHat * v
									+ box.wHat * w;

				B.id[idx] = idx;

				// use pos for new particle
				B.xPos[idx] = pos.x;
				B.yPos[idx] = pos.y;
				B.zPos[idx] = pos.z;

				idx++;
			}
		}
	}

	assert(idx == (N_base + N_add));
	ComputeBoundaryCoefficients(dst.particlesB, params, 0, dst.particlesB.N_);
}

uint32_t ScenarioMgr::BuildRigidBody_OBB_Hollow(
	const OBB& box,
	Sync::UniversalSimState& dst,
	const Sim::SimParams& params,
	const float rigidDensity)
{
	const float size = params.particleSizeB;
	const float radius = params.particleSizeB * 0.5f;
	const uint32_t layers = params.boundaryLayers;

	// shrink OBB lengths by radius in all directions, resulting in "inner" lengths
	const float uLen_i = box.uLen - 2.0f * radius;
	const float vLen_i = box.vLen - 2.0f * radius;
	const float wLen_i = box.wLen - 2.0f * radius;
	if (uLen_i <= 0.0f || vLen_i <= 0.0f || wLen_i <= 0.0f)
	{
		return Sync::InvalidRigidBodyId;
	}

	const glm::vec3 origin_i = box.origin
							 + box.uHat * radius
							 + box.vHat * radius
							 + box.wHat * radius;

	const uint32_t N_u = TilesCount(uLen_i, size);
	const uint32_t N_v = TilesCount(vLen_i, size);
	const uint32_t N_w = TilesCount(wLen_i, size);
	const uint32_t N_keep = ParticleCountHollowBox(N_u, N_v, N_w, layers);
	if (N_keep <= 0) { return Sync::InvalidRigidBodyId; }

	auto& B = dst.particlesB;
	auto& R = dst.particlesR;
	auto& RB = dst.rigidBodies;

	const uint32_t bodyIdx = RB.N_;
	const uint32_t N_base = R.N_;
	const uint32_t N_add = N_keep;

	dst.ResizeRigidBodies(bodyIdx + 1);
	dst.ResizeRigid(N_base + N_add);

	const glm::vec3 center = box.origin
						   + box.uHat * (0.5f * box.uLen)
						   + box.vHat * (0.5f * box.vLen)
						   + box.wHat * (0.5f * box.wLen);

	const float volume = box.uLen * box.vLen * box.wLen;
	const float bodyMass = rigidDensity > 0.0f ? rigidDensity * volume : 0.0f;
	const float bodyMassInv = bodyMass > 0.0f ? 1.0f / bodyMass : 0.0f;

	RB.id[bodyIdx] = bodyIdx;
	RB.xCM[bodyIdx] = center.x;
	RB.yCM[bodyIdx] = center.y;
	RB.zCM[bodyIdx] = center.z;
	RB.rigidParticleBegin[bodyIdx] = N_base;
	RB.rigidParticleCount[bodyIdx] = N_add;
	RB.mass[bodyIdx] = bodyMass;
	RB.massInv[bodyIdx] = bodyMassInv;

	RB.a00[bodyIdx] = box.uHat.x; RB.a01[bodyIdx] = box.vHat.x; RB.a02[bodyIdx] = box.wHat.x;
	RB.a10[bodyIdx] = box.uHat.y; RB.a11[bodyIdx] = box.vHat.y; RB.a12[bodyIdx] = box.wHat.y;
	RB.a20[bodyIdx] = box.uHat.z; RB.a21[bodyIdx] = box.vHat.z; RB.a22[bodyIdx] = box.wHat.z;

	RB.iBodyInv00[bodyIdx] = bodyMass > 0.0f
		? 12.0f / (bodyMass * (box.vLen * box.vLen + box.wLen * box.wLen))
		: 0.0f;
	RB.iBodyInv11[bodyIdx] = bodyMass > 0.0f
		? 12.0f / (bodyMass * (box.uLen * box.uLen + box.wLen * box.wLen))
		: 0.0f;
	RB.iBodyInv22[bodyIdx] = bodyMass > 0.0f
		? 12.0f / (bodyMass * (box.uLen * box.uLen + box.vLen * box.vLen))
		: 0.0f;

	// jittering params
	const float jitAmp = params.jitterFracB * params.particleSizeB;
	std::mt19937_64 rng(params.jitterSeedB);
	std::uniform_real_distribution<float> uniformDist(-jitAmp, +jitAmp);

	uint32_t rigidParticleIdx = N_base;

	for (uint32_t kw = 0; kw < N_w; kw++) // depth
	{
		// interior particle beyond layers in w, and clamp to prevent drift
		const bool wInside = (kw >= layers) && (kw <= N_w - 1 - layers);
		float w = std::min(wLen_i, static_cast<float>(kw) * size);

		for (uint32_t jv = 0; jv < N_v; jv++) // height
		{
			// interior particle beyond layers in v, and clamp to prevent drift
			const bool vInside = (jv >= layers) && (jv <= N_v - 1 - layers);
			float v = std::min(vLen_i, static_cast<float>(jv) * size);

			for (uint32_t iu = 0; iu < N_u; iu++) // width
			{
				// interior particle beyond layers in u, and clamp to prevent drift
				const bool uInside = (iu >= layers) && (iu <= N_u - 1 - layers);
				float u = std::min(uLen_i, static_cast<float>(iu) * size);

				if (uInside && vInside && wInside) { continue; }

				if (jitAmp > 0.0f)
				{
					u += uniformDist(rng);
					v += uniformDist(rng);
					w += uniformDist(rng);
				}

				u = std::clamp(u, 0.0f, uLen_i);
				v = std::clamp(v, 0.0f, vLen_i);
				w = std::clamp(w, 0.0f, wLen_i);

				const glm::vec3 pos = origin_i
									+ box.uHat * u
									+ box.vHat * v
									+ box.wHat * w;
				const glm::vec3 rel = pos - center;

				R.id[rigidParticleIdx] = B.N_ + rigidParticleIdx;
				R.bodyId[rigidParticleIdx] = bodyIdx;
				R.xRelPos[rigidParticleIdx] = Dot(rel, box.uHat);
				R.yRelPos[rigidParticleIdx] = Dot(rel, box.vHat);
				R.zRelPos[rigidParticleIdx] = Dot(rel, box.wHat);

				rigidParticleIdx++;
			}
		}
	}

	assert(rigidParticleIdx == (N_base + N_add));
	ComputeRigidCoefficients(R, params, N_base, N_base + N_add);

	return bodyIdx;
}

uint32_t ScenarioMgr::BuildRigidBody_FromSurfaceSamples(
	const std::vector<SurfaceSample>& bodyLocalSamples,
	const glm::vec3& centerOfMass,
	const glm::mat3& orientation,
	const float mass,
	const glm::mat3& bodyInertiaInv,
	Sync::UniversalSimState& dst,
	const Sim::SimParams& params)
{
	// validate supplied samples and rigid body properties
	if (bodyLocalSamples.empty()) { return Sync::InvalidRigidBodyId; }
	if (bodyLocalSamples.size() > std::numeric_limits<uint32_t>::max())
	{
		throw std::runtime_error("Rigid surface sample count exceeds uint32 limits.");
	}
	if (!IsFinite(centerOfMass) || !IsFinite(orientation) || !IsRotationMatrix(orientation) ||
		!IsFinite(bodyInertiaInv) ||
		!std::isfinite(mass) || mass < 0.0f)
	{
		throw std::runtime_error("Rigid surface body properties are invalid.");
	}

	for (const SurfaceSample& sample : bodyLocalSamples)
	{
		if (!IsFinite(sample.pos))
		{
			throw std::runtime_error("Rigid surface contains a non-finite sample position.");
		}
	}

	// validate state growth before resizing any arrays
	auto& B = dst.particlesB;
	auto& R = dst.particlesR;
	auto& RB = dst.rigidBodies;
	const uint32_t N_add = static_cast<uint32_t>(bodyLocalSamples.size());
	if (RB.N_ == std::numeric_limits<uint32_t>::max() ||
		N_add > std::numeric_limits<uint32_t>::max() - R.N_)
	{
		throw std::runtime_error("Rigid surface state exceeds uint32 limits.");
	}

	const uint32_t bodyIdx = RB.N_;
	const uint32_t N_base = R.N_;

	// append the rigid body and its rigid particle range
	dst.ResizeRigidBodies(bodyIdx + 1);
	dst.ResizeRigid(N_base + N_add);

	RB.id[bodyIdx] = bodyIdx;
	RB.rigidParticleBegin[bodyIdx] = N_base;
	RB.rigidParticleCount[bodyIdx] = N_add;
	RB.xCM[bodyIdx] = centerOfMass.x;
	RB.yCM[bodyIdx] = centerOfMass.y;
	RB.zCM[bodyIdx] = centerOfMass.z;

	// store local-to-world orientation in row-major solver layout
	RB.a00[bodyIdx] = orientation[0][0];
	RB.a01[bodyIdx] = orientation[1][0];
	RB.a02[bodyIdx] = orientation[2][0];
	RB.a10[bodyIdx] = orientation[0][1];
	RB.a11[bodyIdx] = orientation[1][1];
	RB.a12[bodyIdx] = orientation[2][1];
	RB.a20[bodyIdx] = orientation[0][2];
	RB.a21[bodyIdx] = orientation[1][2];
	RB.a22[bodyIdx] = orientation[2][2];

	RB.mass[bodyIdx] = mass;
	RB.massInv[bodyIdx] = mass > 0.0f ? 1.0f / mass : 0.0f;

	// store inverse inertia in body frame
	RB.iBodyInv00[bodyIdx] = bodyInertiaInv[0][0];
	RB.iBodyInv01[bodyIdx] = bodyInertiaInv[1][0];
	RB.iBodyInv02[bodyIdx] = bodyInertiaInv[2][0];
	RB.iBodyInv10[bodyIdx] = bodyInertiaInv[0][1];
	RB.iBodyInv11[bodyIdx] = bodyInertiaInv[1][1];
	RB.iBodyInv12[bodyIdx] = bodyInertiaInv[2][1];
	RB.iBodyInv20[bodyIdx] = bodyInertiaInv[0][2];
	RB.iBodyInv21[bodyIdx] = bodyInertiaInv[1][2];
	RB.iBodyInv22[bodyIdx] = bodyInertiaInv[2][2];

	// append rigid particle ownership and body-local positions
	for (uint32_t localIdx = 0; localIdx < N_add; localIdx++)
	{
		const uint32_t rigidParticleIdx = N_base + localIdx;
		const glm::vec3 localPos = bodyLocalSamples[localIdx].pos;

		R.id[rigidParticleIdx] = B.N_ + rigidParticleIdx;
		R.bodyId[rigidParticleIdx] = bodyIdx;
		R.xRelPos[rigidParticleIdx] = localPos.x;
		R.yRelPos[rigidParticleIdx] = localPos.y;
		R.zRelPos[rigidParticleIdx] = localPos.z;
	}

	// derive per-particle volume and pseudo-mass from the sampled surface
	ComputeRigidCoefficients(R, params, N_base, N_base + N_add);
	return bodyIdx;
}

glm::mat3 ScenarioMgr::MakeOrientation_FromEulerAngles(
	const float yawDeg,
	const float pitchDeg,
	const float rollDeg) const
{
	if (!std::isfinite(yawDeg) ||
		!std::isfinite(pitchDeg) ||
		!std::isfinite(rollDeg))
	{
		throw std::runtime_error("Rigid body Euler angles must be finite.");
	}

	constexpr float DEG_TO_RAD = std::numbers::pi_v<float> / 180.0f;
	const float yaw = yawDeg * DEG_TO_RAD;
	const float pitch = pitchDeg * DEG_TO_RAD;
	const float roll = rollDeg * DEG_TO_RAD;

	const float cosYaw = std::cos(yaw);
	const float sinYaw = std::sin(yaw);
	const float cosPitch = std::cos(pitch);
	const float sinPitch = std::sin(pitch);
	const float cosRoll = std::cos(roll);
	const float sinRoll = std::sin(roll);

	// yaw around world Y
	const glm::mat3 yawRotation
	{
		{ cosYaw, 0.0f, -sinYaw },
		{ 0.0f, 1.0f, 0.0f },
		{ sinYaw, 0.0f, cosYaw },
	};

	// pitch around local Z
	const glm::mat3 pitchRotation
	{
		{ cosPitch, sinPitch, 0.0f },
		{ -sinPitch, cosPitch, 0.0f },
		{ 0.0f, 0.0f, 1.0f },
	};

	// roll around local X
	const glm::mat3 rollRotation
	{
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, cosRoll, sinRoll },
		{ 0.0f, -sinRoll, cosRoll },
	};

	return yawRotation * pitchRotation * rollRotation;
}

OBB ScenarioMgr::MakeOBB_FromAxes(glm::vec3 origin,
								  glm::vec3 uDir, float uLen,
								  glm::vec3 vDir, float vLen,
								  glm::vec3 wDir, float wLen)
{
	// any dir that is zero is set to a default axis
	if (Base::ApproxEqu(uDir, glm::vec3(0))) { uDir = { 1, 0, 0 }; }
	if (Base::ApproxEqu(vDir, glm::vec3(0))) { vDir = { 0, 1, 0 }; }
	if (Base::ApproxEqu(wDir, glm::vec3(0))) { wDir = { 0, 0, 1 }; }

	// TODO: normalization?

	return OBB{ origin, uDir, vDir, wDir, uLen, vLen, wLen };
}

std::array<OBBFace, 6> inline ScenarioMgr::GetFacesFromOBB(const OBB& box)
{
	std::array<OBBFace, 6> faces{};

	// near (zMin)
	faces[0] = { box.origin, box.uHat, box.vHat, -box.wHat, box.uLen, box.vLen };
	// far (zMax)
	faces[1] = { box.origin + box.wHat * box.wLen, box.uHat, box.vHat, +box.wHat, box.uLen, box.vLen };
	// left (xMin)
	faces[2] = { box.origin, box.wHat, box.vHat, -box.uHat, box.wLen, box.vLen };
	// right (xMax)
	faces[3] = { box.origin + box.uHat * box.uLen, box.wHat, box.vHat, +box.uHat, box.wLen, box.vLen };
	// bottom (yMin)
	faces[4] = { box.origin, box.uHat, box.wHat, -box.vHat, box.uLen, box.wLen };
	// top (yMax)
	faces[5] = { box.origin + box.vHat * box.vLen, box.uHat, box.wHat, +box.vHat, box.uLen, box.wLen };

	return faces;
}

OBB ScenarioMgr::ParseOBBFromJson(const json& j, ScenarioMgr& scenarioMgr)
{
	auto readVec3 = [](const json& v, glm::vec3 defaultVal) -> glm::vec3
		{
			if (!v.is_array() || v.size() != 3) return defaultVal;

			return { v[0].get<float>(),
					   v[1].get<float>(),
					   v[2].get<float>() };
		};

	glm::vec3 origin = readVec3(j.value("origin",
		json::array({ 0.0f, 0.0f, 0.0f })), { 0,0,0 });
	glm::vec3 uDir = readVec3(j.value("uDir",
		json::array({ 1.0f, 0.0f, 0.0f })), { 1,0,0 });
	glm::vec3 vDir = readVec3(j.value("vDir",
		json::array({ 0.0f, 1.0f, 0.0f })), { 0,1,0 });
	glm::vec3 wDir = readVec3(j.value("wDir",
		json::array({ 0.0f, 0.0f, 1.0f })), { 0,0,1 });

	float uLen = j.value("uLen", 1.0f);
	float vLen = j.value("vLen", 1.0f);
	float wLen = j.value("wLen", 1.0f);

	return scenarioMgr.MakeOBB_FromAxes(origin, uDir, uLen, vDir, vLen, wDir, wLen);
}

void ScenarioMgr::BuildBoundaryLayersOnFace(const OBBFace& face,
											Sync::UniversalSimState& dst,
											const Sim::SimParams& params)
{
	const float size = params.particleSizeB;
	const uint32_t N_u = TilesCount(face.uLen, size); // num particles in u direction
	const uint32_t N_v = TilesCount(face.vLen, size); // num particles in v direction
	if (N_u <= 0 || N_v <= 0) { return; }

	const float defaultVolB = DefaultBoundaryVolB(params);
	const float defaultPsiB = DefaultBoundaryPsiB(params);

	// pre-reserve memory for new particles
	// (a build region keeps only a slice of the face, so its storage grows on demand instead)
	if (!hasBuildRegion_)
	{
		dst.ReserveBoundaryAdditional(N_u * N_v * params.boundaryLayers);
	}
	auto& B = dst.particlesB;

	for (uint32_t layer = 0; layer < params.boundaryLayers; layer++)
	{
		const float outOffset = params.particleSizeB * 0.5f
							  + static_cast<float>(layer) * size;

		for (uint32_t i = 0; i < N_u; i++)
		{
			const float su = std::min(static_cast<float>(i) * size, face.uLen);
			const glm::vec3 bu = face.uHat * su;

			for (uint32_t j = 0; j < N_v; j++)
			{
				const float sv = std::min(static_cast<float>(j) * size, face.vLen);
				const glm::vec3 bv = face.vHat * sv;

				const glm::vec3 pos = face.origin + bu + bv + face.nHat * outOffset;

				// TODO: add jittering?

				// dense generation index of the first particle, so IDs stay the global scenario indices
				const uint32_t idx = hasBuildRegion_
					? N_builtGlobalB_
					: static_cast<uint32_t>(B.xPos.size());
				N_builtGlobalB_++;

				if (!IsInsideBuildRegion(pos)) { continue; }

				B.id.push_back(idx);

				// use pos for new particle
				B.xPos.push_back(pos.x);
				B.yPos.push_back(pos.y);
				B.zPos.push_back(pos.z);
				B.volB.push_back(defaultVolB);
				B.psiB.push_back(defaultPsiB);

			}
		}
	}

	// update boundary particle count
	dst.particlesB.N_ = static_cast<uint32_t>(dst.particlesB.xPos.size());
}

// deterministic hash-based jitter, safe for parallel use
inline float ScenarioMgr::Jitter3D(uint64_t baseSeed, uint64_t key, float amp)
{
	// using SplitMix64-style hash
	uint64_t x = baseSeed + key * 0x9E3779B97F4A7C15ull;
	x ^= x >> 30;
	x *= 0xBF58476D1CE4E5B9ull;
	x ^= x >> 27;
	x *= 0x94D049BB133111EBull;
	x ^= x >> 31;

	// map to [0,1)
	const double u01 = (x >> 11) * (1.0 / 9007199254740992.0); // 2^53

	// map to [-amp, +amp]
	return static_cast<float>((u01 * 2.0 - 1.0) * amp);
}

inline float ScenarioMgr::StartOffset(float length, float spacing, int N, bool centerToFit)
{
	if (!centerToFit || N <= 1) { return 0.0f; }

	const float covered = (N - 1) * spacing;
	return 0.5f * fmaxf(0.0f, length - covered);
}

inline uint32_t ScenarioMgr::TilesCount(float length, float spacing)
{
	if (spacing <= 0.0f) { return 0; }

	return std::max(1, static_cast<int>(std::floor(length / spacing)) + 1);
}

inline uint32_t ScenarioMgr::ParticleCountHollowBox(
	uint32_t N_u, uint32_t N_v, uint32_t N_w, uint32_t layers)
{
	// total particle count of "outer" box
	const uint32_t N_total = N_u * N_v * N_w;
	const int l = static_cast<int>(layers);
	// compute particle count in each direction for "inner" box
	const uint32_t N_u_i = std::max(static_cast<int>(N_u) - 2 * l, 0);
	const uint32_t N_v_i = std::max(static_cast<int>(N_v) - 2 * l, 0);
	const uint32_t N_w_i = std::max(static_cast<int>(N_w) - 2 * l, 0);
	const uint32_t N_inner = N_u_i * N_v_i * N_w_i;

	return N_total - N_inner;
}
