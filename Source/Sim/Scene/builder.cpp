// builder.cpp

// external
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/vec3.hpp>
#include <numbers>
#include <random>
#include <unordered_map>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// internal
#include "Source/Base/helpers.h"
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
		const Sim::SimParams& params)
	{
		constexpr float PI = std::numbers::pi_v<float>;
		constexpr float ZERO_EQU_LIM_F = 1e-7f;

		const float distSqu = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));

		if (distSqu >= params.kernelRadiusSqu) [[unlikely]]
		{
			return 0.0f;
		}

		const float cubicSplineAlpha =
			(1.0f / (4.0f * PI))
			* params.particleSizeInvF
			* params.particleSizeInvF
			* params.particleSizeInvF;

		if (distSqu < ZERO_EQU_LIM_F) [[unlikely]]
		{
			return 4.0f * cubicSplineAlpha;
		}

		const float dist = std::sqrt(distSqu);
		const float distNorm = dist * params.particleSizeInvF;

		const float t1 = std::max(1.0f - distNorm, 0.0f);
		const float t2 = std::max(2.0f - distNorm, 0.0f);
		const float t1_2 = t1 * t1;
		const float t2_2 = t2 * t2;
		const float t1_3 = t1_2 * t1;
		const float t2_3 = t2_2 * t2;

		return cubicSplineAlpha * (t2_3 - 4.0f * t1_3);
	}

} // namespace anonymous

void ScenarioMgr::ComputeBoundaryCoefficients(
	Sync::BoundaryParticles& boundaryParticles,
	const Sim::SimParams& params)
{
	// temp local cell buckets to avoid O(N_B^2) boundary neighbor search
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

	const uint32_t N_B = boundaryParticles.N_;
	if (N_B == 0) { return; }

	const float defaultVolB = DefaultBoundaryVolB(params);
	const float defaultPsiB = DefaultBoundaryPsiB(params);
	boundaryParticles.volB.resize(N_B);
	boundaryParticles.psiB.resize(N_B);
	const auto& xPosB = boundaryParticles.xPos;
	const auto& yPosB = boundaryParticles.yPos;
	const auto& zPosB = boundaryParticles.zPos;
	auto& volB = boundaryParticles.volB;
	auto& psiB = boundaryParticles.psiB;

	// keep coefficients valid if params are incomplete
	if (params.particleSizeF <= 0.0f ||
		params.particleSizeInvF <= 0.0f ||
		params.kernelRadius <= 0.0f ||
		params.kernelRadiusSqu <= 0.0f ||
		params.gridCellSizeRatio <= 0.0f ||
		params.restDensity <= 0.0f)
	{
		std::ranges::fill(volB, defaultVolB);
		std::ranges::fill(psiB, defaultPsiB);
		return;
	}

	const float cellSize = params.kernelRadius * params.gridCellSizeRatio;
	if (cellSize <= 0.0f)
	{
		std::ranges::fill(volB, defaultVolB);
		std::ranges::fill(psiB, defaultPsiB);
		return;
	}

	// same cell-reach rule as cell-linked list neighbor search
	const float querySpanCells = params.kernelRadius / cellSize;
	const int32_t cellQueryReach = std::max<int32_t>(
		1,
		static_cast<int32_t>(std::ceil(std::max(0.0f, querySpanCells - 1e-6f))));
	const float cellSizeInv = 1.0f / cellSize;

	std::vector<CellCoord> cellCoords(N_B);
	std::unordered_map<CellCoord, std::vector<uint32_t>, CellCoordHash> cellBuckets;
	cellBuckets.reserve(static_cast<uint64_t>(N_B) * 2ull);

	// insert boundary particles into cell buckets
	for (uint32_t i = 0; i < N_B; i++)
	{
		cellCoords[i] = makeCellCoord(
			xPosB[i],
			yPosB[i],
			zPosB[i],
			cellSizeInv);
		cellBuckets[cellCoords[i]].push_back(i);
	}

	// compute non-uniform boundary contribution coefficients for all boundary particles
	const auto& cellBucketsReadOnly = cellBuckets;
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0, N_B),
		[&](const tbb::blocked_range<uint32_t>& r)
		{
			for (uint32_t i = r.begin(); i != r.end(); i++)
			{
				float sumW = 0.0f;
				const CellCoord baseCell = cellCoords[i];

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
									xPosB[i] - xPosB[j],
									yPosB[i] - yPosB[j],
									zPosB[i] - zPosB[j],
									params);
							}
						}
					}
				}

				if (sumW > 0.0f && std::isfinite(sumW))
				{
					// boundary volume from inverse boundary sample density
					volB[i] = 1.0f / sumW;
					psiB[i] = params.restDensity * volB[i];
				}
				else
				{
					volB[i] = defaultVolB;
					psiB[i] = defaultPsiB;
				}
			}
		},
		tbb::auto_partitioner{}
	);
}

void ScenarioMgr::BuildBoundaryContainer_OBB(const OBB& box,
											 Sync::UniversalSimState& dst,
											 const Sim::SimParams& params)
{
	/*
	auto faces = GetFacesFromOBB(box);

	for (const auto& face : faces)
	{
		BuildBoundaryLayersOnFace(face, dst, params);
	}
	*/

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

	ComputeBoundaryCoefficients(dst.boundaryParticles, params);
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

	auto& F = dst.fluidParticles;

	// resize fluid particle data structures
	const uint32_t N_base = F.N_;
	const uint32_t N_add = N_u * N_v * N_w;
	dst.ResizeFluid(N_base + N_add);

	// offsets from box planes
	const float uOff = radius + StartOffset(uLen_i, size, static_cast<int>(N_u));
	const float vOff = radius + StartOffset(vLen_i, size, static_cast<int>(N_v));
	const float wOff = radius + StartOffset(wLen_i, size, static_cast<int>(N_w));

	const float jitAmp = params.jitterFracF * params.particleSizeF;

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

							const glm::vec3 pos = box.origin
												+ box.uHat * u
												+ box.vHat * v
												+ box.wHat * w;

							F.id[idx] = idx;

							// use pos for new particle
							F.xPos[idx] = pos.x;
							F.yPos[idx] = pos.y;
							F.zPos[idx] = pos.z;

							// no velocity by default
							F.xVel[idx] = 0.0f;
							F.yVel[idx] = 0.0f;
							F.zVel[idx] = 0.0f;
							F.xVelPred[idx] = 0.0f;
							F.yVelPred[idx] = 0.0f;
							F.zVelPred[idx] = 0.0f;

							// no acceleration by default
							F.xAccPress[idx] = 0.0f;
							F.yAccPress[idx] = 0.0f;
							F.zAccPress[idx] = 0.0f;

							// default pressure, density, etc...
							F.sourceTerm[idx] = 0.0f;
							F.diagElem[idx] = 0.0f;
							F.pressure[idx] = 0.0f;
							F.density[idx] = params.restDensity;

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

	// first pass to determine how many particles to keep
	uint32_t keepCount = 0;
	for (uint32_t kw = 0; kw < N_w; kw++)
	{
		for (uint32_t jv = 0; jv < N_v; jv++)
		{
			for (uint32_t iu = 0; iu < N_u; iu++)
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
				if (!keep || keep(u, v, w)) { keepCount++; }
			}
		}
	}
	if (keepCount == 0) { return; }

	auto& F = dst.fluidParticles;

	// resize fluid particle data structures
	const uint32_t N_base = F.N_;
	dst.ResizeFluid(N_base + keepCount);

	const float jitAmp = params.jitterFracF * params.particleSizeF;
	uint32_t outIdx = N_base;

	// second pass to actually add particles
	for (uint32_t kw = 0; kw < N_w; kw++)
	{
		for (uint32_t jv = 0; jv < N_v; jv++)
		{
			for (uint32_t iu = 0; iu < N_u; iu++)
			{
				// position components
				float u = uOff + static_cast<float>(iu) * size;
				float v = vOff + static_cast<float>(jv) * size;
				float w = wOff + static_cast<float>(kw) * size;

				// apply filter
				if (keep && !keep(u, v, w)) { continue; }

				// optionally apply jittering
				if (jitAmp > 0.0f)
				{
					// use (idx * 3 + {0,1,2}) as unique keys for x,y,z jitter
					const uint64_t baseSeed = static_cast<uint64_t>(params.jitterSeedF);

					u += Jitter3D(baseSeed, static_cast<uint64_t>(outIdx) * 3ull + 0ull, jitAmp);
					v += Jitter3D(baseSeed, static_cast<uint64_t>(outIdx) * 3ull + 1ull, jitAmp);
					w += Jitter3D(baseSeed, static_cast<uint64_t>(outIdx) * 3ull + 2ull, jitAmp);
				}

				// clamp to make edge particles remain fully inside the box
				u = std::clamp(u, radius, box.uLen - radius);
				v = std::clamp(v, radius, box.vLen - radius);
				w = std::clamp(w, radius, box.wLen - radius);

				const glm::vec3 pos = box.origin
									+ box.uHat * u
									+ box.vHat * v
									+ box.wHat * w;

				F.id[outIdx] = outIdx;

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
	const uint32_t layers = params.boundaryLayers < 0 ? 0 : params.boundaryLayers;

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

	auto& B = dst.boundaryParticles;

	// resize boundary particle data structures
	const uint32_t N_base = B.N_;
	const uint32_t N_add = N_keep;
	dst.ResizeBoundary(N_base + N_add);

	// jittering params
	const float jitAmp = params.jitterFracB * params.particleSizeB;
	std::mt19937_64 rng(params.jitterSeedB);
	std::uniform_real_distribution<float> uniformDist(-jitAmp, +jitAmp);

	uint32_t idx = N_base;

	for (uint32_t kw = 0; kw < N_w; kw++) // loop over depth
	{
		// interior particle beyond layers in w, and clamp to prevent drift
		const bool wInside = (kw >= layers) && (kw <= N_w - 1 - layers);
		float w = std::min(wLen_i, static_cast<float>(kw) * size);

		for (uint32_t jv = 0; jv < N_v; jv++) // loop over height
		{
			// interior particle beyond layers in v, and clamp to prevent drift
			const bool vInside = (jv >= layers) && (jv <= N_v - 1 - layers);
			float v = std::min(vLen_i, static_cast<float>(jv) * size);

			for (uint32_t iu = 0; iu < N_u; iu++) // loop over width
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
	ComputeBoundaryCoefficients(dst.boundaryParticles, params);
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
	dst.ReserveBoundaryAdditional(N_u * N_v * params.boundaryLayers);
	auto& B = dst.boundaryParticles;

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

				// TODO: add jittering

				const uint32_t idx = static_cast<uint32_t>(B.xPos.size());

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
	dst.boundaryParticles.N_ = static_cast<uint32_t>(dst.boundaryParticles.xPos.size());
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
