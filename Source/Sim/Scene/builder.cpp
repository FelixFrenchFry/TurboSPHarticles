// builder.cpp

// external
#include <cassert>
#include <algorithm>
#include <cmath>
#include <glm/vec3.hpp>
#include <omp.h>
#include <random>

// internal
#include "Source/Base/helpers.h"
#include "builder.h"



using namespace Scene;

// public ==============================

void ScenarioMgr::BuildBoundaryContainer_OBB(const OBB& box,
											 Sync::UniversalSimState& dst,
											 const Sim::SimParams& params)
{
	const auto faces = GetFacesFromOBB(box);

	for (const auto& face : faces)
	{
		BuildBoundaryLayersOnFace(face, dst, params);
	}
}

void ScenarioMgr::BuildFluidBlock_OBB(const OBB& box,
									  Sync::UniversalSimState& dst,
									  const Sim::SimParams& params)
{
	const float size = params.particleSizeF;
	const float radius = params.particleRadiusF;

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
	const float uOff = radius + StartOffset(uLen_i, size, N_u);
	const float vOff = radius + StartOffset(vLen_i, size, N_v);
	const float wOff = radius + StartOffset(wLen_i, size, N_w);

	const float jitAmp = params.jitterFracF * params.particleSizeF;

	#pragma omp parallel
	{
		std::mt19937_64 rng(params.jitterSeedF + static_cast<uint32_t>(omp_get_thread_num()));
		std::uniform_real_distribution<float> uniformDist(-jitAmp, +jitAmp);

		#pragma omp for collapse(3) schedule(static)
		for (int kw = 0; kw < N_w; kw++) // loop over depth
		{
			for (int jv = 0; jv < N_v; jv++) // loop over height
			{
				for (int iu = 0; iu < N_u; iu++) // loop over width
				{
					// get global index of currently handled particle
					const uint32_t local = (static_cast<int64_t>(N_v) * kw + jv) * N_u + iu;
					const uint32_t idx = N_base + local;

					// position components
					float u = uOff + iu * size;
					float v = vOff + jv * size;
					float w = wOff + kw * size;

					// optionally apply jittering
					if (jitAmp > 0.0f)
					{
						u += uniformDist(rng);
						v += uniformDist(rng);
						w += uniformDist(rng);
					}

					// clamp to make edge particles remain fully inside the box
					u = std::clamp(u, radius, box.uLen - radius);
					v = std::clamp(v, radius, box.vLen - radius);
					w = std::clamp(w, radius, box.wLen - radius);

					const glm::vec3 pos = box.origin
										+ box.uHat * u
										+ box.vHat * v
										+ box.wHat * w;

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
					F.xAcc[idx] = 0.0f;
					F.yAcc[idx] = 0.0f;
					F.zAcc[idx] = 0.0f;
					F.xAccPress[idx] = 0.0f;
					F.yAccPress[idx] = 0.0f;
					F.zAccPress[idx] = 0.0f;
					F.xAccVisc[idx] = 0.0f;
					F.yAccVisc[idx] = 0.0f;
					F.zAccVisc[idx] = 0.0f;
					F.xAccExt[idx] = 0.0f;
					F.yAccExt[idx] = 0.0f;
					F.zAccExt[idx] = 0.0f;

					// default pressure, density, etc...
					F.sourceTerm[idx] = 0.0f;
					F.diagElem[idx] = 0.0f;
					F.sysVecElem[idx] = 0.0f;
					F.pressure[idx] = 0.0f;
					F.density[idx] = params.restDensityF;

					// use global idx as ID
					F.id[idx] = static_cast<std::uint32_t>(idx);
				}
			}
		}
	}
}

void ScenarioMgr::BuildBoundaryBlockHollow_OBB(const OBB& box,
											   Sync::UniversalSimState& dst,
											   const Sim::SimParams& params)
{
	const float size = params.particleSizeB;
	const float radius = params.particleRadiusB;
	const int layers = std::max(0, params.boundaryLayers);

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
	const int N_u = TilesCount(uLen_i, size);
	const int N_v = TilesCount(vLen_i, size);
	const int N_w = TilesCount(wLen_i, size);
	if (N_u <= 0 || N_v <= 0 || N_w <= 0) { return; }

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

	for (int kw = 0; kw < N_w; kw++) // loop over depth
	{
		// interior particle beyond layers in w, and clamp to prevent drift
		const bool wInside = (kw >= layers) && (kw <= N_w - 1 - layers);
		float w = fminf(wLen_i, kw * size);

		for (int jv = 0; jv < N_v; jv++) // loop over height
		{
			// interior particle beyond layers in v, and clamp to prevent drift
			const bool vInside = (jv >= layers) && (jv <= N_v - 1 - layers);
			float v = fminf(vLen_i, jv * size);

			for (int iu = 0; iu < N_u; iu++) // loop over width
			{
				// interior particle beyond layers in u, and clamp to prevent drift
				const bool uInside = (iu >= layers) && (iu <= N_u - 1 - layers);
				float u = fminf(uLen_i, iu * size);

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

				// use pos for new particle
				B.xPos[idx] = pos.x;
				B.yPos[idx] = pos.y;
				B.zPos[idx] = pos.z;

				// no velocity by default
				B.xVel[idx] = 0.0f;
				B.yVel[idx] = 0.0f;
				B.zVel[idx] = 0.0f;

				// default pressure, density, etc...
				B.pressure[idx] = 0.0f;
				B.density[idx] = params.restDensityB;

				// use global idx as ID
				B.id[idx] = static_cast<std::uint32_t>(idx);

				idx++;
			}
		}
	}

	assert(idx == (N_base + N_add));
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

// private ==============================

void ScenarioMgr::BuildBoundaryLayersOnFace(const OBBFace& face,
											Sync::UniversalSimState& dst,
											const Sim::SimParams& params)
{
	const float size = params.particleSizeB;
	const uint32_t N_u = TilesCount(face.uLen, size); // num particles in u direction
	const uint32_t N_v = TilesCount(face.vLen, size); // num particles in v direction
	if (N_u <= 0 || N_v <= 0) { return; }

	auto& B = dst.boundaryParticles;

	// pre-reserve memory for new particles
	B.reserveAll(B.N_ + N_u * N_v * params.boundaryLayers);

	for (int layer = 0; layer < params.boundaryLayers; layer++)
	{
		const float outOffset = params.particleRadiusB + layer * size;

		for (int i = 0; i < N_u; i++)
		{
			const float su = fminf(i * size, face.uLen);
			const glm::vec3 bu = face.uHat * su;

			for (int j = 0; j < N_v; j++)
			{
				const float sv = fminf(j * size, face.vLen);
				const glm::vec3 bv = face.vHat * sv;

				const glm::vec3 pos = face.origin + bu + bv + face.nHat * outOffset;

				// TODO: add jittering

				// use pos for new particle
				B.xPos.push_back(pos.x);
				B.yPos.push_back(pos.y);
				B.zPos.push_back(pos.z);

				// no velocity by default
				B.xVel.push_back(0.0f);
				B.yVel.push_back(0.0f);
				B.zVel.push_back(0.0f);

				// default pressure, density, etc...
				B.pressure.push_back(0.0f);
				B.density.push_back(params.restDensityB);

				// simple running ID
				B.id.push_back(static_cast<uint32_t>(B.id.size()));
			}
		}
	}

	// update boundary particle count
	dst.boundaryParticles.N_ = static_cast<uint32_t>(dst.boundaryParticles.id.size());
}

float inline ScenarioMgr::StartOffset(float length, float spacing, int N, bool centerToFit)
{
	if (!centerToFit || N <= 1) { return 0.0f; }

	const float covered = (N - 1) * spacing;
	return 0.5f * fmaxf(0.0f, length - covered);
}

uint32_t inline ScenarioMgr::TilesCount(float length, float spacing)
{
	if (spacing <= 0.0f) { return 0; }

	return std::max(1, static_cast<int>(std::floor(length / spacing)) + 1);
}

uint32_t inline ScenarioMgr::ParticleCountHollowBox(int N_u, int N_v, int N_w, int layers)
{
	// total particle count of "outer" box
	const uint32_t N_total = N_u * N_v * N_w;
	// compute particle count in each direction for "inner" box
	const uint32_t N_u_i = std::max(N_u - 2 * layers, 0);
	const uint32_t N_v_i = std::max(N_v - 2 * layers, 0);
	const uint32_t N_w_i = std::max(N_w - 2 * layers, 0);
	const uint32_t N_inner = N_u_i * N_v_i * N_w_i;

	return N_total - N_inner;
}
