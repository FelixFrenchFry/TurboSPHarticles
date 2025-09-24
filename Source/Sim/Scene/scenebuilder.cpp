// scenebuilder.cpp

// external
#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/geometric.hpp>
#include <limits>

// internal
#include "scenebuilder.h"



using namespace Scene;

// public ==============================

void ScenarioMgr::BuildSceneContainer_Rect(const ContainerCorners& corners,
										   Sync::UniversalSimState& dst,
										   const Sim::SimParams& params)
{
	// get the 6 inner faces and their outward normals
	const auto faces = GetPlaneFacesFromCorners(corners);

	// build per-face boundary layers
	// (slabs parallel to face planes)
	for (const auto& face : faces)
	{
		BuildBoundaryLayersOnFace(face, dst, params);
	}
}

// private ==============================

std::array<OBBFace, 6> ScenarioMgr::GetPlaneFacesFromCorners(const ContainerCorners& corners)
{
	std::array<OBBFace, 6> faces{};

	// near face (zMin)
	faces[0] = MakeFace(corners.p000, corners.p100, corners.p010, -1.0f);
	// far face (zMax)
	faces[1] = MakeFace(corners.p001, corners.p101, corners.p011, +1.f);
	// left face (xMin)
	faces[2] = MakeFace(corners.p000, corners.p001, corners.p010, +1.f);
	// right face (xMax)
	faces[3] = MakeFace(corners.p100, corners.p101, corners.p110, -1.f);
	// bottom face (yMin)
	faces[4] = MakeFace(corners.p000, corners.p100, corners.p001, +1.f);
	// top face (yMax)
	faces[5] = MakeFace(corners.p010, corners.p110, corners.p011, -1.f);

	return faces;
}

void ScenarioMgr::BuildBoundaryLayersOnFace(const OBBFace& face,
											Sync::UniversalSimState& dst,
											const Sim::SimParams& params)
{
	const float spacing = params.particleSizeB;
	const int N_u = TilesCount(face.uLen, spacing); // num particles in u direction
	const int N_v = TilesCount(face.vLen, spacing); // num particles in v direction
	if (N_u <= 0 || N_v <= 0 ) { return; }

	// avoid multiple re-allocations
	dst.ReserveBoundaryAdditional(N_u * N_v * params.boundaryLayers);

	// build layers
	for (int layer = 0; layer < params.boundaryLayers; layer++)
	{
		const float outOffset = params.particleRadiusB + layer * spacing;

		for (int i = 0; i < N_u; i++)
		{
			// clamp to inner face
			const float su = std::min(i * spacing, face.uLen);
			const glm::vec3 bu = face.uHat * su;

			for (int j = 0; j < N_v; j++)
			{
				// clamp to inner face
				const float sv = std::min(j * spacing, face.vLen);
				const glm::vec3 bv = face.vHat * sv;

				// center position for this particle and layer
				const glm::vec3 pos = face.p000 + bu + bv + face.nHat * outOffset;

				// use pos for new particle
				dst.boundaryParticles.xPos.push_back(pos.x);
				dst.boundaryParticles.yPos.push_back(pos.y);
				dst.boundaryParticles.zPos.push_back(pos.z);

				// no velocity by default
				dst.boundaryParticles.xVel.push_back(0.0f);
				dst.boundaryParticles.yVel.push_back(0.0f);
				dst.boundaryParticles.zVel.push_back(0.0f);

				// default density, pressure, etc...
				dst.boundaryParticles.density.push_back(params.restDensityB);
				dst.boundaryParticles.pressure.push_back(0.0f);
				dst.boundaryParticles.mass.push_back(params.massB);
				dst.boundaryParticles.size.push_back(params.particleSizeB);

				// simple running ID
				dst.boundaryParticles.id.push_back(dst.boundaryParticles.id.size());
			}
		}
	}

	// update count
	dst.boundaryParticles.N_ = dst.boundaryParticles.id.size();
}

// ===== helpers =====

// determines a face given 3 inner-corner points (a base and its neighbors in u/v dirs)
OBBFace inline ScenarioMgr::MakeFace(const glm::vec3& ptBase,
									 const glm::vec3& ptU,
									 const glm::vec3& ptV,
									 float outwardSign)
{
	const glm::vec3 uVec = ptU - ptBase;
	const glm::vec3 vVec = ptV - ptBase;

	const float uLen = glm::length(uVec);
	const float vLen = glm::length(vVec);
	assert(uLen > 0 && vLen > 0);

	const glm::vec3 uHat = uVec / uLen;
	const glm::vec3 vHat = vVec / vLen;

	glm::vec3 nHat = glm::normalize(glm::cross(uVec, vVec));
	nHat *= outwardSign; // ensure facing outwards

	return OBBFace{ ptBase, uHat, vHat, nHat, uLen, vLen };
}

int inline ScenarioMgr::ArgMinSumXYZ(const ContainerCorners& corners)
{
	int bestIdx = 0;
	float bestVal = std::numeric_limits<float>::infinity();
	const glm::vec3 pts[8] =
	{
		corners.p000, corners.p100, corners.p010, corners.p110,
		corners.p001, corners.p101, corners.p011, corners.p111
	};

	for (int i = 0; i < 8; i++)
	{
		const float sum = pts[i].x + pts[i].y + pts[i].z;

		if (sum < bestVal)
		{
			bestVal = sum;
			bestIdx = i;
		}
	}

	return bestIdx;
}

int inline ScenarioMgr::TilesCount(float length, float spacing)
{
	if (spacing <= 0.0f) { return 0; }

	return std::max(1, static_cast<int>(std::floor(length / spacing)) + 1);
}
