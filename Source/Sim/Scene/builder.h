// builder.h
//
// TODO: comment

#pragma once

// external
#include <array>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"



using json = nlohmann::json;

namespace Scene
{
	struct OBB
	{
		glm::vec3 origin; // near-left-bottom (xMin, yMin, zMin)
		glm::vec3 uHat;   // unit vec along x
		glm::vec3 vHat;   // unit vec along y
		glm::vec3 wHat;   // unit vec along z
		float uLen;		  // width (along uHat)
		float vLen;		  // height (along vHat)
		float wLen;		  // depth (along wHat)
	};

	struct OBBFace
	{
		glm::vec3 origin; // near-left-bottom (xMin, yMin, zMin)
		glm::vec3 uHat;   // unit vec along x
		glm::vec3 vHat;   // unit vec along y
		glm::vec3 nHat;   // face normal (outward)
		float uLen;		  // width (along uHat)
		float vLen;		  // height (along vHat)
	};

	class ScenarioMgr
	{
	public: // -----------------------------------------------------------------s

		ScenarioMgr() = default;

		// builds a box-shaped boundary container around a given OBB
		// (boundary particle layers grow outwards by default)
		void BuildBoundaryContainer_OBB(const OBB& box,
										Sync::UniversalSimState& dst,
										const Sim::SimParams& params);

		// builds a block of fluid particles inside a given OBB
		void BuildFluidBlock_OBB(const OBB& box,
								 Sync::UniversalSimState& dst,
								 const Sim::SimParams& params);

		// TODO: builds a sphere of fluid particles inside a given sphere

		// builds a hollow block of boundary particles inside a given OBB
		// (thickness given boundary layers param)
		void BuildBoundaryBlockHollow_OBB(const OBB& box,
										  Sync::UniversalSimState& dst,
										  const Sim::SimParams& params);

		OBB MakeOBB_FromAxes(glm::vec3 origin,
							 glm::vec3 uDir, float uLen,
							 glm::vec3 vDir, float vLen,
							 glm::vec3 wDir, float wLen);

		std::array<OBBFace, 6> inline GetFacesFromOBB(const OBB& box);

		static OBB ParseOBBFromJson(const json& jfile, ScenarioMgr& scenarioMgr);

	private: // ----------------------------------------------------------------

		// builds a slab of boundary particles on a given OBB face
		void BuildBoundaryLayersOnFace(const OBBFace& face,
									   Sync::UniversalSimState& dst,
									   const Sim::SimParams& params);

		static inline float Jitter3D(uint64_t baseSeed, uint64_t key, float amp);

		static inline float StartOffset(float length, float spacing, int N, bool centerToFit = true);

		static inline uint32_t TilesCount(float length, float spacing);

		static inline uint32_t ParticleCountHollowBox(int N_u, int N_v, int N_w, int layers);
	};

} // namespace Scene
