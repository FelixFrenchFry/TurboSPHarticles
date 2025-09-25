// scenebuilder.h
//
// TODO: comment

#pragma once

// external
#include <array>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"



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

	struct OBBCorners
	{
		glm::vec3 p000; // near-left-bottom  (xMin, yMin, zMin)
		glm::vec3 p100; // near-right-bottom (xMax, yMin, zMin)
		glm::vec3 p010; // near-left-top     (xMin, yMax, zMin)
		glm::vec3 p110; // near-right-top    (xMax, yMax, zMin)
		glm::vec3 p001; // far-left-bottom   (xMin, yMin, zMax)
		glm::vec3 p101; // far-right-bottom  (xMax, yMin, zMax)
		glm::vec3 p011; // far-left-top      (xMin, yMax, zMax)
		glm::vec3 p111; // far-right-top     (xMax, yMax, zMax)
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

	class IScenario
	{
	public: // ==============================

		virtual ~IScenario() = default;

		virtual const char* Name() const = 0;

		// builds a universal sim state according to the scenario and options
		virtual void Build(Sync::UniversalSimState& dst, Sim::SimParams params) const = 0;
	};

	class ScenarioMgr
	{
	public: // ==============================

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

		OBB inline MakeOBB_FromAxes(glm::vec3 origin,
									glm::vec3 uDir, float uLen,
									glm::vec3 vDir, float vLen,
									glm::vec3 wDir, float wLen);

		std::array<OBBFace, 6> inline GetFacesFromOBB(const OBB& box);

	private: // ==============================

		// builds a slab of boundary particles on a given OBB face
		void BuildBoundaryLayersOnFace(const OBBFace& face,
									   Sync::UniversalSimState& dst,
									   const Sim::SimParams& params);

		float inline StartOffset(float length, float spacing, int N, bool centerToFit = true);

		int inline TilesCount(float length, float spacing);

		int64_t inline ParticleCountHollowBox(int N_u, int N_v, int N_w, int layers);
	};

} // namespace Scene
