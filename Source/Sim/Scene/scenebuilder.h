// scenebuilder.h
//
// TODO: comment

#pragma once

// external
#include <array>
#include <glm/vec3.hpp>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"



namespace Scene
{
	struct ContainerCorners
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
		glm::vec3 p000; // near-left-bottom  (xMin, yMin, zMin)
		glm::vec3 uHat; // unit vector along width (x direction)
		glm::vec3 vHat; // unit vector along height (y direction)
		glm::vec3 nHat; // unit vector along depth (z direction)
		float uLen;		// width (along uHat)
		float vLen;		// height (along vHat)
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

		// places boundary particles around a rectangular container
		// defining the domain of a simulated scenario
		void BuildSceneContainer_Rect(const ContainerCorners& corners,
									  Sync::UniversalSimState& dst,
									  const Sim::SimParams& params);

	private: // ==============================

		// derives the 6 faces of an OBB given its 8 corners (orthonormal frames + sizes)
		std::array<OBBFace, 6> GetPlaneFacesFromCorners(const ContainerCorners& corners);

		void BuildBoundaryLayersOnFace(const OBBFace& face,
									   Sync::UniversalSimState& dst,
									   const Sim::SimParams& params);

		// helpers

		OBBFace inline MakeFace(const glm::vec3& ptBase,
								const glm::vec3& ptU,
								const glm::vec3& ptV,
								float outwardSign);

		int inline ArgMinSumXYZ(const ContainerCorners& corners);

		int inline TilesCount(float length, float spacing);
	};

} // namespace Scene
