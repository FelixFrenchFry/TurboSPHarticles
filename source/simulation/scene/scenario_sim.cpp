// scenario_sim.cpp

// external
#include <algorithm>
#include <array>
#include <stdexcept>

// internal
#include "builder.h"
#include "scenario_sim.h"
#include "scenarios/breaking_dam.h"
#include "scenarios/converging_dam.h"
#include "scenarios/big_droplet.h"
#include "scenarios/rigid_cube.h"
#include "scenarios/boundary_meshes.h"
#include "scenarios/rigid_meshes.h"
#include "scenarios/boundary_rigid_meshes.h"



namespace
{
	void IncludePoint(glm::vec3& min, glm::vec3& max, const glm::vec3& point)
	{
		min.x = std::min(min.x, point.x);
		min.y = std::min(min.y, point.y);
		min.z = std::min(min.z, point.z);

		max.x = std::max(max.x, point.x);
		max.y = std::max(max.y, point.y);
		max.z = std::max(max.z, point.z);
	}

	// scenarios use OBBs, while the domain bounds are one axis-aligned box
	void IncludeOBB(glm::vec3& min, glm::vec3& max, const Scene::OBB& box)
	{
		const glm::vec3 u = box.uHat * box.uLen;
		const glm::vec3 v = box.vHat * box.vLen;
		const glm::vec3 w = box.wHat * box.wLen;

		const std::array<glm::vec3, 8> corners =
		{
			box.origin,
			box.origin + u,
			box.origin + v,
			box.origin + w,
			box.origin + u + v,
			box.origin + u + w,
			box.origin + v + w,
			box.origin + u + v + w
		};

		for (const glm::vec3& corner : corners)
		{
			IncludePoint(min, max, corner);
		}
	}

} // namespace anonymous

namespace Scene
{
	std::unique_ptr<ISimScenario> MakeSimScenario(ScenarioMgr& scenarioMgr, 
												  Base::ScenarioType type,
												  const std::string& profile)
	{
		switch (type)
		{
		case Base::ScenarioType::BreakingDam:
		{
			BreakingDamConfig cfg;
			cfg = LoadBreakingDamConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<BreakingDamScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::ConvergingDam:
		{
			ConvergingDamConfig cfg;
			cfg = LoadConvergingDamConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<ConvergingDamScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::BigDroplet:
		{
			BigDropletConfig cfg;
			cfg = LoadBigDropletConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<BigDropletScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::RigidCube:
		{
			RigidCubeConfig cfg;
			cfg = LoadRigidCubeConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<RigidCubeScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::BoundaryMeshes:
		{
			BoundaryMeshesConfig cfg;
			cfg = LoadBoundaryMeshesConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<BoundaryMeshesScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::RigidMeshes:
		{
			RigidMeshesConfig cfg;
			cfg = LoadRigidMeshesConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<RigidMeshesScene>(scenarioMgr, cfg);
		}
		case Base::ScenarioType::BoundaryRigidMeshes:
		{
			BoundaryRigidMeshesConfig cfg;
			cfg = LoadBoundaryRigidMeshesConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<BoundaryRigidMeshesScene>(scenarioMgr, cfg);
		}
		default:
			throw std::runtime_error("Unsupported simulation scenario type requested.");
		}
	}

	bool ComputeSimDomainBounds(
		const std::vector<OBB>& boundaryBoxes,
		const Sim::SimParams& params,
		glm::vec3& min,
		glm::vec3& max)
	{
		if (boundaryBoxes.empty()) { return false; }

		min = boundaryBoxes.front().origin;
		max = boundaryBoxes.front().origin;

		for (const OBB& box : boundaryBoxes)
		{
			IncludeOBB(min, max, box);
		}

		// the boundary particle layers sit outside of the boxes, so the domain has to cover them
		const float boundaryPad = params.boundaryLayers > 0
			? (static_cast<float>(params.boundaryLayers) - 0.5f) * params.particleSizeB
			: 0.0f;

		min -= glm::vec3(boundaryPad);
		max += glm::vec3(boundaryPad);

		return true;
	}

} // namespace Scene
