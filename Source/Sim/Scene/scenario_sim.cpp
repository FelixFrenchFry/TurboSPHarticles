// scenario_sim.cpp

// external
#include <stdexcept>

// internal
#include "builder.h"
#include "scenario_sim.h"
#include "breaking_dam.h"
#include "converging_dam.h"
#include "big_droplet.h"
#include "rigid_cube.h"
#include "sampled_mesh.h"



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
		case Base::ScenarioType::SampledMesh:
		{
			SampledMeshConfig cfg;
			cfg = LoadSampledMeshConfigFromProfile(profile, scenarioMgr);

			return std::make_unique<SampledMeshScene>(scenarioMgr, cfg);
		}
		default:
			throw std::runtime_error("Unsupported simulation scenario type requested.");
		}
	}

} // namespace Scene
