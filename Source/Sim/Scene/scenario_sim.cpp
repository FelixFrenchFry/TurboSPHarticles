// scenario_sim.cpp

// external
#include <stdexcept>

// internal
#include "builder.h"
#include "scenario_sim.h"
#include "breaking_dam.h"
#include "converging_dam.h"
#include "big_droplet.h"



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
		default:
			throw std::runtime_error("Unsupported sim scenario type requested.");
		}
	}

} // namespace Scene
