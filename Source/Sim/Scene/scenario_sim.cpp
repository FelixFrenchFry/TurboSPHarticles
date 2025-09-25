// scenario_sim.cpp

// external
#include <stdexcept>

// internal
#include "builder.h"
#include "scenario_sim.h"
#include "breaking_dam.h"



namespace Scene
{
	std::unique_ptr<ISimScenario> MakeSimScenario(ScenarioMgr& scenarioMgr, 
												  Base::ScenarioType type,
												  const Sim::SimParams& params)
	{
		switch (type)
		{
		case Base::ScenarioType::BreakingDam:
		{
			return std::make_unique<BreakingDamScene>
				(scenarioMgr, MakeBreakingDamConfig_Default(scenarioMgr, params));
		}
		default:
			throw std::runtime_error("Unsupported sim scenario type request");
		}
	}

} // namespace Scene
