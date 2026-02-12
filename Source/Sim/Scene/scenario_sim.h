// scenario_sim.h
//
// TODO: comment

#pragma once

// external
#include <memory>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"
#include "builder.h"



namespace Scene
{
	class ISimScenario
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~ISimScenario() = default;

		virtual const char* Name() const = 0;

		// builds a universal sim state according to the scenario and options
		virtual void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const = 0;

		virtual std::vector<OBB> GetBoundaryBoxes() const = 0;
	};

	// scenario factory
	std::unique_ptr<ISimScenario> MakeSimScenario(ScenarioMgr& scenarioMgr,
												  Base::ScenarioType type,
												  const std::string& profile);

} // namespace Scene
