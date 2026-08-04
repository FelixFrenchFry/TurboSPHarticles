// scenario_sim.h
//
// TODO: comment

#pragma once

// external
#include <memory>

// internal
#include "source/base/enum_hub.h"
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/sync/state_sim.h"
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

	// reduces the boundary boxes of a scenario to the axis-aligned domain the simulation is meant
	// to stay inside, padded by the boundary particle layers placed outside of the boxes
	// (reports false and leaves the bounds untouched when the scenario provides no boxes)
	bool ComputeSimDomainBounds(
		const std::vector<OBB>& boundaryBoxes,
		const Sim::SimParams& params,
		glm::vec3& min,
		glm::vec3& max);

} // namespace Scene
