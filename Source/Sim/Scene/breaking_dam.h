// breaking_dam.h
//
// TODO: comment

#pragma once

// external
#include <algorithm>
#include <glm/vec3.hpp>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/state_sim.h"
#include "builder.h"
#include "scenario_sim.h"



namespace Scene
{
	struct BreakingDamConfig
	{
		// outer boundary container
		OBB container;

		// fluid block placed inside
		OBB fluid;
	};

	class BreakingDamScene final : public ISimScenario
	{
	public: // -----------------------------------------------------------------

		BreakingDamScene(ScenarioMgr& scenarioMgr, BreakingDamConfig config)
			: scenarioMgr_(scenarioMgr), config_(config) {}

		const char* Name() const override { return "Breaking Dam"; }

		void Build(Sync::UniversalSimState& dst, const Sim::SimParams& params) const override
		{
			scenarioMgr_.BuildBoundaryContainer_OBB(config_.container, dst, params);

			scenarioMgr_.BuildFluidBlock_OBB(config_.fluid, dst, params);
		}

		std::vector<OBB> GetBoundaryBoxes() const override
		{
			std::vector<OBB> res{};

			res.push_back(config_.container);

			return res;
		}

	private: // ----------------------------------------------------------------

		ScenarioMgr& scenarioMgr_;
		BreakingDamConfig config_;
	};

	BreakingDamConfig inline MakeBreakingDamConfig_Default(ScenarioMgr& scenarioMgr)
	{
		const glm::vec3 U{ 1, 0, 0 }; // forward
		const glm::vec3 V{ 0, 1, 0 }; // up
		const glm::vec3 W{ 0, 0, 1 }; // right

		OBB container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0,0,0},
			/* uDir, uLen */	U, 5.5f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 5.0f
		);

		OBB fluid = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0.0,0.0,0.0},
			/* uDir, uLen */	U, 5.0f,
			/* vDir, vLen */	V, 0.5f,
			/* wDir, wLen */	W, 5.0f
		);

		return BreakingDamConfig{ container, fluid };
	}

} // namespace Scene
