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
	public: // ==============================

		BreakingDamScene(ScenarioMgr& scenarioMgr, BreakingDamConfig config)
			: scenarioMgr_(scenarioMgr), config_(std::move(config)) {}

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

	private: // ==============================

		ScenarioMgr& scenarioMgr_;
		BreakingDamConfig config_;
	};

	// ============
	// helpers
	// ============

	BreakingDamConfig inline MakeBreakingDamConfig_Default(ScenarioMgr& scenarioMgr,
													       const Sim::SimParams& params)
	{
		const glm::vec3 U{ 1, 0, 0 }, V{ 0, 1, 0 }, W{ 0, 0, 1 };

		OBB container = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0,0,0},
			/* uDir, uLen */	U, 0.6f,
			/* vDir, vLen */	V, 0.4f,
			/* wDir, wLen */	W, 0.2f
		);

		OBB fluid = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0.0,0.0,0.0},
			/* uDir, uLen */	U, 0.6f,
			/* vDir, vLen */	V, 0.4f,
			/* wDir, wLen */	W, 0.2f
		);

		return BreakingDamConfig{ container, fluid };
	}

} // namespace Scene
