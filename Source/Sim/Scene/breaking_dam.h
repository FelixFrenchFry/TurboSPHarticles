// breaking_dam.h
//
// TODO: comment

#pragma once

// external
#include <fstream>
#include <glm/vec3.hpp>
#include <spdlog/spdlog.h>

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
			/* uDir, uLen */	U, 1.0f,
			/* vDir, vLen */	V, 1.0f,
			/* wDir, wLen */	W, 1.0f
		);

		OBB fluid = scenarioMgr.MakeOBB_FromAxes
		(
			/* origin */		{0.0,0.0,0.0},
			/* uDir, uLen */	U, 0.5f,
			/* vDir, vLen */	V, 0.5f,
			/* wDir, wLen */	W, 0.5f
		);

		return BreakingDamConfig{ container, fluid };
	}

	BreakingDamConfig LoadBreakingDamConfigFromJsonFile(ScenarioMgr& scenarioMgr)
	{
		namespace fs = std::filesystem;

		// use TSP_CONFIG_DIR from the build system as base path
		fs::path base(TSP_CONFIG_DIR);
		fs::path full = base / "BreakingDam.json";
		std::string path = full.string();

		std::ifstream in(path);

		if (!in)
		{
			SPDLOG_ERROR("Failed to open BreakingDam config: {}", path);
			// fall back to built-in defaults
			return MakeBreakingDamConfig_Default(scenarioMgr);
		}

		json jfile;
		try
		{
			in >> jfile;
		}
		catch (const std::exception& e)
		{
			SPDLOG_ERROR("Failed to parse BreakingDam config {}: {}",
				path, e.what());
			return MakeBreakingDamConfig_Default(scenarioMgr);
		}

		BreakingDamConfig cfg{};
		if (auto it = jfile.find("container"); it != jfile.end())
		{
			cfg.container = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}
		else
		{
			cfg.container = MakeBreakingDamConfig_Default(scenarioMgr).container;
		}

		if (auto it = jfile.find("fluid"); it != jfile.end())
		{
			cfg.fluid = ScenarioMgr::ParseOBBFromJson(*it, scenarioMgr);
		}
		else
		{
			cfg.fluid = MakeBreakingDamConfig_Default(scenarioMgr).fluid;
		}

		return cfg;
	}

} // namespace Scene
