// user_interface.h
//
// TODO: comment

#pragma once

// external
#include <imgui.h>
#include <string>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"
#include "../Core/parameters_vis.h"

// forward declarations
struct GLFWwindow;



namespace Inter
{
	enum class TimeUnit : uint8_t
	{
		ms, us, ns
	};

	enum class BenchTableView : uint8_t
	{
		Latest, Avg_All, Avg_Used, Min, Max
	};

	enum class StatTableView : uint8_t
	{
		Latest, Avg, Min, Max
	};

	// flags for parameter changes
	struct ParamChanged
	{
		bool gammaPress;
		bool jacobiOmega;
		bool viscosityF;
		bool viscosityB;
		bool gravity;
	};

	// magnitudes of parameter changes
	struct ParamMags
	{
		std::array<float, 2> gammaPress;
		std::array<float, 2> jacobiOmega;
		std::array<float, 2> viscosityF;
		std::array<float, 2> viscosityB;
		std::array<float, 2> gravity;
	};

	// temporary copies for parameter changes
	struct ParamTemp
	{
		float gammaPress;
		float jacobiOmega;
		float viscosityF;
		float viscosityB;
		float gravity;
	};

	// own and manage the ImGui and ImPlot lifecycle
	class UserInterfaceMgr
	{
	public: // ==============================

		UserInterfaceMgr(Sim::SimParamsMgr& simParamsMgr,
						 Vis::VisParamsMgr& visParamsMgr);

		void Init(GLFWwindow* window,
				  const Sim::SimParams& simParams,
				  const char* glslVersion,
				  const char* iniFilename = "TSP_visual_gui.ini",
				  float dpiScale = 1.0f);

		// destroy ImGui and ImPlot contexts and free up resources
		void Shutdown();

		// per-frame user interface lifecycle
		void BeginFrame();

		void BuildUserInterfaceSections(Sim::SimulationMgr& simulationMgr,
										const Sync::BenchSnapshot* benchSnap,
										const Sync::StatSnapshot* statSnap,
										const Sim::SimParams& simParams, 
										const Vis::VisParams& visParams);
		void EndFrame();

		void RenderDrawData();

		// input capture flags whether event is processed by userInterfaceMgr
		bool OnKey(int key, int scancode, int action, int mods);
		bool OnMouseButton(int button, int action, int mods);
		bool OnCursorPos(double xPos, double yPos);
		bool OnScroll(double xOff, double yOff);
		bool OnChar(unsigned int codepoint);

		void OnDPIScaleChanged(float dpiScale);

		// helpers
		void SetDockingEnabled(bool enabled);

	private: // ==============================

		// parameter managers for making changes to parameters
		Sim::SimParamsMgr& simParamsMgr_;
		Vis::VisParamsMgr& visParamsMgr_;

		// context and state
		GLFWwindow* window_ = nullptr;
		std::string iniFilename_;
		bool initialized_ = false;
		bool dockingEnabled_ = true;

		// interface element colors
		ImVec4 RGBA_positive_	= ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
		ImVec4 RGBA_neutral_	= ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
		ImVec4 RGBA_negative_	= ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
		ImVec4 RGBA_highlight_	= ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
		ImVec4 RGBA_tooltip_	= ImVec4(0.6f, 1.0f, 0.6f, 1.0f);
		ImVec4 RGBA_unsaved_	= ImVec4(0.4f, 0.9f, 0.2f, 1.0f);

		// flags, temporary copies, and magnitudes for parameter changes
		// TODO: initialize their values
		ParamChanged changed_
		{
			.gammaPress		= false,
			.jacobiOmega	= false,
			.viscosityF		= false,
			.viscosityB		= false,
			.gravity		= false
		};
		ParamMags mags_
		{
			.gammaPress		= { 0.1f,			0.01f },
			.jacobiOmega	= { 0.1f,			0.01f },
			.viscosityF		= { 0.00001f,		0.000001f },
			.viscosityB		= { 0.00001f,		0.000001f },
			.gravity		= { 1.0f,			0.1f }
		};
		ParamTemp temp_{};

		// parameter section flags
		bool liveParamEditing_ = false;
		bool unsavedParamChanges_ = false;

		// benchmark table state
		BenchTableView benchTableView_ = BenchTableView::Latest;
		TimeUnit benchTimeUnit_ = TimeUnit::us;
		int sparklineContextLength_ = 1024;

		// statistics table state
		StatTableView statTableView_ = StatTableView::Latest;

		// temp string buffer
		char stringBuffer[256];

		// sections
		void BuildSection_SceneManagement(
			Sim::SimulationMgr& simulationMgr,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);

		void BuildSection_SimulationControl(
			const Sim::SimulationMgr& simulationMgr,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);

		void BuildSection_SimulationParameters(
			const Sim::SimulationMgr& simulationMgr,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);

		void BuildSection_SimulationStatistics(
			const Sim::SimulationMgr& simulationMgr,
			const Sync::StatSnapshot* statSnap,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);

		void BuildSection_PerformanceBenchmarks(
			const Sim::SimulationMgr& simulationMgr,
			const Sync::BenchSnapshot* benchSnap,
			const std::vector<Bench::BenchOpGroupDef>& opGroups,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);

		// parameter section elements for parameter changes
		void BuildParamElement_GammaPress(const Sim::SimParams& simParams);
		void BuildParamElement_JacobiOmega(const Sim::SimParams& simParams);
		void BuildParamElement_ViscosityF(const Sim::SimParams& simParams);
		void BuildParamElement_ViscosityB(const Sim::SimParams& simParams);
		void BuildParamElement_Gravity(const Sim::SimParams& simParams);

		// helpers
		void CreateContexts(const char* iniFilename);
		void DestroyContexts();
		void SetupStyle();
		void RebuildFonts(float dpiScale);
		void ApplyParamChanges();
		void ResetParamChanges(const Sim::SimParams& simParams);
		inline double ConvertFromNs(TimeUnit unit, uint64_t ns);
		inline const char* UnitLabel(TimeUnit unit);
		inline float ReadStatData(const Sync::StatSnapshot* snap, Stats::StatMetric metric);
		inline uint64_t ReadOpTiming(const Sync::BenchSnapshot* snap, Bench::BenchOp op);
		inline uint64_t ReadOpGroup_Min(const Sync::BenchSnapshot* snap,
										const Bench::BenchOpGroupDef& opGroup);
		inline std::string MakeGroupAlias(const Bench::BenchOpGroupDef& opGroup);
		void TableTextRightAligned(const char* fmt, ...);
		inline void FormatStatValue(const Stats::StatMetricDef& def, char* out, float x, uint64_t cap);

		// dockspace
		void BeginDockspace();
	};

} // namespace Inter
