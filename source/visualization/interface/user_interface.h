// user_interface.h
#pragma once

// external
#include <imgui.h>
#include <string>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/core/simulation.h"
#include "source/visualization/core/parameters_vis.h"
#include "source/visualization/render/color_maps.h"

// forward declarations
struct GLFWwindow;



namespace Inter
{
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
		bool timeStep;
		bool viscosityF;
		bool viscosityB;
		bool pressGamma;
		bool jacobiOmega;
		bool densityDevTarget;
		bool minSolverIters;
		bool maxSolverIters;
		bool gravity;
	};

	// magnitudes of parameter changes
	struct ParamMags
	{
		std::array<float, 2> timeStep;
		std::array<float, 2> viscosityF;
		std::array<float, 2> viscosityB;
		std::array<float, 2> pressGamma;
		std::array<float, 2> jacobiOmega;
		std::array<float, 2> densityDevTarget;
		std::array<int, 2> minSolverIters;
		std::array<int, 2> maxSolverIters;
		std::array<float, 2> gravity;
	};

	// temporary copies for parameter changes
	struct ParamTemp
	{
		float timeStep;
		float viscosityF;
		float viscosityB;
		float pressGamma;
		float jacobiOmega;
		float densityDevTarget;
		int minSolverIters;
		int maxSolverIters;
		float gravity;
	};

	// own and manage the ImGui lifecycle
	class UserInterfaceMgr
	{
	public:

		UserInterfaceMgr(Sim::SimParamsMgr& simParamsMgr,
						 Vis::VisParamsMgr& visParamsMgr);

		void Init(GLFWwindow* window,
				  const Sim::SimParams& simParams,
				  const char* glslVersion,
				  const char* iniFilename = "TSP_visual_gui.ini",
				  float dpiScale = 1.0f);

		// destroy ImGui context and free up resources
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

	private:

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
			.timeStep = false,
			.viscosityF = false,
			.viscosityB = false,
			.pressGamma = false,
			.jacobiOmega = false,
			.densityDevTarget = false,
			.minSolverIters = false,
			.maxSolverIters = false,
			.gravity = false
		};
		ParamMags mags_
		{
			.timeStep			= { 0.0001f,		0.00001f },
			.viscosityF			= { 0.00001f,		0.000001f },
			.viscosityB			= { 0.00001f,		0.000001f },
			.pressGamma			= { 0.1f,			0.01f },
			.jacobiOmega		= { 0.1f,			0.01f },
			.densityDevTarget	= { 0.01f,		0.001f },
			.minSolverIters		= { 1,			1 },
			.maxSolverIters		= { 1,			1 },
			.gravity			= { 0.1f,			0.01f }
		};
		ParamTemp temp_{};

		// parameter section flags
		bool liveParamEditing_ = false;
		bool unsavedParamChanges_ = false;

		// benchmark table state
		BenchTableView benchTableView_ = BenchTableView::Latest;
		Bench::TimeUnit benchTimeUnit_ = Bench::TimeUnit::ms;

		// statistics table state
		StatTableView statTableView_ = StatTableView::Latest;

		// temp string buffer
		char stringBuffer[256];

		// sections
		void BuildSection_SceneManagement(
			Sim::SimulationMgr& simulationMgr,
			const Vis::VisParams& visParams);

		void BuildSection_VisualizationSettings(
			const Vis::VisParams& visParams);

		void BuildSection_SimulationControl(
			const Sim::SimulationMgr& simulationMgr,
			const Sim::SimParams& simParams);

		void BuildSection_SimulationParameters(
			const Sim::SimParams& simParams);

		void BuildSection_SimulationStatistics(
			const Sync::StatSnapshot* statSnap);

		void BuildSection_PerformanceBenchmarks(
			const Sync::BenchSnapshot* benchSnap,
			const Bench::BenchOpList& benchOps);

		// visualization setting elements
		void BuildColorMapElement(
			const char* label,
			Base::ColorMap colorMap,
			const Render::ColorMapData& colorMapValues,
			const Vis::VisParams& visParams);

		// parameter section elements for parameter changes
		void BuildParamElement_TimeStep(const Sim::SimParams& simParams);
		void BuildParamElement_ViscosityF(const Sim::SimParams& simParams);
		void BuildParamElement_ViscosityB(const Sim::SimParams& simParams);
		void BuildParamElement_PressGamma(const Sim::SimParams& simParams);
		void BuildParamElement_JacobiOmega(const Sim::SimParams& simParams);
		void BuildParamElement_DensityDevTarget(const Sim::SimParams& simParams);
		void BuildParamElement_MinSolverIters(const Sim::SimParams& simParams);
		void BuildParamElement_MaxSolverIters(const Sim::SimParams& simParams);
		void BuildParamElement_Gravity(const Sim::SimParams& simParams);

		// helpers
		void CreateContexts(const char* iniFilename);
		void DestroyContexts();
		void SetupStyle();
		void RebuildFonts(float dpiScale);
		void ApplyParamChanges();
		void ResetParamChanges(const Sim::SimParams& simParams);
		inline double ReadStatData(const Sync::StatSnapshot* snap, Stats::StatMetric metric) const;
		inline uint64_t ReadOpTiming(const Sync::BenchSnapshot* snap, Bench::BenchOp op) const;
		static void TableTextRightAligned(const char* fmt, ...);
		static inline void FormatStatValue(const Stats::StatMetricDef& def, char* out, double x, uint64_t cap);

		// dockspace
		static void BeginDockspace();
	};

} // namespace Inter
