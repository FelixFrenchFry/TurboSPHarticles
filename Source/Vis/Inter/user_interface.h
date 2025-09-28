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
	// own and manage the ImGui and ImPlot lifecycle
	class UserInterfaceMgr
	{
	public: // ==============================

		UserInterfaceMgr(Sim::SimParamsMgr& simParamsMgr,
						 Vis::VisParamsMgr& visParamsMgr);

		void Init(GLFWwindow* window,
				  const char* glslVersion,
				  const char* iniFilename = "TSP_visual_gui.ini",
				  float dpiScale = 1.0f);

		// destroy ImGui and ImPlot contexts and free up resources
		void Shutdown();

		// per-frame user interface lifecycle
		void BeginFrame();
		void BuildUserInterfaceSections(const Sim::SimulationMgr& simulationMgr,
										const Sim::SimParams& simParams, 
										const Vis::VisParams& visParams);
		void BuildSection_SimulationControl(const Sim::SimulationMgr& simulationMgr,
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
		bool initialized_ = false;
		bool dockingEnabled_ = true;
		std::string iniFilename_;

		// interface element colors
		ImVec4 color_buttonPositive_ =	ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
		ImVec4 color_buttonNeutral_ =	ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
		ImVec4 color_buttonNegative_ =	ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
		ImVec4 color_tableHighlight_ =	ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
		ImVec4 color_tooltip_ =			ImVec4(0.6f, 1.0f, 0.6f, 1.0f);
		ImVec4 color_unsavedChanges_ =	ImVec4(0.4f, 0.9f, 0.2f, 1.0f);

		// temp
		char stringBuffer[256];

		// helpers
		void CreateContexts(const char* iniFilename);
		void DestroyContexts();
		void SetupStyle();
		void RebuildFonts(float dpiScale);

		// dockspace
		void BeginDockspace();
	};

} // namespace Inter
