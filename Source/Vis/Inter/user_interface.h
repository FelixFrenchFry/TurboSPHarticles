// user_interface.h
//
// TODO: comment

#pragma once

// external
#include <string>

// internal
#include "../../Sim/Core/parameters_sim.h"
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
		void DrawPanels(const Vis::VisParams& visParams);
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

		// helpers
		void CreateContexts(const char* iniFilename);
		void DestroyContexts();
		void SetupStyle();
		void RebuildFonts(float dpiScale);

		// dockspace
		void BeginDockspace();
	};

} // namespace Inter
