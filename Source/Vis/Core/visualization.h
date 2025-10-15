// visualization.h
//
// TODO: comment

#pragma once

// external
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Core/simulation.h"
#include "Source/Sim/Sync/mailboxes.h"
#include "Source/Vis/Input/user_input.h"
#include "Source/Vis/Inter/user_interface.h"
#include "Source/Vis/Render/rendering.h"
#include "parameters_vis.h"
#include "structs_vis.h"



namespace Vis
{
	class VisualizationMgr
	{
	public: // ==============================

		VisualizationMgr(Sim::SimulationMgr& simulationMgr,
						 Bench::BenchmarksMgr& benchmarksMgr,
						 Sim::SimParamsMgr& simParamsMgr,
						 Vis::VisParamsMgr& visParamsMgr);

		// create window & GL context, and initialize sub-systems
		bool Init(const WindowConfig& windowConfig);

		// destroy contexts, sub-systems and free up resources
		void Shutdown();

		// main visualization loop (handle input, interface, render)
		void RunVis();

		void SetVisualMailbox(Sync::VisualSnapMailbox* visualSnapMailbox) { visualSnapMailbox_ = visualSnapMailbox; }
		void SetBenchMailbox(Sync::BenchSnapMailbox* benchSnapMailbox) { benchSnapMailbox_ = benchSnapMailbox; }

	private: // ==============================

		Sim::SimulationMgr& simulationMgr_;
		Bench::BenchmarksMgr& benchmarksMgr_;

		// parameter managers and snapshots to be managed and passed to sub-systems
		Sim::SimParamsMgr& simParamsMgr_;
		Vis::VisParamsMgr& visParamsMgr_;
		Sim::SimParams simParams_;
		Vis::VisParams visParams_;
		uint64_t simParamsVersion_ = 0;
		uint64_t visParamsVersion_ = 0;

		// controlled sub-system managers
		Input::UserInputMgr userInputMgr_;
		Inter::UserInterfaceMgr userInterfaceMgr_;
		Render::RenderingMgr renderingMgr_;

		// double-buffered vis-specific state
		std::vector<glm::vec3> posF_[2];	// positions of fluid particles
		std::vector<float> scalarDataF_[2]; // data according to color coding mode
		std::vector<Scene::OBB> boxesB_[2]; // OBBs of boundary structures
		int front_ = 0; int back_ = 1; // current / next buffer index

		// current view and scalar (coloring) mode for visualization
		Base::ColorCodingMode scalarMode_{ Base::None };
		VisualSnapView view_{};

		// mailbox pointers
		Sync::VisualSnapMailbox* visualSnapMailbox_{ nullptr };
		Sync::BenchSnapMailbox* benchSnapMailbox_{ nullptr };
		std::shared_ptr<Sync::BenchSnapshot> localBenchSnap_{ nullptr };

		// window and GL context
		GLFWwindow* window_ = nullptr;

		// static callbacks for GLFW
		static void GLFW_FramebufferSizeCB(GLFWwindow* window, int width, int height);
		static void GLFW_WindowSizeCB(GLFWwindow* window, int width, int height);
		static void GLFW_WindowContentScaleCB(GLFWwindow* window, float xScale, float yScale);
		static void GLFW_KeyCB(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void GLFW_MouseButtonCB(GLFWwindow* window, int button, int action, int mods);
		static void GLFW_CursorPosCB(GLFWwindow* window, double xPos, double yPos);
		static void GLFW_ScrollCB(GLFWwindow* window, double xOff, double yOff);
		// TODO: char callback?

		// instance methods called by static callbacks
		void OnFramebufferSize(int width, int height);
		void OnWindowSize(int width, int height);
		void OnWindowContentScale(float xScale, float yScale);
		void OnKey(int key, int scancode, int action, int mods);
		void OnMouseButton(int button, int action, int mods);
		void OnCursorPos(double xPos, double yPos);
		void OnScroll(double xOff, double yOff);

		// pulls latest parameter snapshots and versions if they have changed
		void SyncParameters(bool force = false);

		// pulls latest simulation state and packs it into vis-specific data structures
		// (positions + scalar property depending on color coding mode)
		void SyncVisSimState();

		void SyncBenchSnap();

		// utils
		void SwapVisBuffers();
	};

} // namespace Vis
