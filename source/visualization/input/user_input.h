// user_input.h
//
// TODO: comment

#pragma once

// external
#include <bitset>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/visualization/core/parameters_vis.h"



namespace Input
{
	class UserInputMgr
	{
	public: // ---------------------------------------------------------------------------------------

		UserInputMgr(Sim::SimParamsMgr& simParamsMgr,
					 Vis::VisParamsMgr& visParamsMgr);

		void Init(const Vis::VisParams& visParams);

		void Shutdown();

		// per-frame update of camera state
		void UpdateCamera(const Vis::CameraState& camera, 
						  const Vis::FrameBufferConfig& frameBuffer, float dt);

		// input capture flags return true if input event is consumed
		bool OnKey(const Sim::SimParams& simParams,
				   const Vis::CameraState& camera,
				   const Vis::RenderConfig& render,
				   int key, int scancode, int action, int mods);
		bool OnMouseButton(int button, int action, int mods);
		bool OnCursorPos(double xPos, double yPos);
		bool OnScroll(double xOff, double yOff);

	private: // --------------------------------------------------------------------------------------

		// parameter managers for making changes to parameters
		Sim::SimParamsMgr& simParamsMgr_;
		Vis::VisParamsMgr& visParamsMgr_;

		// default state
		Vis::CameraState defaultCamera_;

		// input state
		std::bitset<512> keys_{}; // key pressed state
	};

} // namespace Input
