// parameters_vis.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>

// internal
#include "structs_vis.h"


namespace Vis
{
	// static non-mutable constants
	struct VisConstants
	{
		static constexpr GLVersion glVersion
		{
			.major	= 4,
			.minor	= 5,
			.name	= "#version 450"
		};
	};

	// snapshot of visualization parameters
	struct VisParams
	{
		WindowConfig window
		{
			.size			= {1920, 1080},
			.title			= "TSP Visual",
			.resizable		= true,
			.vsync			= false
		};

		FrameBufferConfig frameBuffer
		{
			.size			= {1920, 1080},
			.dpiScale		= 1.0f
		};

		CameraState camera
		{
			.viewMat		= glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), // eye
										  glm::vec3(0.0f, 0.0f, 0.0f), // center
										  glm::vec3(0.0f, 1.0f, 0.0f)  // up
										  ),
			// TODO: adjust aspect ratio
			.projMat		= glm::perspective(glm::radians(45.0f), 
											   16.0f / 9.0f, 0.01f, 100.0f),
			.camPos			= {0.0f, 0.0f, 0.0f},
			.yawDeg			= 0.0f,
			.pitchDeg		= 0.0f,
			.moveSpeed		= 2.0f,
			.turnSpeedDeg	= 90.0f,
			.speedMult		= 4.0f,
			.yFovDeg		= 60.0f,
			.zNear			= 0.01f,
			.zFar			= 500.0f
		};

		RenderConfig render
		{
			.lightDirVS		= {0.3f, 0.6f, 0.6f},
			.clearColor		= {0.15f, 0.15f, 0.15f, 1.0f},
			.baseColor		= {1.0f, 0.8f, 0.0f},
			.alpha			= 1.0f,
			.radius			= 0.05f,
			.scalarMin		= 0.0f,
			.scalarMax		= 1.0f,
			.useScalar		= false
		};
	};

	// patches for setters
	struct WindowPatch
	{
		std::optional<glm::ivec2> size;
		std::optional<const char*> title;
		std::optional<bool> resizable;
		std::optional<bool>	vsync;
	};

	struct FrameBufferPatch
	{
		std::optional<glm::ivec2> size;
		std::optional<float> dpiScale;
	};

	struct CameraPatch
	{
		std::optional<glm::mat4> viewMat;
		std::optional<glm::mat4> projMat;
		std::optional<glm::vec3> camPos;
		std::optional<float> yawDeg;
		std::optional<float> pitchDeg;
		std::optional<float> moveSpeed;
		std::optional<float> turnSpeedDeg;
		std::optional<float> yFovDeg;
		std::optional<float> zNear;
		std::optional<float> zFar;
	};

	struct RenderPatch
	{
		std::optional<glm::vec3> lightDirVS;
		std::optional<glm::vec3> baseColor;
		std::optional<float> alpha;
		std::optional<float> radius;
		std::optional<float> scalarMin;
		std::optional<float> scalarMax;
		std::optional<bool> useScalar;
	};

	class VisParamsMgr
	{
	public: // ==============================

		VisParamsMgr() = default;

		// returns snapshot and version of current visualization parameters
		std::pair<VisParams, int64_t> SnapshotParams() const;

		// returns current stable version number
		int64_t StableVersion() const;

		// patch-style setters for individual parameters and their dependent values
		void SetParam(const WindowPatch& patch);
		void SetParam(const FrameBufferPatch& patch);
		void SetParam(const CameraPatch& patch);
		void SetParam(const RenderPatch& patch);

	private: // ==============================

		alignas(64) VisParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<int64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Vis
