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
			.size			= {2120, 1316},
			.title			= "TSP Visual",
			.resizable		= true,
			.vsync			= true
		};

		FrameBufferConfig frameBuffer
		{
			.size			= {2120, 1316},
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
			.camPos			= {1.3f, 0.5f, -0.9f},
			.yawDeg			= -230.0f,
			.pitchDeg		= 0.0f,
			.moveSpeed		= 1.0f,
			.turnSpeedDeg	= 90.0f,
			.speedMult		= 4.0f,
			.yFovDeg		= 60.0f,
			.zNear			= 0.01f,
			.zFar			= 500.0f
		};

		RenderConfig render
		{
			.lightDirVS             = {0.3f, 0.6f, 0.6f},
			.clearColor             = {0.2f, 0.2f, 0.2f, 1.0f},
			.baseColor              = {0.2f, 0.4f, 1.0f},
			.rigidBoundaryColor     = {0.65f, 0.65f, 0.65f},
			.centerOfMassXAxisColor = {250.0f / 255.0f, 0.0f, 50.0f / 255.0f},
			.centerOfMassYAxisColor = {40.0f / 255.0f, 180.0f / 255.0f, 80.0f / 255.0f},
			.centerOfMassZAxisColor = {40.0f / 255.0f, 80.0f / 255.0f, 180.0f / 255.0f},
			.lineColor              = {1.0f, 1.0f, 1.0f},
			.alpha                  = 1.0f,
			.centerOfMassXRayAlpha  = 0.5f,
			.scalarMin              = 0.0f,
			.scalarMax              = 3.0f,
			.colorMap               = Base::ColorMap::Blues,
			.scalarModeF            = Base::ColorCodingMode::VelocityMag,
			.showFluidParticles     = true,
			.showRigidBoundaryParticles = true,
			.showRigidBodyCentersOfMass = false,
			.rigidBoundaryRadiusScale = 1.0f,
			.lineWidth              = 0.02f
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
		std::optional<glm::vec3> rigidBoundaryColor;
		std::optional<glm::vec3> centerOfMassXAxisColor;
		std::optional<glm::vec3> centerOfMassYAxisColor;
		std::optional<glm::vec3> centerOfMassZAxisColor;
		std::optional<float> alpha;
		std::optional<float> centerOfMassXRayAlpha;
		std::optional<float> scalarMin;
		std::optional<float> scalarMax;
		std::optional<Base::ColorMap> colorMap;
		std::optional<Base::ColorCodingMode> scalarModeF;
		std::optional<bool> showFluidParticles;
		std::optional<bool> showRigidBoundaryParticles;
		std::optional<bool> showRigidBodyCentersOfMass;
		std::optional<float> rigidBoundaryRadiusScale;
	};

	class VisParamsMgr
	{
	public: // ---------------------------------------------------------------------------------------

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

	private: // --------------------------------------------------------------------------------------

		alignas(64) VisParams params_{}; // current active copy of visualization parameters
		alignas(64) std::atomic<int64_t> version_{ 0 }; // even = stable, odd = writing

		inline void BeginWrite(); // mark beginning of write operation
		inline void EndWrite(); // mark end of write operation
	};

} // namespace Vis
