// structs_vis.h
//
// TODO: comment

#pragma once

// external
#include <span>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// internal
#include "../../Base/enum_hub.h"



namespace Vis
{
	struct VisualSnapView
	{
		std::span<const glm::vec3> position;
		std::span<const float> scalarData; // data according to color coding mode
		Base::ColorCodingMode scalarMode{ Base::None };
		std::int64_t N_particles{ 0 };
	};

	struct WindowConfig
	{
		glm::ivec2 size; // window size in screen coordinates
		const char* title;
		bool resizable;
		bool vsync;
	};

	struct FrameBufferConfig
	{
		glm::ivec2 size; // window size in pixels
		float dpiScale; // pixels per screen coordinate
	};

	struct CameraState
	{
		glm::mat4 viewMat;
		glm::mat4 projMat;
		glm::vec3 camPos;
		float yawDeg;			// left/right angle in degrees
		float pitchDeg;			// up/down angle in degrees
		float moveSpeed;
		float turnSpeedDeg;
		float speedMult;		// speed multiplier for fast movement toggle
		float yFovDeg;			// vertical field of view in degrees
		float zNear;			// near clipping plane
		float zFar;				// far clipping plane
	};

	struct GLVersion
	{
		int major;
		int minor;
		const char* name;
	};

	struct RenderConfig
	{
		glm::vec3 lightDirVS;		// light direction in view space
		glm::vec4 clearColor;		// background color RGBA
		glm::vec3 baseColor;		// base sphere color RGB
		float alpha;
		float radius;
		float scalarMin;
		float scalarMax;
		bool useScalar;
	};

} // namespace Vis
