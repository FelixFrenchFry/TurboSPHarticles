// structs_vis.h
//
// TODO: comment

#pragma once

// external
#include <cstdint>
#include <span>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// internal
#include "source/base/enum_hub.h"
#include "source/simulation/scene/builder.h"



namespace Vis
{
	struct VisualSnapView
	{
		// metadata
		uint64_t simStep;                   // simulation step at this snapshot

		// data for fluid particle vis
		uint32_t N_F;                       // number of fluid particles
		std::span<const glm::vec3> posF;    // positions of all fluid particles
		std::span<const float> scalarDataF; // data according to color coding mode
		Base::ColorCodingMode scalarModeF;  // color coding mode

		// data for boundary structures vis
		uint32_t N_B;                       // number of boundary particles
		std::span<const glm::vec3> posB;    // positions of all boundary particles
		std::span<const float> volB;        // contribution volumes of all boundary particles
		std::span<const Scene::OBB> boxesB; // OBBs of boundary structures

		// data for rigid particle vis
		uint32_t N_R;                       // number of rigid particles
		std::span<const glm::vec3> posR;    // positions of all rigid particles
		std::span<const float> volR;        // contribution volumes of all rigid particles

		// data for rigid body vis
		uint32_t N_RB;                      // number of rigid bodies
		std::span<const glm::vec3> centerOfMassRB;
		std::span<const glm::mat3> orientationRB;
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
		float yawDeg;           // left/right angle in degrees
		float pitchDeg;         // up/down angle in degrees
		float moveSpeed;
		float turnSpeedDeg;
		float speedMult;        // speed multiplier for fast movement toggle
		float yFovDeg;          // vertical field of view in degrees
		float zNear;            // near clipping plane
		float zFar;             // far clipping plane
	};

	struct GLVersion
	{
		int major;
		int minor;
		const char* name;
	};

	struct RenderConfig
	{
		glm::vec3 lightDirVS;           // light direction in view space
		glm::vec4 clearColor;           // background color RGBA
		glm::vec3 baseColor;            // base sphere impostor color RGB
		glm::vec3 rigidParticleColor;   // rigid particle color RGB
		glm::vec3 centerOfMassXAxisColor; // center-of-mass local X-axis color RGB
		glm::vec3 centerOfMassYAxisColor; // center-of-mass local Y-axis color RGB
		glm::vec3 centerOfMassZAxisColor; // center-of-mass local Z-axis color RGB
		glm::vec3 lineColor;            // wireframe color RGB
		float alpha;
		float centerOfMassXRayAlpha;
		float scalarMin;
		float scalarMax;
		Base::ColorMap colorMap;
		Base::ColorCodingMode scalarModeF;
		bool showParticlesF;
		bool showParticlesR;
		bool showRigidBodyCentersOfMass;
		float rigidParticleRadiusScale;

		float lineWidth;
	};

} // namespace Vis
