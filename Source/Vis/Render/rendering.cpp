// rendering.cpp

// external
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <exception>
#include <glad/glad.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include <cuda_gl_interop.h>
#endif

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "rendering.h"
#include "shader.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "Source/Sim/Sync/mailboxes_cuda.h"
	#include "cuda_interop.cuh"
#endif



using namespace Render;

namespace
{
	static constexpr float CENTER_OF_MASS_MARKER_LENGTH_RATIO = 2.5f;
	static constexpr float CENTER_OF_MASS_MARKER_THICKNESS_RATIO = 0.25f;

	// builds one oriented marker box for each rigid body axis
	std::array<glm::mat4, 3> BuildCenterOfMassMarkerTransforms(
		const glm::vec3& centerOfMass,
		const glm::mat3& orientation,
		const float particleSizeF)
	{
		// size marker boxes relative to fluid particle diameter
		const float size = std::max(particleSizeF, 0.0f);
		const float length = CENTER_OF_MASS_MARKER_LENGTH_RATIO * size;
		const float thickness = CENTER_OF_MASS_MARKER_THICKNESS_RATIO * size;
		const std::array<glm::vec3, 3> dimensions
		{
			glm::vec3{ length, thickness, thickness },
			glm::vec3{ thickness, length, thickness },
			glm::vec3{ thickness, thickness, length },
		};

		// orient and place each marker box at the rigid body center of mass
		std::array<glm::mat4, 3> transforms{};
		for (uint32_t axis = 0; axis < transforms.size(); axis++)
		{
			glm::mat4 transform{ 1.0f };
			transform[0] = glm::vec4(orientation[0] * dimensions[axis].x, 0.0f);
			transform[1] = glm::vec4(orientation[1] * dimensions[axis].y, 0.0f);
			transform[2] = glm::vec4(orientation[2] * dimensions[axis].z, 0.0f);
			transform[3] = glm::vec4(centerOfMass, 1.0f);
			transforms[axis] = transform;
		}
		return transforms;
	}

	// computes rigid boundary particle radius from represented volume
	float ComputeRigidBoundaryRenderRadius(
		const float volume,
		const float particleSizeB,
		const float radiusScale)
	{
		// use nominal particle radius if volume cannot produce a valid radius
		const float nonNegativeScale = std::max(radiusScale, 0.0f);
		const float fallbackRadius = particleSizeB * 0.5f * nonNegativeScale;
		if (!std::isfinite(volume) || volume <= 0.0f)
		{
			return fallbackRadius;
		}

		const float radius = 0.5f * std::cbrt(volume) * nonNegativeScale;
		return std::isfinite(radius) ? radius : fallbackRadius;
	}

} // anonymous

RenderingMgr::RenderingMgr(Sim::SimParamsMgr& simParamsMgr,
						   Vis::VisParamsMgr& visParamsMgr)
	: simParamsMgr_{ simParamsMgr }
	, visParamsMgr_{ visParamsMgr }
{
	// handled by Init()
}

void RenderingMgr::Init(glm::ivec2 fbSize, float dpiScale)
{
	EnableDebugOutput();

	// create shader programs
	CreateSphereImpostorProgram();
	CreateCenterOfMassProgram();
	CreateLineProgram();

	// create VAOs and buffers
	CreateSphereImpostorVAOandBuffers();
	CreateCenterOfMassVAOandBuffers();
	CreateLineVAOandBuffers();
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		InitParticleInteropCUDA();
	#endif

	// upload default color map for scalar particle data
	UpdateColorMap(BuildColorMap_Turbo());

	// enable alpha blending for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void RenderingMgr::Shutdown()
{
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		ShutdownParticleInteropCUDA();
	#endif

	if (progSphereImpostors_)
	{
		glDeleteProgram(progSphereImpostors_);
		progSphereImpostors_ = 0;
	}

	if (progCenterOfMass_)
	{
		glDeleteProgram(progCenterOfMass_);
		progCenterOfMass_ = 0;
	}

	if (progLines_)
	{
		glDeleteProgram(progLines_);
		progLines_ = 0;
	}

	DestroySphereImpostorVAOandBuffers();
	DestroyCenterOfMassVAOandBuffers();
	DestroyLineVAOandBuffers();
}

#ifdef TURBOSPHARTICLES_WITH_CUDA
void RenderingMgr::InitParticleInteropCUDA()
{
	if (!visualFrameMailboxCUDA_) { return; }

	// get current CUDA device and devices used by the OpenGL context
	int deviceCUDA = -1;
	if (!HandleErrorCUDA(cudaGetDevice(&deviceCUDA), "cudaGetDevice()")) { return; }

	std::array<int, 8> glDevicesCUDA{};
	unsigned int N_glDevices = 0;
	if (!HandleErrorCUDA(
		cudaGLGetDevices(
			&N_glDevices,
			glDevicesCUDA.data(),
			static_cast<unsigned int>(glDevicesCUDA.size()),
			cudaGLDeviceListAll),
		"cudaGLGetDevices()"))
	{
		return;
	}

	// make sure current CUDA device is associated with the OpenGL context
	if (std::find(
		glDevicesCUDA.begin(),
		glDevicesCUDA.begin() + N_glDevices,
		deviceCUDA) == glDevicesCUDA.begin() + N_glDevices)
	{
		SPDLOG_WARN(
			"CUDA visual fast path unavailable: simulation device {} is not associated "
			"with the current OpenGL context.", deviceCUDA);
		return;
	}

	// create CUDA rendering stream and enable the interop mailbox
	if (!HandleErrorCUDA(
		cudaStreamCreateWithFlags(&renderStreamCUDA_, cudaStreamNonBlocking),
		"cudaStreamCreateWithFlags() for rendering"))
	{
		return;
	}

	interopReadyCUDA_ = true;
	visualFrameMailboxCUDA_->Enable(deviceCUDA);
}

void RenderingMgr::ShutdownParticleInteropCUDA() noexcept
{
	interopReadyCUDA_ = false;

	// unregister CUDA access to the OpenGL instance buffer
	if (instanceResourceCUDA_)
	{
		cudaGraphicsUnregisterResource(instanceResourceCUDA_);
		instanceResourceCUDA_ = nullptr;
	}

	// destroy CUDA rendering stream
	if (renderStreamCUDA_)
	{
		cudaStreamDestroy(renderStreamCUDA_);
		renderStreamCUDA_ = nullptr;
	}

	// delete OpenGL instance buffer
	if (vboInstancesCUDA_)
	{
		glDeleteBuffers(1, &vboInstancesCUDA_);
		vboInstancesCUDA_ = 0;
	}
	capacityCUDA_ = 0;
}
#endif

void RenderingMgr::UpdateParticles(const Vis::VisualSnapView& view,
								   const Sim::SimParams& simParams,
								   const Vis::VisParams& visParams)
{
	N_fluidImpostorInstances_ = view.N_F;
	N_rigidBoundaryImpostorInstances_ = 0;

	const bool hasBodyIdB = view.bodyIdB.size() == view.N_B;
	if (hasBodyIdB)
	{
		for (uint32_t i = 0; i < view.N_B; i++)
		{
			if (view.bodyIdB[i] != Sync::InvalidRigidBodyId)
			{
				N_rigidBoundaryImpostorInstances_++;
			}
		}
	}

	const uint32_t N_total =
		N_fluidImpostorInstances_ + N_rigidBoundaryImpostorInstances_;
	if (N_total == 0) { return; }

	EnsureInstanceCapacity(N_total);
	glVertexArrayVertexBuffer(vao_, 1, vboInstances_, 0, sizeof(InstanceData));

	auto* out = reinterpret_cast<InstanceData*>(mappedBase_);
	const float fluidRad = simParams.particleSizeF * 0.5f;
	const bool hasVolB = view.volB.size() == view.N_B;
	const bool hasScalar = (view.scalarModeF != Base::ColorCodingMode::None) &&
							(view.scalarDataF.size() == view.N_F);

	// write fluid particles instance data first
	for (uint32_t i = 0; i < N_fluidImpostorInstances_; i++)
	{
		const glm::vec3 pos = view.posF[i];
		out[i] = InstanceData{
			pos.x, pos.y, pos.z, fluidRad,
			hasScalar ? view.scalarDataF[i] : 0.0f,
			{ 0.0f, 0.0f, 0.0f } };
	}

	// append rigid boundary instance data
	uint32_t dst = N_fluidImpostorInstances_;
	if (hasBodyIdB)
	{
		for (uint32_t i = 0; i < view.N_B; i++)
		{
			if (view.bodyIdB[i] == Sync::InvalidRigidBodyId) { continue; }

			const glm::vec3 pos = view.posB[i];
			const float volume = hasVolB ? view.volB[i] : 0.0f;
			const float radius = ComputeRigidBoundaryRenderRadius(
				volume,
				simParams.particleSizeB,
				visParams.render.rigidBoundaryRadiusScale);
			out[dst] = InstanceData{
				pos.x, pos.y, pos.z, radius, 0.0f,
				{ 0.0f, 0.0f, 0.0f } };
			dst++;
		}
	}

	// no need to flush, because buffer persistently mapped
}

#ifdef TURBOSPHARTICLES_WITH_CUDA
bool RenderingMgr::UpdateParticlesCUDA(
	const Sync::VisualFrameReadCUDA& frame,
	const Sim::SimParams& simParams,
	const Vis::VisParams& visParams)
{
	N_fluidImpostorInstances_ = frame.N_F;
	N_rigidBoundaryImpostorInstances_ = frame.N_RBP;

	const uint32_t N_total =
		N_fluidImpostorInstances_ + N_rigidBoundaryImpostorInstances_;

	if (!interopReadyCUDA_ || !visualFrameMailboxCUDA_)
	{
		N_fluidImpostorInstances_ = 0;
		N_rigidBoundaryImpostorInstances_ = 0;
		return false;
	}

	// wait for the simulation stream to finish packing the frame
	bool success = HandleErrorCUDA(
		cudaStreamWaitEvent(renderStreamCUDA_, frame.ready, 0),
		"cudaStreamWaitEvent() for visual snapshot");

	// ensure capacity for fluid and rigid boundary particle instances
	if (success && N_total > 0)
	{
		success = EnsureInstanceCapacityCUDA(N_total);
	}

	// map CUDA-OpenGL instance buffer
	bool mapped = false;
	InstanceData* out = nullptr;
	if (success && N_total > 0)
	{
		mapped = HandleErrorCUDA(
			cudaGraphicsMapResources(
				1, &instanceResourceCUDA_, renderStreamCUDA_),
			"cudaGraphicsMapResources() for particle instances");
		success = mapped;
	}

	if (success && N_total > 0)
	{
		size_t mappedBytes = 0;
		success = HandleErrorCUDA(
			cudaGraphicsResourceGetMappedPointer(
				reinterpret_cast<void**>(&out),
				&mappedBytes,
				instanceResourceCUDA_),
			"cudaGraphicsResourceGetMappedPointer() for particle instances");
		if (success && mappedBytes < static_cast<size_t>(N_total) * sizeof(InstanceData))
		{
			interopReadyCUDA_ = false;
			visualFrameMailboxCUDA_->Disable("mapped OpenGL instance buffer is too small");
			success = false;
		}
	}

	const float fluidRad = simParams.particleSizeF * 0.5f;

	// write fluid particles instance data first and append rigid boundary instance data
	if (success && N_total > 0)
	{
		try
		{
			Launch_PackVisualInstancesCUDA(
				frame.fluid, frame.rigid,
				fluidRad,
				simParams.particleSizeB,
				visParams.render.rigidBoundaryRadiusScale,
				frame.N_F, frame.N_RBP, renderStreamCUDA_, out);
		}
		catch (const std::exception& ex)
		{
			interopReadyCUDA_ = false;
			visualFrameMailboxCUDA_->Disable(ex.what());
			success = false;
		}
	}

	// return instance buffer to OpenGL and release mailbox slot
	if (mapped)
	{
		const bool unmapped = HandleErrorCUDA(
			cudaGraphicsUnmapResources(
				1, &instanceResourceCUDA_, renderStreamCUDA_),
			"cudaGraphicsUnmapResources() for particle instances");
		success = success && unmapped;
	}

	const bool released = visualFrameMailboxCUDA_->ReleaseRead(frame, renderStreamCUDA_);
	if (!success || !released)
	{
		N_fluidImpostorInstances_ = 0;
		N_rigidBoundaryImpostorInstances_ = 0;
		return false;
	}

	// bind CUDA instance buffer for particle rendering
	if (N_total > 0)
	{
		glVertexArrayVertexBuffer(
			vao_, 1, vboInstancesCUDA_, 0, sizeof(InstanceData));
	}
	return true;
}
#endif

void RenderingMgr::UpdateCentersOfMass(
	const Vis::VisualSnapView& view,
	const Sim::SimParams& simParams)
{
	N_centerOfMassInstances_ = 0;
	if (view.centerOfMassR.size() != view.N_R ||
		view.orientationR.size() != view.N_R)
	{
		return;
	}

	std::vector<glm::mat4> transforms;
	transforms.reserve(static_cast<size_t>(view.N_R) * 3);
	for (uint32_t bodyIdx = 0; bodyIdx < view.N_R; bodyIdx++)
	{
		const auto bodyTransforms = BuildCenterOfMassMarkerTransforms(
			view.centerOfMassR[bodyIdx],
			view.orientationR[bodyIdx],
			simParams.particleSizeF);
		transforms.insert(
			transforms.end(), bodyTransforms.begin(), bodyTransforms.end());
	}

	N_centerOfMassInstances_ = static_cast<uint32_t>(transforms.size());
	if (N_centerOfMassInstances_ == 0) { return; }

	const GLsizeiptr bytes =
		static_cast<GLsizeiptr>(transforms.size() * sizeof(glm::mat4));
	glNamedBufferData(
		vboCenterOfMassInstances_, bytes, transforms.data(), GL_DYNAMIC_DRAW);
}

void RenderingMgr::UpdateWireframes(const Vis::VisualSnapView& view, const Vis::VisParams& params)
{
	// get line vertices for all boxes
	std::vector<glm::vec3> lineVertices{};
	for (const auto& box : view.boxesB)
	{
		BuildWireframeEdgesFromOBB(box, lineVertices);
	}

	N_lineVertices_ = lineVertices.size();
	if (N_lineVertices_ == 0) { return; }

	const GLsizeiptr bytes = static_cast<GLsizeiptr>(lineVertices.size() * sizeof(glm::vec3));
	glNamedBufferData(vboLines_, bytes, lineVertices.data(), GL_DYNAMIC_DRAW);
}

void RenderingMgr::UpdateColorMap(const std::vector<glm::vec4>& colorMap256)
{
	if (colorMap256.size() != 256)
	{
		SPDLOG_WARN("Cannot update colormap: Expected 256 entries, got {}.",
			colorMap256.size());
		return;
	}

	glTextureSubImage1D(texColorMap_, 0, 0, 256, GL_RGBA, GL_FLOAT,
		colorMap256.data());
}

// builds Google's Turbo color map from its polynomial approximation
std::vector<glm::vec4> RenderingMgr::BuildColorMap_Turbo()
{
	const glm::vec4 red4{ 0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f };
	const glm::vec4 green4{ 0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f };
	const glm::vec4 blue4{ 0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f };
	const glm::vec2 red2{ -152.94239396f, 59.28637943f };
	const glm::vec2 green2{ 4.27729857f, 2.82956604f };
	const glm::vec2 blue2{ -89.90310912f, 27.34824973f };

	// evaluate the polynomial at each color lookup-table entry
	std::vector<glm::vec4> colorMap(256);
	for (uint32_t i = 0; i < colorMap.size(); i++)
	{
		const float x = static_cast<float>(i) /
			static_cast<float>(colorMap.size() - 1);
		const float x2 = x * x;
		const glm::vec4 x4{ 1.0f, x, x2, x2 * x };
		const glm::vec2 x2High{ x2 * x2, x2 * x2 * x };
		const glm::vec3 color
		{
			glm::dot(x4, red4) + glm::dot(x2High, red2),
			glm::dot(x4, green4) + glm::dot(x2High, green2),
			glm::dot(x4, blue4) + glm::dot(x2High, blue2)
		};
		colorMap[i] = glm::vec4(glm::clamp(color, 0.0f, 1.0f), 1.0f);
	}
	return colorMap;
}

// builds Matplotlib's Blues color map from its control colors
std::vector<glm::vec4> RenderingMgr::BuildColorMap_Blues()
{
	const std::array<glm::vec3, 9> controlColors
	{
		glm::vec3{ 8.0f, 48.0f, 107.0f } / 255.0f,
		glm::vec3{ 8.0f, 81.0f, 156.0f } / 255.0f,
		glm::vec3{ 33.0f, 113.0f, 181.0f } / 255.0f,
		glm::vec3{ 66.0f, 146.0f, 198.0f } / 255.0f,
		glm::vec3{ 107.0f, 174.0f, 214.0f } / 255.0f,
		glm::vec3{ 158.0f, 202.0f, 225.0f } / 255.0f,
		glm::vec3{ 198.0f, 219.0f, 239.0f } / 255.0f,
		glm::vec3{ 222.0f, 235.0f, 247.0f } / 255.0f,
		glm::vec3{ 247.0f, 251.0f, 255.0f } / 255.0f
	};

	// interpolate the control colors at each color lookup-table entry
	std::vector<glm::vec4> colorMap(256);
	for (uint32_t i = 0; i < colorMap.size(); i++)
	{
		const float x = static_cast<float>(i) /
			static_cast<float>(colorMap.size() - 1);
		const float controlPosition = x *
			static_cast<float>(controlColors.size() - 1);
		const uint32_t lowerIdx = std::min(
			static_cast<uint32_t>(controlPosition),
			static_cast<uint32_t>(controlColors.size() - 2));
		const float blend = controlPosition - static_cast<float>(lowerIdx);
		const glm::vec3 color = controlColors[lowerIdx] + blend *
			(controlColors[lowerIdx + 1] - controlColors[lowerIdx]);
		colorMap[i] = glm::vec4(color, 1.0f);
	}
	return colorMap;
}

void RenderingMgr::Render(const Vis::RenderConfig& render, const Vis::CameraState& cam)
{
	// upload selected color map if it changed
	if (render.colorMap != colorMap_)
	{
		switch (render.colorMap)
		{
		case Base::ColorMap::Turbo:
			UpdateColorMap(BuildColorMap_Turbo());
			break;
		case Base::ColorMap::Blues:
			UpdateColorMap(BuildColorMap_Blues());
			break;
		}
		colorMap_ = render.colorMap;
	}

	// make sure params cached on the GPU are up to date
	UpdateCameraUBO(cam);

	if (N_fluidImpostorInstances_ > 0 || N_rigidBoundaryImpostorInstances_ > 0)
	{
		// bind state and bind color LUT
		glUseProgram(progSphereImpostors_);
		glBindVertexArray(vao_);
		glBindTextureUnit(0, texColorMap_);

		// draw 4-vertex quads (2 triangles), instanced N times
		glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		if (render.showFluidParticles && N_fluidImpostorInstances_ > 0)
		{
			// render fluid particle instances
			UpdateParamsUBO(render);
			glDrawArraysInstanced(
				GL_TRIANGLE_FAN,
				0, 4,
				static_cast<GLsizei>(N_fluidImpostorInstances_));
		}
		if (render.showRigidBoundaryParticles && N_rigidBoundaryImpostorInstances_ > 0)
		{
			Vis::RenderConfig rigidRender = render;
			rigidRender.baseColor = render.rigidBoundaryColor;
			rigidRender.alpha = 1.0f;
			rigidRender.scalarModeF = Base::ColorCodingMode::None;

			// render rigid boundary particle instances
			UpdateParamsUBO(rigidRender);
			glDrawArraysInstancedBaseInstance(
				GL_TRIANGLE_FAN,
				0, 4,
				static_cast<GLsizei>(N_rigidBoundaryImpostorInstances_),
				N_fluidImpostorInstances_);
		}
		glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
	}

	if (render.showRigidBodyCentersOfMass && N_centerOfMassInstances_ > 0)
	{
		glUseProgram(progCenterOfMass_);
		glBindVertexArray(vaoCenterOfMass_);

		glUniform3fv(uniCenterOfMassXAxisColorLoc_,
			1, &render.centerOfMassXAxisColor.x);
		glUniform3fv(uniCenterOfMassYAxisColorLoc_,
			1, &render.centerOfMassYAxisColor.x);
		glUniform3fv(uniCenterOfMassZAxisColorLoc_,
			1, &render.centerOfMassZAxisColor.x);

		// draw a faint x-ray pass so centers inside closed bodies remain visible
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glUniform1f(uniCenterOfMassAlphaLoc_, render.centerOfMassXRayAlpha);
		glDrawArraysInstanced(
			GL_TRIANGLES, 0, 36, static_cast<GLsizei>(N_centerOfMassInstances_));

		// draw visible marker surfaces normally against the scene depth buffer
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glUniform1f(uniCenterOfMassAlphaLoc_, 1.0f);
		glDrawArraysInstanced(
			GL_TRIANGLES, 0, 36, static_cast<GLsizei>(N_centerOfMassInstances_));
	}

	if (N_lineVertices_ > 0)
	{
		// bind state and bind color uniform
		glUseProgram(progLines_);
		glBindVertexArray(vaoLines_);
		glUniform4fv(uniColorLoc_, 1, &render.lineColor.x);
		glLineWidth(render.lineWidth);
		glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(N_lineVertices_));
		glBindVertexArray(0);
		glUseProgram(0);
	}
}

// creates the shader program for rendering sphere impostors
void RenderingMgr::CreateSphereImpostorProgram()
{
	GLuint vertS = 0;
	GLuint fragS = 0;
	try
	{
		vertS = Compile(GL_VERTEX_SHADER, VS_sphereImpostor, "VS_sphereImpostor");
		fragS = Compile(GL_FRAGMENT_SHADER, FS_sphereImpostor, "FS_sphereImpostor");
		progSphereImpostors_ = Link(vertS, fragS, "sphere impostor");
	}
	catch (...)
	{
		if (vertS != 0) { glDeleteShader(vertS); }
		if (fragS != 0) { glDeleteShader(fragS); }
		throw;
	}

	glDeleteShader(vertS);
	glDeleteShader(fragS);
}

// creates the shader program for rendering rigid body centers of mass
void RenderingMgr::CreateCenterOfMassProgram()
{
	GLuint vertS = 0;
	GLuint fragS = 0;
	try
	{
		vertS = Compile(
			GL_VERTEX_SHADER, VS_centerOfMassMarker, "VS_centerOfMassMarker");
		fragS = Compile(
			GL_FRAGMENT_SHADER, FS_centerOfMassMarker, "FS_centerOfMassMarker");
		progCenterOfMass_ = Link(vertS, fragS, "center of mass marker");
	}
	catch (...)
	{
		if (vertS != 0) { glDeleteShader(vertS); }
		if (fragS != 0) { glDeleteShader(fragS); }
		throw;
	}

	glDeleteShader(vertS);
	glDeleteShader(fragS);

	uniCenterOfMassXAxisColorLoc_ = glGetUniformLocation(
		progCenterOfMass_, "uXAxisColor");
	uniCenterOfMassYAxisColorLoc_ = glGetUniformLocation(
		progCenterOfMass_, "uYAxisColor");
	uniCenterOfMassZAxisColorLoc_ = glGetUniformLocation(
		progCenterOfMass_, "uZAxisColor");
	uniCenterOfMassAlphaLoc_ = glGetUniformLocation(
		progCenterOfMass_, "uAlpha");
	if (uniCenterOfMassXAxisColorLoc_ < 0 ||
		uniCenterOfMassYAxisColorLoc_ < 0 ||
		uniCenterOfMassZAxisColorLoc_ < 0 ||
		uniCenterOfMassAlphaLoc_ < 0)
	{
		SPDLOG_WARN(
			"Failed to find center-of-mass marker uniform locations.");
	}
}

// creates the shader program for rendering lines
void RenderingMgr::CreateLineProgram()
{
	GLuint vertS = 0;
	GLuint fragS = 0;
	try
	{
		vertS = Compile(GL_VERTEX_SHADER, VS_line, "VS_line");
		fragS = Compile(GL_FRAGMENT_SHADER, FS_line, "FS_line");
		progLines_ = Link(vertS, fragS, "line");
	}
	catch (...)
	{
		if (vertS != 0) { glDeleteShader(vertS); }
		if (fragS != 0) { glDeleteShader(fragS); }
		throw;
	}

	glDeleteShader(vertS);
	glDeleteShader(fragS);

	uniColorLoc_ = glGetUniformLocation(progLines_, "uColor");
	if (uniColorLoc_ < 0)
	{
		SPDLOG_WARN("Failed to find uniform location for 'uColor' in line shader.");
	}
}

// compiles a shader of given type from source and returns its ID
GLuint RenderingMgr::Compile(GLenum type, const char* src, const char* shaderName)
{
	GLuint res = glCreateShader(type);
	GLint ok = 0;

	// compile and check shader
	glShaderSource(res, 1, &src, nullptr);
	glCompileShader(res);
	glGetShaderiv(res, GL_COMPILE_STATUS, &ok);

	if (!ok) // shader compilation failed
	{
		GLint len = 0;
		glGetShaderiv(res, GL_INFO_LOG_LENGTH, &len);
		std::string log(len, '\0');
		glGetShaderInfoLog(res, len, nullptr, log.data());
		glDeleteShader(res);

		throw std::runtime_error(
			std::string("Failed to compile shader '") + shaderName + "': " + log);
	}

	return res;
}

// links a shader program given compiled vertex and fragment shaders and returns its ID
GLuint RenderingMgr::Link(GLuint vertS, GLuint fragS, const char* programName)
{
	GLuint res = glCreateProgram();
	GLint ok = 0;

	// attach, link, and check program
	glAttachShader(res, vertS);
	glAttachShader(res, fragS);
	glLinkProgram(res);
	glGetProgramiv(res, GL_LINK_STATUS, &ok);

	if (!ok) // program linking failed
	{
		GLint len = 0;
		glGetProgramiv(res, GL_INFO_LOG_LENGTH, &len);
		std::string log(len, '\0');
		glGetProgramInfoLog(res, len, nullptr, log.data());
		glDeleteProgram(res);

		throw std::runtime_error(
			std::string("Failed to link shader program '") + programName + "': " + log);
	}

	// detach shaders before the caller releases them
	glDetachShader(res, vertS);
	glDetachShader(res, fragS);

	return res;
}

void RenderingMgr::UpdateParamsUBO(const Vis::RenderConfig& render)
{
	// layout(std140, binding=0) uniform Params
	// {
	//     vec4 uLightDirVS;       // xyz = light direction in view space, w unused (0)
	//     vec4 uBaseColorAlpha;   // xyz = base RGB, w = alpha
	//     vec4 uScalarMinMaxUse;  // x = min, y = max, z = useScalar (0/1), w unused (0)
	// };

	// normalize light direction
	glm::vec3 lightDirVS = glm::normalize(render.lightDirVS);

	// pack each vec4 to match std140 layout (0, 16, 32 bytes offset)
	const glm::vec4 p0(lightDirVS, 0.0f);
	const glm::vec4 p1(render.baseColor, render.alpha);
	const glm::vec4 p2(render.scalarMin, render.scalarMax,
		render.scalarModeF != Base::ColorCodingMode::None ? 1.0f : 0.0f, 0.0f);

	glNamedBufferSubData(uboParams_, 0, sizeof(glm::vec4), &p0);
	glNamedBufferSubData(uboParams_, 16, sizeof(glm::vec4), &p1);
	glNamedBufferSubData(uboParams_, 32, sizeof(glm::vec4), &p2);
}

void RenderingMgr::UpdateCameraUBO(const Vis::CameraState& cam)
{
	struct alignas(16) CameraUBO
	{
		glm::mat4 viewMat;		// offset 0
		glm::mat4 projMat;		// offset 64
		glm::mat4 viewProjMat;  // offset 128
		glm::vec4 camPos;		// offset 192
	} cachedCam;
	static_assert(sizeof(CameraUBO) == 208,	"CameraUBO does not adhere to std140 layout rules");
	static_assert(offsetof(CameraUBO, viewMat) == 0,		"Wrong offset breaks std140 layout");
	static_assert(offsetof(CameraUBO, projMat) == 64,		"Wrong offset breaks std140 layout");
	static_assert(offsetof(CameraUBO, viewProjMat) == 128,	"Wrong offset breaks std140 layout");
	static_assert(offsetof(CameraUBO, camPos) == 192,		"Wrong offset breaks std140 layout");

	cachedCam.viewMat = cam.viewMat;
	cachedCam.projMat = cam.projMat;
	cachedCam.viewProjMat = cam.projMat * cam.viewMat;
	cachedCam.camPos = glm::vec4(cam.camPos, 1.0f);

	glNamedBufferSubData(uboCamera_, 0, sizeof(CameraUBO), &cachedCam);
}

void RenderingMgr::CreateSphereImpostorVAOandBuffers()
{
	// create VAO for tracking vertex attributes
	glCreateVertexArrays(1, &vao_);

	// create static quad VBO with 4 corners
	glCreateBuffers(1, &vboQuad_);
	const glm::vec2 quad[4] =
	{
		{-1.0f, -1.0f},
		{+1.0f, -1.0f},
		{+1.0f, +1.0f},
		{-1.0f, +1.0f},
	};
	glNamedBufferStorage(vboQuad_, sizeof(quad), quad, 0); // immutable storage

	// bind quad VBO to index 0
	glVertexArrayVertexBuffer(vao_, 0, vboQuad_, 0, sizeof(glm::vec2));

	// attribute 0 = vec2 corner
	glEnableVertexArrayAttrib(vao_, 0);
	glVertexArrayAttribFormat(vao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao_, 0, 0); // attribute 0 pulls from binding 0

	// instance buffer
	glCreateBuffers(1, &vboInstances_);
	capacity_ = 0;
	mappedBase_ = nullptr;
	mappedSize_ = 0;

	// bind instance VBO to index 1
	glVertexArrayVertexBuffer(vao_, 1, vboInstances_, 0, sizeof(InstanceData));

	// attribute 1 = vec4 posAndRad (with offset 0)
	glEnableVertexArrayAttrib(vao_, 1);
	glVertexArrayAttribFormat(vao_, 1, 4, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao_, 1, 1);
	glVertexArrayBindingDivisor(vao_, 1, 1);

	// attribute 2 = float scalar (with offset 16)
	glEnableVertexArrayAttrib(vao_, 2);
	glVertexArrayAttribFormat(vao_, 2, 1, GL_FLOAT, GL_FALSE, 16);
	glVertexArrayAttribBinding(vao_, 2, 1);
	glVertexArrayBindingDivisor(vao_, 2, 1);

	// create UBO for caching params on the GPU
	glCreateBuffers(1, &uboParams_);
	glNamedBufferStorage(uboParams_, 3 * sizeof(glm::vec4), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboParams_); // binding to 0
	glCreateBuffers(1, &uboCamera_);
	glNamedBufferStorage(uboCamera_, 3 * sizeof(glm::mat4) + sizeof(glm::vec4), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboCamera_); // binding to 1

	// color lookup-table texture (allocate empty 256x1 row)
	glCreateTextures(GL_TEXTURE_1D, 1, &texColorMap_);
	glTextureStorage1D(texColorMap_, 1, GL_SRGB8_ALPHA8, 256); // store in sRGB
	glTextureParameteri(texColorMap_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(texColorMap_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(texColorMap_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

	// TODO: add GPU timer?
	// glGenQueries(1, &timerQuery_);
}

void RenderingMgr::CreateCenterOfMassVAOandBuffers()
{
	constexpr float H = 0.5f;
	const glm::vec3 cubeVertices[36]
	{
		{-H, -H, +H}, {+H, -H, +H}, {+H, +H, +H},
		{-H, -H, +H}, {+H, +H, +H}, {-H, +H, +H},
		{+H, -H, -H}, {-H, -H, -H}, {-H, +H, -H},
		{+H, -H, -H}, {-H, +H, -H}, {+H, +H, -H},
		{-H, -H, -H}, {-H, -H, +H}, {-H, +H, +H},
		{-H, -H, -H}, {-H, +H, +H}, {-H, +H, -H},
		{+H, -H, +H}, {+H, -H, -H}, {+H, +H, -H},
		{+H, -H, +H}, {+H, +H, -H}, {+H, +H, +H},
		{-H, +H, +H}, {+H, +H, +H}, {+H, +H, -H},
		{-H, +H, +H}, {+H, +H, -H}, {-H, +H, -H},
		{-H, -H, -H}, {+H, -H, -H}, {+H, -H, +H},
		{-H, -H, -H}, {+H, -H, +H}, {-H, -H, +H},
	};

	glCreateVertexArrays(1, &vaoCenterOfMass_);
	glCreateBuffers(1, &vboCenterOfMassCube_);
	glNamedBufferStorage(
		vboCenterOfMassCube_, sizeof(cubeVertices), cubeVertices, 0);
	glVertexArrayVertexBuffer(
		vaoCenterOfMass_, 0, vboCenterOfMassCube_, 0, sizeof(glm::vec3));

	glEnableVertexArrayAttrib(vaoCenterOfMass_, 0);
	glVertexArrayAttribFormat(vaoCenterOfMass_, 0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoCenterOfMass_, 0, 0);

	glCreateBuffers(1, &vboCenterOfMassInstances_);
	glVertexArrayVertexBuffer(
		vaoCenterOfMass_, 1, vboCenterOfMassInstances_, 0, sizeof(glm::mat4));

	for (GLuint column = 0; column < 4; column++)
	{
		const GLuint location = 1 + column;
		glEnableVertexArrayAttrib(vaoCenterOfMass_, location);
		glVertexArrayAttribFormat(
			vaoCenterOfMass_,
			location,
			4,
			GL_FLOAT,
			GL_FALSE,
			static_cast<GLuint>(column * sizeof(glm::vec4)));
		glVertexArrayAttribBinding(vaoCenterOfMass_, location, 1);
	}
	glVertexArrayBindingDivisor(vaoCenterOfMass_, 1, 1);
}
void RenderingMgr::CreateLineVAOandBuffers()
{
	// create VAO and VBO for rendering lines
	glCreateVertexArrays(1, &vaoLines_);
	glCreateBuffers(1, &vboLines_);

	// bind line VBO to index 0
	glVertexArrayVertexBuffer(vaoLines_, 0, vboLines_, 0, sizeof(glm::vec3));

	// attribute 0 = vec3 aPos 
	glEnableVertexArrayAttrib(vaoLines_, 0);
	glVertexArrayAttribFormat(vaoLines_, 0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoLines_, 0, 0);
}

void RenderingMgr::DestroySphereImpostorVAOandBuffers()
{
	if (mappedBase_)
	{
		glUnmapNamedBuffer(vboInstances_);
		mappedBase_ = nullptr;
		mappedSize_ = 0;
	}

	if (texColorMap_) { glDeleteTextures(1, &texColorMap_); texColorMap_ = 0; }
	if (uboParams_) { glDeleteBuffers(1, &uboParams_); uboParams_ = 0; }
	if (uboCamera_) { glDeleteBuffers(1, &uboCamera_); uboCamera_ = 0; }
	if (vboInstances_) { glDeleteBuffers(1, &vboInstances_); vboInstances_ = 0; }
	if (vboQuad_) { glDeleteBuffers(1, &vboQuad_); vboQuad_ = 0; }
	if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
}

void RenderingMgr::DestroyCenterOfMassVAOandBuffers()
{
	if (vboCenterOfMassInstances_)
	{
		glDeleteBuffers(1, &vboCenterOfMassInstances_);
		vboCenterOfMassInstances_ = 0;
	}
	if (vboCenterOfMassCube_)
	{
		glDeleteBuffers(1, &vboCenterOfMassCube_);
		vboCenterOfMassCube_ = 0;
	}
	if (vaoCenterOfMass_)
	{
		glDeleteVertexArrays(1, &vaoCenterOfMass_);
		vaoCenterOfMass_ = 0;
	}
}

void RenderingMgr::DestroyLineVAOandBuffers()
{
	if (vboLines_) { glDeleteBuffers(1, &vboLines_); vboLines_ = 0; }
	if (vaoLines_) { glDeleteVertexArrays(1, &vaoLines_); vaoLines_ = 0; }
}

static void APIENTRY DebugCallback(GLenum source, GLenum type, GLuint id,
								   GLenum severity, GLsizei length,
								   const GLchar* message, const void* userParam)
{
	if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) { return; } // too noisy

	SPDLOG_WARN("OpenGL debug message (type=0x{:x}, severity=0x{:x}): {}.",
		type, severity, message);
}

void RenderingMgr::EnsureInstanceCapacity(uint32_t N_required)
{
	if (N_required <= capacity_) { return; }

	// grow capacity to next power of two
	uint64_t newCap = capacity_ ? capacity_ : 1024;
	while (newCap < N_required)
	{
		newCap *= 2;
	}
	uint64_t newSize = newCap * sizeof(InstanceData);

	// (re)allocate immutable storage and map persistently
	const GLbitfield storageFlags = GL_MAP_WRITE_BIT |
		GL_MAP_PERSISTENT_BIT |
		GL_MAP_COHERENT_BIT;
	const GLbitfield mapFlags = GL_MAP_WRITE_BIT |
		GL_MAP_PERSISTENT_BIT |
		GL_MAP_COHERENT_BIT;

	// immutable buffer storage cannot be allocated twice on the same object
	if (mappedBase_)
	{
		glUnmapNamedBuffer(vboInstances_);
		mappedBase_ = nullptr;
		mappedSize_ = 0;
	}
	if (vboInstances_)
	{
		glDeleteBuffers(1, &vboInstances_);
		vboInstances_ = 0;
	}
	glCreateBuffers(1, &vboInstances_);

	glNamedBufferStorage(vboInstances_, newSize, nullptr, storageFlags);
	mappedBase_ = static_cast<unsigned char*>(
		glMapNamedBufferRange(vboInstances_, 0, newSize, mapFlags));
	if (!mappedBase_)
	{
		throw std::runtime_error("Failed to map instance buffer.");
	}

	mappedSize_ = newSize;
	capacity_ = newCap;

	// re-bind instance VBO to index 1
	glVertexArrayVertexBuffer(vao_, 1, vboInstances_, 0, sizeof(InstanceData));
}

#ifdef TURBOSPHARTICLES_WITH_CUDA
bool RenderingMgr::EnsureInstanceCapacityCUDA(const uint32_t N_required)
{
	if (N_required <= capacityCUDA_) { return true; }

	// grow capacity to next power of two
	uint64_t newCap = capacityCUDA_ ? capacityCUDA_ : 1024;
	while (newCap < N_required) { newCap *= 2; }
	uint64_t newSize = newCap * sizeof(InstanceData);

	// unregister and delete old buffer if existing
	if (instanceResourceCUDA_)
	{
		if (!HandleErrorCUDA(
			cudaGraphicsUnregisterResource(instanceResourceCUDA_),
			"cudaGraphicsUnregisterResource() while growing particle instances"))
		{
			return false;
		}
		instanceResourceCUDA_ = nullptr;
	}
	if (vboInstancesCUDA_)
	{
		glDeleteBuffers(1, &vboInstancesCUDA_);
		vboInstancesCUDA_ = 0;
	}

	// create OpenGL instance buffer and register it with CUDA
	glCreateBuffers(1, &vboInstancesCUDA_);
	glNamedBufferData(
		vboInstancesCUDA_,
		static_cast<GLsizeiptr>(newSize),
		nullptr,
		GL_DYNAMIC_DRAW);

	if (!HandleErrorCUDA(
		cudaGraphicsGLRegisterBuffer(
			&instanceResourceCUDA_,
			vboInstancesCUDA_,
			cudaGraphicsRegisterFlagsWriteDiscard),
		"cudaGraphicsGLRegisterBuffer() for particle instances"))
	{
		glDeleteBuffers(1, &vboInstancesCUDA_);
		vboInstancesCUDA_ = 0;
		return false;
	}

	capacityCUDA_ = static_cast<uint32_t>(newCap);
	return true;
}

bool RenderingMgr::HandleErrorCUDA(
	const cudaError_t error,
	const char* operation)
{
	if (error == cudaSuccess) { return true; }

	SPDLOG_WARN(
		"{} failed: {} ({}).",
		operation,
		cudaGetErrorString(error),
		cudaGetErrorName(error));

	// disable CUDA fast path after first interop error
	interopReadyCUDA_ = false;
	if (visualFrameMailboxCUDA_) { visualFrameMailboxCUDA_->Disable(operation); }
	return false;
}
#endif

void RenderingMgr::BuildWireframeEdgesFromOBB(const Scene::OBB& box, std::vector<glm::vec3>& dst)
{
	const glm::vec3 p000 = box.origin;
	const glm::vec3 p100 = box.origin + box.uHat * box.uLen;
	const glm::vec3 p010 = box.origin + box.vHat * box.vLen;
	const glm::vec3 p110 = box.origin + box.uHat * box.uLen + box.vHat * box.vLen;
	const glm::vec3 p001 = box.origin + box.wHat * box.wLen;
	const glm::vec3 p101 = box.origin + box.uHat * box.uLen + box.wHat * box.wLen;
	const glm::vec3 p011 = box.origin + box.vHat * box.vLen + box.wHat * box.wLen;
	const glm::vec3 p111 = box.origin + box.uHat * box.uLen + box.vHat * box.vLen + box.wHat * box.wLen;

	auto edge = [&](const glm::vec3& a, const glm::vec3& b) { dst.push_back(a); dst.push_back(b); };

	// near (zMin)
	edge(p000, p100); edge(p100, p110); edge(p110, p010); edge(p010, p000);
	// far (zMax)
	edge(p001, p101); edge(p101, p111); edge(p111, p011); edge(p011, p001);
	// connectors
	edge(p000, p001); edge(p100, p101); edge(p110, p111); edge(p010, p011);
}

void RenderingMgr::EnableDebugOutput()
{
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(DebugCallback, nullptr);
}
