// rendering.cpp

// external
#include <algorithm>
#include <cassert>
#include <spdlog/spdlog.h>

// internal
#include "Source/Sim/Sync/state_sim.h"
#include "rendering.h"
#include "shader.h"



using namespace Render;

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
	CreateLineProgram();

	// create VAOs and buffers
	CreateSphereImpostorVAOandBuffers();
	CreateLineVAOandBuffers();

	// default grayscale colormap
	std::vector<glm::vec4> lut(256);
	for (int i = 0; i < 256; i++)
	{
		float t = static_cast<float>(i) / 255.0f;
		lut[i] = glm::vec4(t, t, t, 1.0f);
	}
	UpdateColorMap(lut);

	// enable alpha blending for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void RenderingMgr::Shutdown()
{
	if (progSphereImpostors_)
	{
		glDeleteProgram(progSphereImpostors_);
		progSphereImpostors_ = 0;
	}

	if (progLines_)
	{
		glDeleteProgram(progLines_);
		progLines_ = 0;
	}

	DestroySphereImpostorVAOandBuffers();
	DestroyLineVAOandBuffers();
}

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

	auto* out = reinterpret_cast<InstanceData*>(mappedBase_);
	const float fluidRad = simParams.particleSizeF * 0.5f;
	const float rigidRad =
		simParams.particleSizeB * 0.5f *
		std::max(visParams.render.rigidBoundaryRadiusScale, 0.0f);
	const bool hasScalar = (view.scalarModeF != Base::ColorCodingMode::None) &&
							(view.scalarDataF.size() == view.N_F);

	// write fluid particles instance data first
	for (uint32_t i = 0; i < N_fluidImpostorInstances_; i++)
	{
		const glm::vec3 pos = view.posF[i];
		out[i].posAndRad = glm::vec4(pos, fluidRad);
		out[i].scalar = hasScalar ? view.scalarDataF[i] : 0.0f;
	}

	// append rigid boundary instance data
	uint32_t dst = N_fluidImpostorInstances_;
	if (hasBodyIdB)
	{
		for (uint32_t i = 0; i < view.N_B; i++)
		{
			if (view.bodyIdB[i] == Sync::InvalidRigidBodyId) { continue; }

			const glm::vec3 pos = view.posB[i];
			out[dst].posAndRad = glm::vec4(pos, rigidRad);
			out[dst].scalar = 0.0f;
			dst++;
		}
	}

	// no need to flush, because buffer persistently mapped
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
		SPDLOG_WARN("Failed to update colormap, got {} instead of expected 256 entries",
			colorMap256.size());
		return;
	}

	glTextureSubImage1D(texColorMap_, 0, 0, 256, GL_RGBA, GL_FLOAT, 
		colorMap256.data());
}

void RenderingMgr::Render(const Vis::RenderConfig& render, const Vis::CameraState& cam)
{
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
			rigidRender.useScalar = false;

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
	GLuint vertS = Compile(GL_VERTEX_SHADER, VS_sphereImpostor);
	GLuint fragS = Compile(GL_FRAGMENT_SHADER, FS_sphereImpostor);

	if (vertS == 0) { SPDLOG_ERROR("Failed to compile vertex shader (VS_sphereImpostor)"); }
	if (fragS == 0) { SPDLOG_ERROR("Failed to compile fragment shader (FS_sphereImpostor)"); }

	progSphereImpostors_ = Link(vertS, fragS);
	if (!progSphereImpostors_)
	{
		SPDLOG_ERROR("Failed to create sphere impostor shader program");
		throw std::runtime_error("Failed to create sphere impostor shader program");
	}
}

// creates the shader program for rendering lines
void RenderingMgr::CreateLineProgram()
{
	GLuint vertS = Compile(GL_VERTEX_SHADER, VS_line);
	GLuint fragS = Compile(GL_FRAGMENT_SHADER, FS_line);

	if (vertS == 0) { SPDLOG_ERROR("Failed to compile vertex shader (VS_line)"); }
	if (fragS == 0) { SPDLOG_ERROR("Failed to compile fragment shader (FS_line)"); }

	progLines_ = Link(vertS, fragS);
	if (!progLines_)
	{
		SPDLOG_ERROR("Failed to create line shader program");
		throw std::runtime_error("Failed to create line shader program");
	}

	uniColorLoc_ = glGetUniformLocation(progLines_, "uColor");
	if (uniColorLoc_ < 0)
	{
		SPDLOG_WARN("Failed to find uniform location for 'uColor' in line shader");
	}
}

// compiles a shader of given type from source and returns its ID
GLuint RenderingMgr::Compile(GLenum type, const char* src)
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

		SPDLOG_ERROR("Shader compile error: {}", log);
		return 0;
	}

	return res;
}

// links a shader program given compiled vertex and fragment shaders and returns its ID
GLuint RenderingMgr::Link(GLuint vertS, GLuint fragS)
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

		SPDLOG_ERROR("Program link error: {}", log);
		return 0;
	}

	// cleanup
	glDetachShader(res, vertS);
	glDetachShader(res, fragS);
	glDeleteShader(vertS);
	glDeleteShader(fragS);

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
	const glm::vec4 p2(render.scalarMin, render.scalarMax, render.useScalar ? 1.0f : 0.0f, 0.0f);

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

	SPDLOG_WARN("GL debug type=0x{:x}, severity=0x{:x}, msg={}", type, severity, message);
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

	// unmap and delete old buffer if existing
	if (mappedBase_)
	{
		glUnmapNamedBuffer(vboInstances_);
		mappedBase_ = nullptr;
		mappedSize_ = 0;
	}

	glNamedBufferStorage(vboInstances_, newSize, nullptr, storageFlags);
	mappedBase_ = static_cast<unsigned char*>(
		glMapNamedBufferRange(vboInstances_, 0, newSize, mapFlags));
	if (!mappedBase_)
	{
		SPDLOG_ERROR("Failed to map instance buffer");
		throw std::runtime_error("Failed to map instance buffer");
	}

	mappedSize_ = newSize;
	capacity_ = newCap;

	// re-bind instance VBO to index 1
	glVertexArrayVertexBuffer(vao_, 1, vboInstances_, 0, sizeof(InstanceData));
}

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
