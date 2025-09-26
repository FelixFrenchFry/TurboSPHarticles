// rendering.h
//
// TODO: comment

#pragma once

// external
#include <vector>
#include <glad/glad.h>
#include <glm/glm.hpp>

// internal
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Vis/Core/parameters_vis.h"

// forward declarations



namespace Render
{
	// 32 bytes per rendered particle instance
	struct InstanceData
	{
		glm::vec4 posAndRad; // (x, y, z, radius)
		float scalar; // scalar value for color coding
		float pad[3]; // padding to multiple of 16 bytes
	};
	static_assert(sizeof(InstanceData) == 32, "Instance must be 32 bytes");

	// manages the rendering of particles and other objects
	// (instancing, shaders, buffers, draw calls, etc...)
	class RenderingMgr
	{
	public: // ==============================

		RenderingMgr(Sim::SimParamsMgr& simParamsMgr,
					 Vis::VisParamsMgr& visParamsMgr);

		// create GL resources (buffers, shaders, mesh)
		void Init(glm::ivec2 fbSize, float dpiScale);

		void Shutdown();

		// upload latest render-specific sim state to the GPU instance buffers
		// (positions required, scalar data for coloring optional)
		void UpdateParticles(const Vis::VisualSnapView& view,
							 const Vis::VisParams& params);

		// upload latest render-specific sim state to the GPU instance buffers
		void UpdateWireframes(const Vis::VisualSnapView& view,
							  const Vis::VisParams& params);

		// upload a new 256-entry RGBA colormap (assumed to be linear RGB)
		void UpdateColorMap(const std::vector<glm::vec4>& colorMap256);

		// draw the particles in the current frame (as imposter spheres)
		void Render(const Vis::RenderConfig& render, const Vis::CameraState& cam);

	private: // ==============================

		// parameter managers for making changes to parameters
		Sim::SimParamsMgr& simParamsMgr_;
		Vis::VisParamsMgr& visParamsMgr_;

		// addresses of GL objects (sphere impostors -> particles)
		GLuint vao_ = 0;					// vertex array object
		GLuint vboQuad_ = 0;				// quad for impostor sphere instances
		GLuint vboInstances_ = 0;			// instance data
		GLuint progSphereImpostors_ = 0;	// impostor sphere shader program
		GLuint texColorMap_ = 0;			// color lookup-table

		// addresses of GL objects (lines -> wireframes)
		GLuint vaoLines_ = 0;
		GLuint vboLines_ = 0;
		GLuint progLines_ = 0;
		GLint uniColorLoc_ = -1;

		// cached params
		GLuint uboParams_ = 0; // cached render params
		GLuint uboCamera_ = 0; // cached camera config

		// persistent map pointer
		unsigned char* mappedBase_ = nullptr;
		std::int64_t mappedSize_ = 0; // total bytes reserved
		std::int64_t capacity_ = 0; // number of instances capacity

		// draw state
		int64_t N_impostorInstances_ = 0;
		int64_t N_lineVertices_ = 0;

		// helpers
		void CreateSphereImpostorProgram();
		void CreateLineProgram();
		GLuint Compile(GLenum type, const char* src);
		GLuint Link(GLuint vertS, GLuint fragS);

		void CreateSphereImpostorVAOandBuffers();
		void CreateLineVAOandBuffers();

		void DestroySphereImpostorVAOandBuffers();
		void DestroyLineVAOandBuffers();

		void UpdateParamsUBO(const Vis::RenderConfig& render);
		void UpdateCameraUBO(const Vis::CameraState& cam);

		void EnsureInstanceCapacity(std::int64_t N_required);

		void BuildWireframeEdgesFromOBB(const Scene::OBB& box, std::vector<glm::vec3>& dst);

		// debug callback
		void EnableDebugOutput();
	};

} // namespace Render
