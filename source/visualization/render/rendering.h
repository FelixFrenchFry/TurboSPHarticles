// rendering.h
#pragma once

// external
#include <cstdint>
#include <vector>
#include <glad/glad.h>
#include <glm/glm.hpp>
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include <cuda_runtime_api.h>
#endif

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/visualization/core/parameters_vis.h"
#include "color_maps.h"



// forward declarations
#ifdef TURBOSPHARTICLES_WITH_CUDA
struct cudaGraphicsResource;

namespace Sync
{
	class VisualFrameMailboxCUDA;
	struct VisualFrameReadCUDA;
} // namespace Sync
#endif

namespace Render
{
	// 32 bytes per rendered particle instance
	struct InstanceData
	{
		float x;
		float y;
		float z;
		float radius;
		float scalar;
		float pad[3];
	};
	static_assert(sizeof(InstanceData) == 32, "Instance must be 32 bytes");

	// manages the rendering of particles and other objects
	// (instancing, shaders, buffers, draw calls, etc...)
	class RenderingMgr
	{
	public:

		// create GL resources (buffers, shaders, mesh)
		void Init();

		void Shutdown();

		// upload latest render-specific sim state to the GPU instance buffers
		// (positions required, scalar data for coloring optional)
		void UpdateParticles(const Vis::VisualSnapView& view,
							 const Sim::SimParams& simParams,
							 const Vis::VisParams& visParams);

	#ifdef TURBOSPHARTICLES_WITH_CUDA
		void SetVisualMailboxCUDA(Sync::VisualFrameMailboxCUDA* visualMailboxCUDA)
			{ visualFrameMailboxCUDA_ = visualMailboxCUDA; }

		bool UpdateParticlesCUDA(
			const Sync::VisualFrameReadCUDA& frame,
			const Sim::SimParams& simParams,
			const Vis::VisParams& visParams);
	#endif

		// upload boundary mesh particle instances from the latest visual snapshot
		void UpdateBoundaryMeshParticles(
			const Vis::VisualSnapView& view,
			const Sim::SimParams& simParams);

		// upload rigid body center-of-mass marker transforms
		void UpdateCentersOfMass(const Vis::VisualSnapView& view,
								 const Sim::SimParams& simParams);

		// upload latest render-specific sim state to the GPU instance buffers
		void UpdateWireframes(const Vis::VisualSnapView& view);

		// upload a new 256-entry RGB color map (assumed to be sRGB)
		void UpdateColorMap(const ColorMapData& colorMap);
		static const ColorMapData& GetColorMap_Turbo();
		static const ColorMapData& GetColorMap_Viridis();
		static const ColorMapData& GetColorMap_Inferno();
		static const ColorMapData& GetColorMap_Fast();
		static const ColorMapData& GetColorMap_Coolwarm();
		static const ColorMapData& GetColorMap_Blues();
		static const ColorMapData& GetColorMap_Aurora();
		static const ColorMapData& GetColorMap_Earth();

		// draw the particles in the current frame (as imposter spheres)
		void Render(const Vis::RenderConfig& render, const Vis::CameraState& cam);

	private:

		// addresses of GL objects (sphere impostors -> particles)
		GLuint vao_ = 0;                    // vertex array object
		GLuint vboQuad_ = 0;                // quad for impostor sphere instances
		GLuint vboInstances_ = 0;           // instance data
		GLuint progSphereImpostors_ = 0;    // impostor sphere shader program
		GLuint texColorMap_ = 0;            // color lookup-table
		Base::ColorMap colorMap_ = Base::ColorMap::Coolwarm; // currently uploaded color map

		// addresses of GL objects (sphere impostors -> boundary mesh particles)
		GLuint vaoMeshB_ = 0;               // vertex array object sharing the impostor quad
		GLuint vboMeshInstancesB_ = 0;      // instance data

		// addresses of GL objects (cuboids -> rigid body centers of mass)
		GLuint vaoCenterOfMass_ = 0;
		GLuint vboCenterOfMassCube_ = 0;
		GLuint vboCenterOfMassInstances_ = 0;
		GLuint progCenterOfMass_ = 0;
		GLint uniCenterOfMassXAxisColorLoc_ = -1;
		GLint uniCenterOfMassYAxisColorLoc_ = -1;
		GLint uniCenterOfMassZAxisColorLoc_ = -1;
		GLint uniCenterOfMassAlphaLoc_ = -1;

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
		uint64_t mappedSize_ = 0; // total bytes reserved
		uint32_t capacity_ = 0; // number of instances capacity

		// draw state
		uint32_t N_fluidImpostorInstances_ = 0;
		uint32_t N_meshImpostorInstancesB_ = 0;
		uint32_t N_rigidImpostorInstances_ = 0;
		uint32_t N_centerOfMassInstances_ = 0;
		uint64_t N_lineVertices_ = 0;

		// additional CUDA-OpenGL interop resources
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		Sync::VisualFrameMailboxCUDA* visualFrameMailboxCUDA_{ nullptr };
		cudaGraphicsResource* instanceResourceCUDA_{ nullptr };
		cudaStream_t renderStreamCUDA_{ nullptr };
		GLuint vboInstancesCUDA_ = 0;
		uint32_t capacityCUDA_ = 0;
		bool interopReadyCUDA_ = false;
	#endif

		// helpers
		void CreateSphereImpostorProgram();
		void CreateCenterOfMassProgram();
		void CreateLineProgram();
		GLuint Compile(GLenum type, const char* src, const char* shaderName);
		GLuint Link(GLuint vertS, GLuint fragS, const char* programName);

		void CreateSphereImpostorVAOandBuffers();
		void CreateBoundaryMeshImpostorVAOandBuffers();
		void CreateCenterOfMassVAOandBuffers();
		void CreateLineVAOandBuffers();

		void DestroySphereImpostorVAOandBuffers();
		void DestroyBoundaryMeshImpostorVAOandBuffers();
		void DestroyCenterOfMassVAOandBuffers();
		void DestroyLineVAOandBuffers();

		void UpdateParamsUBO(const Vis::RenderConfig& render);
		void UpdateCameraUBO(const Vis::CameraState& cam);

		void EnsureInstanceCapacity(uint32_t N_required);

		void BuildWireframeEdgesFromOBB(const Scene::OBB& box, std::vector<glm::vec3>& dst);

		// debug callback
		void EnableDebugOutput();

		// additional CUDA-OpenGL interop helpers
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		void InitParticleInteropCUDA();
		void ShutdownParticleInteropCUDA() noexcept;
		bool EnsureInstanceCapacityCUDA(uint32_t N_required);
		bool HandleErrorCUDA(cudaError_t error, const char* operation);
	#endif
	};

} // namespace Render
