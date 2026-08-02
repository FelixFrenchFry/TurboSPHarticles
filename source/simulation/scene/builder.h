// builder.h

#pragma once

// external
#include <array>
#include <cstdint>
#include <functional>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <nlohmann/json.hpp>
#include <string_view>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "source/simulation/scene/geometry/surface_sampling.h"
#include "source/simulation/sync/state_sim.h"



using json = nlohmann::json;
using ParticleKeepPredicate = std::function<bool(float u, float v, float w)>;

namespace Scene
{
	struct OBB
	{
		glm::vec3 origin; // near-left-bottom (xMin, yMin, zMin)
		glm::vec3 uHat;   // unit vec along x
		glm::vec3 vHat;   // unit vec along y
		glm::vec3 wHat;   // unit vec along z
		float uLen;		  // width (along uHat)
		float vLen;		  // height (along vHat)
		float wLen;		  // depth (along wHat)
	};

	struct OBBFace
	{
		glm::vec3 origin; // near-left-bottom (xMin, yMin, zMin)
		glm::vec3 uHat;   // unit vec along x
		glm::vec3 vHat;   // unit vec along y
		glm::vec3 nHat;   // face normal (outward)
		float uLen;		  // width (along uHat)
		float vLen;		  // height (along vHat)
	};

	class ScenarioMgr
	{
	public:

		ScenarioMgr() = default;

		// restricts fluid and boundary particle generation to the given world-space region,
		// distributed ranks build their own regions while particle IDs stay the global scenario indices
		void BeginBuildRegion(const glm::vec3& min, const glm::vec3& max);
		void EndBuildRegion();

		// unfiltered particle counts of the build region session, identical on every rank
		// because the generated particle set does not depend on the region
		uint32_t GetBuiltGlobalFluidParticleCount() const { return N_builtGlobalF_; }
		uint32_t GetBuiltGlobalBoundaryParticleCount() const { return N_builtGlobalB_; }

		// builds a box-shaped boundary container around a given OBB
		// (boundary particle layers grow outwards by default)
		void BuildBoundaryContainer_OBB(const OBB& box,
										Sync::UniversalSimState& dst,
										const Sim::SimParams& params);

		// builds a block of fluid particles inside a given OBB
		void BuildFluidBlock_OBB(const OBB& box,
								 Sync::UniversalSimState& dst,
								 const Sim::SimParams& params);

		// builds a block of fluid particles inside a given OBB,
		// but only keeps particles for which the given predicate is true
		void BuildFluidBlock_OBB_Filtered(const OBB& box,
										  Sync::UniversalSimState& dst,
										  const Sim::SimParams& params,
										  const ParticleKeepPredicate& keep);

		// TODO: builds a sphere of fluid particles inside a given hollow sphere of boundary particles

		// builds a hollow block of boundary particles inside a given OBB
		// (thickness given boundary layers param)
		void BuildBoundaryBlock_OBB_Hollow(const OBB& box,
										   Sync::UniversalSimState& dst,
										   const Sim::SimParams& params);

		// builds a hollow rigid body sampled by boundary particles
		uint32_t BuildRigidBody_OBB_Hollow(const OBB& box,
										   Sync::UniversalSimState& dst,
										   const Sim::SimParams& params,
										   float rigidDensity);

		// appends body-local surface samples to the existing rigid particle state
		uint32_t BuildRigidBody_FromSurfaceSamples(
			const std::vector<SurfaceSample>& bodyLocalSamples,
			const glm::vec3& centerOfMass,
			const glm::mat3& orientation,
			float mass,
			const glm::mat3& bodyInertiaInv,
			Sync::UniversalSimState& dst,
			const Sim::SimParams& params);

		// builds a local-to-world orientation from yaw, pitch, and roll angles
		glm::mat3 MakeOrientation_FromEulerAngles(
			float yawDeg,
			float pitchDeg,
			float rollDeg) const;

		OBB MakeOBB_FromAxes(glm::vec3 origin,
							 glm::vec3 uDir, float uLen,
							 glm::vec3 vDir, float vLen,
							 glm::vec3 wDir, float wLen);

		std::array<OBBFace, 6> inline GetFacesFromOBB(const OBB& box);

		static OBB ParseOBBFromJson(const json& j, ScenarioMgr& scenarioMgr);

	private:

		// tests a world-space particle position against the current build region
		// (always true while no build region is active)
		bool IsInsideBuildRegion(const glm::vec3& pos) const;

		// builds a slab of boundary particles on a given OBB face
		void BuildBoundaryLayersOnFace(const OBBFace& face,
									   Sync::UniversalSimState& dst,
									   const Sim::SimParams& params);

		// computes the non-uniform boundary contribution coefficients ("pseudo-mass") for boundary particles
		// (inspired by "Versatile Rigid-Fluid Coupling for Incompressible SPH" Akinci et al. 2012)
		static void ComputeBoundaryCoefficients(
			Sync::BoundaryParticles& boundaryParticles,
			const Sim::SimParams& params,
			uint32_t beginB,
			uint32_t endB);

		static void ComputeRigidCoefficients(
			Sync::RigidParticles& rigidParticles,
			const Sim::SimParams& params,
			uint32_t beginR,
			uint32_t endR);

		static void ComputeParticleCoefficients(
			const std::vector<float>& xPos,
			const std::vector<float>& yPos,
			const std::vector<float>& zPos,
			std::vector<float>& vol,
			std::vector<float>& psi,
			uint32_t N,
			const Sim::SimParams& params,
			uint32_t begin,
			uint32_t end);

		static inline float Jitter3D(uint64_t baseSeed, uint64_t key, float amp);

		static inline float StartOffset(float length, float spacing, int N, bool centerToFit = true);

		static inline uint32_t TilesCount(float length, float spacing);

		static inline uint32_t ParticleCountHollowBox(
			uint32_t N_u, uint32_t N_v, uint32_t N_w, uint32_t layers);

		// build region state, reset by every BeginBuildRegion() call
		glm::vec3 buildRegionMin_{ 0.0f };
		glm::vec3 buildRegionMax_{ 0.0f };
		bool hasBuildRegion_ = false;
		uint32_t N_builtGlobalF_ = 0; // total fluid particles generated, before the region filter
		uint32_t N_builtGlobalB_ = 0; // total boundary particles generated, before the region filter
	};

} // namespace Scene
