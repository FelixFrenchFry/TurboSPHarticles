// surface_sampling.h
#pragma once

// external
#include <cstdint>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>

// internal
#include "source/simulation/core/parameters_sim.h"
#include "triangle_surface.h"



namespace Scene
{
	// accepted surface position with temp metadata about source-triangle
	struct SurfaceSample
	{
		glm::vec3 pos{ 0.0f };
		glm::vec3 normal{ 0.0f };
		glm::vec3 barycentric{ 0.0f };
		uint32_t triangleIdx = 0;
	};

	// generated input and output counts
	struct SurfaceSamplingStats
	{
		uint32_t N_candidates = 0;
		uint32_t N_initialSamples = 0;
		uint32_t N_coverageRepairs = 0;
		uint32_t N_coverageProbes = 0;
		float surfaceArea = 0.0f;
	};

	struct SurfaceSamplingResult
	{
		std::vector<SurfaceSample> samples;
		SurfaceSamplingStats stats;
	};

	// performs deterministic Poisson surface sampling with extensive coverage constraints
	// (minimum/maximum spacing and coverage repair)
	class SurfaceSampler
	{
	public:

		SurfaceSampler(
			const TriangleSurfaceMesh& mesh,
			const Sim::SimParams& params);
		~SurfaceSampler();

		SurfaceSampler(const SurfaceSampler&) = delete;
		SurfaceSampler& operator=(const SurfaceSampler&) = delete;

		SurfaceSamplingResult Sample();

	private:

		// bounds temporary candidate and coverage probe buffers
		static constexpr uint32_t N_samplingBatch = 1000000u;

		void PreprocessMesh(const TriangleSurfaceMesh& mesh);
		void SelectPoissonSamples();
		void RepairCoverage();
		void ValidateSamples();

		struct Storage;
		std::unique_ptr<Storage> storage_;
		Sim::SimParams params_; // param snapshot used by sampling stages
	};

} // namespace Scene
