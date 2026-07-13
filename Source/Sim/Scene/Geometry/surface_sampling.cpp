// surface_sampling.cpp

// external
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <unordered_map>
#include <utility>
#include <vector>

// internal
#include "surface_sampling.h"



namespace
{
	// preprocessed triangle geometry shared by all sampling stages
	struct TriangleData
	{
		glm::vec3 v0{ 0.0f };
		glm::vec3 v1{ 0.0f };
		glm::vec3 v2{ 0.0f };
		glm::vec3 normal{ 0.0f };
		float area = 0.0f; // triangle area before cumulative prefix
		double cumulativeArea = 0.0;
		float longestEdge = 0.0f;
	};

	// integer coordinate used by the temporary sample spatial hash
	struct CellCoord
	{
		int64_t x = 0;
		int64_t y = 0;
		int64_t z = 0;

		bool operator==(const CellCoord&) const = default;
	};

	struct CellCoordHash
	{
		uint64_t operator()(const CellCoord& cell) const noexcept
		{
			uint64_t h = static_cast<uint64_t>(cell.x);
			h ^= static_cast<uint64_t>(cell.y)
				+ 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			h ^= static_cast<uint64_t>(cell.z)
				+ 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			return h;
		}
	};

	bool IsFinite(const glm::vec3& value)
	{
		return std::isfinite(value.x) &&
			std::isfinite(value.y) &&
			std::isfinite(value.z);
	}

	double Random01(std::mt19937_64& rng)
	{
		// explicit conversion keeps seeded candidates consistent across standard libraries
		return static_cast<double>(rng() >> 11) * 0x1.0p-53;
	}

	float DistanceEpsilon(const float distance)
	{
		return 32.0f * std::numeric_limits<float>::epsilon() *
			std::max(std::abs(distance), std::numeric_limits<float>::min());
	}

	CellCoord MakeCellCoord(const glm::vec3& pos, const float cellSizeInv)
	{
		const auto makeAxis = [cellSizeInv](const float value) -> int64_t
			{
				const long double scaled = std::floor(
					static_cast<long double>(value) * static_cast<long double>(cellSizeInv));
				constexpr int64_t cellLimit = std::numeric_limits<int64_t>::max() / 2;
				if (!std::isfinite(scaled) ||
					scaled < static_cast<long double>(-cellLimit) ||
					scaled > static_cast<long double>(cellLimit))
				{
					throw std::runtime_error("Surface sample grid coordinate exceeds int64 limits.");
				}
				return static_cast<int64_t>(scaled);
			};

		return CellCoord
		{
			.x = makeAxis(pos.x),
			.y = makeAxis(pos.y),
			.z = makeAxis(pos.z),
		};
	}

	float SurfaceDistance(
		const Scene::SurfaceSample& sampleA,
		const Scene::SurfaceSample& sampleB)
	{
		return glm::distance(sampleA.pos, sampleB.pos);
	}

	Scene::SurfaceSample MakeSurfacePoint(
		const TriangleData& triangle,
		const uint32_t triangleIdx,
		const glm::vec3& barycentric)
	{
		return Scene::SurfaceSample
		{
			.pos = triangle.v0 * barycentric.x
				+ triangle.v1 * barycentric.y
				+ triangle.v2 * barycentric.z,
			.normal = triangle.normal,
			.barycentric = barycentric,
			.triangleIdx = triangleIdx,
		};
	}

	// temporary spatial index for nearby accepted sample queries
	class SampleSpatialHash
	{
	public:
		SampleSpatialHash(
			const float cellSize,
			const std::vector<Scene::SurfaceSample>& samples)
			:
			cellSize_(cellSize),
			cellSizeInv_(1.0f / cellSize),
			samples_(samples)
		{}

		void Reserve(const uint32_t N)
		{
			cellBuckets_.reserve(static_cast<uint64_t>(N) * 2ull);
		}

		void Insert(const uint32_t sampleIdx)
		{
			cellBuckets_[MakeCellCoord(samples_[sampleIdx].pos, cellSizeInv_)].push_back(sampleIdx);
		}
		template<typename Func>
		bool AnyNearby(const glm::vec3& pos, const float queryRadius, Func&& func) const
		{
			const CellCoord baseCell = MakeCellCoord(pos, cellSizeInv_);
			const double cellQueryReachD = std::ceil(
				static_cast<double>(queryRadius) / static_cast<double>(cellSize_));
			if (!std::isfinite(cellQueryReachD) ||
				cellQueryReachD >= std::numeric_limits<int32_t>::max())
			{
				throw std::runtime_error("Surface sample grid query exceeds int32 limits.");
			}
			const int32_t cellQueryReach = std::max<int32_t>(
				1,
				static_cast<int32_t>(cellQueryReachD));

			for (int32_t dz = -cellQueryReach; dz <= cellQueryReach; dz++)
			{
				for (int32_t dy = -cellQueryReach; dy <= cellQueryReach; dy++)
				{
					for (int32_t dx = -cellQueryReach; dx <= cellQueryReach; dx++)
					{
						const CellCoord cell
						{
							.x = baseCell.x + dx,
							.y = baseCell.y + dy,
							.z = baseCell.z + dz,
						};
						const auto it = cellBuckets_.find(cell);
						if (it == cellBuckets_.end()) { continue; }

						for (const uint32_t sampleIdx : it->second)
						{
							if (func(sampleIdx)) { return true; }
						}
					}
				}
			}

			return false;
		}

	private:
		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		const std::vector<Scene::SurfaceSample>& samples_;
		std::unordered_map<CellCoord, std::vector<uint32_t>, CellCoordHash> cellBuckets_;
	};

} // namespace anonymous

namespace Scene

{
	// owns preprocessed mesh data and temporary sampling buffers
	struct SurfaceSampler::Storage
	{
		double surfaceArea = 0.0;
		std::vector<TriangleData> triangles;
		std::vector<SurfaceSample> candidates;
		std::vector<SurfaceSample> probes;
		std::vector<uint32_t> probeBegin; // first coverage probe for each triangle
		std::vector<SurfaceSample> samples;
		SurfaceSamplingStats stats;
	};

	void SurfaceSampler::PreprocessMesh(const TriangleSurfaceMesh& mesh)
	{
		const Sim::SimParams& params = params_;
		auto& triangles = storage_->triangles;
		double& surfaceArea = storage_->surfaceArea;

		// validate mesh structure and count limits
		if (mesh.vertices.empty() || mesh.triangles.empty())
		{
			throw std::runtime_error("Cannot sample an empty triangle surface.");
		}
		if (mesh.vertices.size() > std::numeric_limits<uint32_t>::max() ||
			mesh.triangles.size() > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Triangle surface vertex or triangle count exceeds uint32 limits.");
		}

		// validate SPH particle scales used by the sampling policy
		if (!std::isfinite(params.particleSizeF) || params.particleSizeF <= 0.0f ||
			!std::isfinite(params.particleSizeB) || params.particleSizeB <= 0.0f)
		{
			throw std::runtime_error("Surface sampling requires finite positive particle sizes.");
		}

		// prevent boundary sampling from becoming coarser than the fluid
		const float particleSizeEpsilon = DistanceEpsilon(params.particleSizeF);
		if (params.surfaceSamplingMaxDistance > params.particleSizeF + particleSizeEpsilon)
		{
			throw std::runtime_error("Rigid surface sampling is coarser than the fluid spacing.");
		}

		// validate sampling distance constraints
		if (!std::isfinite(params.surfaceSamplingMinDistance) || params.surfaceSamplingMinDistance <= 0.0f)
		{
			throw std::runtime_error("Surface sample minimum distance must be finite and positive.");
		}
		if (!std::isfinite(params.surfaceSamplingMaxDistance) || params.surfaceSamplingMaxDistance < params.surfaceSamplingMinDistance)
		{
			throw std::runtime_error("Surface sample maximum distance must be finite and not smaller than the minimum distance.");
		}
		if (!std::isfinite(params.surfaceSamplingCoverageProbeSpacing) || params.surfaceSamplingCoverageProbeSpacing <= 0.0f)
		{
			throw std::runtime_error("Surface coverage probe spacing must be finite and positive.");
		}

		const float distanceEpsilon = DistanceEpsilon(params.surfaceSamplingMaxDistance);
		if (params.surfaceSamplingMaxDistance + distanceEpsilon <
			params.surfaceSamplingMinDistance + params.surfaceSamplingCoverageProbeSpacing)
		{
			throw std::runtime_error("Surface coverage maximum distance must be at least the minimum distance plus the probe spacing.");
		}

		// validate candidate controls
		if (params.surfaceSamplingCandidateMultiplier == 0 || params.surfaceSamplingMaxTrialsPerCell == 0)
		{
			throw std::runtime_error("Surface sampling candidate multiplier and trials per cell must be positive.");
		}

		// validate source positions before computing triangle geometry
		for (const glm::vec3& vertex : mesh.vertices)
		{
			if (!IsFinite(vertex))
			{
				throw std::runtime_error("Triangle surface contains a non-finite vertex.");
			}
		}

		// validate triangle vertex references
		for (const glm::uvec3& indices : mesh.triangles)
		{
			if (indices.x >= mesh.vertices.size() ||
				indices.y >= mesh.vertices.size() ||
				indices.z >= mesh.vertices.size())
			{
				throw std::runtime_error("Triangle surface contains an invalid vertex index.");
			}
		}

		// precompute triangle geometry
		triangles.clear();
		triangles.resize(mesh.triangles.size());
		surfaceArea = 0.0;

		// compute per-triangle geometry in parallel
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(mesh.triangles.size())),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t triangleIdx = r.begin(); triangleIdx != r.end(); triangleIdx++)
				{
					const glm::uvec3 indices = mesh.triangles[triangleIdx];
					const glm::vec3 v0 = mesh.vertices[indices.x];
					const glm::vec3 v1 = mesh.vertices[indices.y];
					const glm::vec3 v2 = mesh.vertices[indices.z];
					const glm::vec3 cross = glm::cross(v1 - v0, v2 - v0);
					const float crossLength = glm::length(cross);
					const float area = 0.5f * crossLength;
					const glm::vec3 normal = std::isfinite(area) && area > 0.0f
						? cross / crossLength
						: glm::vec3{ 0.0f };

					triangles[triangleIdx] = TriangleData
					{
						.v0 = v0,
						.v1 = v1,
						.v2 = v2,
						.normal = normal,
						.area = area,
						.cumulativeArea = 0.0,
						.longestEdge = std::max(
							glm::distance(v0, v1),
							std::max(glm::distance(v1, v2), glm::distance(v2, v0))),
					};
				}
			},
			tbb::auto_partitioner{}
		);

		// assign cumulative areas in stable triangle order
		for (uint32_t triangleIdx = 0;
			triangleIdx < static_cast<uint32_t>(triangles.size());
			triangleIdx++)
		{
			const float area = triangles[triangleIdx].area;
			if (!std::isfinite(area) || area <= 0.0f)
			{
				throw std::runtime_error("Triangle surface contains a degenerate triangle.");
			}

			surfaceArea += static_cast<double>(area);
			if (!std::isfinite(surfaceArea) ||
				surfaceArea > std::numeric_limits<float>::max())
			{
				throw std::runtime_error("Triangle surface area exceeds float limits.");
			}

			triangles[triangleIdx].cumulativeArea = surfaceArea;
		}

	}

	void SurfaceSampler::GeneratePoissonCandidates()
	{
		const auto& triangles = storage_->triangles;
		const double surfaceArea = storage_->surfaceArea;
		const Sim::SimParams& params = params_;
		auto& candidates = storage_->candidates;
		uint32_t& candidateCount = storage_->stats.candidateCount;

		// area-weighted candidates with uniform barycentric positions
		const double estimatedSampleCount = std::ceil(
			surfaceArea /
			(static_cast<double>(params.surfaceSamplingMinDistance) * static_cast<double>(params.surfaceSamplingMinDistance)));
		const double candidateCountD = estimatedSampleCount * params.surfaceSamplingCandidateMultiplier;
		if (!std::isfinite(candidateCountD) ||
			candidateCountD > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Surface candidate count exceeds uint32 limits.");
		}

		candidateCount = std::max<uint32_t>(1, static_cast<uint32_t>(candidateCountD));
		candidates.resize(candidateCount);
		std::vector<double> areaTargets(candidateCount);
		std::vector<float> barycentricScales(candidateCount);
		std::vector<float> barycentricOffsets(candidateCount);

		// generate random inputs serially to preserve the seeded candidate stream
		std::mt19937_64 rng(params.surfaceSamplingSeed);
		for (uint32_t candidateIdx = 0; candidateIdx < candidateCount; candidateIdx++)
		{
			areaTargets[candidateIdx] = Random01(rng) * surfaceArea;
			barycentricScales[candidateIdx] = std::sqrt(
				static_cast<float>(Random01(rng)));
			barycentricOffsets[candidateIdx] = static_cast<float>(Random01(rng));
		}

		// resolve candidates independently into preallocated slots
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, candidateCount),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t candidateIdx = r.begin(); candidateIdx != r.end(); candidateIdx++)
				{
					// choose a triangle proportional to its surface area
					const auto triangleIt = std::lower_bound(
						triangles.begin(),
						triangles.end(),
						areaTargets[candidateIdx],
						[](const TriangleData& triangle, const double target)
						{
							return triangle.cumulativeArea < target;
						});
					const uint32_t triangleIdx = triangleIt == triangles.end()
						? static_cast<uint32_t>(triangles.size() - 1)
						: static_cast<uint32_t>(triangleIt - triangles.begin());

					// sample a uniform position on the selected triangle
					const float b1 = barycentricScales[candidateIdx] *
						(1.0f - barycentricOffsets[candidateIdx]);
					const glm::vec3 barycentric
					{
						1.0f - barycentricScales[candidateIdx],
						b1,
						barycentricScales[candidateIdx] - b1,
					};
					candidates[candidateIdx] = MakeSurfacePoint(
						triangles[triangleIdx],
						triangleIdx,
						barycentric);
				}
			},
			tbb::auto_partitioner{}
		);

	}

	SurfaceSampler::SurfaceSampler(
		const TriangleSurfaceMesh& mesh,
		const Sim::SimParams& params)
		:
		storage_(std::make_unique<Storage>())
		, params_(params)
	{
		// temporary sampling timing
		const auto t0 = std::chrono::steady_clock::now();
		PreprocessMesh(mesh);
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[surface sampling] PreprocessMesh: "
			<< std::chrono::duration<double, std::milli>(t1 - t0).count()
			<< " ms" << std::endl;
	}

	SurfaceSampler::~SurfaceSampler() = default;

	void SurfaceSampler::SelectPoissonSamples()
	{
		const Sim::SimParams& params = params_;
		const auto& candidates = storage_->candidates;
		auto& samples = storage_->samples;

		// select candidates with a minimum surface distance
		samples.reserve(candidates.size());
		SampleSpatialHash spatialHash(params.surfaceSamplingMinDistance, samples);
		spatialHash.Reserve(static_cast<uint32_t>(candidates.size()));

		const float cellSizeInv = 1.0f / params.surfaceSamplingMinDistance;

		// bound candidate attempts per spatial cell
		std::unordered_map<CellCoord, uint32_t, CellCoordHash> trialsPerCell;
		trialsPerCell.reserve(candidates.size());

		for (const SurfaceSample& candidate : candidates)
		{
			const CellCoord cell = MakeCellCoord(candidate.pos, cellSizeInv);
			uint32_t& trialCount = trialsPerCell[cell];
			if (trialCount >= params.surfaceSamplingMaxTrialsPerCell) { continue; }
			trialCount++;

			// reject candidates inside an existing Poisson disk
			const bool hasDistanceConflict = spatialHash.AnyNearby(
				candidate.pos,
				params.surfaceSamplingMinDistance,
				[&](const uint32_t sampleIdx)
				{
					return SurfaceDistance(
						candidate,
						samples[sampleIdx]) < params.surfaceSamplingMinDistance;
				});
			if (hasDistanceConflict) { continue; }

			const uint32_t sampleIdx = static_cast<uint32_t>(samples.size());
			samples.push_back(candidate);
			spatialHash.Insert(sampleIdx);
		}

		storage_->stats.initialSampleCount = static_cast<uint32_t>(samples.size());
	}

	void SurfaceSampler::GenerateCoverageProbes()
	{
		const auto& triangles = storage_->triangles;
		const Sim::SimParams& params = params_;
		auto& probes = storage_->probes;

		// barycentric lattices cover triangle vertices, edges, and interiors
		// first pass counts probes and validates allocation limits
		uint64_t probeCount = 0;
		storage_->probeBegin.resize(triangles.size());
		for (uint32_t triangleIdx = 0;
			triangleIdx < static_cast<uint32_t>(triangles.size());
			triangleIdx++)
		{
			const TriangleData& triangle = triangles[triangleIdx];
			const double subdivisionCountD = std::ceil(
				static_cast<double>(triangle.longestEdge) /
				static_cast<double>(params.surfaceSamplingCoverageProbeSpacing));
			if (!std::isfinite(subdivisionCountD) ||
				subdivisionCountD > std::numeric_limits<uint32_t>::max())
			{
				throw std::runtime_error("Surface probe subdivision count exceeds uint32 limits.");
			}

			const uint64_t subdivisionCount = std::max<uint64_t>(
				1,
				static_cast<uint64_t>(subdivisionCountD));
			const uint64_t factorA = subdivisionCount + 1ull;
			const uint64_t factorB = subdivisionCount + 2ull;
			if (factorA > std::numeric_limits<uint64_t>::max() / factorB)
			{
				throw std::runtime_error("Surface coverage probe count exceeds uint64 limits.");
			}

			const uint64_t triangleProbeCount = factorA * factorB / 2ull;
			if (triangleProbeCount > std::numeric_limits<uint32_t>::max() - probeCount)
			{
				throw std::runtime_error("Surface coverage probe count exceeds uint32 limits.");
			}
			storage_->probeBegin[triangleIdx] = static_cast<uint32_t>(probeCount);
			probeCount += triangleProbeCount;
		}

		probes.resize(static_cast<uint32_t>(probeCount));

		// second pass generates probes in stable triangle and lattice order
		const auto& probeBegin = storage_->probeBegin;
		tbb::parallel_for(
			tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(triangles.size())),
			[&](const tbb::blocked_range<uint32_t>& r)
			{
				for (uint32_t triangleIdx = r.begin(); triangleIdx != r.end(); triangleIdx++)
				{
					const TriangleData& triangle = triangles[triangleIdx];
					const uint32_t subdivisionCount = std::max<uint32_t>(
						1,
						static_cast<uint32_t>(std::ceil(
							triangle.longestEdge / params.surfaceSamplingCoverageProbeSpacing)));
					const float subdivisionCountInv = 1.0f / static_cast<float>(subdivisionCount);
					uint32_t probeIdx = probeBegin[triangleIdx];

					for (uint32_t i = 0; i <= subdivisionCount; i++)
					{
						for (uint32_t j = 0; j <= subdivisionCount - i; j++)
						{
							const uint32_t k = subdivisionCount - i - j;
							const glm::vec3 barycentric
							{
								static_cast<float>(i) * subdivisionCountInv,
								static_cast<float>(j) * subdivisionCountInv,
								static_cast<float>(k) * subdivisionCountInv,
							};
							probes[probeIdx] = MakeSurfacePoint(
								triangle,
								triangleIdx,
								barycentric);
							probeIdx++;
						}
					}
				}
			},
			tbb::auto_partitioner{}
		);

		storage_->stats.coverageProbeCount = static_cast<uint32_t>(probes.size());
	}

	void SurfaceSampler::RepairCoverage()
	{
		const auto& probes = storage_->probes;
		const Sim::SimParams& params = params_;
		auto& samples = storage_->samples;

		// fill uncovered probes while preserving the minimum sample distance
		// rebuild the spatial hash from the initially accepted samples
		SampleSpatialHash spatialHash(params.surfaceSamplingMinDistance, samples);
		const uint64_t maxSampleCount = static_cast<uint64_t>(samples.size()) + probes.size();
		if (maxSampleCount > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Repaired surface sample count exceeds uint32 limits.");
		}
		spatialHash.Reserve(static_cast<uint32_t>(maxSampleCount));
		for (uint32_t sampleIdx = 0; sampleIdx < static_cast<uint32_t>(samples.size()); sampleIdx++)
		{
			spatialHash.Insert(sampleIdx);
		}

		// reserve probe spacing as the continuous coverage error budget
		const float probeTargetDistance = std::max(
			params.surfaceSamplingMinDistance,
			params.surfaceSamplingMaxDistance - params.surfaceSamplingCoverageProbeSpacing);
		uint32_t& repairCount = storage_->stats.coverageRepairCount;
		repairCount = 0;

		const uint32_t N_probes = static_cast<uint32_t>(probes.size());
		constexpr uint32_t N_probeBatch = 65536u; // bound temporary coverage flags
		std::vector<uint8_t> needsRepair(N_probeBatch, 0u);

		for (uint32_t probeBegin = 0; probeBegin < N_probes;)
		{
			const uint32_t N_batch = std::min(
				N_probeBatch,
				N_probes - probeBegin);

			// query one probe batch in parallel against the read-only sample hash
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_batch),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t batchIdx = r.begin(); batchIdx != r.end(); batchIdx++)
					{
						const SurfaceSample& probe = probes[probeBegin + batchIdx];
						const bool isCovered = spatialHash.AnyNearby(
							probe.pos,
							probeTargetDistance,
							[&](const uint32_t sampleIdx)
							{
								return SurfaceDistance(
									probe,
									samples[sampleIdx]) <= probeTargetDistance;
							});
						needsRepair[batchIdx] = static_cast<uint8_t>(!isCovered);
					}
				},
				tbb::auto_partitioner{}
			);

			// recheck and insert repairs in stable probe order
			for (uint32_t batchIdx = 0; batchIdx < N_batch; batchIdx++)
			{
				if (needsRepair[batchIdx] == 0u) { continue; }

				const SurfaceSample& probe = probes[probeBegin + batchIdx];
				const bool isCovered = spatialHash.AnyNearby(
					probe.pos,
					probeTargetDistance,
					[&](const uint32_t sampleIdx)
					{
						return SurfaceDistance(
							probe,
							samples[sampleIdx]) <= probeTargetDistance;
					});
				if (isCovered) { continue; }
				if (samples.size() >= std::numeric_limits<uint32_t>::max())
				{
					throw std::runtime_error("Repaired surface sample count exceeds uint32 limits.");
				}

				const uint32_t sampleIdx = static_cast<uint32_t>(samples.size());
				samples.push_back(probe);
				spatialHash.Insert(sampleIdx);
				repairCount++;
			}

			probeBegin += N_batch;
		}
	}

	void SurfaceSampler::ValidateSamples()
	{
		const auto& triangles = storage_->triangles;
		const auto& samples = storage_->samples;

		// validate the final sample count and generated sample data
		if (samples.empty())
		{
			throw std::runtime_error("Surface sampling produced no samples.");
		}
		if (samples.size() > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Surface sample count exceeds uint32 limits.");
		}

		// validate generated sample references and coordinates
		for (const SurfaceSample& sample : samples)
		{
			if (sample.triangleIdx >= triangles.size())
			{
				throw std::runtime_error("Surface sample references an invalid triangle.");
			}
			if (!IsFinite(sample.pos))
			{
				throw std::runtime_error("Surface sampling produced a non-finite sample position.");
			}
			if (!IsFinite(sample.normal))
			{
				throw std::runtime_error("Surface sampling produced a non-finite sample normal.");
			}
			if (!IsFinite(sample.barycentric))
			{
				throw std::runtime_error("Surface sampling produced non-finite barycentric coordinates.");
			}
		}
	}

	SurfaceSamplingResult SurfaceSampler::Sample()
	{
		storage_->stats = {};
		storage_->stats.surfaceArea = static_cast<float>(storage_->surfaceArea);
		storage_->candidates.clear();
		storage_->probes.clear();
		storage_->samples.clear();
		const auto tSampleStart = std::chrono::steady_clock::now();
		auto tStageStart = tSampleStart;
		double sampleTotalMs = 0.0;
		double stageTimeMs = 0.0;

		// generate and select the initial Poisson sample set
		GeneratePoissonCandidates();
		auto tStageEnd = std::chrono::steady_clock::now();
		stageTimeMs = std::chrono::duration<double, std::milli>(
			tStageEnd - tStageStart).count();
		sampleTotalMs += stageTimeMs;
		std::cout << "[surface sampling] GeneratePoissonCandidates: "
			<< stageTimeMs
			<< " ms" << std::endl;
		tStageStart = tStageEnd;

		SelectPoissonSamples();
		tStageEnd = std::chrono::steady_clock::now();
		stageTimeMs = std::chrono::duration<double, std::milli>(
			tStageEnd - tStageStart).count();
		sampleTotalMs += stageTimeMs;
		std::cout << "[surface sampling] SelectPoissonSamples: "
			<< stageTimeMs
			<< " ms" << std::endl;
		tStageStart = tStageEnd;

		// detect and repair remaining surface coverage gaps
		GenerateCoverageProbes();
		tStageEnd = std::chrono::steady_clock::now();
		stageTimeMs = std::chrono::duration<double, std::milli>(
			tStageEnd - tStageStart).count();
		sampleTotalMs += stageTimeMs;
		std::cout << "[surface sampling] GenerateCoverageProbes: "
			<< stageTimeMs
			<< " ms" << std::endl;
		tStageStart = tStageEnd;

		RepairCoverage();
		tStageEnd = std::chrono::steady_clock::now();
		stageTimeMs = std::chrono::duration<double, std::milli>(
			tStageEnd - tStageStart).count();
		sampleTotalMs += stageTimeMs;
		std::cout << "[surface sampling] RepairCoverage: "
			<< stageTimeMs
			<< " ms" << std::endl;
		tStageStart = tStageEnd;

		ValidateSamples();
		tStageEnd = std::chrono::steady_clock::now();
		stageTimeMs = std::chrono::duration<double, std::milli>(
			tStageEnd - tStageStart).count();
		sampleTotalMs += stageTimeMs;
		std::cout << "[surface sampling] ValidateSamples: "
			<< stageTimeMs
			<< " ms" << std::endl;
		std::cout << "[surface sampling] Sample total: "
			<< sampleTotalMs
			<< " ms" << std::endl;

		SurfaceSamplingResult result;
		result.stats = storage_->stats;
		result.samples = std::move(storage_->samples);
		return result;
	}

} // namespace Scene
