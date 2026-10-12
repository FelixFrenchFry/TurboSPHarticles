// surface_sampling.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
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
		const glm::vec3& sampleA,
		const glm::vec3& sampleB)
	{
		return glm::distance(sampleA, sampleB);
	}

	glm::vec3 MakeSurfacePoint(
		const TriangleData& triangle,
		const glm::vec3& barycentric)
	{
		return triangle.v0 * barycentric.x
			+ triangle.v1 * barycentric.y
			+ triangle.v2 * barycentric.z;
	}

	// temporary spatial index for nearby accepted sample queries
	class SampleSpatialHash
	{
	public:
		SampleSpatialHash(
			const float cellSize,
			const std::vector<glm::vec3>& acceptedSamples)
			:
			cellSize_(cellSize),
			cellSizeInv_(1.0f / cellSize),
			acceptedSamples_(acceptedSamples)
		{}

		void Reserve(const uint32_t N)
		{
			cellBuckets_.reserve(N);
		}

		void Insert(const uint32_t sampleIdx)
		{
			cellBuckets_[MakeCellCoord(acceptedSamples_[sampleIdx], cellSizeInv_)].push_back(sampleIdx);
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
		const std::vector<glm::vec3>& acceptedSamples_;
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
		std::vector<glm::vec3> acceptedSamples;
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
		if (!std::isfinite(params.particleSizeF) || params.particleSizeF <= 0.0f)
		{
			throw std::runtime_error("Surface sampling requires a finite positive fluid particle size.");
		}

		// prevent boundary sampling from becoming coarser than the fluid
		const float particleSizeEpsilon = DistanceEpsilon(params.particleSizeF);
		if (maxDistance_ > params.particleSizeF + particleSizeEpsilon)
		{
			throw std::runtime_error("Surface sampling is coarser than the fluid spacing.");
		}

		// validate sampling distance constraints
		if (!std::isfinite(minDistance_) || minDistance_ <= 0.0f)
		{
			throw std::runtime_error("Surface sample minimum distance must be finite and positive.");
		}
		if (!std::isfinite(maxDistance_) || maxDistance_ < minDistance_)
		{
			throw std::runtime_error("Surface sample maximum distance must be finite and not smaller than the minimum distance.");
		}
		if (!std::isfinite(coverageProbeSpacing_) || coverageProbeSpacing_ <= 0.0f)
		{
			throw std::runtime_error("Surface coverage probe spacing must be finite and positive.");
		}

		const float distanceEpsilon = DistanceEpsilon(maxDistance_);
		if (maxDistance_ + distanceEpsilon <
			minDistance_ + coverageProbeSpacing_)
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

					triangles[triangleIdx] = TriangleData
					{
						.v0 = v0,
						.v1 = v1,
						.v2 = v2,
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

	void SurfaceSampler::SelectPoissonSamples()
	{
		const auto& triangles = storage_->triangles;
		const double surfaceArea = storage_->surfaceArea;
		const Sim::SimParams& params = params_;
		auto& acceptedSamples = storage_->acceptedSamples;
		uint32_t& N_candidates = storage_->stats.N_candidates;

		// derive candidate and estimated sample counts from surface area
		const double N_estimatedSamplesD = std::ceil(
			surfaceArea /
			(static_cast<double>(minDistance_) * static_cast<double>(minDistance_)));
		const double N_candidatesD =
			N_estimatedSamplesD * params.surfaceSamplingCandidateMultiplier;
		if (!std::isfinite(N_candidatesD) ||
			N_candidatesD > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Surface candidate count exceeds uint32 limits.");
		}

		N_candidates = std::max<uint32_t>(1, static_cast<uint32_t>(N_candidatesD));
		const uint32_t N_estimatedSamples = std::max<uint32_t>(
			1,
			static_cast<uint32_t>(N_estimatedSamplesD));

		// reserve persistent selection state from estimated sample count
		acceptedSamples.reserve(N_estimatedSamples);
		SampleSpatialHash spatialHash(minDistance_, acceptedSamples);
		spatialHash.Reserve(N_estimatedSamples);
		std::unordered_map<CellCoord, uint32_t, CellCoordHash> trialsPerCell;
		trialsPerCell.reserve(N_estimatedSamples);
		const float cellSizeInv = 1.0f / minDistance_;

		// generate and select candidate batches in stable global order
		const uint32_t N_candidateBatch = std::min(N_samplingBatch, N_candidates);
		std::vector<glm::vec3> candidates(N_candidateBatch);
		std::vector<double> areaTargets(N_candidateBatch);
		std::vector<float> barycentricScales(N_candidateBatch);
		std::vector<float> barycentricOffsets(N_candidateBatch);
		std::mt19937_64 rng(params.surfaceSamplingSeed);

		for (uint32_t candidateBegin = 0; candidateBegin < N_candidates;)
		{
			const uint32_t N_batch = std::min(
				N_candidateBatch,
				N_candidates - candidateBegin);

			// generate random inputs serially to preserve the seeded candidate stream
			for (uint32_t batchIdx = 0; batchIdx < N_batch; batchIdx++)
			{
				areaTargets[batchIdx] = Random01(rng) * surfaceArea;
				barycentricScales[batchIdx] = std::sqrt(
					static_cast<float>(Random01(rng)));
				barycentricOffsets[batchIdx] = static_cast<float>(Random01(rng));
			}

			// resolve candidates independently into preallocated slots
			tbb::parallel_for(
				tbb::blocked_range<uint32_t>(0, N_batch),
				[&](const tbb::blocked_range<uint32_t>& r)
				{
					for (uint32_t batchIdx = r.begin(); batchIdx != r.end(); batchIdx++)
					{
						// choose a triangle proportional to its surface area
						const auto triangleIt = std::lower_bound(
							triangles.begin(),
							triangles.end(),
							areaTargets[batchIdx],
							[](const TriangleData& triangle, const double target)
							{
								return triangle.cumulativeArea < target;
							});
						const uint32_t triangleIdx = triangleIt == triangles.end()
							? static_cast<uint32_t>(triangles.size() - 1)
							: static_cast<uint32_t>(triangleIt - triangles.begin());

						// sample a uniform position on the selected triangle
						const float b1 = barycentricScales[batchIdx] *
							(1.0f - barycentricOffsets[batchIdx]);
						const glm::vec3 barycentric
						{
							1.0f - barycentricScales[batchIdx],
							b1,
							barycentricScales[batchIdx] - b1,
						};
						candidates[batchIdx] = MakeSurfacePoint(
							triangles[triangleIdx],
							barycentric);
					}
				},
				tbb::auto_partitioner{}
			);

			// select candidates with a minimum surface distance
			for (uint32_t batchIdx = 0; batchIdx < N_batch; batchIdx++)
			{
				const glm::vec3& candidate = candidates[batchIdx];
				const CellCoord cell = MakeCellCoord(candidate, cellSizeInv);
				uint32_t& N_trials = trialsPerCell[cell];
				if (N_trials >= params.surfaceSamplingMaxTrialsPerCell) { continue; }
				N_trials++;

				const bool hasDistanceConflict = spatialHash.AnyNearby(
					candidate,
					minDistance_,
					[&](const uint32_t sampleIdx)
					{
						return SurfaceDistance(
							candidate,
							acceptedSamples[sampleIdx]) < minDistance_;
					});
				if (hasDistanceConflict) { continue; }

				const uint32_t sampleIdx = static_cast<uint32_t>(acceptedSamples.size());
				acceptedSamples.push_back(candidate);
				spatialHash.Insert(sampleIdx);
			}

			candidateBegin += N_batch;
		}

		storage_->stats.N_initialSamples = static_cast<uint32_t>(acceptedSamples.size());
	}

	SurfaceSampler::SurfaceSampler(
		const TriangleSurfaceMesh& mesh,
		const Sim::SimParams& params,
		const float minDistance,
		const float maxDistance,
		const float coverageProbeSpacing)
		:
		storage_(std::make_unique<Storage>())
		, params_(params)
		, minDistance_(minDistance)
		, maxDistance_(maxDistance)
		, coverageProbeSpacing_(coverageProbeSpacing)
	{
		PreprocessMesh(mesh);
	}

	SurfaceSampler::~SurfaceSampler() = default;

	void SurfaceSampler::RepairCoverage()
	{
		const auto& triangles = storage_->triangles;
		auto& acceptedSamples = storage_->acceptedSamples;

		// count barycentric coverage probes and validate allocation limits
		uint64_t N_probes = 0;
		for (const TriangleData& triangle : triangles)
		{
			const double N_subdivisionsD = std::ceil(
				static_cast<double>(triangle.longestEdge) /
				static_cast<double>(coverageProbeSpacing_));
			if (!std::isfinite(N_subdivisionsD) ||
				N_subdivisionsD > std::numeric_limits<uint32_t>::max())
			{
				throw std::runtime_error("Surface probe subdivision count exceeds uint32 limits.");
			}

			const uint64_t N_subdivisions = std::max<uint64_t>(
				1,
				static_cast<uint64_t>(N_subdivisionsD));
			const uint64_t factorA = N_subdivisions + 1ull;
			const uint64_t factorB = N_subdivisions + 2ull;
			if (factorA > std::numeric_limits<uint64_t>::max() / factorB)
			{
				throw std::runtime_error("Surface coverage probe count exceeds uint64 limits.");
			}

			const uint64_t N_triangleProbes = factorA * factorB / 2ull;
			if (N_triangleProbes > std::numeric_limits<uint32_t>::max() - N_probes)
			{
				throw std::runtime_error("Surface coverage probe count exceeds uint32 limits.");
			}
			N_probes += N_triangleProbes;
		}
		storage_->stats.N_coverageProbes = static_cast<uint32_t>(N_probes);

		const uint64_t N_maxSamples = static_cast<uint64_t>(acceptedSamples.size()) + N_probes;
		if (N_maxSamples > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Repaired surface sample count exceeds uint32 limits.");
		}

		// rebuild the spatial hash from the initially accepted samples
		SampleSpatialHash spatialHash(minDistance_, acceptedSamples);
		spatialHash.Reserve(static_cast<uint32_t>(acceptedSamples.size()));
		for (uint32_t sampleIdx = 0; sampleIdx < static_cast<uint32_t>(acceptedSamples.size()); sampleIdx++)
		{
			spatialHash.Insert(sampleIdx);
		}

		// reserve probe spacing as the continuous coverage error budget
		const float probeTargetDistance = std::max(
			minDistance_,
			maxDistance_ - coverageProbeSpacing_);
		uint32_t& N_repairs = storage_->stats.N_coverageRepairs;
		N_repairs = 0;

		// generate and repair coverage probes in bounded batches
		const uint32_t N_probeBatch = std::min(
			N_samplingBatch,
			static_cast<uint32_t>(N_probes));
		std::vector<glm::vec3> probes;
		probes.reserve(N_probeBatch);
		std::vector<uint8_t> needsRepair(N_probeBatch, 0u);

		const auto repairProbeBatch = [&]()
			{
				const uint32_t N_batch = static_cast<uint32_t>(probes.size());
				if (N_batch == 0) { return; }

				// query one probe batch in parallel against the read-only sample hash
				tbb::parallel_for(
					tbb::blocked_range<uint32_t>(0, N_batch),
					[&](const tbb::blocked_range<uint32_t>& r)
					{
						for (uint32_t batchIdx = r.begin(); batchIdx != r.end(); batchIdx++)
						{
							const glm::vec3& probe = probes[batchIdx];
							const bool isCovered = spatialHash.AnyNearby(
								probe,
								probeTargetDistance,
								[&](const uint32_t sampleIdx)
								{
									return SurfaceDistance(
										probe,
										acceptedSamples[sampleIdx]) <= probeTargetDistance;
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

					const glm::vec3& probe = probes[batchIdx];
					const bool isCovered = spatialHash.AnyNearby(
						probe,
						probeTargetDistance,
						[&](const uint32_t sampleIdx)
						{
							return SurfaceDistance(
								probe,
								acceptedSamples[sampleIdx]) <= probeTargetDistance;
						});
					if (isCovered) { continue; }
					if (acceptedSamples.size() >= std::numeric_limits<uint32_t>::max())
					{
						throw std::runtime_error("Repaired surface sample count exceeds uint32 limits.");
					}

					const uint32_t sampleIdx = static_cast<uint32_t>(acceptedSamples.size());
					acceptedSamples.push_back(probe);
					spatialHash.Insert(sampleIdx);
					N_repairs++;
				}

				probes.clear();
			};

		// generate probes in stable triangle and barycentric lattice order
		for (uint32_t triangleIdx = 0;
			triangleIdx < static_cast<uint32_t>(triangles.size());
			triangleIdx++)
		{
			const TriangleData& triangle = triangles[triangleIdx];
			const uint32_t N_subdivisions = std::max<uint32_t>(
				1,
				static_cast<uint32_t>(std::ceil(
					triangle.longestEdge / coverageProbeSpacing_)));
			const float N_subdivisionsInv = 1.0f / static_cast<float>(N_subdivisions);

			for (uint32_t i = 0; i <= N_subdivisions; i++)
			{
				for (uint32_t j = 0; j <= N_subdivisions - i; j++)
				{
					const uint32_t k = N_subdivisions - i - j;
					const glm::vec3 barycentric
					{
						static_cast<float>(i) * N_subdivisionsInv,
						static_cast<float>(j) * N_subdivisionsInv,
						static_cast<float>(k) * N_subdivisionsInv,
					};
					probes.push_back(MakeSurfacePoint(
						triangle,
						barycentric));

					if (probes.size() == N_probeBatch)
					{
						repairProbeBatch();
					}
				}
			}
		}

		repairProbeBatch();
	}

	void SurfaceSampler::ValidateSamples()
	{
		const auto& acceptedSamples = storage_->acceptedSamples;

		// validate the final sample count and generated sample data
		if (acceptedSamples.empty())
		{
			throw std::runtime_error("Surface sampling produced no samples.");
		}
		if (acceptedSamples.size() > std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Surface sample count exceeds uint32 limits.");
		}

		// validate generated sample positions
		for (const glm::vec3& sample : acceptedSamples)
		{
			if (!IsFinite(sample))
			{
				throw std::runtime_error("Surface sampling produced a non-finite sample position.");
			}
		}
	}

	SurfaceSamplingResult SurfaceSampler::Sample()
	{
		storage_->stats = {};
		storage_->stats.surfaceArea = static_cast<float>(storage_->surfaceArea);
		storage_->acceptedSamples.clear();
		// generate and select the initial Poisson sample set
		SelectPoissonSamples();

		// detect and repair remaining surface coverage gaps
		RepairCoverage();
		ValidateSamples();

		SurfaceSamplingResult result;
		result.stats = storage_->stats;
		result.acceptedSamples = std::move(storage_->acceptedSamples);
		return result;
	}

} // namespace Scene
