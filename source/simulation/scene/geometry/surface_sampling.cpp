// surface_sampling.cpp

// external
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/geometric.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
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

	// per-coordinate mixed hash for the linear-probing cell table
	struct CellCoordHash
	{
		// splitmix64 finalizer
		static uint64_t Mix(uint64_t h)
		{
			h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
			h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
			return h ^ (h >> 31);
		}

		uint64_t operator()(const CellCoord& cell) const noexcept
		{
			uint64_t h = Mix(static_cast<uint64_t>(cell.x));
			h = Mix(h ^ static_cast<uint64_t>(cell.y));
			return Mix(h ^ static_cast<uint64_t>(cell.z));
		}
	};

	// stores cell values in contiguous entries with an index-only hash table
	template<typename Value>
	class SampleCellTable
	{
	public:

		void Reserve(const uint32_t N)
		{
			const uint64_t N_slots = std::max<uint64_t>(16ull, static_cast<uint64_t>(N) + N / 2ull + 1ull);
			if (N_slots > slots_.max_size())
			{
				throw std::runtime_error("Surface sample cell table exceeds allocation limits.");
			}

			entries_.reserve(N);
			if (N_slots <= slots_.size()) { return; }

			// rehash entry indices without copying cell values
			std::vector<uint32_t> slots(static_cast<size_t>(N_slots), InvalidEntryIdx);
			for (uint32_t entryIdx = 0; entryIdx < entries_.size(); entryIdx++)
			{
				size_t slotIdx = CellCoordHash{}(entries_[entryIdx].cell) % slots.size();
				while (slots[slotIdx] != InvalidEntryIdx)
				{
					if (++slotIdx == slots.size()) { slotIdx = 0; }
				}
				slots[slotIdx] = entryIdx;
			}
			slots_.swap(slots);
		}

		const Value* Find(const CellCoord& cell) const
		{
			if (slots_.empty()) { return nullptr; }
			const uint32_t entryIdx = slots_[FindSlot(cell)];
			return entryIdx == InvalidEntryIdx ? nullptr : &entries_[entryIdx].value;
		}

		Value& GetOrInsert(const CellCoord& cell)
		{
			if (slots_.empty()) { Reserve(1); }
			size_t slotIdx = FindSlot(cell);
			if (slots_[slotIdx] != InvalidEntryIdx) { return entries_[slots_[slotIdx]].value; }

			const uint64_t N_entries = entries_.size() + 1ull;
			if (N_entries > std::numeric_limits<uint32_t>::max())
			{
				throw std::runtime_error("Surface sample cell count exceeds uint32 limits.");
			}

			// grow before insertion to keep the hash load near two thirds
			if (N_entries > entries_.capacity() ||
				N_entries > slots_.size() - slots_.size() / 3)
			{
				const uint32_t N_reserve = static_cast<uint32_t>(std::min<uint64_t>(
					std::numeric_limits<uint32_t>::max(),
					std::max<uint64_t>(16ull, N_entries + N_entries / 2ull)));
				Reserve(N_reserve);
				slotIdx = FindSlot(cell);
			}

			const uint32_t entryIdx = static_cast<uint32_t>(entries_.size());
			entries_.push_back(Entry{ .cell = cell });
			slots_[slotIdx] = entryIdx;
			return entries_.back().value;
		}

	private:

		static constexpr uint32_t InvalidEntryIdx = std::numeric_limits<uint32_t>::max();

		struct Entry
		{
			CellCoord cell;
			Value value{};
		};

		size_t FindSlot(const CellCoord& cell) const
		{
			size_t slotIdx = CellCoordHash{}(cell) % slots_.size();
			while (slots_[slotIdx] != InvalidEntryIdx && entries_[slots_[slotIdx]].cell != cell)
			{
				if (++slotIdx == slots_.size()) { slotIdx = 0; }
			}
			return slotIdx;
		}

		std::vector<Entry> entries_;
		std::vector<uint32_t> slots_;
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
			cellBuckets_.Reserve(N);
			nextSampleIdx_.reserve(N);
		}

		void Insert(const uint32_t sampleIdx)
		{
			CellBucket& bucket = cellBuckets_.GetOrInsert(MakeCellCoord(acceptedSamples_[sampleIdx], cellSizeInv_));
			nextSampleIdx_.push_back(InvalidSampleIdx);

			// append samples in their original per-cell order
			if (bucket.lastSampleIdx != InvalidSampleIdx) { nextSampleIdx_[bucket.lastSampleIdx] = sampleIdx; }
			else { bucket.firstSampleIdx = sampleIdx; }
			bucket.lastSampleIdx = sampleIdx;
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
						const CellBucket* bucket = cellBuckets_.Find(cell);
						if (!bucket) { continue; }

						for (uint32_t sampleIdx = bucket->firstSampleIdx;
							sampleIdx != InvalidSampleIdx;
							sampleIdx = nextSampleIdx_[sampleIdx])
						{
							if (func(sampleIdx)) { return true; }
						}
					}
				}
			}

			return false;
		}

	private:

		static constexpr uint32_t InvalidSampleIdx = std::numeric_limits<uint32_t>::max();

		struct CellBucket
		{
			uint32_t firstSampleIdx = InvalidSampleIdx;
			uint32_t lastSampleIdx = InvalidSampleIdx;
		};

		float cellSize_ = 0.0f;
		float cellSizeInv_ = 0.0f;
		const std::vector<glm::vec3>& acceptedSamples_;
		SampleCellTable<CellBucket> cellBuckets_;
		std::vector<uint32_t> nextSampleIdx_;
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
		SampleCellTable<uint32_t> trialsPerCell;
		trialsPerCell.Reserve(N_estimatedSamples);
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
				uint32_t& N_trials = trialsPerCell.GetOrInsert(cell);
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
