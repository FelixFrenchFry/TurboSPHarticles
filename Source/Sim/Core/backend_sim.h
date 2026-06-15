// backend_sim.h

#pragma once

// external
#include <cstdint>
#include <memory>
#include <vector>

// internal
#include "Source/Base/enum_hub.h"
#include "Source/Eval/Bench/benchmarks.h"
#include "Source/Eval/Mem/memory.h"
#include "Source/Eval/Stats/statistics.h"
#include "Source/Sim/Core/parameters_sim.h"
#include "Source/Sim/Sync/snapshots.h"
#include "Source/Sim/Sync/state_sim.h"



namespace Sim
{
	// forward declarations
	class I_NeighborSearch;
	struct NeighborSpan;

	// -----------------------------------------------------------------------------------------------
	// --- [ simulation backend interface ] ----------------------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class I_SimBackend
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_SimBackend() = default;

		virtual const char* Name() const = 0;

		virtual void SetNeighborSearchMethod(std::unique_ptr<I_NeighborSearch> ns) = 0;

		// copies & translates a universal simulation state into -> backend-specific data structures
		virtual void PullFromUniversalSimState(Sync::UniversalSimState& src, const SimParams& params) = 0;

		// copies & translates backend-specific data structures into -> universal simulation state
		virtual void PushToUniversalSimState(Sync::UniversalSimState& dst) const = 0;

		// packs the current simulation state into a snapshot used for exporting or visualization
		// (implementation is backend-specific for better performance)
		virtual void PackExportSnapshot(Sync::ExportSnapshot& snap, const SimParams& params) const = 0;
		virtual void PackVisualSnapshot(Sync::VisualSnapshot& snap) const = 0;

		// computes & fills in the scalar data based on different color coding modes
		// (backend-specific so that possible transformations can be applied efficiently)
		virtual void FillScalarData(Sync::VisualSnapshot& snap, Base::ColorCodingMode mode) const = 0;

		// computes & fills in the memory usage data for the current sim step
		// (backend-specific so that the memory usage can be computed from the custom data structures efficiently)
		virtual void FillMemoryRecord(Mem::MemRecordStep& record, const SimParams& params) const = 0;

		// computes & fills in the stat data for the current sim step
		// (backend-specific so that the stats can be computed from the custom data structures efficiently)
		virtual void FillStatisticsRecord(Stats::StatRecordStep& record, const SimParams& params) const = 0;

		// executes only the neighbor search (construction and query of the data NS structure)
		virtual void ExecuteNeighborSearch(const SimParams& params, bool bench = false) = 0;

		// provides the latest local re-order permutation if the backend exposes it
		virtual bool TryGetLastParticleReOrder(
			std::vector<uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::vector<uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
		{
			(void)oldFluidLocalIdx_FromNewLocalIdx;
			(void)oldBoundaryLocalIdx_FromNewLocalIdx;
			return false;
		}

		// sets which local fluid particles are authoritative for integration
		virtual bool SetActiveFluidParticleLocalIdxs(const std::vector<int>& localIdxs)
		{
			(void)localIdxs;
			return false;
		}

		// clears distributed integration filtering
		virtual void ClearActiveFluidParticleLocalIdxs() {}

		// advances the simulation by N_steps
		virtual void ExecuteFullSimStep(const SimParams& params, uint32_t N_steps = 1) = 0;

		// IISPH fluid solver step
		virtual void Step_IISPH(const SimParams& params) = 0;

		// provides the list defining how computation operations are benchmarked in groups and/or individually
		// (backend-specific so that important cache-efficiency optimization can be applied)
		virtual void GetBenchOpGroups(std::vector<Bench::BenchOpGroupDef>& out) const = 0;

		// info
		virtual uint32_t GetNumFluidParticles() const = 0;
		virtual uint32_t GetNumBoundaryParticles() const = 0;
	};

	// backend factory
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendVersion version,
												 Sim::SimParamsMgr& simParamsMgr,
												 Bench::BenchmarksTracker& benchmarksTracker,
												 const SimParams& params);

	// -----------------------------------------------------------------------------------------------
	// --- [ neighbor search and neighbor list interfaces ] ------------------------------------------
	// -----------------------------------------------------------------------------------------------

	class I_NeighborSearch
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_NeighborSearch() = default;

		virtual const char* Name() const = 0;

		virtual void Init(const SimParams& params) = 0;

		virtual void Construct(const SimParams& params) = 0;

		virtual void Query_Count(const SimParams& params) = 0;

		virtual void Query_Fill(const SimParams& params) = 0;

		virtual void DataReOrder(const SimParams& params) = 0;

		// provides the latest local re-order permutation if this neighbor search exposes it
		virtual bool TryGetLastParticleReOrder(
			std::vector<uint32_t>& oldFluidLocalIdx_FromNewLocalIdx,
			std::vector<uint32_t>& oldBoundaryLocalIdx_FromNewLocalIdx) const
		{
			(void)oldFluidLocalIdx_FromNewLocalIdx;
			(void)oldBoundaryLocalIdx_FromNewLocalIdx;
			return false;
		}

		virtual std::pair<Mem::MemUsage, Mem::MemUsage> GetMemUsage(
			const SimParams& params) const = 0;
	};

	struct NeighborSpan
	{
		const uint32_t* data = nullptr;
		uint32_t count = 0;
	};

	class I_NeighborList
	{
	public: // ---------------------------------------------------------------------------------------

		virtual ~I_NeighborList() = default;

		virtual void Resize(uint32_t N_F, uint32_t N_B, const SimParams& params) = 0;

		// applies optional post-processing on the data structures after the count-query
		// (e.g. construction of the CSR neighbor storage)
		virtual void Finalize_Count() = 0;

		// applies optional post-processing on the data structures after the fill-query
		// (e.g. concatenation of individual neighbor lists into CSR neighbor storage)
		virtual void Finalize_Fill() = 0;

		// sets the neighbor count of fluid particle i, computed by the count-query
		virtual void SetNbrCountFF(uint32_t i, uint32_t count) = 0;
		virtual void SetNbrCountFB(uint32_t i, uint32_t count) = 0;

		// sets the full neighbor list js of fluid particle i, accumulated in a scratch array by the fill-query
		// (optionally applying compression and other post-processing if the implementation uses it)
		virtual void SetNbrListFF(uint32_t i, const uint32_t* js, uint32_t count) = 0;
		virtual void SetNbrListFB(uint32_t i, const uint32_t* js, uint32_t count) = 0;

		// gets a "span" of the fluid/boundary neighbor list of fluid particle i
		// (for compressed neighbor lists, the span points to per-thread scratch space, which
		// gets overwritten on each call, so the data must be processed before that)
		virtual NeighborSpan FF(uint32_t i) const = 0;
		virtual NeighborSpan FB(uint32_t i) const = 0;

		// default implementations for neighbor counts, overwritten for compressed neighbor lists
		virtual uint32_t CountFF(uint32_t i) const { return FF(i).count; }
		virtual uint32_t CountFB(uint32_t i) const { return FB(i).count; }

		// stats
		virtual std::pair<Mem::MemUsage, Mem::MemUsage>GetMemUsage(
			const SimParams& params) const = 0;

		// flags
		virtual bool RunCountQuery() const = 0;
		virtual bool RunFillQuery() const = 0;
	};

} // namespace Sim
