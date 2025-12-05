// neighbor_lists.h
//
// TODO: comment

#pragma once

// external
#include <vector>
#include <boost/container/small_vector.hpp>

// internal
#include "Source/Sim/Core/backend_sim.h"



namespace Sim
{
	// -------------------------------------------------------------------------
	// --- [ small vectors neighbor lists implementation ] ---------------------
	// -------------------------------------------------------------------------

	class NeighborLists_SmallVectors_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// neighbor storage
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountF>> neighborsFF_;
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountB>> neighborsFB_;
	};

	// -------------------------------------------------------------------------
	// --- [ fixed arrays neighbor lists implementation ] ----------------------
	// -------------------------------------------------------------------------

	class NeighborLists_FixedArrays_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// max neighbor counts
		static constexpr uint32_t N_MAX_FF_ = 64;
		static constexpr uint32_t N_MAX_FB_ = 32;

		std::vector<uint32_t> nbrCountFF_;
		std::vector<uint32_t> nbrCountFB_;
		std::vector<uint32_t> neighborsFF_; // N_F_ * N_MAX_FF_
		std::vector<uint32_t> neighborsFB_; // N_F_ * N_MAX_FB_
	};

	// -------------------------------------------------------------------------
	// --- [ compact vector neighbor lists implementation ] --------------------
	// -------------------------------------------------------------------------

	class NeighborLists_CompactVector_CPU01 final : public I_NeighborLists
	{
	public: // -----------------------------------------------------------------

		void Resize(uint32_t N_F, uint32_t N_B) override;

		void ClearOne(uint32_t i) override;

		void ClearAll() override;

		void Finalize() override;

		void AddNbrFF(uint32_t i, uint32_t j) override;
		void AddNbrFB(uint32_t i, uint32_t j) override;

		NeighborSpan FF(uint32_t i) const override;
		NeighborSpan FB(uint32_t i) const override;

	private: // ----------------------------------------------------------------

		// metadata
		uint32_t N_F_ = 0;
		uint32_t N_B_ = 0;

		// temp neighbor lists during construction
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountF>> tmpFF_;
		std::vector<boost::container::small_vector<uint32_t, N_expctNbrCountB>> tmpFB_;

		std::vector<uint64_t> offsetsFF_;
		std::vector<uint64_t> offsetsFB_;
		std::vector<uint32_t> neighborsFF_;
		std::vector<uint32_t> neighborsFB_;
	};

} // namespace Sim
