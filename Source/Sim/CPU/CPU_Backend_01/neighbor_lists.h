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
	// --- [ small vector neighbor lists implementation ] ----------------------
	// -------------------------------------------------------------------------

	class NeighborLists_SmallVector_CPU01 final : public I_NeighborLists
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

} // namespace Sim
