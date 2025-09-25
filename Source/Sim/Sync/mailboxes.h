// mailboxes.h
//
// TODO: comment

#pragma once

// external
#include <atomic>
#include <memory>

// internal
#include "snapshots.h"



namespace Sync
{
	// single-producer, single-consumer, lock-free mailbox for vis-specific simulation state snapshots
	class VisualSnapMailbox
	{
	public: // ==============================

		void StoreSnapshot(std::shared_ptr<VisualSnapshot> snap) noexcept
		{
			latestSnap_.store(std::move(snap), std::memory_order_release);
		};

		bool TryFetchLatestSnapshot(std::shared_ptr<VisualSnapshot>& dst) noexcept
		{
			auto snap = latestSnap_.exchange(nullptr, std::memory_order_acq_rel);
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		};

	private: // ==============================

		// latest snapshot of the vis-specific simulation state
		std::atomic<std::shared_ptr<VisualSnapshot>> latestSnap_{ nullptr };
	};

} // namespace Sim
