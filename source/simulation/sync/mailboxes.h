// mailboxes.h
#pragma once

// external
#include <atomic>
#include <memory>
#include <mutex>

// internal
#include "snapshots.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "mailboxes_cuda.h"
#endif



namespace Sync
{
	// single-producer, single-consumer mailbox for the latest snapshot
	template <typename Snapshot>
	class LatestSnapMailbox
	{
	public:

		void StoreSnapshot(std::shared_ptr<Snapshot> snap) noexcept
		{
			std::scoped_lock lock(mutex_);
			latestSnap_.swap(snap);
		}

		bool TryFetchLatestSnapshot(std::shared_ptr<Snapshot>& dst) noexcept
		{
			std::shared_ptr<Snapshot> snap{};
			{
				std::scoped_lock lock(mutex_);
				snap = std::move(latestSnap_);
			}
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		}

	private:

		std::mutex mutex_;
		std::shared_ptr<Snapshot> latestSnap_{ nullptr };
	};

	// mailbox for vis-specific simulation state snapshots
	class VisualSnapMailbox : public LatestSnapMailbox<VisualSnapshot>
	{
	public:

		void RequestSnapshot() noexcept
		{
			snapshotRequested_.store(true, std::memory_order_release);
		}

		bool TakeSnapshotRequest() noexcept
		{
			return snapshotRequested_.exchange(false, std::memory_order_acq_rel);
		}

		// access to the additional mailbox for CUDA-device buffer snapshot
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		VisualFrameMailboxCUDA& GetFrameMailboxCUDA() noexcept { return visualFrameMailboxCUDA_; }
	#endif

	private:

		std::atomic<bool> snapshotRequested_{ false };

		// additional mailbox for CUDA-device buffer snapshot
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		VisualFrameMailboxCUDA visualFrameMailboxCUDA_{};
	#endif
	};

	using BenchSnapMailbox = LatestSnapMailbox<BenchSnapshot>;
	using MemSnapMailbox = LatestSnapMailbox<MemSnapshot>;
	using StatSnapMailbox = LatestSnapMailbox<StatSnapshot>;

} // namespace Sync
