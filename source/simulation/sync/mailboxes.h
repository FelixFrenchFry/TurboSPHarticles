// mailboxes.h
#pragma once

// external
#include <atomic>
#include <memory>

// internal
#include "snapshots.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "mailboxes_cuda.h"
#endif



namespace Sync
{
	// single-producer, single-consumer, lock-free mailbox for vis-specific simulation state snapshots
	class VisualSnapMailbox
	{
	public:

		void StoreSnapshot(std::shared_ptr<VisualSnapshot> snap) noexcept
		{
			std::atomic_store_explicit(&latestSnap_, std::move(snap), std::memory_order_release);
		}

		bool TryFetchLatestSnapshot(std::shared_ptr<VisualSnapshot>& dst) noexcept
		{
			auto snap = std::atomic_exchange_explicit(&latestSnap_, std::shared_ptr<VisualSnapshot>{}, std::memory_order_acq_rel);
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		}

		// access to the additional mailbox for CUDA-device buffer snapshot
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		VisualFrameMailboxCUDA& GetFrameMailboxCUDA() noexcept { return visualFrameMailboxCUDA_; }
	#endif

	private:

		// latest snapshot of the vis-specific simulation state
		std::shared_ptr<VisualSnapshot> latestSnap_{ nullptr };

		// additional mailbox for CUDA-device buffer snapshot
	#ifdef TURBOSPHARTICLES_WITH_CUDA
		VisualFrameMailboxCUDA visualFrameMailboxCUDA_{};
	#endif
	};

	// single-producer, single-consumer, lock-free mailbox for operation benchmark snapshots
	class BenchSnapMailbox
	{
	public:

		void StoreSnapshot(std::shared_ptr<BenchSnapshot> snap) noexcept
		{
			std::atomic_store_explicit(&latestSnap_, std::move(snap), std::memory_order_release);
		}

		bool TryFetchLatestSnapshot(std::shared_ptr<BenchSnapshot>& dst) noexcept
		{
			auto snap = std::atomic_exchange_explicit(&latestSnap_, std::shared_ptr<BenchSnapshot>{}, std::memory_order_acq_rel);
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		}

	private:

		// latest snapshot of the operation benchmarks
		std::shared_ptr<BenchSnapshot> latestSnap_{ nullptr };
	};

	// single-producer, single-consumer, lock-free mailbox for memory usage snapshots
	class MemSnapMailbox
	{
	public:

		void StoreSnapshot(std::shared_ptr<MemSnapshot> snap) noexcept
		{
			std::atomic_store_explicit(&latestSnap_, std::move(snap), std::memory_order_release);
		}

		bool TryFetchLatestSnapshot(std::shared_ptr<MemSnapshot>& dst) noexcept
		{
			auto snap = std::atomic_exchange_explicit(&latestSnap_, std::shared_ptr<MemSnapshot>{}, std::memory_order_acq_rel);
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		}

	private:

		// latest snapshot of the operation benchmarks
		std::shared_ptr<MemSnapshot> latestSnap_{ nullptr };
	};

	// single-producer, single-consumer, lock-free mailbox for statistics data snapshots
	class StatSnapMailbox
	{
	public:

		void StoreSnapshot(std::shared_ptr<StatSnapshot> snap) noexcept
		{
			std::atomic_store_explicit(&latestSnap_, std::move(snap), std::memory_order_release);
		}

		bool TryFetchLatestSnapshot(std::shared_ptr<StatSnapshot>& dst) noexcept
		{
			auto snap = std::atomic_exchange_explicit(&latestSnap_, std::shared_ptr<StatSnapshot>{}, std::memory_order_acq_rel);
			if (!snap) { return false; } // failed fetch

			// successful fetch
			dst = std::move(snap);
			return true;
		}

	private:

		// latest snapshot of the statistics data
		std::shared_ptr<StatSnapshot> latestSnap_{ nullptr };
	};

} // namespace Sync
