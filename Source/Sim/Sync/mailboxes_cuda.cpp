// mailboxes_cuda.cpp

#ifdef TURBOSPHARTICLES_WITH_CUDA

// external
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <utility>
#include <spdlog/spdlog.h>

// internal
#include "mailboxes_cuda.h"



namespace Sync
{
	struct VisualFrameMailboxCUDA::Impl
	{
		enum class SlotState : uint8_t
		{
			Free,
			Writing,
			Ready,
			Reading,
			Retired,
		};

		struct Slot
		{
			VisualParticleSample* samples = nullptr;
			size_t capacity = 0;
			uint32_t N_F = 0;
			uint32_t N_RBP = 0;
			cudaEvent_t completion = nullptr; // completion of the last stream using this slot
			std::shared_ptr<VisualSnapshot> auxiliarySnap{};
			SlotState state = SlotState::Free;
		};

		static constexpr uint32_t InvalidSlot = std::numeric_limits<uint32_t>::max();

		bool EnsureCapacity(
			Slot& slot,
			const size_t N_required,
			VisualFrameMailboxCUDA& mailbox)
		{
			// create completion event on first use
			if (!slot.completion && cudaEventCreateWithFlags(
				&slot.completion, cudaEventDisableTiming) != cudaSuccess)
			{
				mailbox.Disable("failed to create a visual frame event");
				return false;
			}

			// resize the reusable particle buffer if required
			if (N_required <= slot.capacity) { return true; }

			if (slot.samples)
			{
				cudaFree(slot.samples);
				slot.samples = nullptr;
				slot.capacity = 0;
			}
			if (cudaMalloc(
				reinterpret_cast<void**>(&slot.samples),
				N_required * sizeof(VisualParticleSample)) != cudaSuccess)
			{
				mailbox.Disable("failed to allocate a visual snapshot slot");
				return false;
			}

			slot.capacity = N_required;
			return true;
		}

		void ReclaimRetiredSlots(VisualFrameMailboxCUDA& mailbox)
		{
			// reclaim slots after the last CUDA stream using them has completed
			for (auto& slot : slots_)
			{
				if (slot.state != SlotState::Retired) { continue; }

				const cudaError_t query = cudaEventQuery(slot.completion);
				if (query == cudaSuccess)
				{
					slot.auxiliarySnap.reset();
					slot.state = SlotState::Free;
				}
				else if (query != cudaErrorNotReady)
				{
					mailbox.Disable(cudaGetErrorString(query));
					return;
				}
			}
		}

		std::array<Slot, 2> slots_{};
		std::mutex mutex_;
		std::atomic<bool> enabled_{ false };
		uint32_t latestSlot_ = InvalidSlot;
		int device_ = -1;
	};

	VisualFrameMailboxCUDA::VisualFrameMailboxCUDA()
		: impl_{ std::make_unique<Impl>() }
	{
	}

	VisualFrameMailboxCUDA::~VisualFrameMailboxCUDA()
	{
		Shutdown();
	}

	void VisualFrameMailboxCUDA::Enable(const int device)
	{
		std::scoped_lock lock(impl_->mutex_);
		impl_->device_ = device;
		impl_->enabled_.store(true, std::memory_order_release);
	}

	void VisualFrameMailboxCUDA::Disable(const char* reason) noexcept
	{
		if (impl_->enabled_.exchange(false, std::memory_order_acq_rel))
		{
			SPDLOG_WARN("CUDA visual fast path disabled: {}", reason);
		}
	}

	bool VisualFrameMailboxCUDA::IsEnabled() const noexcept
	{
		return impl_->enabled_.load(std::memory_order_acquire);
	}

	bool VisualFrameMailboxCUDA::TryAcquireWrite(
		const uint32_t N_F,
		const uint32_t N_RBP,
		VisualFrameWriteCUDA& dst)
	{
		std::scoped_lock lock(impl_->mutex_);
		if (!IsEnabled()) { return false; }

		// select CUDA device used by the interop mailbox
		int currentDevice = -1;
		cudaError_t error = cudaGetDevice(&currentDevice);
		if (error == cudaSuccess && currentDevice != impl_->device_)
		{
			error = cudaSetDevice(impl_->device_);
		}
		if (error != cudaSuccess)
		{
			Disable(cudaGetErrorString(error));
			return false;
		}

		// reclaim completed slots and acquire a free slot for packing
		impl_->ReclaimRetiredSlots(*this);
		const size_t N_required = static_cast<size_t>(N_F) + N_RBP;
		for (uint32_t i = 0; i < impl_->slots_.size(); i++)
		{
			auto& slot = impl_->slots_[i];
			if (slot.state != Impl::SlotState::Free) { continue; }
			if (!impl_->EnsureCapacity(slot, N_required, *this)) { return false; }

			slot.state = Impl::SlotState::Writing;
			slot.N_F = N_F;
			slot.N_RBP = N_RBP;
			dst.slot = i;
			dst.fluid = N_F > 0 ? slot.samples : nullptr;
			dst.rigid = N_RBP > 0 ? slot.samples + N_F : nullptr;
			return true;
		}

		return false;
	}

	bool VisualFrameMailboxCUDA::Publish(
		const VisualFrameWriteCUDA& src,
		cudaStream_t stream,
		std::shared_ptr<VisualSnapshot> auxiliarySnap)
	{
		if (src.slot >= impl_->slots_.size() || !IsEnabled()) { return false; }

		// record when producer stream finishes packing this slot
		auto& slot = impl_->slots_[src.slot];
		slot.auxiliarySnap = std::move(auxiliarySnap);
		const cudaError_t error = cudaEventRecord(slot.completion, stream);
		if (error != cudaSuccess)
		{
			Disable(cudaGetErrorString(error));
			return false;
		}

		// publish new frame and retire the previous unread frame
		std::scoped_lock lock(impl_->mutex_);
		if (slot.state != Impl::SlotState::Writing || !IsEnabled()) { return false; }
		if (impl_->latestSlot_ < impl_->slots_.size())
		{
			auto& previous = impl_->slots_[impl_->latestSlot_];
			if (previous.state == Impl::SlotState::Ready)
			{
				previous.state = Impl::SlotState::Retired;
			}
		}

		slot.state = Impl::SlotState::Ready;
		impl_->latestSlot_ = src.slot;
		return true;
	}

	bool VisualFrameMailboxCUDA::TryFetchLatest(VisualFrameReadCUDA& dst)
	{
		std::scoped_lock lock(impl_->mutex_);
		if (!IsEnabled() || impl_->latestSlot_ >= impl_->slots_.size()) { return false; }

		auto& slot = impl_->slots_[impl_->latestSlot_];
		if (slot.state != Impl::SlotState::Ready) { return false; }

		// transfer latest frame to the visualization thread
		slot.state = Impl::SlotState::Reading;
		dst.slot = impl_->latestSlot_;
		dst.fluid = slot.N_F > 0 ? slot.samples : nullptr;
		dst.rigid = slot.N_RBP > 0 ? slot.samples + slot.N_F : nullptr;
		dst.N_F = slot.N_F;
		dst.N_RBP = slot.N_RBP;
		dst.ready = slot.completion;
		dst.auxiliarySnap = std::move(slot.auxiliarySnap);
		impl_->latestSlot_ = Impl::InvalidSlot;
		return true;
	}

	bool VisualFrameMailboxCUDA::ReleaseRead(
		const VisualFrameReadCUDA& src,
		cudaStream_t stream)
	{
		if (src.slot >= impl_->slots_.size()) { return false; }

		// record when visualization stream finishes reading this slot
		auto& slot = impl_->slots_[src.slot];
		const cudaError_t error = cudaEventRecord(slot.completion, stream);
		if (error != cudaSuccess)
		{
			Disable(cudaGetErrorString(error));
			return false;
		}

		// retire slot until its completion event is ready
		std::scoped_lock lock(impl_->mutex_);
		if (slot.state != Impl::SlotState::Reading) { return false; }
		slot.state = Impl::SlotState::Retired;
		return true;
	}

	void VisualFrameMailboxCUDA::Shutdown() noexcept
	{
		impl_->enabled_.store(false, std::memory_order_release);
		std::scoped_lock lock(impl_->mutex_);
		if (impl_->device_ < 0) { return; }

		// release CUDA resources owned by all mailbox slots
		cudaSetDevice(impl_->device_);
		for (auto& slot : impl_->slots_)
		{
			if (slot.samples) { cudaFree(slot.samples); slot.samples = nullptr; }
			if (slot.completion) { cudaEventDestroy(slot.completion); slot.completion = nullptr; }
			slot.auxiliarySnap.reset();
			slot.state = Impl::SlotState::Free;
		}
		impl_->latestSlot_ = Impl::InvalidSlot;
		impl_->device_ = -1;
	}

} // namespace Sync

#endif
