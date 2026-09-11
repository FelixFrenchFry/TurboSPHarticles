// mailboxes_cuda.h
#pragma once

#ifdef TURBOSPHARTICLES_WITH_CUDA

// external
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <memory>



namespace Sync
{
	struct VisualSnapshot;

	// particle data packed for CUDA-OpenGL rendering
	struct VisualParticleSample
	{
		float x;
		float y;
		float z;
		float value;
	};
	static_assert(sizeof(VisualParticleSample) == 16);

	struct VisualFrameWriteCUDA
	{
		uint32_t slot = std::numeric_limits<uint32_t>::max();
		VisualParticleSample* fluid = nullptr;
		VisualParticleSample* rigid = nullptr;
	};

	struct VisualFrameReadCUDA
	{
		uint32_t slot = std::numeric_limits<uint32_t>::max();
		const VisualParticleSample* fluid = nullptr;
		const VisualParticleSample* rigid = nullptr;
		uint32_t N_F = 0;
		uint32_t N_R = 0;
		cudaEvent_t ready = nullptr;
		std::shared_ptr<VisualSnapshot> auxiliarySnap{};
	};

	// reusable CUDA frames shared by the simulation and visualization threads
	class VisualFrameMailboxCUDA
	{
	public:

		VisualFrameMailboxCUDA();
		~VisualFrameMailboxCUDA();

		VisualFrameMailboxCUDA(const VisualFrameMailboxCUDA&) = delete;
		VisualFrameMailboxCUDA& operator=(const VisualFrameMailboxCUDA&) = delete;

		void Enable(int device);
		void Disable(const char* reason) noexcept;
		bool IsEnabled() const noexcept;

		bool TryAcquireWrite(
			uint32_t N_F,
			uint32_t N_R,
			VisualFrameWriteCUDA& dst);

		bool Publish(
			const VisualFrameWriteCUDA& src,
			cudaStream_t stream,
			std::shared_ptr<VisualSnapshot> auxiliarySnap);

		bool TryFetchLatest(VisualFrameReadCUDA& dst);
		bool ReleaseRead(const VisualFrameReadCUDA& src, cudaStream_t stream);

	private:

		struct Impl;
		std::unique_ptr<Impl> impl_;

		void Shutdown() noexcept;
	};

} // namespace Sync

#endif
