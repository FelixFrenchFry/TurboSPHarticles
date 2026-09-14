// backend_sim.cpp

// external
#include <memory>
#include <stdexcept>

// internal
#include "backend_sim.h"
#include "source/simulation/CPU/backend/cpu_backend.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "source/simulation/GPU/backend/gpu_backend.h"
#endif



namespace Sim
{
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendType backendType,
												 Bench::BenchmarksTracker& benchmarksTracker,
												 SimParams const& params)
	{
		switch (backendType)
		{
		case Base::BackendType::CPU:
			{
				return std::make_unique<SimBackend_CPU>(benchmarksTracker, params);
			}
		case Base::BackendType::GPU:
			{
			#ifdef TURBOSPHARTICLES_WITH_CUDA
				return std::make_unique<SimBackend_GPU>(benchmarksTracker, params);
			#else
				throw std::runtime_error("GPU sim backend requested, but CUDA support is disabled.");
			#endif
			}
		}

		throw std::runtime_error("Unsupported sim backend type requested.");
	}

} // namespace Sim
