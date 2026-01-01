// backend_sim.cpp

// external
#include <memory>
#include <stdexcept>

// internal
#include "backend_sim.h"
#include "Source/Sim/CPU/CPU_Backend_01/cpu_backend_01.h"
#ifdef TURBOSPHARTICLES_WITH_CUDA
	#include "Source/Sim/GPU/GPU_Backend_01/gpu_backend_01.h"
#endif



namespace Sim
{
	std::unique_ptr<I_SimBackend> MakeSimBackend(Base::BackendVersion version,
												 Bench::BenchmarksMgr& benchmarksMgr,
												 SimParams const& params)
	{
		switch (version)
		{
		case Base::BackendVersion::CPU01:
			{
				return std::make_unique<SimBackend_CPU01>(benchmarksMgr, params);
			}
		#ifdef TURBOSPHARTICLES_WITH_CUDA
		case Base::BackendVersion::GPU01:
			{
				return std::make_unique<SimBackend_GPU01>(benchmarksMgr, params);
			}
		#endif
		}

		throw std::runtime_error("Unsupported sim backend version requested");
	}

} // namespace Sim
