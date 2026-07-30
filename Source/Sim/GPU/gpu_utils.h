// gpu_utils.h
//
// TODO: comment

#pragma once

// external
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

// detect std::source_location availability
#if defined(__cpp_lib_source_location) && (__cpp_lib_source_location >= 201907L)
	#include <source_location>
	#define SIM_HAVE_SOURCE_LOCATION 1
#else
	#define SIM_HAVE_SOURCE_LOCATION 0
#endif



namespace GPU
{
	// helper for checking potential CUDA error code and throwing an exception if not successful
	inline void Check(
		cudaError_t err,
		std::string_view what = "CUDA call failed"
	#if SIM_HAVE_SOURCE_LOCATION
		, std::source_location loc = std::source_location::current()
	#endif
	)
	{
		if (err == cudaSuccess) { return; }

		const char* name = cudaGetErrorName(err);
		const char* desc = cudaGetErrorString(err);

	#if SIM_HAVE_SOURCE_LOCATION
		throw std::runtime_error(
			std::string(what) + ": " + desc + " (" + name + ") at " +
			loc.file_name() + ":" + std::to_string((int)loc.line()) +
			" in " + loc.function_name() + ".");
	#else
		throw std::runtime_error(std::string(what) + ": " + desc + " (" + name + ").");
	#endif
	}

	// check the last launched kernel for errors
	inline void CheckLastKernel(std::string_view what = "CUDA kernel launch")
	{
		Check(cudaPeekAtLastError(), what);
	}

	// sync a given stream and check for errors
	inline void StreamSync(cudaStream_t stream)
	{
		Check(cudaStreamSynchronize(stream), "cudaStreamSynchronize()");
	}

	// helper for copying buffer data from host to device
	template <class T>
	inline void Copy_HostToDevice(T* dstDevice, const T* srcHost, 
								  size_t N_elements, cudaStream_t stream)
	{
		if (N_elements == 0) { return; }

		if (!dstDevice)
		{
			throw std::runtime_error("GPU::Copy_HostToDevice received a null destination pointer "
				"with a nonzero element count.");
		}

		if (!srcHost)
		{
			throw std::runtime_error("GPU::Copy_HostToDevice received a null source pointer "
				"with a nonzero element count.");
		}

		Check(cudaMemcpyAsync(dstDevice, srcHost, N_elements * sizeof(T),
			  cudaMemcpyHostToDevice, stream),
			  "cudaMemcpyAsync() H->D");
	}

	// helper for copying buffer data from device to host
	template <class T>
	inline void Copy_DeviceToHost(T* dstHost, const T* srcDevice,
								  size_t N_elements, cudaStream_t stream)
	{
		if (N_elements == 0) { return; }

		if (!dstHost)
		{
			throw std::runtime_error("GPU::Copy_DeviceToHost received a null destination pointer "
				"with a nonzero element count.");
		}

		if (!srcDevice)
		{
			throw std::runtime_error("GPU::Copy_DeviceToHost received a null source pointer "
				"with a nonzero element count.");
		}

		Check(cudaMemcpyAsync(dstHost, srcDevice, N_elements * sizeof(T),
			  cudaMemcpyDeviceToHost, stream), 
			  "cudaMemcpyAsync() D to H");
	}

	// helper for setting device buffer memory to a given byte value
	template <class T>
	inline void MemsetBytes(T* dstDevice, int value,
							size_t N_elements, cudaStream_t stream)
	{
		if (N_elements == 0) { return; }

		if (!dstDevice)
		{
			throw std::runtime_error("GPU::MemsetBytes received a null destination pointer "
				"with a nonzero element count.");
		}

		Check(cudaMemsetAsync(dstDevice, value, N_elements * sizeof(T), stream),
			  "cudaMemsetAsync()");
	}

	// helper for setting device buffer memory to zero
	template <class T>
	inline void MemsetZero(T* dstDevice,
						   size_t N_elements, cudaStream_t stream)
	{
		MemsetBytes(dstDevice, 0, N_elements, stream);
	}

} // namespace GPU
