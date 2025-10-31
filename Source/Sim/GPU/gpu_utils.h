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
			" in " + loc.function_name());
	#else
		throw std::runtime_error(std::string(what) + ": " + desc + " (" + name + ")");
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
								  size_t countElems, cudaStream_t stream)
	{
		if (!dstDevice || !srcHost || countElems == 0) { return; }

		Check(cudaMemcpyAsync(dstDevice, srcHost, countElems * sizeof(T),
			cudaMemcpyHostToDevice, stream), 
			"cudaMemcpyAsync() H to D");
	}

	// helper for copying buffer data from device to host
	template <class T>
	inline void Copy_DeviceToHost(T* dstHost, const T* srcDevice,
								  size_t countElems, cudaStream_t stream)
	{
		if (!dstHost || !srcDevice || countElems == 0) { return; }

		Check(cudaMemcpyAsync(dstHost, srcDevice, countElems * sizeof(T),
			  cudaMemcpyDeviceToHost, stream), 
			  "cudaMemcpyAsync() D to H");
	}

	// helper for setting device buffer memory to a given byte value
	template <class T>
	inline void MemsetBytes(T* dstDevice, int value,
							size_t countElems, cudaStream_t stream)
	{
		if (!dstDevice || countElems == 0) { return; }

		Check(cudaMemsetAsync(dstDevice, value, countElems * sizeof(T), stream), 
			  "cudaMemsetAsync()");
	}

	// helper for setting device buffer memory to zero
	template <class T>
	inline void MemsetZero(T* dstDevice,
						   size_t countElems, cudaStream_t stream)
	{
		MemsetBytes(dstDevice, 0, countElems, stream);
	}

} // namespace GPU
