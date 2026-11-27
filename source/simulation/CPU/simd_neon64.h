// simd_neon64.h
#pragma once

// external
#include <cstdint>
#include <arm_neon.h>
#include <xsimd/xsimd.hpp>



namespace CPU
{
	constexpr uint32_t N_SIMD_WIDTH = 4u;
	constexpr uint32_t N_SIMD_ALIGNMENT = 16u;

	// 4-wide CPU SIMD types used by SELL-style kernels
	using SimdArch = xsimd::neon64;
	using Float32xSIMD = xsimd::batch<float, SimdArch>;
	using Int32xSIMD = xsimd::batch<int32_t, SimdArch>;
	using UInt32xSIMD = xsimd::batch<uint32_t, SimdArch>;
	using Mask32xSIMD = xsimd::batch_bool<float, SimdArch>;
	using UIntMask32xSIMD = xsimd::batch_bool<uint32_t, SimdArch>;

	// load contiguous uint16 values and widen them to uint32 lanes
	inline UInt32xSIMD LoadUInt16AsUInt32xSIMD(
		const uint16_t* p) noexcept
	{
		const uint16x4_t packed = vld1_u16(p);
		return UInt32xSIMD(vmovl_u16(packed));
	}

	// use the native square root to preserve accuracy near the kernel cutoff
	inline Float32xSIMD SqrtFloat32xSIMD(
		const Float32xSIMD v) noexcept
	{
		return Float32xSIMD(vsqrtq_f32(v));
	}

} // namespace CPU
