// simd_avx512.h
#pragma once

// external
#include <cstdint>
#include <immintrin.h>
#include <xsimd/xsimd.hpp>



namespace CPU
{
	constexpr uint32_t N_SIMD_WIDTH = 16u;
	constexpr uint32_t N_SIMD_ALIGNMENT = 64u;

	// 16-wide CPU SIMD types used by SELL-style kernels
	using SimdArch = xsimd::avx512f;
	using Float32xSIMD = xsimd::batch<float, SimdArch>;
	using Int32xSIMD = xsimd::batch<int32_t, SimdArch>;
	using UInt32xSIMD = xsimd::batch<uint32_t, SimdArch>;
	using Mask32xSIMD = xsimd::batch_bool<float, SimdArch>;
	using UIntMask32xSIMD = xsimd::batch_bool<uint32_t, SimdArch>;

	// load contiguous uint8 values and widen them to uint32 lanes
	inline UInt32xSIMD LoadUInt8AsUInt32xSIMD(
		const uint8_t* p) noexcept
	{
		const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
		return UInt32xSIMD(_mm512_cvtepu8_epi32(lo));
	}

	// load contiguous uint16 values and widen them to uint32 lanes
	inline UInt32xSIMD LoadUInt16AsUInt32xSIMD(
		const uint16_t* p) noexcept
	{
		const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
		return UInt32xSIMD(_mm512_cvtepu16_epi32(lo));
	}

} // namespace CPU
