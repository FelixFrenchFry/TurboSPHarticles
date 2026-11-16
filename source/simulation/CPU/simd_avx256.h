// simd_avx256.h
#pragma once

// external
#include <cstdint>
#include <immintrin.h>
#include <xsimd/xsimd.hpp>



namespace CPU
{
	constexpr uint32_t N_SIMD_WIDTH = 8u;
	constexpr uint32_t N_SIMD_ALIGNMENT = 32u;

	// 8-wide CPU SIMD types used by SELL-style kernels
	using SimdArch = xsimd::fma3<xsimd::avx2>;
	using Float32xSIMD = xsimd::batch<float, SimdArch>;
	using Int32xSIMD = xsimd::batch<int32_t, SimdArch>;
	using UInt32xSIMD = xsimd::batch<uint32_t, SimdArch>;
	using Mask32xSIMD = xsimd::batch_bool<float, SimdArch>;
	using UIntMask32xSIMD = xsimd::batch_bool<uint32_t, SimdArch>;

	// load contiguous uint8 values and widen them to uint32 lanes
	inline UInt32xSIMD LoadUInt8AsUInt32xSIMD(
		const uint8_t* p) noexcept
	{
		const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
		return UInt32xSIMD(_mm256_cvtepu8_epi32(lo));
	}

	// load contiguous uint16 values and widen them to uint32 lanes
	inline UInt32xSIMD LoadUInt16AsUInt32xSIMD(
		const uint16_t* p) noexcept
	{
		const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
		return UInt32xSIMD(_mm256_cvtepu16_epi32(lo));
	}

	// gather scattered uint16 values by index and widen them to uint32 lanes
	// (every lane reads 2 bytes past its value)
	inline UInt32xSIMD GatherUInt16AsUInt32xSIMD(
		const uint8_t* p,
		const UInt32xSIMD idx) noexcept
	{
		const __m256i v = _mm256_i32gather_epi32(reinterpret_cast<const int*>(p), idx, 2);
		return UInt32xSIMD(_mm256_and_si256(v, _mm256_set1_epi32(0xffff)));
	}

	// gather scattered uint32 values by index from unaligned bytes
	inline UInt32xSIMD GatherUInt32xSIMD(
		const uint8_t* p,
		const UInt32xSIMD idx) noexcept
	{
		return UInt32xSIMD(_mm256_i32gather_epi32(reinterpret_cast<const int*>(p), idx, 4));
	}

} // namespace CPU
