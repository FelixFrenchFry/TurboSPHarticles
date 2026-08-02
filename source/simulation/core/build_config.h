// build_config.h

#pragma once

// external
#include <cstdint>



#ifndef TSP_GIT_COMMIT_HASH
	#define TSP_GIT_COMMIT_HASH "unknown"
#endif

#ifndef TSP_BUILD_TYPE
	#define TSP_BUILD_TYPE "unknown"
#endif

#ifndef TSP_CXX_COMPILER_ID
	#define TSP_CXX_COMPILER_ID "unknown"
#endif

#ifndef TSP_CXX_COMPILER_VERSION
	#define TSP_CXX_COMPILER_VERSION "unknown"
#endif

#ifndef TSP_CPU_SIMD_ARCH_NAME
	#define TSP_CPU_SIMD_ARCH_NAME "unknown"
#endif

#ifndef TSP_CPU_SIMD_WIDTH
	#define TSP_CPU_SIMD_WIDTH 0
#endif

#ifndef TSP_CPU_ISA_COMPILER_FLAGS
	#define TSP_CPU_ISA_COMPILER_FLAGS "unknown"
#endif



namespace Sim
{
	namespace BuildConfig
	{
		constexpr const char* gitCommitHash = TSP_GIT_COMMIT_HASH;
		constexpr const char* buildType = TSP_BUILD_TYPE;
		constexpr const char* cxxCompilerId = TSP_CXX_COMPILER_ID;
		constexpr const char* cxxCompilerVersion = TSP_CXX_COMPILER_VERSION;
		constexpr const char* cpuSimdArch = TSP_CPU_SIMD_ARCH_NAME;
		constexpr uint32_t cpuSimdWidth = TSP_CPU_SIMD_WIDTH;
		constexpr const char* cpuIsaCompilerFlags = TSP_CPU_ISA_COMPILER_FLAGS;

	} // namespace BuildConfig

} // namespace Sim
