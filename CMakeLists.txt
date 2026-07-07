# CMakeLists.txt for TurboSPHarticles
cmake_minimum_required(VERSION 3.24)
project(TurboSPHarticles LANGUAGES CXX)

# --- [ language standards / build options ] ---
# ----------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_CUDA "Build GPU backend with CUDA" OFF)
option(ENABLE_VIS "Build window with visual interface and particle rendering" ON)
option(ENABLE_MPI "Build distributed headless application with MPI" OFF)

set(TSP_CUDA_ARCHITECTURE "native" CACHE STRING "CUDA architecture")
set(TSP_CPU_SIMD_ARCH "AVX256" CACHE STRING "CPU SIMD architecture")
set_property(CACHE TSP_CPU_SIMD_ARCH PROPERTY STRINGS AVX256 AVX512)

if(TSP_CPU_SIMD_ARCH STREQUAL "AVX256")
    set(TSP_CPU_SIMD_WIDTH 8)
    set(TSP_CPU_SIMD_ARCH_DEFINE TSP_CPU_SIMD_ARCH_AVX256=1)
    set(TSP_CPU_ISA_COMPILER_FLAGS_MSVC "/arch:AVX2")
    set(TSP_CPU_ISA_COMPILER_FLAGS_GNU "-mavx2 -mfma")
elseif(TSP_CPU_SIMD_ARCH STREQUAL "AVX512")
    set(TSP_CPU_SIMD_WIDTH 16)
    set(TSP_CPU_SIMD_ARCH_DEFINE TSP_CPU_SIMD_ARCH_AVX512=1)
    set(TSP_CPU_ISA_COMPILER_FLAGS_MSVC "/arch:AVX512")
    set(TSP_CPU_ISA_COMPILER_FLAGS_GNU "-mavx512f -mavx512cd -mavx512dq -mavx512bw -mfma")
else()
    message(FATAL_ERROR
        "Unsupported TSP_CPU_SIMD_ARCH='${TSP_CPU_SIMD_ARCH}'. "
        "Expected AVX256 or AVX512.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(TSP_CPU_ISA_COMPILER_FLAGS "${TSP_CPU_ISA_COMPILER_FLAGS_MSVC}")
else()
    set(TSP_CPU_ISA_COMPILER_FLAGS "${TSP_CPU_ISA_COMPILER_FLAGS_GNU}")
endif()

message(STATUS "CPU SIMD: ${TSP_CPU_SIMD_ARCH} (${TSP_CPU_SIMD_WIDTH} lanes)")

# default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# build output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# paths to certain directiories
set(TSP_EXPORT_DIR "${CMAKE_SOURCE_DIR}/Export")
set(TSP_CONFIG_DIR "${CMAKE_SOURCE_DIR}/Configs")
set(TSP_LOGS_DIR "${CMAKE_SOURCE_DIR}/Logs")

# --- [ dependencies via vcpkg ] ---
# ----------------------------------

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR
    "vcpkg toolchain not set. Use the provided CMakePresets.json or pass:\n"
    "  -DCMAKE_TOOLCHAIN_FILE=<path>/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

find_package(TBB CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)

# probe for /openmp:llvm on MSVC and fall back
if(MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/openmp:llvm" HAS_OPENMP_LLVM_FLAG)
    check_cxx_compiler_flag("/openmp:experimental" HAS_OPENMP_EXP_FLAG)
    add_library(MSVC_LLVM_OpenMP INTERFACE)

    if(HAS_OPENMP_LLVM_FLAG)
        target_compile_options(MSVC_LLVM_OpenMP INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)  # modern LLVM libomp
        if(HAS_OPENMP_EXP_FLAG)
            target_compile_options(MSVC_LLVM_OpenMP INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
        endif()
        message(STATUS "OpenMP: MSVC using /openmp:llvm (+experimental for simd)")
    else()
        # fallback to classic /openmp (older vcomp runtime)
        target_compile_options(MSVC_LLVM_OpenMP INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
        message(WARNING "OpenMP: /openmp:llvm not supported by this MSVC; falling back to /openmp.")
    endif()
else()
    # explicitly request CXX component
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(ENABLE_VIS)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(imgui CONFIG REQUIRED)
    find_package(implot CONFIG REQUIRED)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# --- [ sources ] ---
# -------------------

file(GLOB_RECURSE SRC_BASE CONFIGURE_DEPENDS "Source/Base/*.cpp")
file(GLOB_RECURSE SRC_EVAL CONFIGURE_DEPENDS "Source/Eval/*.cpp")
file(GLOB_RECURSE SRC_SIM_CORE CONFIGURE_DEPENDS "Source/Sim/Core/*.cpp")
file(GLOB_RECURSE SRC_SIM_SCENE CONFIGURE_DEPENDS "Source/Sim/Scene/*.cpp")
file(GLOB_RECURSE SRC_SIM_SYNC CONFIGURE_DEPENDS "Source/Sim/Sync/*.cpp")
file(GLOB_RECURSE SRC_SIM_CPU CONFIGURE_DEPENDS "Source/Sim/CPU/*.cpp")
list(FILTER SRC_SIM_CPU EXCLUDE REGEX "/Legacy/")

if(ENABLE_CUDA)
    file(GLOB_RECURSE SRC_SIM_GPU CONFIGURE_DEPENDS
        "Source/Sim/GPU/*.cpp"
        "Source/Sim/GPU/*.cu")
endif()

if(ENABLE_VIS)
    file(GLOB_RECURSE SRC_VIS CONFIGURE_DEPENDS "Source/Vis/*.cpp")
endif()

if(ENABLE_MPI)
    file(GLOB_RECURSE SRC_SIM_DIST CONFIGURE_DEPENDS "Source/Sim/Dist/*.cpp")
endif()

# --- [ project-wide helpers ] ---
# --------------------------------

# warnings
add_library(Project_Warnings INTERFACE)
target_compile_options(Project_Warnings INTERFACE
    # MSVC C++ compilation
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:
        /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /EHsc /utf-8
    >
    # NVCC compiling CUDA (.cu) with MSVC as host compiler
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:
        -Xcompiler=/utf-8
        -Xcompiler=/Zc:preprocessor
    >
    # non-MSVC C++ compilation
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:
        -Wall -Wextra -Wpedantic
    >
)

# platform defines
add_library(Project_Defines INTERFACE)
target_compile_definitions(Project_Defines INTERFACE
    $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN NOMINMAX _UNICODE>
    TSP_BUILD_TYPE=\"$<CONFIG>\"
    TSP_CXX_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\"
    TSP_CXX_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\"
    TSP_CPU_SIMD_ARCH_NAME=\"${TSP_CPU_SIMD_ARCH}\"
    TSP_CPU_SIMD_WIDTH=${TSP_CPU_SIMD_WIDTH}
    TSP_CPU_ISA_COMPILER_FLAGS=\"${TSP_CPU_ISA_COMPILER_FLAGS}\"
    ${TSP_CPU_SIMD_ARCH_DEFINE}
)

# C++ optimization flags
add_library(Project_Options INTERFACE)
target_compile_options(Project_Options INTERFACE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
    >
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    >
    # CUDA-specific options, separate from MSVC/GCC flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2 -lineinfo>
    >
)

# CPU SIMD ISA flags required by the selected SELL vector width
if(TSP_CPU_SIMD_ARCH STREQUAL "AVX512") # 16-wide SIMD processing of floats
    target_compile_options(Project_Options INTERFACE
        # MSVC C++ only
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:
            /arch:AVX512
        >
        # GCC/Clang C++ only
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:
            -mavx512f -mavx512cd -mavx512dq -mavx512bw -mfma
        >
    )
else() # 8-wide SIMD processing of floats
    target_compile_options(Project_Options INTERFACE
        # MSVC C++ only
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:
            /arch:AVX2
        >
        # GCC/Clang C++ only
        $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:
            -mavx2 -mfma
        >
    )
endif()

# link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

# consistent debug info format
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")

# --- [ headless library ] ---
# ----------------------------

add_library(TSP_H STATIC
    ${SRC_BASE}
    ${SRC_EVAL}
    ${SRC_SIM_CORE}
    ${SRC_SIM_CPU}
    $<$<BOOL:${ENABLE_CUDA}>:${SRC_SIM_GPU}>
    ${SRC_SIM_SCENE}
    ${SRC_SIM_SYNC}
)

target_include_directories(TSP_H PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_compile_definitions(TSP_H PUBLIC
    TSP_EXPORT_DIR=\"${TSP_EXPORT_DIR}\"
    TSP_CONFIG_DIR=\"${TSP_CONFIG_DIR}\"
    TSP_LOGS_DIR=\"${TSP_LOGS_DIR}\"
)

target_link_libraries(TSP_H PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::boost
    glm::glm
    xsimd
    TBB::tbb
    TBB::tbbmalloc
    $<$<BOOL:${ENABLE_CUDA}>:CUDA::cudart>
    Project_Warnings
    Project_Defines
    Project_Options
    $<$<CXX_COMPILER_ID:MSVC>:MSVC_LLVM_OpenMP>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>
)

if(ENABLE_CUDA)
    target_compile_definitions(TSP_H PUBLIC TURBOSPHARTICLES_WITH_CUDA=1)
endif()

if(ENABLE_CUDA AND SRC_SIM_GPU)
    set_target_properties(TSP_H PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "${TSP_CUDA_ARCHITECTURE}"
    )
    # keep IPO off when CUDA is enabled to avoid nvlink/LTO clashes
    set_property(TARGET TSP_H PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# --- [ visual library (optional add-on) ] ---
# --------------------------------------------

if(ENABLE_VIS)
    add_library(TSP_V STATIC
        ${SRC_VIS}
    )
    target_link_libraries(TSP_V PUBLIC
        TSP_H # build on top of the headless base
        OpenGL::GL
        glad::glad
        glfw
        imgui::imgui
        implot::implot
    )
    # avoid GLFW including <GL/gl.h> before glad
    target_compile_definitions(TSP_V PUBLIC
        TURBOSPHARTICLES_WITH_VIZ=1
        GLFW_INCLUDE_NONE
        IMGUI_IMPL_OPENGL_LOADER_GLAD
        TSP_CONFIG_DIR=\"${TSP_CONFIG_DIR}\"
    )
    if(ENABLE_CUDA)
        # final executables perform CUDA device linking
        set_target_properties(TSP_V PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    endif()
endif()

# --- [ distributed library (optional add-on) ] ---
# -------------------------------------------------

if(ENABLE_MPI)
    add_library(TSP_D STATIC
        ${SRC_SIM_DIST}
    )
    target_link_libraries(TSP_D PUBLIC
        TSP_H # build on top of the headless base
        MPI::MPI_CXX
    )
    target_compile_definitions(TSP_D PUBLIC
        TURBOSPHARTICLES_WITH_MPI=1
    )
endif()

# --- [ apps ] ---
# ----------------

add_executable(TSP_Headless
    "Apps/main_headless.cpp"
    "Apps/app_runtime.cpp")
target_link_libraries(TSP_Headless PRIVATE 
    TSP_H
    CLI11::CLI11
)

if(ENABLE_VIS)
    add_executable(TSP_Visual
        "Apps/main_visual.cpp"
        "Apps/app_runtime.cpp")
    target_link_libraries(TSP_Visual PRIVATE
        TSP_V
        CLI11::CLI11
    )
endif()

if(ENABLE_MPI)
    add_executable(TSP_Distributed
        "Apps/main_distributed.cpp"
        "Apps/app_runtime.cpp")
    target_link_libraries(TSP_Distributed PRIVATE
        TSP_D
        CLI11::CLI11
    )
endif()

# make sure device link works on final executables and turn IPO off when CUDA is on
if(ENABLE_CUDA)
    set_target_properties(TSP_Headless PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET TSP_Headless PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    if(ENABLE_VIS)
        set_target_properties(TSP_Visual  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_property(TARGET TSP_Visual  PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    if(ENABLE_MPI)
        set_target_properties(TSP_Distributed PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_property(TARGET TSP_Distributed PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
endif()
