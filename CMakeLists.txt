# CMakeLists.txt for TurboSPHarticles
cmake_minimum_required(VERSION 3.24)
project(TurboSPHarticles LANGUAGES CXX)

# ==============================================
# ===== language standards / build options =====

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_CUDA "Build GPU backend with CUDA" OFF)
option(ENABLE_VIS "Build visual interface" ON)

# default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# build output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ==================================
# ===== dependencies via vcpkg =====

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR
    "vcpkg toolchain not set. Use the provided CMakePresets.json or pass:\n"
    "  -DCMAKE_TOOLCHAIN_FILE=<path>/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

find_package(benchmark CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(HighFive CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# probe for /openmp:llvm on MSVC and fall back
if(MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/openmp:llvm" HAS_OPENMP_LLVM_FLAG)
    add_library(MSVC_LLVM_OpenMP INTERFACE)

    if(HAS_OPENMP_LLVM_FLAG)
        target_compile_options(MSVC_LLVM_OpenMP INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)  # modern LLVM libomp
        message(STATUS "OpenMP: MSVC using /openmp:llvm (LLVM libomp)")
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
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89 90 100 120)
    #if(NOT CMAKE_CUDA_ARCHITECTURES)
    #    set(CMAKE_CUDA_ARCHITECTURES native)
    #endif()
endif()

if(ENABLE_VIS)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(imgui CONFIG REQUIRED)
    find_package(implot CONFIG REQUIRED)
endif()

# ===================
# ===== sources =====

file(GLOB_RECURSE SRC_BASE CONFIGURE_DEPENDS "Source/Base/*.cpp")
file(GLOB_RECURSE SRC_EVAL CONFIGURE_DEPENDS "Source/Eval/*.cpp")
file(GLOB_RECURSE SRC_SIM_CORE CONFIGURE_DEPENDS "Source/Sim/Core/*.cpp")
file(GLOB_RECURSE SRC_SIM_SCENE CONFIGURE_DEPENDS "Source/Sim/Scene/*.cpp")
file(GLOB_RECURSE SRC_SIM_SYNC CONFIGURE_DEPENDS "Source/Sim/Sync/*.cpp")
file(GLOB_RECURSE SRC_SIM_CPU CONFIGURE_DEPENDS "Source/Sim/CPU/*.cpp")

if(ENABLE_CUDA)
    file(GLOB_RECURSE SRC_SIM_GPU CONFIGURE_DEPENDS
        "Source/Sim/GPU/*.cpp"
        "Source/Sim/GPU/*.cu")
endif()

if(ENABLE_VIS)
    file(GLOB_RECURSE SRC_VIS CONFIGURE_DEPENDS "Source/Vis/*.cpp")
endif()

# ================================
# ===== project-wide helpers =====

# warnings
add_library(Project_Warnings INTERFACE)
target_compile_options(Project_Warnings INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /EHsc /utf-8>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# platform defines
add_library(Project_Defines INTERFACE)
target_compile_definitions(Project_Defines INTERFACE
    $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN NOMINMAX _UNICODE>
)

# C++ optimization flags
add_library(Project_Options INTERFACE)
target_compile_options(Project_Options INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
    >
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    >
)

# optional native archtitecture-specific optimizations
# (activated with "-DENABLE_NATIVE_OPT=ON")
option(ENABLE_NATIVE_OPT "Enable -march=native (or /arch:AVX2) for Release-like builds" OFF)
if(ENABLE_NATIVE_OPT)
    target_compile_options(Project_Options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/arch:AVX2>
        >
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-march=native>
        >
    )
endif()

# link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

# consistent debug info format
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")

# ============================
# ===== headless library =====

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

target_link_libraries(TSP_H PUBLIC
    benchmark::benchmark
    spdlog::spdlog
    HighFive
    CLI11::CLI11
    glm::glm
    $<$<BOOL:${ENABLE_CUDA}>:CUDA::cudart>
    Project_Warnings
    Project_Defines
    Project_Options
    $<$<CXX_COMPILER_ID:MSVC>:MSVC_LLVM_OpenMP>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>
)

if(ENABLE_CUDA AND SRC_SIM_GPU)
    set_target_properties(TSP_H PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # CUDA optimization flags
    target_compile_options(TSP_H PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Debug>:-O0 -g>
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2 -lineinfo>
        >
    )
    # TODO: comment this out if compilation works fine without it?
    set_property(TARGET TSP_H PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# ============================================
# ===== visual library (optional add-on) =====

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
        Project_Warnings
        Project_Defines
        Project_Options
    )
    # avoid GLFW including <GL/gl.h> before glad
    target_compile_definitions(TSP_V PUBLIC
        TURBOSPHARTICLES_WITH_VIZ=1
        GLFW_INCLUDE_NONE
        IMGUI_IMPL_OPENGL_LOADER_GLAD
    )
endif()

# ================
# ===== apps =====

add_executable(TSP_Headless "Apps/main_headless.cpp")
target_link_libraries(TSP_Headless PRIVATE 
    TSP_H Project_Warnings Project_Defines Project_Options
    $<$<CXX_COMPILER_ID:MSVC>:MSVC_LLVM_OpenMP>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>
)
target_compile_definitions(TSP_Headless PRIVATE TURBOSPHARTICLES_HEADLESS=1)

if(ENABLE_VIS)
    add_executable(TSP_Visual "Apps/main_visual.cpp")
    target_link_libraries(TSP_Visual PRIVATE
    TSP_V Project_Warnings Project_Defines Project_Options
    $<$<CXX_COMPILER_ID:MSVC>:MSVC_LLVM_OpenMP>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:OpenMP::OpenMP_CXX>
)
endif()
