find_package(GTest CONFIG REQUIRED)

# keep the maintained suite explicit; other tests in this directory are opt-in
add_executable(TSP_Tests
    "unit/cell_frame.cpp"
    "unit/scratch_storage.cpp"
    "unit/snapshot_format.cpp"
    "unit/sph_kernels.cpp"
    "unit/triangle_surface.cpp"
)
target_link_libraries(TSP_Tests PRIVATE TSP_H GTest::gtest_main)

if(ENABLE_CUDA)
    set_target_properties(TSP_Tests PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
endif()

include(GoogleTest)
gtest_discover_tests(TSP_Tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS unit TIMEOUT 30
)
