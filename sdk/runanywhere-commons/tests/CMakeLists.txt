# CMakeLists.txt for RunAnywhere Commons Tests
#
# Build with: cmake -B build-test -DRAC_BUILD_TESTS=ON && cmake --build build-test --target run_tests

cmake_minimum_required(VERSION 3.16)

# =============================================================================
# GoogleTest Setup
# =============================================================================

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

# =============================================================================
# Test Sources
# =============================================================================

set(RAC_TEST_SOURCES
    # Core tests
    core/test_unified_memory_manager.cpp

    # Backend allocator tests (no external backend dependency)
    backends/llamacpp/test_llamacpp_pooled_buffer.cpp

    # Feature tests
    features/voice_agent/test_hybrid_voice_pipeline.cpp
    features/voice_agent/test_rac_hybrid_voice_pipeline.cpp
)

# LlamaCpp pooled buffer source - has no dependency on actual llama.cpp,
# only on MemorySegment from rac_commons
set(RAC_BACKEND_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/backends/llamacpp/llamacpp_pooled_buffer.cpp
)

# =============================================================================
# Test Executable
# =============================================================================

add_executable(rac_tests ${RAC_TEST_SOURCES} ${RAC_BACKEND_TEST_SOURCES})

target_include_directories(rac_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(rac_tests
    PRIVATE
        rac_commons
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

# C++17 required
target_compile_features(rac_tests PRIVATE cxx_std_17)

# =============================================================================
# Test Registration
# =============================================================================

include(GoogleTest)
gtest_discover_tests(rac_tests)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS rac_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# =============================================================================
# Test Data Setup
# =============================================================================

# Create test data directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data)

# Set environment variable for the test executable via target property
target_compile_definitions(rac_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/test_data"
)

# =============================================================================
# Coverage (optional)
# =============================================================================

option(RAC_ENABLE_COVERAGE "Enable code coverage" OFF)

if(RAC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(rac_tests PRIVATE --coverage)
        target_link_options(rac_tests PRIVATE --coverage)
        message(STATUS "Code coverage enabled")
    else()
        message(WARNING "Coverage not supported with this compiler")
    endif()
endif()
