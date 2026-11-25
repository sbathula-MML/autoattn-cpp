
cmake_minimum_required(VERSION 3.16)

project(AutoAttn VERSION 1.0 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared header files
add_library(autoattn_headers INTERFACE)

target_include_directories(
    autoattn_headers
    INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

# Helper function for creating executables
function(add_autoattn_program name source)
    add_executable(${name} ${source})

    target_link_libraries(
        ${name}
        PRIVATE autoattn_headers
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(
            ${name}
            PRIVATE
            -Wall
            -Wextra
            -Werror
        )
    endif()
endfunction()

# Main application
add_autoattn_program(
    autoattn
    src/main.cpp
)

# Enable automated testing
enable_testing()

# Helper function for creating tests
function(add_autoattn_test name source)
    add_autoattn_program(${name} ${source})

    add_test(
        NAME ${name}
        COMMAND $<TARGET_FILE:${name}>
    )

    # Tests that use CSV files expect
    # the project root as their working directory
    set_tests_properties(
        ${name}
        PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
endfunction()

# Register all C++ tests
add_autoattn_test(
    test_matrix
    tests/test_matrix.cpp
)

add_autoattn_test(
    test_attention
    tests/test_attention.cpp
)

add_autoattn_test(
    test_tiled_attention
    tests/test_tiled_attention.cpp
)

add_autoattn_test(
    test_memory_estimator
    tests/test_memory_estimator.cpp
)

add_autoattn_test(
    test_dispatch_policy
    tests/test_dispatch_policy.cpp
)

add_autoattn_test(
    test_attention_engine
    tests/test_attention_engine.cpp
)

add_autoattn_test(
    test_calibration_io
    tests/test_calibration_io.cpp
)

add_autoattn_test(
    test_csv_io
    tests/test_csv_io.cpp
)


# Keep assertions enabled in test programs
# even when building in Release mode.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(test_target
        test_matrix
        test_attention
        test_tiled_attention
        test_memory_estimator
        test_dispatch_policy
        test_attention_engine
        test_calibration_io
        test_csv_io
    )
        target_compile_options(
            ${test_target} PRIVATE -UNDEBUG
        )
    endforeach()
endif()

# Python integration tests
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME test_cli_integration
    COMMAND
        ${CMAKE_COMMAND} -E env
        "AUTOATTN_EXECUTABLE=$<TARGET_FILE:autoattn>"
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/tests/test_cli_integration.py"
)

set_tests_properties(
    test_cli_integration
    PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# Matrix boundary and allocation safety tests
add_autoattn_test(
    test_matrix_boundaries
    tests/test_matrix_boundaries.cpp
)

# Protect input files from accidental overwriting
add_test(
    NAME test_path_safety
    COMMAND
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/tests/test_path_safety.py"
        "$<TARGET_FILE:autoattn>"
        "${PROJECT_SOURCE_DIR}/benchmarks/calibration_grid.csv"
)

# Calibration file validation
add_test(
    NAME test_calibration_validation
    COMMAND
        ${CMAKE_COMMAND} -E env
        "AUTOATTN_EXECUTABLE=$<TARGET_FILE:autoattn>"
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/tests/test_calibration_validation.py"
)

set_tests_properties(
    test_calibration_validation
    PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
