
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "csv_io.hpp"

using namespace autoattn;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_test_file(
    const std::string& filename,
    const std::string& content
) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot create test file"
        );
    }

    file << content;

    if (!file) {
        throw std::runtime_error(
            "Cannot write test file"
        );
    }
}

void expect_failure(const std::string& filename) {
    bool failed = false;

    try {
        auto matrix = load_csv_matrix(filename);
        (void)matrix;
    } catch (const std::exception&) {
        failed = true;
    }

    require(failed, "Expected CSV loading to fail");
}

int main() {
    const std::string filename =
        "tests/test_input_temp.csv";

    try {
        // Test 1 - Valid matrix
        write_test_file(
            filename,
            "1,2,3\n4,5,6\n"
        );

        auto matrix = load_csv_matrix(filename);

        require(matrix.rows() == 2, "Wrong row count");
        require(matrix.cols() == 3, "Wrong column count");
        require(matrix(0, 0) == 1, "Wrong first value");
        require(matrix(1, 2) == 6, "Wrong last value");

        std::cout << "Test 1 passed - Valid CSV\n";

        // Test 2 - Inconsistent dimensions
        write_test_file(
            filename,
            "1,2,3\n4,5\n"
        );

        expect_failure(filename);

        std::cout
            << "Test 2 passed - Dimension validation\n";

        // Test 3 - Invalid numeric value
        write_test_file(
            filename,
            "1,2\n3,hello\n"
        );

        expect_failure(filename);

        std::cout
            << "Test 3 passed - Invalid numeric data\n";

        // Test 4 - Non-finite input
        write_test_file(
            filename,
            "1,2\n3,nan\n"
        );

        expect_failure(filename);

        std::cout
            << "Test 4 passed - Non-finite rejection\n";

        // Test 5 - Trailing comma
        write_test_file(
            filename,
            "1,2,\n3,4\n"
        );

        expect_failure(filename);

        std::cout
            << "Test 5 passed - Trailing comma rejection\n";

        // Test 6 - Empty file
        write_test_file(filename, "");

        expect_failure(filename);

        std::cout
            << "Test 6 passed - Empty file rejection\n";

        // Test 7 - Missing file
        std::remove(filename.c_str());

        expect_failure(filename);

        std::cout
            << "Test 7 passed - Missing file handling\n";

        std::cout
            << "\nAll CSV input tests passed!\n";

    } catch (const std::exception& error) {
        std::remove(filename.c_str());

        std::cerr
            << "FAILED: " << error.what() << "\n";

        return 1;
    }

    return 0;
}

