#include <iostream>
#include <limits>
#include <stdexcept>

#include "matrix.hpp"

int main() {
    Matrix<float> A(2, 3);

    A(0, 0) = 10.0f;
    A(1, 2) = 20.0f;

    if (A(0, 0) != 10.0f ||
        A(1, 2) != 20.0f) {
        throw std::runtime_error(
            "Element access failed"
        );
    }

    bool caught = false;

    // Column overflow that previously
    // could access the next row.
    try {
        A(0, 3) = 5.0f;
    } catch (const std::out_of_range&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Column bounds check failed"
        );
    }

    caught = false;

    try {
        A(2, 0) = 5.0f;
    } catch (const std::out_of_range&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Row bounds check failed"
        );
    }

    caught = false;

    try {
        Matrix<float> huge(
            std::numeric_limits<std::size_t>::max(),
            2
        );
    } catch (const std::overflow_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Allocation overflow check failed"
        );
    }

    std::cout
        << "All matrix boundary tests passed!\n";

    return 0;
}
