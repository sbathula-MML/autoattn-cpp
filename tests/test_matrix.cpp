
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "../include/matrix.hpp"

int main() {
    // Create a 2 x 3 matrix
    Matrix<float> A(2, 3);

    // Assign values
    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(0, 2) = 3.0f;
    A(1, 0) = 4.0f;
    A(1, 1) = 5.0f;
    A(1, 2) = 6.0f;

    // Check dimensions
    assert(A.rows() == 2);
    assert(A.cols() == 3);

    // Check element access
    assert(A(0, 0) == 1.0f);
    assert(A(1, 2) == 6.0f);

    // Check row-major storage
    assert(A.data()[3] == 4.0f);

    // Check invalid dimensions
    bool caught = false;

    try {
        Matrix<float> B(0, 3);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    assert(caught);

    // Check out-of-bounds access
    caught = false;

    try {
        A(2, 0) = 10.0f;
    } catch (const std::out_of_range&) {
        caught = true;
    }

    assert(caught);

    std::cout << "All Matrix tests passed!\n";
    return 0;
}

