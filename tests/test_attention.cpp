
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "matrix.hpp"
#include "standard_attention.hpp"

void check_close(float actual, float expected,
                 float tolerance = 1e-5f) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error("Numerical test failed");
    }
}

int main() {
    // Test 1: Standard attention
    Matrix<float> Q(2, 2);
    Matrix<float> K(2, 2);
    Matrix<float> V(2, 2);

    Q(0, 0) = 1; Q(0, 1) = 0;
    Q(1, 0) = 0; Q(1, 1) = 1;

    K(0, 0) = 1; K(0, 1) = 0;
    K(1, 0) = 0; K(1, 1) = 1;

    V(0, 0) = 10; V(0, 1) = 0;
    V(1, 0) = 0;  V(1, 1) = 20;

    Matrix<float> O = standard_attention(Q, K, V);

    float a = 1.0f / std::sqrt(2.0f);
    float w = std::exp(a) / (std::exp(a) + 1.0f);

    check_close(O(0, 0), 10 * w);
    check_close(O(0, 1), 20 * (1 - w));
    check_close(O(1, 0), 10 * (1 - w));
    check_close(O(1, 1), 20 * w);

    std::cout << "Test 1 passed - Standard attention\n";

    // Test 2: Causal attention
    Matrix<float> C = standard_attention(Q, K, V, true);

    // First token can attend only to itself
    check_close(C(0, 0), 10);
    check_close(C(0, 1), 0);

    // Second token can attend to both tokens
    check_close(C(1, 0), 10 * (1 - w));
    check_close(C(1, 1), 20 * w);

    std::cout << "Test 2 passed - Causal masking\n";

    // Test 3: Uniform attention weights
    Matrix<float> Z(2, 2);

    Matrix<float> U = standard_attention(Z, Z, V);

    check_close(U(0, 0), 5);
    check_close(U(0, 1), 10);
    check_close(U(1, 0), 5);
    check_close(U(1, 1), 10);

    std::cout << "Test 3 passed - Uniform softmax\n";

    // Test 4: Large attention scores
    Matrix<float> Qbig(1, 2);
    Matrix<float> Kbig(2, 2);

    Qbig(0, 0) = 1e10f;
    Qbig(0, 1) = 0;

    Kbig(0, 0) = 1e10f;
    Kbig(0, 1) = 0;
    Kbig(1, 0) = -1e10f;
    Kbig(1, 1) = 0;

    Matrix<float> B =
        standard_attention(Qbig, Kbig, V);

    check_close(B(0, 0), 10);
    check_close(B(0, 1), 0);

    std::cout << "Test 4 passed - Numerical stability\n";

    // Test 5: Incompatible dimensions
    bool caught = false;

    try {
        Matrix<float> badK(3, 4);
        standard_attention(Q, badK, V);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Dimension validation failed"
        );
    }

    std::cout << "Test 5 passed - Dimension validation\n";

    // Test 6: Reject non-finite inputs
    Q(0, 0) = std::numeric_limits<float>::quiet_NaN();
    caught = false;

    try {
        standard_attention(Q, K, V);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Non-finite input validation failed"
        );
    }

    std::cout << "Test 6 passed - Invalid input handling\n";

    std::cout << "\nAll attention tests passed!\n";
    return 0;
}

