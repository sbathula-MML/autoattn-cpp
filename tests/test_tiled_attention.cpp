
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "matrix.hpp"
#include "standard_attention.hpp"
#include "tiled_attention.hpp"

// Compare two matrices element by element.
double compare_matrices(
    const Matrix<float>& A,
    const Matrix<float>& B
) {
    if (A.rows() != B.rows() ||
        A.cols() != B.cols()) {
        throw std::runtime_error("Output shapes differ");
    }

    double max_error = 0.0;

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            double expected = A(i, j);
            double actual = B(i, j);

            if (!std::isfinite(actual)) {
                throw std::runtime_error(
                    "Non-finite tiled attention output"
                );
            }

            double error = std::abs(actual - expected);
            max_error = std::max(max_error, error);

            double tolerance =
                1e-5 + 1e-5 * std::abs(expected);

            if (error > tolerance) {
                throw std::runtime_error(
                    "Tiled attention does not match baseline"
                );
            }
        }
    }

    return max_error;
}

// Test different shapes and block sizes.
void run_test(
    std::size_t N,
    std::size_t M,
    std::size_t d,
    std::size_t dv,
    bool causal
) {
    Matrix<float> Q(N, d);
    Matrix<float> K(M, d);
    Matrix<float> V(M, dv);

    // Deterministic input generation.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            Q(i, j) = static_cast<float>(
                std::sin((i * 7 + j * 3) * 0.17)
            );
        }
    }

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            K(i, j) = static_cast<float>(
                std::cos((i * 5 + j * 11) * 0.13)
            );
        }

        for (std::size_t j = 0; j < dv; ++j) {
            V(i, j) = static_cast<float>(
                std::sin((i * 13 + j * 7) * 0.09)
            );
        }
    }

    Matrix<float> reference =
        standard_attention(Q, K, V, causal);

    const std::size_t block_sizes[] = {
        1, 3, 8, 16, 32, 64, 128
    };

    for (std::size_t block : block_sizes) {
        Matrix<float> result =
            tiled_attention(Q, K, V, block, causal);

        double error =
            compare_matrices(reference, result);

        std::cout
            << "PASS N=" << N
            << " M=" << M
            << " block=" << block
            << " causal=" << causal
            << " max_error=" << error
            << "\n";
    }
}

int main() {
    try {
        // Single-token attention
        run_test(1, 1, 1, 1, false);

        // Small self-attention
        run_test(3, 3, 4, 2, false);

        // Causal self-attention
        run_test(7, 7, 8, 4, true);

        // Different query and key lengths
        run_test(7, 11, 5, 3, false);

        // Non-multiple block boundaries
        run_test(17, 17, 16, 8, true);

        // Larger attention matrices
        run_test(64, 64, 32, 16, false);
        run_test(64, 64, 32, 16, true);

        // Verify invalid block size is rejected
        Matrix<float> X(2, 2);
        bool rejected = false;

        try {
            tiled_attention(X, X, X, 0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }

        if (!rejected) {
            throw std::runtime_error(
                "Zero block size was not rejected"
            );
        }

        std::cout
            << "\nAll tiled attention tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

