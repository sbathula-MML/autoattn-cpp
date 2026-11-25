
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "attention_engine.hpp"

using namespace autoattn;

// Compare two attention outputs.
void compare_outputs(
    const Matrix<float>& A,
    const Matrix<float>& B
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("Output shapes differ");
    }

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            double error = std::abs(
                static_cast<double>(A(i, j)) - B(i, j)
            );

            if (!std::isfinite(error) ||
                error > 1e-5) {
                throw std::runtime_error(
                    "Attention outputs do not match"
                );
            }
        }
    }
}

int main() {
    try {
        const std::size_t N = 17;
        const std::size_t d = 8;
        const std::size_t dv = 6;

        Matrix<float> Q(N, d);
        Matrix<float> K(N, d);
        Matrix<float> V(N, dv);

        // Generate deterministic input matrices.
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                Q(i, j) = std::sin((i + j) * 0.13);
                K(i, j) = std::cos((i * 3 + j) * 0.17);
            }

            for (std::size_t j = 0; j < dv; ++j) {
                V(i, j) = std::sin((i * 5 + j) * 0.11);
            }
        }

        // Independent execution of the baseline.
        Matrix<float> reference =
            standard_attention(Q, K, V, false);

        // Test 1 - Automatic fallback uses tiled attention.
        auto result = run_attention(
            Q, K, V, false, 512
        );

        if (result.choice.algorithm != Algorithm::Tiled ||
            result.choice.calibrated ||
            result.choice.scratch_bytes > 512) {
            throw std::runtime_error(
                "Automatic selection failed"
            );
        }

        compare_outputs(reference, result.output);

        std::cout
            << "Test 1 passed - Automatic tiled execution\n";

        // Test 2 - Causal attention.
        auto causal_result = run_attention(
            Q, K, V, true, 512
        );

        Matrix<float> causal_reference =
            standard_attention(Q, K, V, true);

        compare_outputs(
            causal_reference,
            causal_result.output
        );

        std::cout
            << "Test 2 passed - Causal attention\n";

        // Test 3 - Simulated calibration data.
        // These timings are test fixtures, not benchmarks.
        std::vector<BenchmarkEntry> timings = {
            {N, N, d, dv, false,
             Algorithm::Standard, 0, 1.0},

            {N, N, d, dv, false,
             Algorithm::Tiled, 128, 2.0}
        };

        auto calibrated_result = run_attention(
            Q, K, V, false, 4096, timings
        );

        if (calibrated_result.choice.algorithm
                != Algorithm::Standard ||
            !calibrated_result.choice.calibrated) {
            throw std::runtime_error(
                "Calibrated selection failed"
            );
        }

        compare_outputs(
            reference,
            calibrated_result.output
        );

        std::cout
            << "Test 3 passed - Calibrated execution\n";

        // Test 4 - Reject insufficient memory.
        bool rejected = false;

        try {
            run_attention(Q, K, V, false, 1);
        } catch (const std::runtime_error&) {
            rejected = true;
        }

        if (!rejected) {
            throw std::runtime_error(
                "Memory limit was not enforced"
            );
        }

        std::cout
            << "Test 4 passed - Memory budget\n";

        // Test 5 - Verify very small standard attention.
        Matrix<float> smallQ(1, 2);
        Matrix<float> smallK(1, 2);
        Matrix<float> smallV(1, 64);

        auto small_result = run_attention(
            smallQ, smallK, smallV, false, 4
        );

        if (small_result.choice.algorithm
            != Algorithm::Standard) {
            throw std::runtime_error(
                "Small standard fallback failed"
            );
        }

        std::cout
            << "Test 5 passed - Standard fallback\n";

        std::cout
            << "\nAll attention engine tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED - " << e.what() << "\n";
        return 1;
    }

    return 0;
}

