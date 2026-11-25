
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "attention_engine.hpp"
#include "calibration_io.hpp"

using namespace autoattn;

int main() {
    try {
        // Load actual measurements from disk.
        std::vector<BenchmarkEntry> profile =
            load_profile(
                "benchmarks/calibration_profile.csv"
            );

        std::cout << "Loaded "
                  << profile.size()
                  << " calibration entries\n";

        if (profile.size() != 8) {
            throw std::runtime_error(
                "Expected eight calibration entries"
            );
        }

        const std::size_t N = 256;
        const std::size_t d = 64;

        Matrix<float> Q(N, d);
        Matrix<float> K(N, d);
        Matrix<float> V(N, d);

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                Q(i, j) = std::sin(
                    (i * 7 + j * 3) * 0.17
                );

                K(i, j) = std::cos(
                    (i * 5 + j * 11) * 0.13
                );

                V(i, j) = std::sin(
                    (i * 13 + j * 7) * 0.09
                );
            }
        }

        // Independent baseline execution.
        auto reference =
            standard_attention(Q, K, V, false);

        for (std::size_t budget :
             {600u, 4096u, 300000u}) {

            auto result = run_attention(
                Q, K, V, false, budget, profile
            );

            if (!result.choice.calibrated) {
                throw std::runtime_error(
                    "Calibration data was not used"
                );
            }

            if (result.choice.scratch_bytes > budget) {
                throw std::runtime_error(
                    "Memory budget exceeded"
                );
            }

            double max_error = 0.0;

            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    double error = std::abs(
                        static_cast<double>(
                            reference(i, j)
                        ) - result.output(i, j)
                    );

                    if (!std::isfinite(error)) {
                        throw std::runtime_error(
                            "Non-finite output error"
                        );
                    }

                    if (error > max_error) {
                        max_error = error;
                    }
                }
            }

            if (max_error > 1e-5) {
                throw std::runtime_error(
                    "Output correctness test failed"
                );
            }

            std::cout
                << "Budget = " << budget
                << " | Algorithm = "
                << (result.choice.algorithm ==
                    Algorithm::Tiled
                    ? "tiled" : "standard")
                << " | Block = "
                << result.choice.block_size
                << " | Error = "
                << max_error
                << "\n";
        }

        std::cout
            << "\nAll calibration loading tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

