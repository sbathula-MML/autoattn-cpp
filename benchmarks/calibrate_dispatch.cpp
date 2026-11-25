
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention_engine.hpp"

using Clock = std::chrono::steady_clock;
using namespace autoattn;

volatile double checksum = 0.0;

// Measure median runtime after warm-up.
template <typename Function>
double measure(Function function) {
    constexpr int repeats = 9;

    std::vector<double> times;

    for (int i = 0; i < 3; ++i) {
        auto output = function();
        checksum += output(0, 0);
    }

    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now();

        auto output = function();

        auto end = Clock::now();

        checksum += output(0, 0);

        double ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    return times[repeats / 2];
}

std::string algorithm_name(Algorithm algorithm) {
    return algorithm == Algorithm::Standard
        ? "standard"
        : "tiled";
}

// Save calibration results to CSV.
void save_profile(
    const std::vector<BenchmarkEntry>& profile
) {
    std::ofstream file(
        "benchmarks/calibration_profile.csv"
    );

    if (!file) {
        throw std::runtime_error(
            "Could not create calibration file"
        );
    }

    file << "N,M,d,dv,causal,algorithm,"
         << "block_size,median_ms\n";

    file << std::setprecision(17);

    for (const auto& entry : profile) {
        file
            << entry.N << ","
            << entry.M << ","
            << entry.d << ","
            << entry.dv << ","
            << static_cast<int>(entry.causal) << ","
            << algorithm_name(entry.algorithm) << ","
            << entry.block_size << ","
            << entry.median_ms << "\n";
    }

    file.close();

    if (!file) {
        throw std::runtime_error(
            "Failed to save calibration results"
        );
    }

    std::cout
        << "\nCalibration profile saved successfully\n";
}

int main() {
    try {
        const std::size_t N = 256;
        const std::size_t d = 64;
        const std::size_t dv = 64;
        const bool causal = false;

        Matrix<float> Q(N, d);
        Matrix<float> K(N, d);
        Matrix<float> V(N, dv);

        // Generate deterministic inputs.
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

        std::vector<BenchmarkEntry> profile;

        std::cout << std::fixed
                  << std::setprecision(4);

        std::cout << "Calibrating AutoAttn\n";
        std::cout << "Sequence length = "
                  << N << "\n\n";

        // Benchmark standard attention.
        double standard_ms = measure([&]() {
            return standard_attention(
                Q, K, V, causal
            );
        });

        profile.push_back({
            N, N, d, dv, causal,
            Algorithm::Standard, 0, standard_ms
        });

        std::cout
            << "Standard attention = "
            << standard_ms << " ms\n\n";

        // Benchmark tiled attention.
        for (std::size_t block :
             {1u, 8u, 16u, 32u, 64u, 128u, 256u}) {

            double tiled_ms = measure([&]() {
                return tiled_attention(
                    Q, K, V, block, causal
                );
            });

            profile.push_back({
                N, N, d, dv, causal,
                Algorithm::Tiled, block, tiled_ms
            });

            std::cout
                << "Tiled block "
                << std::setw(3) << block
                << " = " << tiled_ms << " ms\n";
        }

        // Save the measured calibration data.
        save_profile(profile);

        // Compute reference output.
        Matrix<float> reference =
            standard_attention(Q, K, V, causal);

        std::cout << "\nAdaptive selection\n";

        for (std::size_t budget :
             {600u, 4096u, 300000u}) {

            auto result = run_attention(
                Q, K, V,
                causal,
                budget,
                profile
            );

            double max_error = 0.0;

            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < dv; ++j) {
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

                    max_error = std::max(
                        max_error, error
                    );
                }
            }

            if (max_error > 1e-5 ||
                result.choice.scratch_bytes > budget ||
                !result.choice.calibrated) {
                throw std::runtime_error(
                    "Dispatcher validation failed"
                );
            }

            std::cout
                << "\nBudget = " << budget << " bytes"
                << "\nSelected = "
                << algorithm_name(
                    result.choice.algorithm
                )
                << "\nBlock size = "
                << result.choice.block_size
                << "\nEstimated scratch = "
                << result.choice.scratch_bytes
                << " bytes"
                << "\nMaximum output error = "
                << max_error
                << "\n";
        }

        std::cout
            << "\nAll calibration checks passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

