
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "attention_engine.hpp"

using Clock = std::chrono::steady_clock;
using namespace autoattn;

volatile double checksum = 0.0;

// Measure the median runtime.
template <typename Function>
double measure(Function function) {
    constexpr int repeats = 7;

    std::vector<double> times;
    times.reserve(repeats);

    // Warm-up executions.
    for (int i = 0; i < 2; ++i) {
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

// Generate deterministic matrices.
void generate_inputs(
    Matrix<float>& Q,
    Matrix<float>& K,
    Matrix<float>& V
) {
    for (std::size_t i = 0; i < Q.rows(); ++i) {
        for (std::size_t j = 0; j < Q.cols(); ++j) {
            Q(i, j) = static_cast<float>(
                std::sin((i * 7 + j * 3) * 0.17)
            );
        }
    }

    for (std::size_t i = 0; i < K.rows(); ++i) {
        for (std::size_t j = 0; j < K.cols(); ++j) {
            K(i, j) = static_cast<float>(
                std::cos((i * 5 + j * 11) * 0.13)
            );
        }

        for (std::size_t j = 0; j < V.cols(); ++j) {
            V(i, j) = static_cast<float>(
                std::sin((i * 13 + j * 7) * 0.09)
            );
        }
    }
}

// Write one measurement to the profile.
void save_entry(
    std::ofstream& file,
    const BenchmarkEntry& entry
) {
    file << entry.N << ","
         << entry.M << ","
         << entry.d << ","
         << entry.dv << ","
         << static_cast<int>(entry.causal) << ","
         << (entry.algorithm == Algorithm::Standard
                ? "standard" : "tiled") << ","
         << entry.block_size << ","
         << entry.median_ms << "\n";

    if (!file) {
        throw std::runtime_error(
            "Failed to write calibration data"
        );
    }
}

int main() {
    try {
        std::ofstream file(
            "benchmarks/calibration_grid.csv"
        );

        if (!file) {
            throw std::runtime_error(
                "Cannot create calibration CSV"
            );
        }

        file << "N,M,d,dv,causal,algorithm,"
             << "block_size,median_ms\n";

        file << std::setprecision(17);

        std::size_t total_entries = 0;

        const std::size_t lengths[] = {
            64, 256, 512
        };

        const std::size_t dimensions[] = {
            32, 64
        };

        const std::size_t blocks[] = {
            16, 32, 64, 128, 256
        };

        // Calibrate multiple attention workloads.
        for (std::size_t N : lengths) {
            for (std::size_t d : dimensions) {
                for (bool causal : {false, true}) {

                    Matrix<float> Q(N, d);
                    Matrix<float> K(N, d);
                    Matrix<float> V(N, d);

                    generate_inputs(Q, K, V);

                    std::cout
                        << "\nN = " << N
                        << " d = " << d
                        << " causal = " << causal
                        << "\n";

                    // Standard attention.
                    double standard_ms = measure([&]() {
                        return standard_attention(
                            Q, K, V, causal
                        );
                    });

                    save_entry(file, {
                        N, N, d, d, causal,
                        Algorithm::Standard,
                        0, standard_ms
                    });

                    ++total_entries;

                    std::cout
                        << "Standard = "
                        << standard_ms << " ms\n";

                    // Tiled attention.
                    for (std::size_t block : blocks) {

                        double tiled_ms = measure([&]() {
                            return tiled_attention(
                                Q, K, V, block, causal
                            );
                        });

                        save_entry(file, {
                            N, N, d, d, causal,
                            Algorithm::Tiled,
                            block, tiled_ms
                        });

                        ++total_entries;

                        std::cout
                            << "Tiled block "
                            << block << " = "
                            << tiled_ms << " ms\n";
                    }

                    // Preserve measurements after
                    // each completed workload.
                    file.flush();

                    if (!file) {
                        throw std::runtime_error(
                            "Failed to flush calibration file"
                        );
                    }
                }
            }
        }

        file.close();

        if (!file) {
            throw std::runtime_error(
                "Failed to close calibration file"
            );
        }

        if (total_entries != 72) {
            throw std::runtime_error(
                "Unexpected number of entries"
            );
        }

        std::cout
            << "\nCalibration complete\n"
            << "Total entries = "
            << total_entries << "\n"
            << "Saved to benchmarks/"
            << "calibration_grid.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

