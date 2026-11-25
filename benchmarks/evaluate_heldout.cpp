
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "attention_engine.hpp"
#include "calibration_io.hpp"

using namespace autoattn;

using Clock = std::chrono::steady_clock;

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

    for (int i = 0; i < repeats; ++i) {
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

// Generate new deterministic evaluation inputs.
void generate_inputs(
    Matrix<float>& Q,
    Matrix<float>& K,
    Matrix<float>& V,
    std::uint32_t seed
) {
    std::mt19937 rng(seed);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (std::size_t i = 0; i < Q.rows(); ++i) {
        for (std::size_t j = 0; j < Q.cols(); ++j) {
            Q(i, j) = dist(rng);
            K(i, j) = dist(rng);
            V(i, j) = dist(rng);
        }
    }
}

// Verify numerical correctness.
void verify_output(
    const Matrix<float>& reference,
    const Matrix<float>& actual
) {
    if (reference.rows() != actual.rows() ||
        reference.cols() != actual.cols()) {
        throw std::runtime_error(
            "Output dimensions do not match"
        );
    }

    for (std::size_t i = 0;
         i < reference.rows(); ++i) {

        for (std::size_t j = 0;
             j < reference.cols(); ++j) {

            double expected = reference(i, j);
            double observed = actual(i, j);

            double tolerance =
                1e-5 + 1e-5 * std::abs(expected);

            if (!std::isfinite(observed) ||
                std::abs(expected - observed) >
                    tolerance) {
                throw std::runtime_error(
                    "Numerical correctness failed"
                );
            }
        }
    }
}

int main() {
    try {
        const auto profile = load_profile(
            "benchmarks/calibration_grid.csv"
        );

        const std::size_t budget = 4096;

        std::ofstream csv(
            "benchmarks/heldout_results.csv"
        );

        if (!csv) {
            throw std::runtime_error(
                "Cannot create results file"
            );
        }

        csv << "N,d,causal,selected_block,"
            << "best_block,standard_ms,"
            << "selected_ms,best_ms,gap_percent\n";

        csv << std::setprecision(17);

        double total_gap = 0.0;
        std::size_t count = 0;

        std::cout << std::fixed
                  << std::setprecision(4);

        for (std::size_t N : {128u, 384u}) {
            for (std::size_t d : {32u, 64u}) {
                for (bool causal : {false, true}) {

                    Matrix<float> Q(N, d);
                    Matrix<float> K(N, d);
                    Matrix<float> V(N, d);

                    generate_inputs(
                        Q, K, V,
                        static_cast<std::uint32_t>(
                            N * 100 + d * 10 + causal
                        )
                    );

                    // Compute reference output.
                    auto reference =
                        standard_attention(
                            Q, K, V, causal
                        );

                    // Get the dispatcher's choice.
                    auto result = run_attention(
                        Q, K, V,
                        causal,
                        budget,
                        profile
                    );

                    if (result.choice.calibrated) {
                        throw std::runtime_error(
                            "Unexpected calibration match"
                        );
                    }

                    if (result.choice.scratch_bytes >
                        budget) {
                        throw std::runtime_error(
                            "Memory budget exceeded"
                        );
                    }

                    verify_output(
                        reference, result.output
                    );

                    // Measure standard attention.
                    double standard_ms = measure([&]() {
                        return standard_attention(
                            Q, K, V, causal
                        );
                    });

                    double best_ms =
                        std::numeric_limits<double>
                            ::infinity();

                    double selected_ms = -1.0;
                    std::size_t best_block = 0;

                    // Measure all feasible tiled kernels.
                    for (std::size_t block :
                         {16u, 32u, 64u, 128u, 256u}) {

                        std::size_t memory =
                            tiled_scratch_bytes(
                                N, d, block
                            );

                        if (memory > budget) {
                            continue;
                        }

                        auto output =
                            tiled_attention(
                                Q, K, V,
                                block, causal
                            );

                        verify_output(
                            reference, output
                        );

                        double ms = measure([&]() {
                            return tiled_attention(
                                Q, K, V,
                                block, causal
                            );
                        });

                        if (ms < best_ms) {
                            best_ms = ms;
                            best_block = block;
                        }

                        if (result.choice.algorithm ==
                                Algorithm::Tiled &&
                            result.choice.block_size ==
                                block) {
                            selected_ms = ms;
                        }
                    }

                    if (selected_ms <= 0 ||
                        !std::isfinite(best_ms)) {
                        throw std::runtime_error(
                            "Selected kernel was not measured"
                        );
                    }

                    // Performance gap to the best
                    // measured feasible tiled kernel.
                    double gap = 100.0 *
                        (selected_ms / best_ms - 1.0);

                    total_gap += gap;
                    ++count;

                    std::cout
                        << "\nN = " << N
                        << " d = " << d
                        << " causal = " << causal
                        << "\nSelected block = "
                        << result.choice.block_size
                        << "\nBest block = "
                        << best_block
                        << "\nStandard = "
                        << standard_ms << " ms"
                        << "\nSelected = "
                        << selected_ms << " ms"
                        << "\nBest tiled = "
                        << best_ms << " ms"
                        << "\nGap = "
                        << gap << "%\n";

                    csv
                        << N << ","
                        << d << ","
                        << static_cast<int>(causal) << ","
                        << result.choice.block_size << ","
                        << best_block << ","
                        << standard_ms << ","
                        << selected_ms << ","
                        << best_ms << ","
                        << gap << "\n";
                }
            }
        }

        csv.close();

        if (!csv) {
            throw std::runtime_error(
                "Failed to save results"
            );
        }

        std::cout
            << "\nEvaluation complete\n"
            << "Workloads = " << count << "\n"
            << "Mean gap to best tiled kernel = "
            << total_gap / count << "%\n"
            << "Saved to benchmarks/heldout_results.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

