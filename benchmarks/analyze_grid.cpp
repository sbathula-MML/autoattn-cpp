
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "calibration_io.hpp"
#include "dispatch_policy.hpp"

using namespace autoattn;

int main() {
    try {
        const auto profile = load_profile(
            "benchmarks/calibration_grid.csv"
        );

        if (profile.size() != 72) {
            throw std::runtime_error(
                "Expected 72 calibration entries"
            );
        }

        std::cout << "Loaded "
                  << profile.size()
                  << " calibration entries\n\n";

        double total_improvement = 0.0;
        double max_improvement = 0.0;
        int workloads = 0;

        std::cout << std::fixed
                  << std::setprecision(3);

        for (std::size_t N : {64u, 256u, 512u}) {
            for (std::size_t d : {32u, 64u}) {
                for (bool causal : {false, true}) {

                    const std::size_t budget = 4096;

                    auto choice = choose_attention(
                        N, N, d, d, causal,
                        budget, profile
                    );

                    if (!choice.calibrated ||
                        choice.algorithm != Algorithm::Tiled) {
                        throw std::runtime_error(
                            "Unexpected dispatcher choice"
                        );
                    }

                    double fixed_time = -1.0;
                    double selected_time = -1.0;

                    for (const auto& entry : profile) {
                        if (entry.N != N ||
                            entry.M != N ||
                            entry.d != d ||
                            entry.dv != d ||
                            entry.causal != causal ||
                            entry.algorithm != Algorithm::Tiled) {
                            continue;
                        }

                        if (entry.block_size == 128) {
                            fixed_time = entry.median_ms;
                        }

                        if (entry.block_size ==
                            choice.block_size) {
                            selected_time = entry.median_ms;
                        }
                    }

                    if (fixed_time <= 0 ||
                        selected_time <= 0) {
                        throw std::runtime_error(
                            "Missing benchmark measurements"
                        );
                    }

                    double improvement =
                        100.0 * (fixed_time - selected_time)
                        / fixed_time;

                    total_improvement += improvement;

                    if (improvement > max_improvement) {
                        max_improvement = improvement;
                    }

                    ++workloads;

                    std::cout
                        << "N=" << N
                        << " d=" << d
                        << " causal=" << causal
                        << " selected=" << choice.block_size
                        << " fixed128=" << fixed_time
                        << " ms"
                        << " adaptive=" << selected_time
                        << " ms"
                        << " improvement=" << improvement
                        << "%\n";
                }
            }
        }

        std::cout
            << "\nWorkloads evaluated = "
            << workloads
            << "\nMean improvement = "
            << total_improvement / workloads
            << "%"
            << "\nMaximum improvement = "
            << max_improvement
            << "%\n";

        std::cout
            << "\nCalibration analysis completed!\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

