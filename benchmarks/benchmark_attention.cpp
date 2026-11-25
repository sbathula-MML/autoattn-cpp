
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "standard_attention.hpp"
#include "tiled_attention.hpp"

using Clock = std::chrono::steady_clock;

// Prevent the compiler from discarding results.
volatile double checksum = 0.0;

// Measure median execution time in milliseconds.
template <typename Function>
double benchmark(Function function, int repeats = 5) {
    std::vector<double> times;

    // Warm-up run
    auto warmup = function();
    checksum += warmup(0, 0);

    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now();

        auto output = function();

        auto end = Clock::now();

        checksum += output(0, 0);

        double milliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(milliseconds);
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

void run_benchmark(std::size_t N) {
    const std::size_t d = 64;

    Matrix<float> Q(N, d);
    Matrix<float> K(N, d);
    Matrix<float> V(N, d);

    // Deterministic inputs, generated outside timing.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            Q(i, j) = static_cast<float>(
                std::sin((i * 7 + j * 3) * 0.17)
            );

            K(i, j) = static_cast<float>(
                std::cos((i * 5 + j * 11) * 0.13)
            );

            V(i, j) = static_cast<float>(
                std::sin((i * 13 + j * 7) * 0.09)
            );
        }
    }

    double standard_ms = benchmark([&]() {
        return standard_attention(Q, K, V, false);
    });

    double tiled_ms = benchmark([&]() {
        return tiled_attention(Q, K, V, 32, false);
    });

    std::cout
        << std::setw(8) << N
        << std::setw(18) << standard_ms
        << std::setw(18) << tiled_ms
        << std::setw(14) << standard_ms / tiled_ms
        << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout
        << std::setw(8) << "N"
        << std::setw(18) << "Standard(ms)"
        << std::setw(18) << "Tiled(ms)"
        << std::setw(14) << "Speedup"
        << "\n";

    for (std::size_t N : {64, 128, 256, 512, 1024}) {
        run_benchmark(N);
    }

    return 0;
}

