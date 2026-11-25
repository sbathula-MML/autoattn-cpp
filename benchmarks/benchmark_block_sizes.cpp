
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "tiled_attention.hpp"

using Clock = std::chrono::steady_clock;

volatile double checksum = 0.0;

template <typename Function>
double measure(Function function, int repeats = 7) {
    std::vector<double> times;

    // Warm up the implementation.
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

    return times[times.size() / 2];
}

void run_experiment(std::size_t N) {
    const std::size_t d = 64;

    Matrix<float> Q(N, d);
    Matrix<float> K(N, d);
    Matrix<float> V(N, d);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            Q(i, j) = std::sin((i * 7 + j * 3) * 0.17);
            K(i, j) = std::cos((i * 5 + j * 11) * 0.13);
            V(i, j) = std::sin((i * 13 + j * 7) * 0.09);
        }
    }

    std::cout << "\nSequence length N = " << N << "\n";

    double best_time = 1e100;
    std::size_t best_block = 0;

    for (std::size_t block :
         {1, 8, 16, 32, 64, 128, 256}) {

        double ms = measure([&]() {
            return tiled_attention(
                Q, K, V, block, false
            );
        });

        std::cout
            << "Block " << std::setw(3) << block
            << " | " << std::fixed
            << std::setprecision(4)
            << ms << " ms\n";

        if (ms < best_time) {
            best_time = ms;
            best_block = block;
        }
    }

    std::cout
        << "Best block size = " << best_block
        << " | Runtime = " << best_time
        << " ms\n";
}

int main() {
    for (std::size_t N : {64, 256, 1024}) {
        run_experiment(N);
    }

    return 0;
}

