#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "tiled_attention.hpp"

using Clock = std::chrono::steady_clock;
volatile double checksum = 0.0;

double measure(
    const Matrix<float>& Q,
    const Matrix<float>& K,
    const Matrix<float>& V,
    std::size_t block,
    bool causal
) {
    auto start = Clock::now();

    auto output = tiled_attention(
        Q, K, V, block, causal
    );

    auto end = Clock::now();

    checksum += output(0, 0);

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void run_case(
    std::size_t N,
    std::size_t d,
    bool causal,
    std::size_t block_a,
    std::size_t block_b,
    std::ofstream& csv
) {
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

    // Check output equivalence.
    auto A = tiled_attention(Q, K, V, block_a, causal);
    auto B = tiled_attention(Q, K, V, block_b, causal);

    double max_error = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            max_error = std::max(
                max_error,
                std::abs(
                    static_cast<double>(A(i, j)) -
                    static_cast<double>(B(i, j))
                )
            );
        }
    }

    if (!std::isfinite(max_error) || max_error > 1e-5) {
        throw std::runtime_error("Output mismatch");
    }

    // Warm-up both implementations.
    for (int i = 0; i < 5; ++i) {
        measure(Q, K, V, block_a, causal);
        measure(Q, K, V, block_b, causal);
    }

    std::mt19937 rng(42);
    std::bernoulli_distribution coin(0.5);

    std::vector<double> times_a;
    std::vector<double> times_b;
    std::vector<double> improvements;

    int wins_b = 0;
    constexpr int repeats = 30;

    for (int r = 0; r < repeats; ++r) {
        double ms_a = 0.0;
        double ms_b = 0.0;

        // Randomize execution order.
        if (coin(rng)) {
            ms_a = measure(Q, K, V, block_a, causal);
            ms_b = measure(Q, K, V, block_b, causal);
        } else {
            ms_b = measure(Q, K, V, block_b, causal);
            ms_a = measure(Q, K, V, block_a, causal);
        }

        times_a.push_back(ms_a);
        times_b.push_back(ms_b);

        improvements.push_back(
            100.0 * (ms_a - ms_b) / ms_a
        );

        if (ms_b < ms_a) {
            ++wins_b;
        }

        csv << N << "," << d << ","
            << causal << ","
            << block_a << "," << block_b << ","
            << r << "," << ms_a << "," << ms_b
            << "\n";
    }

    std::cout << "\nN = " << N
              << " d = " << d
              << " causal = " << causal << "\n";

    std::cout << "Block " << block_a << " median = "
              << median(times_a) << " ms\n";

    std::cout << "Block " << block_b << " median = "
              << median(times_b) << " ms\n";

    std::cout << "Median paired improvement = "
              << median(improvements) << "%\n";

    std::cout << "Block " << block_b
              << " faster in " << wins_b
              << " / " << repeats << " trials\n";

    std::cout << "Maximum numerical error = "
              << max_error << "\n";
}

int main() {
    try {
        std::ofstream csv(
            "benchmarks/timing_stability.csv"
        );

        if (!csv) {
            throw std::runtime_error(
                "Cannot create results file"
            );
        }

        csv << "N,d,causal,block_a,block_b,"
            << "trial,ms_a,ms_b\n";

        csv << std::setprecision(17);

        std::cout << std::fixed
                  << std::setprecision(5);

        run_case(128, 32, false, 128, 256, csv);
        run_case(128, 64, false, 128, 256, csv);
        run_case(384, 64, false, 128, 64, csv);
        run_case(384, 64, true, 128, 64, csv);

        csv.close();

        if (!csv) {
            throw std::runtime_error(
                "Failed to save results"
            );
        }

        std::cout
            << "\nTiming stability experiment complete\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
