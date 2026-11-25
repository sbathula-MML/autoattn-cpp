
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include "matrix.hpp"
#include "standard_attention.hpp"
#include "tiled_attention.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr
            << "Usage: ./benchmark_memory "
            << "standard|tiled N\n";
        return 1;
    }

    const std::string mode = argv[1];
    const std::size_t N = std::stoull(argv[2]);
    const std::size_t d = 64;

    if ((mode != "standard" && mode != "tiled")
        || N == 0 || N > 8192) {
        throw std::invalid_argument(
            "Invalid mode or sequence length"
        );
    }

    Matrix<float> Q(N, d);
    Matrix<float> K(N, d);
    Matrix<float> V(N, d);

    // Generate identical inputs for both algorithms.
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

    Matrix<float> output =
        (mode == "standard")
        ? standard_attention(Q, K, V, false)
        : tiled_attention(Q, K, V, 128, false);

    // Ensure the output is used.
    std::cout << "Algorithm: " << mode << "\n";
    std::cout << "Sequence length: " << N << "\n";
    std::cout << "Output checksum: "
              << output(0, 0) + output(N - 1, d - 1)
              << "\n";

    return 0;
}

