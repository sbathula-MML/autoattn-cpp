
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "matrix.hpp"
#include "standard_attention.hpp"

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr
            << "Usage: attention_reference_driver "
            << "N M d dv causal\n";
        return 1;
    }

    try {
        const std::size_t N = std::stoull(argv[1]);
        const std::size_t M = std::stoull(argv[2]);
        const std::size_t d = std::stoull(argv[3]);
        const std::size_t dv = std::stoull(argv[4]);
        const bool causal = std::stoi(argv[5]) != 0;

        Matrix<float> Q(N, d);
        Matrix<float> K(M, d);
        Matrix<float> V(M, dv);

        // Deterministic test inputs
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                int x = static_cast<int>((i * 7 + j * 3) % 19);
                Q(i, j) = (x - 9) * 0.15f;
            }
        }

        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                int x = static_cast<int>((i * 5 + j * 11) % 23);
                K(i, j) = (x - 11) * 0.12f;
            }
        }

        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < dv; ++j) {
                int x = static_cast<int>((i * 13 + j * 7) % 29);
                V(i, j) = (x - 14) * 0.08f;
            }
        }

        Matrix<float> O =
            standard_attention(Q, K, V, causal);

        std::cout << std::setprecision(9);

        for (std::size_t i = 0; i < O.rows(); ++i) {
            for (std::size_t j = 0; j < O.cols(); ++j) {
                if (j > 0) std::cout << " ";
                std::cout << O(i, j);
            }
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

