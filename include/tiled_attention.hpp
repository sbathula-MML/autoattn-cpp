
#pragma once

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline Matrix<float> tiled_attention(
    const Matrix<float>& Q,
    const Matrix<float>& K,
    const Matrix<float>& V,
    std::size_t block_size = 32,
    bool causal = false
) {
    // Step 1 - Validate inputs

    if (block_size == 0) {
        throw std::invalid_argument(
            "Block size must be positive"
        );
    }

    if (Q.cols() != K.cols()) {
        throw std::invalid_argument(
            "Q and K feature dimensions must match"
        );
    }

    if (K.rows() != V.rows()) {
        throw std::invalid_argument(
            "K and V sequence lengths must match"
        );
    }

    if (causal && Q.rows() != K.rows()) {
        throw std::invalid_argument(
            "Causal attention requires equal lengths"
        );
    }

    for (float x : Q.data()) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "Non-finite Q value"
            );
        }
    }

    for (float x : K.data()) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "Non-finite K value"
            );
        }
    }

    for (float x : V.data()) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "Non-finite V value"
            );
        }
    }

    const std::size_t N = Q.rows();
    const std::size_t M = K.rows();
    const std::size_t d = Q.cols();
    const std::size_t dv = V.cols();

    const double scale =
        1.0 / std::sqrt(static_cast<double>(d));

    Matrix<float> output(N, dv);

    // Memory is independent of sequence length
    // apart from the output matrix.

    std::vector<double> scores(
        std::min(block_size, M)
    );

    std::vector<double> accumulator(dv);

    // Step 2 - Process each query row

    for (std::size_t i = 0; i < N; ++i) {

        std::fill(
            accumulator.begin(),
            accumulator.end(),
            0.0
        );

        double running_max =
            -std::numeric_limits<double>::infinity();

        double running_sum = 0.0;

        // Causal attention excludes future keys.
        const std::size_t valid_keys =
            causal ? i + 1 : M;

        // Step 3 - Process keys in blocks

        for (std::size_t start = 0;
             start < valid_keys;) {

            const std::size_t count =
                std::min(
                    block_size,
                    valid_keys - start
                );

            double block_max =
                -std::numeric_limits<double>::infinity();

            // Compute scores for this block

            for (std::size_t b = 0;
                 b < count; ++b) {

                const std::size_t j = start + b;

                double dot = 0.0;

                for (std::size_t k = 0;
                     k < d; ++k) {

                    dot +=
                        static_cast<double>(Q(i, k)) *
                        static_cast<double>(K(j, k));
                }

                double score = dot * scale;

                if (!std::isfinite(score) ||
                    std::abs(score) >
                    std::numeric_limits<float>::max()) {
                    throw std::overflow_error(
                        "Attention score exceeds "
                        "float32 range"
                    );
                }

                scores[b] = static_cast<float>(score);

                block_max = std::max(
                    block_max,
                    scores[b]
                );
            }

            // Step 4 - Update the online softmax

            const double new_max = std::max(
                running_max,
                block_max
            );

            double correction = 0.0;

            if (std::isfinite(running_max)) {
                correction = std::exp(
                    running_max - new_max
                );
            }

            running_sum *= correction;

            for (std::size_t c = 0; c < dv; ++c) {
                accumulator[c] *= correction;
            }

            // Add the current block contribution

            for (std::size_t b = 0;
                 b < count; ++b) {

                const std::size_t j = start + b;

                const double weight = std::exp(
                    scores[b] - new_max
                );

                running_sum += weight;

                for (std::size_t c = 0;
                     c < dv; ++c) {

                    accumulator[c] +=
                        weight *
                        static_cast<double>(V(j, c));
                }
            }

            running_max = new_max;
            start += count;
        }

        // Step 5 - Normalize the output

        for (std::size_t c = 0; c < dv; ++c) {
            output(i, c) = static_cast<float>(
                accumulator[c] / running_sum
            );
        }
    }

    return output;
}

