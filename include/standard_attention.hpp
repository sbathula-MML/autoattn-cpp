
#pragma once

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

inline Matrix<float> standard_attention(
    const Matrix<float>& Q,
    const Matrix<float>& K,
    const Matrix<float>& V,
    bool causal = false
) {
    // Step 1: Validate dimensions

    if (Q.cols() != K.cols()) {
        throw std::invalid_argument(
            "Q and K must have the same feature dimension"
        );
    }

    if (K.rows() != V.rows()) {
        throw std::invalid_argument(
            "K and V must have the same number of rows"
        );
    }

    if (causal && Q.rows() != K.rows()) {
        throw std::invalid_argument(
            "Causal attention requires equal sequence lengths"
        );
    }

    for (float value : Q.data()) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Non-finite Q value");
    }

    for (float value : K.data()) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Non-finite K value");
    }

    for (float value : V.data()) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Non-finite V value");
    }

    const std::size_t N = Q.rows();
    const std::size_t M = K.rows();
    const std::size_t d = Q.cols();
    const std::size_t dv = V.cols();

    const double scale =
        1.0 / std::sqrt(static_cast<double>(d));

    Matrix<float> scores(N, M);
    Matrix<float> output(N, dv);

    // Step 2: Compute scaled QK^T

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {

            if (causal && j > i) {
                scores(i, j) =
                    -std::numeric_limits<float>::infinity();
                continue;
            }

            double dot = 0.0;

            for (std::size_t k = 0; k < d; ++k) {
                dot += static_cast<double>(Q(i, k)) *
                       static_cast<double>(K(j, k));
            }

            double score = dot * scale;

            if (!std::isfinite(score) ||
                std::abs(score) >
                    std::numeric_limits<float>::max()) {
                throw std::overflow_error(
                    "Attention score exceeds float32 range"
                );
            }

            scores(i, j) = static_cast<float>(score);
        }
    }

    // Step 3: Apply numerically stable softmax

    for (std::size_t i = 0; i < N; ++i) {

        float max_score =
            -std::numeric_limits<float>::infinity();

        for (std::size_t j = 0; j < M; ++j) {
            max_score = std::max(max_score, scores(i, j));
        }

        double denominator = 0.0;

        for (std::size_t j = 0; j < M; ++j) {
            double e = std::exp(
                static_cast<double>(scores(i, j)) -
                static_cast<double>(max_score)
            );

            scores(i, j) = static_cast<float>(e);
            denominator += e;
        }

        for (std::size_t j = 0; j < M; ++j) {
            scores(i, j) = static_cast<float>(
                scores(i, j) / denominator
            );
        }
    }

    // Step 4: Multiply attention weights by V

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < dv; ++j) {

            double sum = 0.0;

            for (std::size_t k = 0; k < M; ++k) {
                sum += static_cast<double>(scores(i, k)) *
                       static_cast<double>(V(k, j));
            }

            output(i, j) = static_cast<float>(sum);
        }
    }

    return output;
}

