
#pragma once

#include "matrix.hpp"
#include "standard_attention.hpp"
#include "tiled_attention.hpp"
#include "dispatch_policy.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace autoattn {

// Stores the attention output and the algorithm selected.
struct AttentionResult {
    Matrix<float> output;
    DispatchChoice choice;
};

// Main adaptive attention interface.
inline AttentionResult run_attention(
    const Matrix<float>& Q,
    const Matrix<float>& K,
    const Matrix<float>& V,
    bool causal,
    std::size_t scratch_budget_bytes,
    const std::vector<BenchmarkEntry>& benchmarks = {}
) {
    // Validate matrix dimensions.
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

    // Select an algorithm.
    DispatchChoice choice = choose_attention(
        Q.rows(),
        K.rows(),
        Q.cols(),
        V.cols(),
        causal,
        scratch_budget_bytes,
        benchmarks
    );

    // Execute standard attention.
    if (choice.algorithm == Algorithm::Standard) {
        return {
            standard_attention(Q, K, V, causal),
            choice
        };
    }

    // Execute memory-efficient tiled attention.
    return {
        tiled_attention(
            Q,
            K,
            V,
            choice.block_size,
            causal
        ),
        choice
    };
}

} // namespace autoattn

