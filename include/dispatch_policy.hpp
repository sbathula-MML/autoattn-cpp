
#pragma once

#include "memory_estimator.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace autoattn {

enum class Algorithm { Standard, Tiled };

// Timings must be measured for the exact workload.
struct BenchmarkEntry {
    std::size_t N, M, d, dv;
    bool causal;
    Algorithm algorithm;
    std::size_t block_size;
    double median_ms;
};

struct DispatchChoice {
    Algorithm algorithm;
    std::size_t block_size;
    std::size_t scratch_bytes;
    bool calibrated;
};

inline DispatchChoice choose_attention(
    std::size_t N,
    std::size_t M,
    std::size_t d,
    std::size_t dv,
    bool causal,
    std::size_t scratch_budget_bytes,
    const std::vector<BenchmarkEntry>& benchmarks = {}
) {
    if (!N || !M || !d || !dv || (causal && N != M)) {
        throw std::invalid_argument(
            "Invalid attention shape"
        );
    }

    double best_time =
        std::numeric_limits<double>::infinity();

    DispatchChoice best{Algorithm::Tiled, 0, 0, false};
    bool found = false;

    // Examine measured configurations.
    for (const auto& entry : benchmarks) {
        if (entry.N != N || entry.M != M ||
            entry.d != d || entry.dv != dv ||
            entry.causal != causal) {
            continue;
        }

        if (!std::isfinite(entry.median_ms) ||
            entry.median_ms <= 0) {
            throw std::invalid_argument(
                "Invalid benchmark time"
            );
        }

        const std::size_t bytes =
            entry.algorithm == Algorithm::Standard
                ? standard_scratch_bytes(N, M)
                : tiled_scratch_bytes(
                    M, dv, entry.block_size
                );

        if (bytes <= scratch_budget_bytes &&
            entry.median_ms < best_time) {

            best_time = entry.median_ms;

            best = {
                entry.algorithm,
                entry.block_size,
                bytes,
                true
            };

            found = true;
        }
    }

    if (found) {
        return best;
    }

    // Fallback when no measured candidate fits.
    // Prefer a large feasible tile.
    for (std::size_t block :
         {128u, 64u, 32u, 16u, 8u, 1u}) {

        const std::size_t bytes =
            tiled_scratch_bytes(M, dv, block);

        if (bytes <= scratch_budget_bytes) {
            return {
                Algorithm::Tiled,
                block,
                bytes,
                false
            };
        }
    }

    // Standard attention can use less scratch
    // for very small matrices.
    const std::size_t bytes =
        standard_scratch_bytes(N, M);

    if (bytes <= scratch_budget_bytes) {
        return {
            Algorithm::Standard,
            0,
            bytes,
            false
        };
    }

    throw std::runtime_error(
        "No kernel fits the scratch memory budget"
    );
}

} // namespace autoattn

