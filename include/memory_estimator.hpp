
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace autoattn {

inline std::size_t checked_multiply(
    std::size_t a,
    std::size_t b
) {
    if (a != 0 &&
        b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error("Memory size overflow");
    }

    return a * b;
}

inline std::size_t checked_add(
    std::size_t a,
    std::size_t b
) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("Memory size overflow");
    }

    return a + b;
}

// Temporary score matrix used by standard attention.
inline std::size_t standard_scratch_bytes(
    std::size_t N,
    std::size_t M
) {
    if (N == 0 || M == 0) {
        throw std::invalid_argument("Invalid dimensions");
    }

    return checked_multiply(
        checked_multiply(N, M),
        sizeof(float)
    );
}

// Temporary score buffer and output accumulator.
inline std::size_t tiled_scratch_bytes(
    std::size_t M,
    std::size_t dv,
    std::size_t block_size
) {
    if (M == 0 || dv == 0 || block_size == 0) {
        throw std::invalid_argument("Invalid dimensions");
    }

    const std::size_t scores = checked_multiply(
        std::min(M, block_size),
        sizeof(double)
    );

    const std::size_t accumulator = checked_multiply(
        dv,
        sizeof(double)
    );

    return checked_add(scores, accumulator);
}

}

