
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dispatch_policy.hpp"

int main() {
    using namespace autoattn;

    // Hypothetical timings to test selection behavior.
    // These are not experimental measurements.
    const std::vector<BenchmarkEntry> timings = {
        {256, 256, 64, 64, false,
         Algorithm::Standard, 0, 1.0},

        {256, 256, 64, 64, false,
         Algorithm::Tiled, 128, 2.0},

        {256, 256, 64, 64, false,
         Algorithm::Tiled, 32, 3.0}
    };

    // Test 1 - Fastest measured algorithm.
    auto a = choose_attention(
        256, 256, 64, 64, false, 300000, timings
    );

    assert(a.algorithm == Algorithm::Standard);
    assert(a.calibrated);

    std::cout << "Test 1 passed - Fastest selection\n";

    // Test 2 - Standard exceeds the memory budget.
    auto b = choose_attention(
        256, 256, 64, 64, false, 2000, timings
    );

    assert(b.algorithm == Algorithm::Tiled);
    assert(b.block_size == 128);
    assert(b.scratch_bytes == 1536);
    assert(b.calibrated);

    std::cout << "Test 2 passed - Memory constraint\n";

    // Test 3 - Only the smaller measured tile fits.
    auto c = choose_attention(
        256, 256, 64, 64, false, 800, timings
    );

    assert(c.algorithm == Algorithm::Tiled);
    assert(c.block_size == 32);
    assert(c.scratch_bytes == 768);

    std::cout << "Test 3 passed - Block selection\n";

    // Test 4 - Use fallback when measured
    // configurations exceed the budget.
    auto d = choose_attention(
        256, 256, 64, 64, false, 600, timings
    );

    assert(d.algorithm == Algorithm::Tiled);
    assert(d.block_size == 8);
    assert(d.scratch_bytes == 576);
    assert(!d.calibrated);

    std::cout << "Test 4 passed - Fallback policy\n";

    // Test 5 - Reject impossible memory budgets.
    bool rejected = false;

    try {
        choose_attention(
            256, 256, 64, 64, false, 519, timings
        );
    } catch (const std::runtime_error&) {
        rejected = true;
    }

    assert(rejected);

    std::cout << "Test 5 passed - Budget rejection\n";

    // Test 6 - Do not reuse noncausal timings
    // for a causal workload.
    auto e = choose_attention(
        256, 256, 64, 64, true, 2000, timings
    );

    assert(e.algorithm == Algorithm::Tiled);
    assert(!e.calibrated);

    std::cout << "Test 6 passed - Workload matching\n";

    std::cout
        << "\nAll dispatcher tests passed!\n";

    return 0;
}

