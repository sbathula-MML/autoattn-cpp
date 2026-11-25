
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "memory_estimator.hpp"

int main() {
    using namespace autoattn;

    const auto standard =
        standard_scratch_bytes(4096, 4096);

    const auto tiled =
        tiled_scratch_bytes(4096, 64, 128);

    assert(standard == 67108864);
    assert(tiled == 1536);

    bool caught = false;

    try {
        standard_scratch_bytes(
            std::numeric_limits<std::size_t>::max(),
            4096
        );
    } catch (const std::overflow_error&) {
        caught = true;
    }

    assert(caught);

    std::cout << "Standard scratch: "
              << standard << " bytes\n";

    std::cout << "Tiled scratch: "
              << tiled << " bytes\n";

    std::cout << "All memory estimator tests passed!\n";
}

