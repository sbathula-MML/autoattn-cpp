#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoattn {

inline void validate_output_paths(
    const std::string& output,
    const std::string& metadata,
    const std::vector<std::string>& input_files
) {
    namespace fs = std::filesystem;

    if (output.empty() || metadata.empty()) {
        throw std::invalid_argument(
            "Output filenames cannot be empty"
        );
    }

    auto normalize = [](const std::string& name) {
        return fs::weakly_canonical(
            fs::absolute(fs::path(name))
        );
    };

    const auto output_path = normalize(output);
    const auto metadata_path = normalize(metadata);

    if (output_path == metadata_path) {
        throw std::invalid_argument(
            "Output and metadata paths must differ"
        );
    }

    for (const auto& input : input_files) {
        if (input.empty()) {
            continue;
        }

        const auto input_path = normalize(input);

        if (output_path == input_path ||
            metadata_path == input_path) {
            throw std::invalid_argument(
                "Cannot overwrite input or calibration files"
            );
        }
    }
}

} // namespace autoattn
