
#pragma once

#include "dispatch_policy.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoattn {

// Convert a CSV field into a nonnegative integer.
inline std::size_t parse_size(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty numeric field");
    }

    for (char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(
                "Invalid integer in calibration file"
            );
        }
    }

    std::size_t pos = 0;
    unsigned long long value = std::stoull(text, &pos);

    if (pos != text.size()) {
        throw std::invalid_argument("Invalid integer");
    }

    return static_cast<std::size_t>(value);
}

// Read a calibration profile from a CSV file.
inline std::vector<BenchmarkEntry> load_profile(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open calibration file: " + filename
        );
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error(
            "Calibration file is empty"
        );
    }

    const std::string expected_header =
        "N,M,d,dv,causal,algorithm,block_size,median_ms";

    // Support Windows-style line endings as well.
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    if (line != expected_header) {
        throw std::runtime_error(
            "Invalid calibration CSV header"
        );
    }

    std::vector<BenchmarkEntry> profile;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);
        std::vector<std::string> fields;
        std::string field;

        while (std::getline(stream, field, ',')) {
            fields.push_back(field);
        }

        if (fields.size() != 8) {
            throw std::runtime_error(
                "Invalid calibration CSV row"
            );
        }

        const std::size_t N = parse_size(fields[0]);
        const std::size_t M = parse_size(fields[1]);
        const std::size_t d = parse_size(fields[2]);
        const std::size_t dv = parse_size(fields[3]);

        if (!N || !M || !d || !dv) {
            throw std::runtime_error(
                "Calibration dimensions must be positive"
            );
        }

        if (fields[4] != "0" && fields[4] != "1") {
            throw std::runtime_error(
                "Invalid causal flag"
            );
        }

        bool causal = fields[4] == "1";

        if (causal && N != M) {
            throw std::runtime_error(
                "Invalid causal sequence lengths"
            );
        }

        Algorithm algorithm;

        if (fields[5] == "standard") {
            algorithm = Algorithm::Standard;
        } else if (fields[5] == "tiled") {
            algorithm = Algorithm::Tiled;
        } else {
            throw std::runtime_error(
                "Unknown attention algorithm"
            );
        }

        std::size_t block = parse_size(fields[6]);

        if ((algorithm == Algorithm::Standard && block != 0) ||
            (algorithm == Algorithm::Tiled && block == 0)) {
            throw std::runtime_error(
                "Invalid block size for algorithm"
            );
        }

        std::size_t pos = 0;
        double median_ms = std::stod(fields[7], &pos);

        if (pos != fields[7].size() ||
            !std::isfinite(median_ms) ||
            median_ms <= 0.0) {
            throw std::runtime_error(
                "Invalid benchmark runtime"
            );
        }

        profile.push_back({
            N, M, d, dv, causal,
            algorithm, block, median_ms
        });
    }

    if (profile.empty()) {
        throw std::runtime_error(
            "Calibration profile has no measurements"
        );
    }

    return profile;
}

} // namespace autoattn

