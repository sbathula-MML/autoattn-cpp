
#pragma once

#include "matrix.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoattn {

// Convert a CSV value into a valid float32 number.
inline float parse_csv_float(const std::string& text) {
    if (text.empty()) {
        throw std::runtime_error("Empty CSV value");
    }

    std::size_t position = 0;
    double value = std::stod(text, &position);

    // Allow trailing whitespace.
    while (position < text.size() &&
           std::isspace(
               static_cast<unsigned char>(text[position])
           )) {
        ++position;
    }

    if (position != text.size() ||
        !std::isfinite(value) ||
        std::abs(value) >
            std::numeric_limits<float>::max()) {
        throw std::runtime_error(
            "Invalid or non-finite CSV value"
        );
    }

    return static_cast<float>(value);
}

// Load a matrix from a CSV file.
inline Matrix<float> load_csv_matrix(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open CSV file: " + filename
        );
    }

    std::vector<float> values;

    std::size_t rows = 0;
    std::size_t cols = 0;

    std::string line;

    while (std::getline(file, line)) {

        // Reject blank rows.
        if (line.find_first_not_of(" \t\r\n") ==
            std::string::npos) {
            throw std::runtime_error(
                "Blank row in CSV file"
            );
        }

        // Reject missing final values.
        if (line.back() == ',') {
            throw std::runtime_error(
                "Trailing comma in CSV row"
            );
        }

        std::stringstream stream(line);
        std::vector<float> row;

        std::string cell;

        while (std::getline(stream, cell, ',')) {
            row.push_back(parse_csv_float(cell));
        }

        if (row.empty()) {
            throw std::runtime_error(
                "Empty CSV row"
            );
        }

        // All rows must contain the same
        // number of columns.
        if (rows == 0) {
            cols = row.size();

            if (cols > 1024) {
                throw std::runtime_error(
                    "Too many CSV columns"
                );
            }
        } else if (row.size() != cols) {
            throw std::runtime_error(
                "Inconsistent CSV row dimensions"
            );
        }

        if (rows >= 8192) {
            throw std::runtime_error(
                "Too many CSV rows"
            );
        }

        values.insert(
            values.end(),
            row.begin(),
            row.end()
        );

        ++rows;
    }

    if (!file.eof()) {
        throw std::runtime_error(
            "Error while reading CSV file"
        );
    }

    if (rows == 0) {
        throw std::runtime_error(
            "CSV file contains no data"
        );
    }

    Matrix<float> matrix(rows, cols);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix(i, j) = values[i * cols + j];
        }
    }

    return matrix;
}

} // namespace autoattn

