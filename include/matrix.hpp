#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

template <typename T>
class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;

    static std::size_t checked_size(
        std::size_t rows,
        std::size_t cols
    ) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument(
                "Matrix dimensions must be positive"
            );
        }

        if (rows >
            std::numeric_limits<std::size_t>::max()
                / cols) {
            throw std::overflow_error(
                "Matrix dimensions overflow"
            );
        }

        return rows * cols;
    }

public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows),
          cols_(cols),
          data_(checked_size(rows, cols)) {}

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    T& operator()(
        std::size_t i,
        std::size_t j
    ) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range(
                "Matrix index out of bounds"
            );
        }

        return data_[i * cols_ + j];
    }

    const T& operator()(
        std::size_t i,
        std::size_t j
    ) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range(
                "Matrix index out of bounds"
            );
        }

        return data_[i * cols_ + j];
    }

    const std::vector<T>& data() const {
        return data_;
    }
};
