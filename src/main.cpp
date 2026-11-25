
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention_engine.hpp"
#include "calibration_io.hpp"
#include "csv_io.hpp"
#include "path_safety.hpp"

using namespace autoattn;

struct Options {
    std::size_t N = 256;
    std::size_t d = 64;
    std::size_t block = 128;
    std::size_t budget = 134217728;

    std::uint32_t seed = 7;
    bool causal = false;

    std::string mode = "auto";

    std::string profile =
        "benchmarks/calibration_grid.csv";

    std::string q_file;
    std::string k_file;
    std::string v_file;

    std::string output = "attention.csv";
    std::string metadata = "metrics.json";
};

std::size_t parse_number(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty number");
    }

    for (char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(
                "Expected a nonnegative integer"
            );
        }
    }

    unsigned long long value = std::stoull(text);

    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range("Integer is too large");
    }

    return static_cast<std::size_t>(value);
}

Options parse_arguments(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) != "run") {
        throw std::invalid_argument(
            "Usage: ./autoattn run "
            "--mode standard|tiled|auto "
            "[--n N --d D | --q Q.csv --k K.csv --v V.csv]"
        );
    }

    Options opt;
    bool explicit_shape = false;

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];

        if (key == "--causal") {
            opt.causal = true;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for " + key
            );
        }

        std::string value = argv[++i];

        if (key == "--n") {
            opt.N = parse_number(value);
            explicit_shape = true;
        } else if (key == "--d") {
            opt.d = parse_number(value);
            explicit_shape = true;
        } else if (key == "--block") {
            opt.block = parse_number(value);
        } else if (key == "--budget-bytes") {
            opt.budget = parse_number(value);
        } else if (key == "--seed") {
            std::size_t parsed = parse_number(value);

            if (parsed >
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "Seed is too large"
                );
            }

            opt.seed = static_cast<std::uint32_t>(
                parsed
            );
        } else if (key == "--mode") {
            opt.mode = value;
        } else if (key == "--profile") {
            opt.profile = value;
        } else if (key == "--q") {
            opt.q_file = value;
        } else if (key == "--k") {
            opt.k_file = value;
        } else if (key == "--v") {
            opt.v_file = value;
        } else if (key == "--out") {
            opt.output = value;
        } else if (key == "--meta") {
            opt.metadata = value;
        } else {
            throw std::invalid_argument(
                "Unknown argument " + key
            );
        }
    }

    if (opt.mode != "standard" &&
        opt.mode != "tiled" &&
        opt.mode != "auto") {
        throw std::invalid_argument(
            "Invalid attention mode"
        );
    }

    if (opt.N == 0 || opt.d == 0 ||
        opt.N > 8192 || opt.d > 1024 ||
        opt.block == 0) {
        throw std::invalid_argument(
            "Invalid dimensions or block size"
        );
    }

    int file_count =
        static_cast<int>(!opt.q_file.empty()) +
        static_cast<int>(!opt.k_file.empty()) +
        static_cast<int>(!opt.v_file.empty());

    if (file_count != 0 && file_count != 3) {
        throw std::invalid_argument(
            "Provide all three CSV files Q, K, and V"
        );
    }

    if (file_count == 3 && explicit_shape) {
        throw std::invalid_argument(
            "Do not combine CSV inputs with --n or --d"
        );
    }

    if (opt.output.empty() ||
        opt.metadata.empty() ||
        opt.output == opt.metadata) {
        throw std::invalid_argument(
            "Invalid output filenames"
        );
    }

    return opt;
}

void generate_inputs(
    Matrix<float>& Q,
    Matrix<float>& K,
    Matrix<float>& V,
    std::uint32_t seed
) {
    std::mt19937 generator(seed);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (std::size_t i = 0; i < Q.rows(); ++i) {
        for (std::size_t j = 0; j < Q.cols(); ++j) {
            Q(i, j) = dist(generator);
            K(i, j) = dist(generator);
            V(i, j) = dist(generator);
        }
    }
}

void save_matrix(
    const Matrix<float>& matrix,
    const std::string& filename
) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot create output CSV"
        );
    }

    file << std::setprecision(9);

    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0;
             j < matrix.cols(); ++j) {

            if (j > 0) {
                file << ",";
            }

            file << matrix(i, j);
        }

        file << "\n";
    }

    file.close();

    if (!file) {
        throw std::runtime_error(
            "Failed to save output matrix"
        );
    }
}

void save_metadata(
    const std::string& filename,
    const Options& opt,
    const DispatchChoice& choice,
    const Matrix<float>& Q,
    const Matrix<float>& K,
    const Matrix<float>& V,
    double runtime_ms
) {
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot create metadata file"
        );
    }

    const std::string algorithm =
        choice.algorithm == Algorithm::Standard
        ? "standard" : "tiled";

    const bool csv_input = !opt.q_file.empty();

    file << std::fixed << std::setprecision(6);

    file << "{\n"
         << "  \"input_type\": \""
         << (csv_input ? "csv" : "generated")
         << "\",\n"
         << "  \"query_length\": "
         << Q.rows() << ",\n"
         << "  \"key_length\": "
         << K.rows() << ",\n"
         << "  \"query_dimension\": "
         << Q.cols() << ",\n"
         << "  \"value_dimension\": "
         << V.cols() << ",\n"
         << "  \"causal\": "
         << (opt.causal ? "true" : "false") << ",\n"
         << "  \"algorithm\": \""
         << algorithm << "\",\n"
         << "  \"block_size\": "
         << choice.block_size << ",\n"
         << "  \"scratch_budget_bytes\": "
         << opt.budget << ",\n"
         << "  \"estimated_scratch_bytes\": "
         << choice.scratch_bytes << ",\n"
         << "  \"calibrated\": "
         << (choice.calibrated ? "true" : "false")
         << ",\n"
         << "  \"execution_time_ms\": "
         << runtime_ms << "\n"
         << "}\n";

    file.close();

    if (!file) {
        throw std::runtime_error(
            "Failed to save metadata"
        );
    }
}

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_arguments(
            argc, argv
        );

        validate_output_paths(
            opt.output,
            opt.metadata,
            {
                opt.q_file,
                opt.k_file,
                opt.v_file,
                opt.mode == "auto" ? opt.profile : ""
            }
        );

        const bool csv_input = !opt.q_file.empty();

        Matrix<float> Q = csv_input
            ? load_csv_matrix(opt.q_file)
            : Matrix<float>(opt.N, opt.d);

        Matrix<float> K = csv_input
            ? load_csv_matrix(opt.k_file)
            : Matrix<float>(opt.N, opt.d);

        Matrix<float> V = csv_input
            ? load_csv_matrix(opt.v_file)
            : Matrix<float>(opt.N, opt.d);

        if (!csv_input) {
            generate_inputs(Q, K, V, opt.seed);
        }

        // Validate attention dimensions.
        if (Q.cols() != K.cols()) {
            throw std::invalid_argument(
                "Q and K dimensions must match"
            );
        }

        if (K.rows() != V.rows()) {
            throw std::invalid_argument(
                "K and V row counts must match"
            );
        }

        if (opt.causal && Q.rows() != K.rows()) {
            throw std::invalid_argument(
                "Causal attention requires equal lengths"
            );
        }

        std::vector<BenchmarkEntry> profile;

        if (opt.mode == "auto") {
            profile = load_profile(opt.profile);
        }

        auto execute = [&]() -> AttentionResult {

            if (opt.mode == "auto") {
                return run_attention(
                    Q, K, V,
                    opt.causal,
                    opt.budget,
                    profile
                );
            }

            if (opt.mode == "standard") {
                std::size_t scratch =
                    standard_scratch_bytes(
                        Q.rows(), K.rows()
                    );

                if (scratch > opt.budget) {
                    throw std::runtime_error(
                        "Standard attention exceeds "
                        "scratch memory budget"
                    );
                }

                return {
                    standard_attention(
                        Q, K, V, opt.causal
                    ),
                    {
                        Algorithm::Standard,
                        0, scratch, false
                    }
                };
            }

            std::size_t scratch =
                tiled_scratch_bytes(
                    K.rows(), V.cols(), opt.block
                );

            if (scratch > opt.budget) {
                throw std::runtime_error(
                    "Tiled attention exceeds "
                    "scratch memory budget"
                );
            }

            return {
                tiled_attention(
                    Q, K, V,
                    opt.block, opt.causal
                ),
                {
                    Algorithm::Tiled,
                    opt.block, scratch, false
                }
            };
        };

        const auto start =
            std::chrono::steady_clock::now();

        AttentionResult result = execute();

        const auto end =
            std::chrono::steady_clock::now();

        double runtime_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        save_matrix(result.output, opt.output);

        save_metadata(
            opt.metadata,
            opt,
            result.choice,
            Q, K, V,
            runtime_ms
        );

        const std::string selected =
            result.choice.algorithm == Algorithm::Standard
            ? "standard" : "tiled";

        std::cout << std::fixed
                  << std::setprecision(4);

        std::cout
            << "\nAutoAttn execution completed\n"
            << "Input = "
            << (csv_input ? "CSV" : "generated") << "\n"
            << "Algorithm = " << selected << "\n"
            << "Block size = "
            << result.choice.block_size << "\n"
            << "Output shape = "
            << result.output.rows() << " x "
            << result.output.cols() << "\n"
            << "Execution time = "
            << runtime_ms << " ms\n"
            << "Estimated scratch = "
            << result.choice.scratch_bytes
            << " bytes\n"
            << "Calibrated = "
            << (result.choice.calibrated
                ? "yes" : "no") << "\n"
            << "Output = " << opt.output << "\n"
            << "Metadata = " << opt.metadata << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: "
                  << e.what() << "\n";
        return 1;
    }

    return 0;
}

