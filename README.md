# AutoAttn

### Adaptive and Memory-Efficient Transformer Attention in C++

AutoAttn is a CPU-based Transformer attention engine implemented in C++17. It provides standard scaled dot-product attention, memory-efficient tiled attention using online softmax, and an adaptive dispatcher that selects an implementation based on measured performance and a scratch-memory budget.

The project investigates the trade-offs between execution time, temporary memory usage, numerical accuracy, and attention block size.

## Features

- Standard scaled dot-product attention
- Memory-efficient attention using key-block tiling and online softmax
- Causal and noncausal attention
- Adaptive algorithm and block-size selection
- Benchmark-driven calibration profiles
- Configurable scratch-memory budgets
- CSV input and output
- Deterministic input generation
- Automated C++ and Python correctness tests
- Reproducible CPU benchmarks

## Requirements

- C++17-compatible compiler
- CMake 3.16 or newer
- Python 3 with NumPy for integration tests

The implementation uses the C++ standard library and does not require CUDA or an external attention library.

## Building

Clone the repository.

```bash
git clone https://github.com/sbathula-MML/autoattn-cpp.git
cd autoattn-cpp
```

Configure and compile the optimized executable.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j 4
```

The executable is created at `build-release/autoattn`.

## Usage

### Adaptive attention

```bash
./build-release/autoattn run \
  --n 256 \
  --d 64 \
  --seed 7 \
  --mode auto \
  --budget-bytes 4096 \
  --out attention.csv \
  --meta metrics.json
```

The engine loads the calibration profile and selects a feasible attention implementation.

### Standard attention

```bash
./build-release/autoattn run \
  --n 256 \
  --d 64 \
  --mode standard \
  --budget-bytes 300000 \
  --out standard.csv \
  --meta standard.json
```

### Tiled attention

```bash
./build-release/autoattn run \
  --n 256 \
  --d 64 \
  --mode tiled \
  --block 128 \
  --budget-bytes 4096 \
  --out tiled.csv \
  --meta tiled.json
```

Add `--causal` to enable causal masking.

### Custom CSV input

AutoAttn also accepts user-provided Q, K, and V matrices.

```bash
./build-release/autoattn run \
  --q tests/Q_sample.csv \
  --k tests/K_sample.csv \
  --v tests/V_sample.csv \
  --mode auto \
  --out output.csv \
  --meta output.json
```

Q and K must have matching feature dimensions. K and V must have matching row counts.

The output is saved as a CSV matrix. Execution metadata is saved as JSON.

## Attention Algorithms

### Standard attention

The baseline computes scaled query-key similarity scores, applies a numerically stable softmax, and multiplies the resulting attention weights by V.

It materializes an attention-score matrix requiring O(NM) temporary storage for N queries and M keys.

### Memory-efficient tiled attention

The tiled implementation processes one query row at a time and divides the keys into blocks.

It uses online softmax to maintain a running maximum, normalization factor, and weighted output accumulator.

This avoids materializing the full attention matrix.

The main scratch buffers require O(B + dv) memory, where B is the block size and dv is the value dimension.

### Adaptive dispatcher

The dispatcher selects the fastest measured configuration that satisfies the scratch-memory budget.

For workloads with no exact calibration match, it uses a memory-aware fallback policy.

The memory budget applies to estimated algorithm scratch buffers, not total process memory.

## Experimental Results

Experiments were conducted on an Apple Silicon MacBook Pro using optimized C++17 CPU implementations.

### Runtime and memory

For sequence length 4096 and attention dimension 64, one measured execution produced the following results.

| Metric | Standard | Tiled |
|---|---:|---:|
| Runtime | 1.59 s | 0.43 s |
| Peak resident memory | 69.45 MiB | 5.41 MiB |

The tiled implementation achieved approximately 3.70x lower wall-clock runtime and 92.2% lower peak resident memory in this comparison.

These timings are individual observations rather than repeated-run statistical estimates.

### Adaptive block selection

Calibration covered 12 workloads and 72 algorithm configurations.

The best measured configuration improved runtime by an average of 1.913% over fixed block size 128 on the calibration workloads.

On eight held-out workloads, the fallback policy had an initial mean measured gap of 2.462% relative to the fastest tested tiled configuration.

Follow-up repeated measurements showed that some small differences arose from benchmark variability. These results should therefore be interpreted as exploratory rather than definitive generalization gains.

## Testing

Run the automated test suite with:

```bash
ctest --test-dir build-release --output-on-failure
```

The suite covers matrix operations, attention correctness, causal masking, adaptive dispatch, memory estimates, CSV parsing, calibration loading, CLI integration, and file safety.

Independent NumPy comparisons are also used to verify numerical correctness.

## Project Structure

- `include/` - Matrix class, attention kernels, dispatcher, and utilities
- `src/` - Command-line application
- `tests/` - C++ and Python tests
- `benchmarks/` - Performance experiments, calibration profiles, and results
- `CMakeLists.txt` - Build and testing configuration

## Limitations

AutoAttn is a CPU-oriented educational ML systems implementation, not a production Transformer inference engine.

The current tiled kernel processes one query row at a time rather than using a fully optimized two-dimensional tiled GPU implementation.

The dispatcher primarily relies on exact workload calibration. Unseen workloads use a fixed-priority memory-aware fallback.

GPU kernels, automatic differentiation, full Transformer models, and distributed execution are outside the project scope.

## References

1. Milakov and Gimelshein, Online normalizer calculation for softmax, 2018.
   https://arxiv.org/abs/1805.02867

2. Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, NeurIPS 2022.
   https://arxiv.org/abs/2205.14135

## Author

**Satwik Bathula**

University of Southern California

**Email** — sbathula@usc.edu

