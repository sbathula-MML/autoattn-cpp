
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APP = Path(os.environ.get("AUTOATTN_EXECUTABLE", str(ROOT / "build-release" / "autoattn")))


def run(args, expect_success=True):
    result = subprocess.run(
        [str(APP), "run", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if expect_success and result.returncode != 0:
        raise RuntimeError(result.stderr)

    if not expect_success and result.returncode == 0:
        raise RuntimeError("Invalid input was accepted")

    return result


def check_close(actual, expected):
    np.testing.assert_allclose(
        actual, expected,
        rtol=1e-5,
        atol=1e-6,
    )


with tempfile.TemporaryDirectory() as directory:
    tmp = Path(directory)

    outputs = {}

    # Test all three execution modes.
    for mode in ["standard", "tiled", "auto"]:
        output = tmp / f"{mode}.csv"
        metadata = tmp / f"{mode}.json"

        run([
            "--n", 256,
            "--d", 64,
            "--seed", 7,
            "--mode", mode,
            "--budget-bytes", 300000,
            "--out", output,
            "--meta", metadata,
        ])

        outputs[mode] = np.loadtxt(
            output, delimiter=","
        )

        with open(metadata) as file:
            info = json.load(file)

        assert outputs[mode].shape == (256, 64)
        assert info["estimated_scratch_bytes"] <= 300000

        print(f"PASS - {mode} execution")

    check_close(outputs["tiled"], outputs["standard"])
    check_close(outputs["auto"], outputs["standard"])

    print("PASS - Numerical equivalence")

    # Test CSV input against independent NumPy.
    Q = np.array([[1, 0], [0, 1]], dtype=float)
    K = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    V = np.array([[10, 0], [0, 20], [5, 5]], dtype=float)

    for name, matrix in [("Q", Q), ("K", K), ("V", V)]:
        np.savetxt(
            tmp / f"{name}.csv",
            matrix,
            delimiter=","
        )

    output = tmp / "csv_result.csv"

    run([
        "--q", tmp / "Q.csv",
        "--k", tmp / "K.csv",
        "--v", tmp / "V.csv",
        "--mode", "auto",
        "--out", output,
        "--meta", tmp / "csv_metrics.json",
    ])

    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores -= scores.max(axis=1, keepdims=True)

    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)

    expected = weights @ V
    actual = np.loadtxt(output, delimiter=",")

    check_close(actual, expected)

    print("PASS - CSV input and NumPy reference")

    # Invalid memory budget.
    run([
        "--n", 256,
        "--d", 64,
        "--mode", "standard",
        "--budget-bytes", 1,
    ], expect_success=False)

    print("PASS - Memory budget rejection")

    # Incomplete CSV input.
    run([
        "--q", tmp / "Q.csv",
        "--mode", "tiled",
    ], expect_success=False)

    print("PASS - Incomplete CSV rejection")

    # Invalid algorithm.
    run([
        "--mode", "unknown",
    ], expect_success=False)

    print("PASS - Invalid algorithm rejection")

print("\nAll CLI integration tests passed!")

