
import subprocess
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXECUTABLE = ROOT / "tests" / "attention_reference_driver"


def generate_inputs(N, M, d, dv):
    # Reproduce the deterministic C++ inputs exactly.
    i = np.arange(N)[:, None]
    j = np.arange(d)[None, :]

    Q = (((i * 7 + j * 3) % 19) - 9).astype(
        np.float32
    ) * np.float32(0.15)

    i = np.arange(M)[:, None]

    K = (((i * 5 + j * 11) % 23) - 11).astype(
        np.float32
    ) * np.float32(0.12)

    j = np.arange(dv)[None, :]

    V = (((i * 13 + j * 7) % 29) - 14).astype(
        np.float32
    ) * np.float32(0.08)

    return Q, K, V


def numpy_attention(Q, K, V, causal):
    # Independent reference using float64 arithmetic.
    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)

    scores = (Q @ K.T) / np.sqrt(Q.shape[1])

    if causal:
        mask = np.triu(
            np.ones(scores.shape, dtype=bool),
            k=1
        )
        scores[mask] = -np.inf

    scores -= np.max(scores, axis=1, keepdims=True)

    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)

    return weights @ V


def cpp_attention(N, M, d, dv, causal):
    command = [
        str(EXECUTABLE),
        str(N),
        str(M),
        str(d),
        str(dv),
        str(int(causal))
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    values = np.fromstring(result.stdout, sep=" ")

    if values.size != N * dv:
        raise RuntimeError("Unexpected C++ output shape")

    return values.reshape(N, dv)


def run_test(N, M, d, dv, causal):
    Q, K, V = generate_inputs(N, M, d, dv)

    expected = numpy_attention(Q, K, V, causal)
    actual = cpp_attention(N, M, d, dv, causal)

    error = np.max(np.abs(expected - actual))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6
    )

    print(
        f"PASS N={N}, M={M}, d={d}, dv={dv}, "
        f"causal={causal}, max_error={error:.3e}"
    )

    return error


if __name__ == "__main__":
    configurations = [
        (1, 1, 1, 1, False),
        (3, 3, 4, 2, False),
        (3, 3, 4, 2, True),
        (7, 11, 5, 3, False),
        (17, 17, 8, 4, True),
        (31, 29, 16, 8, False),
        (64, 64, 32, 16, False),
        (64, 64, 32, 16, True),
    ]

    errors = []

    for config in configurations:
        errors.append(run_test(*config))

    print("\nAll NumPy reference tests passed!")
    print(f"Maximum absolute error: {max(errors):.3e}")

