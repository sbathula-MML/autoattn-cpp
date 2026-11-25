import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(sys.argv[1]).resolve()
PROFILE = Path(sys.argv[2]).resolve()

def execute(args, should_pass=False):
    result = subprocess.run(
        [str(APP), "run", *map(str, args)],
        capture_output=True,
        text=True,
    )

    success = result.returncode == 0

    if success != should_pass:
        raise RuntimeError(
            f"Unexpected result\n{result.stdout}\n"
            f"{result.stderr}"
        )

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)

    Q = root / "Q.csv"
    K = root / "K.csv"
    V = root / "V.csv"

    Q.write_text("1,0\n0,1\n")
    K.write_text("1,0\n0,1\n")
    V.write_text("10,0\n0,20\n")

    original_q = Q.read_bytes()
    original_profile = PROFILE.read_bytes()

    inputs = [
        "--q", Q,
        "--k", K,
        "--v", V,
    ]

    # Reject attempts to overwrite an input matrix.
    execute([
        *inputs,
        "--mode", "tiled",
        "--out", Q,
        "--meta", root / "metrics.json",
    ])

    assert Q.read_bytes() == original_q
    print("PASS - Input matrix protected")

    # Reject overwriting the calibration profile.
    execute([
        *inputs,
        "--mode", "auto",
        "--profile", PROFILE,
        "--out", PROFILE,
        "--meta", root / "metrics.json",
    ])

    execute([
        *inputs,
        "--mode", "auto",
        "--profile", PROFILE,
        "--out", root / "output.csv",
        "--meta", PROFILE,
    ])

    assert PROFILE.read_bytes() == original_profile
    print("PASS - Calibration profile protected")

    # Reject identical output and metadata paths.
    execute([
        *inputs,
        "--mode", "tiled",
        "--out", root / "result.csv",
        "--meta", root / "result.csv",
    ])

    print("PASS - Duplicate output paths rejected")

    # Verify normal execution still works.
    execute([
        *inputs,
        "--mode", "tiled",
        "--out", root / "result.csv",
        "--meta", root / "result.json",
    ], should_pass=True)

    assert (root / "result.csv").exists()
    assert (root / "result.json").exists()

    print("PASS - Valid execution")

print("\nAll path safety tests passed!")
