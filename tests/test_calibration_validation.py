import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = Path(os.environ.get("AUTOATTN_EXECUTABLE", str(ROOT / "build-release" / "autoattn")))

HEADER = (
    "N,M,d,dv,causal,algorithm,"
    "block_size,median_ms\n"
)

VALID_ROW = "256,256,64,64,0,tiled,128,1.5\n"


def check_profile(content, should_pass):
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)

        profile = folder / "profile.csv"
        output = folder / "output.csv"
        metadata = folder / "metadata.json"

        profile.write_bytes(content.encode())

        command = [
            str(APP),
            "run",
            "--n", "256",
            "--d", "64",
            "--mode", "auto",
            "--budget-bytes", "4096",
            "--profile", str(profile),
            "--out", str(output),
            "--meta", str(metadata),
        ]

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if should_pass:
            assert result.returncode == 0, result.stderr
            assert output.exists()
            assert metadata.exists()

            info = json.loads(metadata.read_text())
            assert info["calibrated"] is True
            assert info["block_size"] == 128
        else:
            assert result.returncode != 0, (
                "Invalid calibration file was accepted"
            )
            assert not output.exists()


# Valid calibration
check_profile(HEADER + VALID_ROW, True)
print("PASS - Valid calibration file")

# Windows-style line endings
check_profile(
    (HEADER + VALID_ROW).replace("\n", "\r\n"),
    True
)
print("PASS - Windows line endings")

# Invalid header
check_profile("invalid_header\n" + VALID_ROW, False)
print("PASS - Invalid header rejected")

# No measurements
check_profile(HEADER, False)
print("PASS - Empty calibration rejected")

# Invalid dimensions
check_profile(
    HEADER + "0,256,64,64,0,tiled,128,1.5\n",
    False
)
print("PASS - Invalid dimensions rejected")

# Invalid block size
check_profile(
    HEADER + "256,256,64,64,0,tiled,0,1.5\n",
    False
)
print("PASS - Invalid block rejected")

# Non-finite runtime
check_profile(
    HEADER + "256,256,64,64,0,tiled,128,nan\n",
    False
)
print("PASS - Non-finite runtime rejected")

# Missing measurement
check_profile(
    HEADER + "256,256,64,64,0,tiled,128\n",
    False
)
print("PASS - Incomplete row rejected")

# Extra CSV field
check_profile(
    HEADER + "256,256,64,64,0,tiled,128,1.5,extra\n",
    False
)
print("PASS - Extra CSV field rejected")

print("\nAll calibration validation tests passed!")
