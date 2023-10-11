import os
import subprocess


SKIP_CUDA_TESTS = (
    (os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1")
    and (len(subprocess.run(
        "which nvidia-smi",
        shell=True,
        capture_output=True).stdout,
    ) == 0)
)
