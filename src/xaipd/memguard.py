"""Resource guard. Import FIRST (before numpy/torch/tensorflow) in every script:
    import xaipd.memguard  # noqa
- limits BLAS/OMP threads to 4 (user is gaming; CPU must stay available)
- watchdog thread: if process RSS > LIMIT_GB (default 6), print reason and hard-exit (code 99)."""
from __future__ import annotations

import os
import sys
import threading
import time

THREADS = os.environ.get("XAIPD_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
          "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[k] = THREADS if "INTEROP" not in k else "2"

LIMIT_GB = float(os.environ.get("XAIPD_MEM_LIMIT_GB", "6"))

try:
    import psutil
except ImportError:  # refuse to run unguarded
    sys.stderr.write("memguard: psutil missing -> refusing to run without a memory guard\n")
    raise

_proc = psutil.Process()
if os.environ.get("XAIPD_LOWPRIO", "1") == "1" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    _proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # keep GenerativeUI latency measurements stable
peak = {"rss_gb": 0.0}


def _watch():
    while True:
        rss = _proc.memory_info().rss / 2**30
        peak["rss_gb"] = max(peak["rss_gb"], rss)
        if rss > LIMIT_GB:
            sys.stderr.write(f"\nmemguard: RSS {rss:.2f} GB > {LIMIT_GB} GB limit -> aborting\n")
            sys.stderr.flush()
            os._exit(99)
        time.sleep(0.5)


threading.Thread(target=_watch, daemon=True).start()


def limit_torch():
    import torch
    torch.set_num_threads(int(THREADS))
    torch.set_num_interop_threads(1)


def limit_tf():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(THREADS))
    tf.config.threading.set_inter_op_parallelism_threads(2)


def report(tag=""):
    print(f"[memguard]{tag} peak RSS {peak['rss_gb']:.2f} GB (limit {LIMIT_GB} GB)", flush=True)
