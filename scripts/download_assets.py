
"""Download model weights and datasets via HF-Mirror."""

import os
import sys
import time
from pathlib import Path

ROOT = Path("/root/siton-tmp/home/liuxinyu/loopjev")

# 必须在导入 huggingface_hub 之前设置
os.environ["HF_ENDPOINT"] = os.getenv(
    "HF_ENDPOINT", "https://hf-mirror.com"
)

# 将 HF 缓存也限制在项目目录内
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

JOBS = {
    # "ouro": dict(
    #     repo_id="ByteDance/Ouro-1.4B",
    #     repo_type="model",
    #     local_dir=str(ROOT / "models/Ouro-1.4B"),
    # ),
    # "dataset": dict(
    #     repo_id="ZefanCai/Open-Jev-v1.1",
    #     repo_type="dataset",
    #     local_dir=str(ROOT / "data/Open-Jev-v1.1"),
    # ),
    "openjev2b": dict(
        repo_id="ZefanCai/Open-Jev-2B",
        repo_type="model",
        local_dir=str(ROOT / "models/Open-Jev-2B"),
    ),
}


def download(name, retries=3):
    job = JOBS[name]
    Path(job["local_dir"]).mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(
                f"\n[{name}] Attempt {attempt}/{retries}",
                flush=True,
            )
            print(f"Endpoint: {os.environ['HF_ENDPOINT']}")

            path = snapshot_download(
                **job,
                max_workers=8,
            )

            print(f"[done] {name} -> {path}", flush=True)
            return

        except Exception as e:
            print(f"[error] {name}: {e}", flush=True)

            if attempt == retries:
                raise

            time.sleep(5 * attempt)


if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS)

    for name in names:
        if name not in JOBS:
            raise ValueError(f"Unknown job: {name}")

        download(name)
