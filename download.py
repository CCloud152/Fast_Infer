"""Download Llama 3.2 3B Instruct from ModelScope."""

from pathlib import Path

MODEL_ID = "LLM-Research/Llama-3.2-3B-Instruct"
DEFAULT_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / MODEL_ID.replace("/", "__")


def download(cache_dir: str | None = None) -> Path:
    """Download model from ModelScope. Returns path to model directory."""
    from modelscope import snapshot_download

    target = cache_dir or str(DEFAULT_CACHE)
    Path(target).mkdir(parents=True, exist_ok=True)

    model_dir = snapshot_download(
        MODEL_ID,
        cache_dir=target,
    )
    return Path(model_dir)


if __name__ == "__main__":
    path = download()
    print(f"Model downloaded to: {path}")
    # List key files
    for f in sorted(path.iterdir()):
        print(f"  {f.name}")
