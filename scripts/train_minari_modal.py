import os
import sys
from pathlib import Path

import modal


PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096

app = modal.App("decision-diffuser-train")

volume = modal.Volume.from_name(
    "decision-diffuser-train-volume",
    create_if_missing=True,
)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns



image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libosmesa6", "patchelf", "swig")
    .uv_sync()
    .add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
    )
)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    timeout=60 * 60 * 6,
    volumes={VOLUME_PATH: volume},
    env={
        "PYTHONPATH": PROJECT_DIR,
    },
)
def train(*args: str) -> None:
    os.chdir(PROJECT_DIR)
    minari_home = Path(VOLUME_PATH) / "minari"
    minari_home.mkdir(parents=True, exist_ok=True)

    default_minari_home = Path("/root/.minari")
    if default_minari_home.is_dir() and not default_minari_home.is_symlink():
        import shutil
        shutil.rmtree(default_minari_home)
    elif default_minari_home.exists() or default_minari_home.is_symlink():
        default_minari_home.unlink()
    default_minari_home.symlink_to(minari_home)

    from scripts.train_minari import main as train_main

    sys.argv = [
        "train_minari.py",
        *args,
        "--device",
        "cuda",
    ]
    train_main()
    volume.commit()


@app.local_entrypoint()
def main(*args: str) -> None:
    train.remote(*args)
