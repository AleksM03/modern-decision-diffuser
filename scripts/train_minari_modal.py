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

    patterns: list[str] = ["**/exp/**"]
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
    .apt_install(
        "libegl1",
        "libgl1",
        "libgles2",
        "libglib2.0-0",
        "libglvnd0",
        "libosmesa6",
        "patchelf",
        "swig",
    )
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
    import faulthandler

    faulthandler.enable(all_threads=True)
    print("modal: starting train entrypoint", flush=True)

    os.chdir(PROJECT_DIR)
    print("modal: project dir ready", flush=True)

    minari_home = Path(VOLUME_PATH) / "minari"
    minari_home.mkdir(parents=True, exist_ok=True)

    default_minari_home = Path("/root/.minari")
    if default_minari_home.is_dir() and not default_minari_home.is_symlink():
        import shutil
        shutil.rmtree(default_minari_home)
    elif default_minari_home.exists() or default_minari_home.is_symlink():
        default_minari_home.unlink()
    default_minari_home.symlink_to(minari_home)
    print("modal: minari volume ready", flush=True)

    exp_vol = Path(VOLUME_PATH) / "exp"
    exp_vol.mkdir(parents=True, exist_ok=True)

    exp_link = Path(PROJECT_DIR) / "exp"
    if exp_link.is_dir() and not exp_link.is_symlink():
        import shutil
        shutil.rmtree(exp_link)
    elif exp_link.exists() or exp_link.is_symlink():
        exp_link.unlink()
    exp_link.symlink_to(exp_vol)
    print("modal: exp volume ready", flush=True)

    if "--eval-video" in args:
        args = tuple(arg for arg in args if arg != "--eval-video")
        print("modal: eval video disabled in training container", flush=True)

    print("modal: importing train_minari", flush=True)
    from scripts.train_minari import main as train_main
    from scripts.train_minari import make_logger, parse_args

    sys.argv = ["train_minari.py", *args]
    if "--device" not in args:
        sys.argv.extend(["--device", "cuda"])
    print("modal: parsing args", flush=True)
    train_args = parse_args()
    print(f"modal: parsed args {train_args}", flush=True)

    logger = make_logger(train_args)
    print("modal: launching training", flush=True)
    try:
        train_main(logger, train_args)
    finally:
        logger.close()
        print("modal: committing volume", flush=True)
        volume.commit()


@app.local_entrypoint()
def main(*args: str) -> None:
    wait = False
    train_args = list(args)
    if "--wait" in train_args:
        wait = True
        train_args.remove("--wait")

    call = train.spawn(*train_args)
    print(f"modal: submitted training call {call.object_id}", flush=True)
    print("modal: safe to close this terminal now", flush=True)
    print(
        "modal: follow logs with "
        f"`modal app logs {app.name} --function-call {call.object_id} -f`",
        flush=True,
    )
    print(
        "modal: fetch recent logs with "
        f"`modal app logs {app.name} --function-call {call.object_id} --tail 200`",
        flush=True,
    )

    if wait:
        print("modal: --wait supplied; waiting for training to finish", flush=True)
        call.get()
