"""Pack everything a Colab training run needs into one tar (`make colab-bundle`).

Contents: the committed source (git ls-files; the tree must be clean so the run maps to
a commit), the pinned YOLOX source, the starting checkpoint, the training annotations for
each size in configs/train/yolox_tiny_*_v0.yaml, and the referenced images (stored once,
dereferenced from the combined dataset's symlinks). Images and checkpoints never go
into Git; the bundle lives in git-ignored `data/colab/`.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

import yaml

from platedet.eval.baseline import sha256_file
from platedet.train.dataset import TRAINING_SPLITS, training_dir
from platedet.train.yolox_src import ensure_yolox

DEFAULT_CONFIGS = (
    Path("configs/train/yolox_tiny_640_v0.yaml"),
    Path("configs/train/yolox_tiny_416_v0.yaml"),
)
IMAGE_ROOT = Path("data/processed/combined")
BUNDLE_INFO = "bundle_info.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _symlink(tar: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tar.addfile(info)


def build_bundle(configs: tuple[Path, ...], output: Path) -> dict[str, object]:
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("Commit your changes first: the bundle must match a git commit")
    commit = _git("rev-parse", "HEAD").strip()
    yolox = ensure_yolox()
    loaded = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in configs]
    pretrained = {Path(config["pretrained"]["path"]) for config in loaded}
    for config in loaded:
        if sha256_file(Path(config["pretrained"]["path"])) != config["pretrained"]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {config['name']}")

    image_files: set[str] = set()
    data_dirs = [training_dir(config) for config in loaded]
    for data_dir in data_dirs:
        for split in TRAINING_SPLITS:
            coco = json.loads((data_dir / "annotations" / f"{split}.json").read_text("utf-8"))
            image_files.update(image["file_name"] for image in coco["images"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as tar:
        # Colab has no .git folder; train.py records this commit instead.
        stamp = json.dumps({"git_commit": commit, "dirty": False}).encode()
        stamp_info = tarfile.TarInfo(BUNDLE_INFO)
        stamp_info.size = len(stamp)
        tar.addfile(stamp_info, io.BytesIO(stamp))
        for name in _git("ls-files").splitlines():
            tar.add(name, recursive=False)
        tar.add(str(yolox))
        for path in sorted(pretrained):
            tar.add(str(path))
        for data_dir in data_dirs:
            tar.add(str(data_dir / "annotations"))
            tar.add(str(data_dir / "stats.json"))
            _symlink(tar, str(data_dir / "images"), "../combined/images")
        for name in sorted(image_files):
            source = (IMAGE_ROOT / name).resolve(strict=True)
            tar.add(str(source), arcname=str(IMAGE_ROOT / name))
    info = {
        "git_commit": commit,
        "configs": [str(path) for path in configs],
        "images": len(image_files),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("data/colab/platedet_bundle.tar"))
    args = parser.parse_args()
    info = build_bundle(tuple(args.config or DEFAULT_CONFIGS), args.output)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
