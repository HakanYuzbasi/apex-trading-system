"""
models/model_manifest.py - Saved model manifest and checksum verification.

Tracks model artifact integrity for production model directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_MODEL_DIRS: Tuple[Path, ...] = (
    Path("models/saved_advanced"),
    Path("models/saved_ultimate"),
    Path("models/test_enhanced_features"),
)
DEFAULT_PATTERNS: Tuple[str, ...] = ("*.json", "*.pkl", "*.bin", "*.onnx", "*.pt")
DEFAULT_MANIFEST_PATH = Path("models/model_manifest.json")


@dataclass(frozen=True)
class ModelManifestEntry:
    path: str
    sha256: str
    size_bytes: int
    modified_at: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
        }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_model_files(base_dirs: Sequence[Path], patterns: Sequence[str]) -> Iterable[Path]:
    seen = set()
    for base in base_dirs:
        if not base.exists():
            continue
        for pattern in patterns:
            for candidate in base.rglob(pattern):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    yield candidate


def build_manifest(
    base_dirs: Sequence[Path] = DEFAULT_MODEL_DIRS,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> Dict[str, object]:
    entries: List[ModelManifestEntry] = []
    for file_path in sorted(_iter_model_files(base_dirs, patterns)):
        stat = file_path.stat()
        entries.append(
            ModelManifestEntry(
                path=file_path.as_posix(),
                sha256=_hash_file(file_path),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            )
        )

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [entry.to_dict() for entry in entries],
    }


def write_manifest(manifest: Dict[str, object], output_path: Path = DEFAULT_MANIFEST_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return output_path


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> List[str]:
    manifest = load_manifest(path)
    errors: List[str] = []
    for entry in manifest.get("files", []):
        file_path = Path(entry["path"])
        if not file_path.exists():
            errors.append(f"missing: {file_path.as_posix()}")
            continue
        expected = entry["sha256"]
        actual = _hash_file(file_path)
        if expected != actual:
            errors.append(
                f"checksum_mismatch: {file_path.as_posix()} expected={expected} actual={actual}"
            )
    return errors


def _cmd_generate(args: argparse.Namespace) -> int:
    manifest = build_manifest()
    output = write_manifest(manifest, Path(args.output))
    print(f"Wrote manifest: {output}")
    print(f"Tracked files: {len(manifest['files'])}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    errors = verify_manifest(Path(args.path))
    if errors:
        print("Manifest verification failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Manifest verification passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Saved model manifest tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate/update model manifest")
    gen.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    gen.set_defaults(func=_cmd_generate)

    verify = sub.add_parser("verify", help="Verify model files against manifest")
    verify.add_argument("--path", default=str(DEFAULT_MANIFEST_PATH))
    verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
