"""Verify saved model artifacts against manifest checksums."""

from models.model_manifest import main


if __name__ == "__main__":
    raise SystemExit(main(["verify"]))
