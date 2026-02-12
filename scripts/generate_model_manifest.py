"""Generate manifest/checksums for saved model artifacts."""

from models.model_manifest import main


if __name__ == "__main__":
    raise SystemExit(main(["generate"]))
