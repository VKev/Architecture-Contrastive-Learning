import argparse
import json
import os
from typing import Any, Dict

import torch

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # YAML output will be unavailable if PyYAML is not installed

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # Fallback when numpy is not available


def _to_builtin(obj: Any) -> Any:
    """Best-effort conversion of objects to JSON/YAML serializable builtins."""
    # Handle argparse.Namespace
    try:
        from argparse import Namespace  # local import to avoid unused in some envs

        if isinstance(obj, Namespace):
            return {k: _to_builtin(v) for k, v in vars(obj).items()}
    except Exception:
        pass

    # Handle numpy scalars
    if np is not None:
        if isinstance(obj, (np.generic,)):
            return obj.item()

    # Handle tensors by converting to Python number or list
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    # Containers
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(v) for v in obj]

    # Fallback: string conversion for unknown objects
    try:
        json.dumps(obj)  # type: ignore[arg-type]
        return obj
    except Exception:
        return str(obj)


def load_hyperparameters(ckpt_path: str) -> Dict[str, Any]:
    """Load the checkpoint and return its hyperparameters dict.

    Supports PyTorch Lightning checkpoints where hyperparameters are stored under
    the 'hyper_parameters' key.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Common PyTorch Lightning key
    if isinstance(checkpoint, dict) and "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
    else:
        # Try a few common alternatives just in case
        for candidate in ("hparams", "args", "config"):
            if isinstance(checkpoint, dict) and candidate in checkpoint:
                hparams = checkpoint[candidate]
                break
        else:
            raise KeyError(
                "No hyperparameters found in checkpoint. Expected 'hyper_parameters' key."
            )

    return _to_builtin(hparams)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and print hyperparameters stored in a model checkpoint (.ckpt/.pth)"
    )
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--format",
        choices=["plain", "json", "yaml"],
        default="plain",
        help="Output format",
    )
    args = parser.parse_args()

    hparams = load_hyperparameters(args.ckpt)

    if args.format == "json":
        print(json.dumps(hparams, indent=2, ensure_ascii=False))
        return

    if args.format == "yaml":
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Install with 'pip install pyyaml' or use --format json/plain.")
        print(yaml.safe_dump(hparams, sort_keys=True, allow_unicode=True))
        return

    # Plain aligned key: value listing
    keys = sorted(hparams.keys(), key=lambda k: str(k))
    if not keys:
        print("No hyperparameters found.")
        return

    pad = max(len(str(k)) for k in keys)
    for k in keys:
        v = hparams[k]
        print(f"{str(k).ljust(pad)} : {v}")


if __name__ == "__main__":
    main()


