from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.build_dataset import _encode_file
from cs336_basics.build_tokenizer import train_tokenizer
from cs336_basics.lm_trainer import train_lm
from cs336_basics.transfromer.scaled_dot_prod_attention import resolve_attention_kernel
from cs336_systems.Parallelization.DDP.DDP_runner import run_naive_ddp_training
from cs336_systems.Parallelization.FlashDDP.FlashDDP_runner import run_flashddp_training

DATASET_MODES = {"tinystories", "local_text", "local_npy", "auto"}
DDP_WRAPPERS = {"none", "naive", "flashddp"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "default_pipeline_config.json"


def parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = json.loads(json.dumps(config))
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        cursor = updated
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError(f"Invalid override key '{key}'.")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path '{key}' collides with non-object value.")
        cursor[parts[-1]] = parse_override_value(value)
    return updated


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo / path


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def resolve_dataset_mode(dataset_mode: str, paths_cfg: dict[str, Any]) -> str:
    if dataset_mode not in DATASET_MODES:
        raise ValueError(f"Unsupported dataset_mode '{dataset_mode}'.")

    if dataset_mode != "auto":
        return dataset_mode

    def _exists(value: Any) -> bool:
        if not value:
            return False
        return Path(str(value)).exists()

    if all(
        _exists(paths_cfg.get(key))
        for key in ("train_npy", "val_npy", "vocab_path", "merges_path")
    ):
        return "local_npy"

    if _exists(paths_cfg.get("train_text")) and _exists(paths_cfg.get("val_text")):
        return "local_text"

    return "tinystories"


def _require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _ensure_tokenizer_assets(
    train_text: Path,
    tokenizer_cfg: dict[str, Any],
    vocab_path: Path,
    merges_path: Path,
) -> None:
    force_retrain = bool(tokenizer_cfg.get("force_retrain", False))
    if vocab_path.exists() and merges_path.exists() and not force_retrain:
        return

    special_tokens = tokenizer_cfg.get("special_tokens", ["<|endoftext|>"])
    if not isinstance(special_tokens, list) or not special_tokens:
        raise ValueError("tokenizer.special_tokens must be a non-empty list.")

    vocab_size = int(tokenizer_cfg.get("vocab_size", 10_000))
    if not _is_power_of_two(vocab_size):
        print(
            f"Warning: tokenizer vocab_size={vocab_size} is not a power of two; proceeding anyway."
        )

    vocab, merges = train_tokenizer(
        input_path=str(train_text),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=int(tokenizer_cfg.get("num_processes", 4)),
        split_special_token=special_tokens[0].encode("utf-8"),
    )

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    merges_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)


def _ensure_npy_from_text(
    text_path: Path,
    vocab_path: Path,
    merges_path: Path,
    out_path: Path,
    num_workers: int,
    size: int | None,
    force_rebuild: bool,
) -> None:
    if out_path.exists() and not force_rebuild:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = _encode_file(text_path, vocab_path, merges_path, size, num_workers)
    np.save(out_path, tokens)


def prepare_dataset_assets(config: dict[str, Any]) -> dict[str, str]:
    repo = _repo_root()
    paths_cfg = dict(config.get("paths", {}))

    paths_cfg.setdefault("train_text", str(repo / "cs336-basics/data/TinyStoriesV2-GPT4-train.txt"))
    paths_cfg.setdefault("val_text", str(repo / "cs336-basics/data/TinyStoriesV2-GPT4-valid.txt"))
    paths_cfg.setdefault("train_npy", str(repo / "cs336-basics/data/tokenized/ts_train.npy"))
    paths_cfg.setdefault("val_npy", str(repo / "cs336-basics/data/tokenized/ts_valid.npy"))
    paths_cfg.setdefault(
        "vocab_path",
        str(repo / "cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl"),
    )
    paths_cfg.setdefault(
        "merges_path",
        str(repo / "cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl"),
    )

    resolved_paths_cfg = {
        k: str(_resolve_path(repo, v))
        for k, v in paths_cfg.items()
    }

    mode = resolve_dataset_mode(config.get("dataset_mode", "auto"), resolved_paths_cfg)

    train_text = Path(resolved_paths_cfg["train_text"])
    val_text = Path(resolved_paths_cfg["val_text"])
    train_npy = Path(resolved_paths_cfg["train_npy"])
    val_npy = Path(resolved_paths_cfg["val_npy"])
    vocab_path = Path(resolved_paths_cfg["vocab_path"])
    merges_path = Path(resolved_paths_cfg["merges_path"])

    tokenizer_cfg = dict(config.get("tokenizer", {}))
    dataset_build_cfg = dict(config.get("dataset_build", {}))

    if mode == "tinystories":
        ts_cfg = dict(config.get("tinystories", {}))
        train_url = ts_cfg.get(
            "train_url",
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        )
        val_url = ts_cfg.get(
            "val_url",
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        )
        force_download = bool(ts_cfg.get("force_download", False))
        if force_download or not train_text.exists():
            _download_file(train_url, train_text)
        if force_download or not val_text.exists():
            _download_file(val_url, val_text)

    if mode == "local_npy":
        _require_existing(train_npy, "train_npy")
        _require_existing(val_npy, "val_npy")
        _require_existing(vocab_path, "vocab_path")
        _require_existing(merges_path, "merges_path")
        return {
            "dataset_mode": mode,
            "train_npy": str(train_npy),
            "val_npy": str(val_npy),
            "vocab_path": str(vocab_path),
            "merges_path": str(merges_path),
        }

    _require_existing(train_text, "train_text")
    _require_existing(val_text, "val_text")

    _ensure_tokenizer_assets(train_text, tokenizer_cfg, vocab_path, merges_path)

    _ensure_npy_from_text(
        text_path=train_text,
        vocab_path=vocab_path,
        merges_path=merges_path,
        out_path=train_npy,
        num_workers=int(dataset_build_cfg.get("num_workers", 4)),
        size=dataset_build_cfg.get("train_size"),
        force_rebuild=bool(dataset_build_cfg.get("force_rebuild", False)),
    )
    _ensure_npy_from_text(
        text_path=val_text,
        vocab_path=vocab_path,
        merges_path=merges_path,
        out_path=val_npy,
        num_workers=int(dataset_build_cfg.get("num_workers", 4)),
        size=dataset_build_cfg.get("val_size"),
        force_rebuild=bool(dataset_build_cfg.get("force_rebuild", False)),
    )

    return {
        "dataset_mode": mode,
        "train_npy": str(train_npy),
        "val_npy": str(val_npy),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }


def build_training_kwargs(config: dict[str, Any], assets: dict[str, str]) -> tuple[dict[str, Any], str]:
    training_cfg = dict(config.get("training", {}))

    ddp_wrapper = str(
        training_cfg.pop("ddp_wrapper", training_cfg.pop("DDP_WRAPPER", "none"))
    ).lower()
    if ddp_wrapper not in DDP_WRAPPERS:
        raise ValueError(f"Unsupported ddp_wrapper '{ddp_wrapper}'.")

    attention_kernel_raw = str(
        training_cfg.pop("attention_kernel", training_cfg.pop("ATTENTION_KERNEL", "scaled_dot_prod_attention"))
    )
    resolved_kernel_name, _ = resolve_attention_kernel(attention_kernel_raw)

    train_kwargs: dict[str, Any] = {}
    for key, value in training_cfg.items():
        normalized = key if key.isupper() else key.upper()
        train_kwargs[normalized] = value

    train_kwargs["TRAIN_PATH"] = assets["train_npy"]
    train_kwargs["VAL_PATH"] = assets["val_npy"]
    train_kwargs["VOCAB_PATH"] = assets["vocab_path"]
    train_kwargs["MERGES_PATH"] = assets["merges_path"]
    train_kwargs["ATTENTION_KERNEL"] = resolved_kernel_name
    train_kwargs.setdefault("DEVICE", "cuda")

    if not train_kwargs.get("WANDB_PROJECT"):
        raise RuntimeError("training.WANDB_PROJECT is required for pipeline runs.")

    return train_kwargs, ddp_wrapper


def _to_ddp_namespace(train_kwargs: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        EPOCHES=int(train_kwargs.get("EPOCHES", 5)),
        WARMUP_EPOCHS=int(train_kwargs.get("WARMUP_EPOCHS", 2)),
        EVAL_INTERVAL=int(train_kwargs.get("EVAL_INTERVAL", 100)),
        TR_BAT_SIZE=int(train_kwargs.get("TR_BAT_SIZE", 8)),
        VAL_BAT_SIZE=int(train_kwargs.get("VAL_BAT_SIZE", 8)),
        TRAIN_PATH=str(train_kwargs["TRAIN_PATH"]),
        VAL_PATH=str(train_kwargs["VAL_PATH"]),
        DTYPE=str(train_kwargs.get("DTYPE", "float32")),
        CONTEXT_LENGTH=int(train_kwargs.get("CONTEXT_LENGTH", 256)),
        PRINT_EVERY=int(train_kwargs.get("PRINT_EVERY", 1)),
        VOCAB_SIZE=int(train_kwargs.get("VOCAB_SIZE", 10000)),
        ROPE_THETA=float(train_kwargs.get("ROPE_THETA", 10_000.0)),
        NUM_LAYERS=int(train_kwargs.get("NUM_LAYERS", 16)),
        D_MODEL=int(train_kwargs.get("D_MODEL", 256)),
        NUM_HEADS=int(train_kwargs.get("NUM_HEADS", 8)),
        D_FF=int(train_kwargs.get("D_FF", 1024)),
        ATTN_KERNEL=str(train_kwargs["ATTENTION_KERNEL"]),
        LR=float(train_kwargs.get("LR", 3e-4)),
        WEIGHT_DECAY=float(train_kwargs.get("WEIGHT_DECAY", 0.01)),
        BETA1=float(train_kwargs.get("BETA1", 0.9)),
        BETA2=float(train_kwargs.get("BETA2", 0.999)),
        ADAM_EPS=float(train_kwargs.get("ADAM_EPS", 1e-8)),
        BUCKET_SIZE_MB=int(train_kwargs.get("BUCKET_SIZE_MB", 1)),
    )


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("run_pipeline.py is CUDA-only.")

    assets = prepare_dataset_assets(config)
    train_kwargs, ddp_wrapper = build_training_kwargs(config, assets)

    world_size = torch.cuda.device_count()
    distributed_enabled = ddp_wrapper != "none" and world_size > 1

    if distributed_enabled:
        ddp_args = _to_ddp_namespace(train_kwargs)
        if ddp_wrapper == "naive":
            run_result = run_naive_ddp_training(ddp_args)
        elif ddp_wrapper == "flashddp":
            run_result = run_flashddp_training(ddp_args)
        else:
            raise ValueError(f"Unsupported ddp_wrapper '{ddp_wrapper}'.")
    else:
        if ddp_wrapper != "none":
            print(
                f"ddp_wrapper='{ddp_wrapper}' requested, but CUDA device_count={world_size}; "
                "falling back to single-process training."
            )
        run_result = train_lm(**train_kwargs)

    return {
        "dataset_mode": assets["dataset_mode"],
        "ddp_wrapper": ddp_wrapper,
        "distributed_enabled": distributed_enabled,
        "attention_kernel": train_kwargs["ATTENTION_KERNEL"],
        "result": run_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CUDA-only end-to-end LM pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to pipeline JSON config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE (supports dotted keys).",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print final merged config before executing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text())
    merged = apply_overrides(config, args.override)

    if args.print_resolved_config:
        print(json.dumps(merged, indent=2))

    summary = run_pipeline(merged)
    print("Pipeline run summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
