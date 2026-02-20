import json
from pathlib import Path

import pytest

from cs336_systems.experiments.run_pipeline import (
    apply_overrides,
    build_training_kwargs,
    resolve_dataset_mode,
)


def test_apply_overrides_takes_precedence_over_config_file_values():
    config = {
        "training": {
            "LR": 3e-4,
            "EPOCHES": 100,
        }
    }
    merged = apply_overrides(config, ["training.LR=0.001", "training.EPOCHES=12"])

    assert merged["training"]["LR"] == 0.001
    assert merged["training"]["EPOCHES"] == 12


def test_dataset_mode_auto_prefers_local_npy_when_all_assets_exist(tmp_path: Path):
    train_npy = tmp_path / "train.npy"
    val_npy = tmp_path / "val.npy"
    vocab = tmp_path / "vocab.pkl"
    merges = tmp_path / "merges.pkl"
    for p in (train_npy, val_npy, vocab, merges):
        p.write_bytes(b"x")

    mode = resolve_dataset_mode(
        "auto",
        {
            "train_npy": str(train_npy),
            "val_npy": str(val_npy),
            "vocab_path": str(vocab),
            "merges_path": str(merges),
            "train_text": str(tmp_path / "train.txt"),
            "val_text": str(tmp_path / "val.txt"),
        },
    )
    assert mode == "local_npy"


def test_dataset_mode_auto_falls_back_to_local_text_then_tinystories(tmp_path: Path):
    train_text = tmp_path / "train.txt"
    val_text = tmp_path / "val.txt"
    train_text.write_text("hello")
    val_text.write_text("world")

    mode_text = resolve_dataset_mode(
        "auto",
        {
            "train_text": str(train_text),
            "val_text": str(val_text),
            "train_npy": str(tmp_path / "missing_train.npy"),
            "val_npy": str(tmp_path / "missing_val.npy"),
            "vocab_path": str(tmp_path / "missing_vocab.pkl"),
            "merges_path": str(tmp_path / "missing_merges.pkl"),
        },
    )
    assert mode_text == "local_text"

    mode_ts = resolve_dataset_mode(
        "auto",
        {
            "train_text": str(tmp_path / "missing_train.txt"),
            "val_text": str(tmp_path / "missing_val.txt"),
            "train_npy": str(tmp_path / "missing_train.npy"),
            "val_npy": str(tmp_path / "missing_val.npy"),
            "vocab_path": str(tmp_path / "missing_vocab.pkl"),
            "merges_path": str(tmp_path / "missing_merges.pkl"),
        },
    )
    assert mode_ts == "tinystories"


def test_kernel_and_wrapper_selection_with_alias_resolution():
    config = {
        "training": {
            "ddp_wrapper": "flashddp",
            "attention_kernel": "MyTriton",
            "WANDB_PROJECT": "unit-test",
            "TR_BAT_SIZE": 2,
        }
    }
    assets = {
        "train_npy": "train.npy",
        "val_npy": "val.npy",
        "vocab_path": "vocab.pkl",
        "merges_path": "merges.pkl",
    }

    kwargs, wrapper = build_training_kwargs(config, assets)

    assert wrapper == "flashddp"
    assert kwargs["ATTENTION_KERNEL"] == "FlashAttention-2 Triton"


def test_build_training_kwargs_rejects_unknown_wrapper():
    config = {
        "training": {
            "ddp_wrapper": "unknown",
            "WANDB_PROJECT": "unit-test",
        }
    }
    assets = {
        "train_npy": "train.npy",
        "val_npy": "val.npy",
        "vocab_path": "vocab.pkl",
        "merges_path": "merges.pkl",
    }

    with pytest.raises(ValueError):
        _ = build_training_kwargs(config, assets)
