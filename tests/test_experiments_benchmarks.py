from pathlib import Path

import pandas as pd
import pytest

from cs336_systems.experiments.benchmark_attention_sweep import (
    parse_tier_spec,
    validate_tiers,
    write_artifacts as write_attention_artifacts,
)
from cs336_systems.experiments.benchmark_lm_matrix import (
    write_artifacts as write_lm_matrix_artifacts,
)


def test_attention_sweep_parse_tier_and_power_of_two_validation():
    tier = parse_tier_spec("Tiny:1:8:128:256:64")
    assert tier["tier"] == "Tiny"
    validate_tiers([tier])

    with pytest.raises(ValueError):
        validate_tiers([parse_tier_spec("Bad:1:3:128:256:64")])


def test_attention_sweep_tier_format_validation():
    with pytest.raises(ValueError):
        _ = parse_tier_spec("bad-format")


def test_attention_benchmark_artifact_schema_and_presence(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "tier": "Small",
                "pattern": "square",
                "kernel": "scaled_dot_prod_attention",
                "batch": 1,
                "heads": 8,
                "q_n": 128,
                "k_n": 128,
                "d": 64,
                "dtype": "float16",
                "is_causal": False,
                "forward_ms": 1.23,
                "status": "success",
                "error": "",
            },
            {
                "tier": "Small",
                "pattern": "rect_qk",
                "kernel": "FlashAttention-2 Triton",
                "batch": 1,
                "heads": 8,
                "q_n": 128,
                "k_n": 256,
                "d": 64,
                "dtype": "float16",
                "is_causal": False,
                "forward_ms": 0.98,
                "status": "success",
                "error": "",
            },
        ]
    )

    artifacts = write_attention_artifacts(df, tmp_path)
    for path in artifacts.values():
        assert path.exists()

    out_df = pd.read_csv(artifacts["csv"])
    assert {
        "tier",
        "pattern",
        "kernel",
        "forward_ms",
        "status",
    }.issubset(set(out_df.columns))


def test_lm_matrix_artifact_schema_and_presence(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "wrapper": "naive",
                "kernel": "scaled_dot_prod_attention",
                "status": "success",
                "world_size": 2,
                "wall_time_s": 10.0,
                "error": "",
            },
            {
                "wrapper": "flashddp",
                "kernel": "FlashAttention-2 Triton",
                "status": "success",
                "world_size": 2,
                "wall_time_s": 5.0,
                "error": "",
            },
        ]
    )

    artifacts = write_lm_matrix_artifacts(df, tmp_path)
    for path in artifacts.values():
        assert path.exists()

    out_df = pd.read_csv(artifacts["csv"])
    assert {
        "wrapper",
        "kernel",
        "wall_time_s",
        "status",
    }.issubset(set(out_df.columns))
