from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNELS, resolve_attention_kernel
from cs336_systems.Parallelization.DDP.DDP_runner import run_naive_ddp_training
from cs336_systems.Parallelization.FlashDDP.FlashDDP_runner import run_flashddp_training
from cs336_systems.experiments.run_pipeline import apply_overrides, default_config_path, prepare_dataset_assets

WRAPPERS = ["naive", "flashddp"]


def _rows_to_markdown(rows: list[dict[str, Any]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep, *body])


def _training_value(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, cfg.get(key.lower(), default))


def _to_ddp_namespace(
    training_cfg: dict[str, Any],
    assets: dict[str, str],
    attention_kernel: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        EPOCHES=int(_training_value(training_cfg, "EPOCHES", 5)),
        WARMUP_EPOCHS=int(_training_value(training_cfg, "WARMUP_EPOCHS", 2)),
        EVAL_INTERVAL=int(_training_value(training_cfg, "EVAL_INTERVAL", 100)),
        TR_BAT_SIZE=int(_training_value(training_cfg, "TR_BAT_SIZE", 8)),
        VAL_BAT_SIZE=int(_training_value(training_cfg, "VAL_BAT_SIZE", 8)),
        TRAIN_PATH=assets["train_npy"],
        VAL_PATH=assets["val_npy"],
        DTYPE=str(_training_value(training_cfg, "DTYPE", "float32")),
        CONTEXT_LENGTH=int(_training_value(training_cfg, "CONTEXT_LENGTH", 256)),
        PRINT_EVERY=int(_training_value(training_cfg, "PRINT_EVERY", 1)),
        VOCAB_SIZE=int(_training_value(training_cfg, "VOCAB_SIZE", 10_000)),
        ROPE_THETA=float(_training_value(training_cfg, "ROPE_THETA", 10_000.0)),
        NUM_LAYERS=int(_training_value(training_cfg, "NUM_LAYERS", 16)),
        D_MODEL=int(_training_value(training_cfg, "D_MODEL", 256)),
        NUM_HEADS=int(_training_value(training_cfg, "NUM_HEADS", 8)),
        D_FF=int(_training_value(training_cfg, "D_FF", 1024)),
        ATTN_KERNEL=attention_kernel,
        LR=float(_training_value(training_cfg, "LR", 3e-4)),
        WEIGHT_DECAY=float(_training_value(training_cfg, "WEIGHT_DECAY", 0.01)),
        BETA1=float(_training_value(training_cfg, "BETA1", 0.9)),
        BETA2=float(_training_value(training_cfg, "BETA2", 0.999)),
        ADAM_EPS=float(_training_value(training_cfg, "ADAM_EPS", 1e-8)),
        BUCKET_SIZE_MB=int(_training_value(training_cfg, "BUCKET_SIZE_MB", 1)),
    )


def write_artifacts(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lm_matrix_results.csv"
    png_path = output_dir / "lm_matrix_plot.png"
    md_path = output_dir / "lm_matrix_summary.md"

    df.to_csv(csv_path, index=False)

    success_df = df[df["status"] == "success"].copy()
    if not success_df.empty:
        kernel_order = list(ATTENTION_KERNELS.keys())
        wrapper_order = WRAPPERS
        pivot = success_df.pivot(index="kernel", columns="wrapper", values="wall_time_s")
        pivot = pivot.reindex(kernel_order)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = range(len(kernel_order))
        bar_w = 0.35
        for idx, wrapper in enumerate(wrapper_order):
            values = [pivot.loc[k, wrapper] if wrapper in pivot.columns else float("nan") for k in kernel_order]
            offsets = [v + (idx - 0.5) * bar_w for v in x]
            axes[0].bar(offsets, values, width=bar_w, label=wrapper)

        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(kernel_order, rotation=20, ha="right")
        axes[0].set_ylabel("Wall Time (s)")
        axes[0].set_yscale("log")
        axes[0].set_title("LM Training Runtime by Kernel/Wrapper")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        baseline = success_df[
            (success_df["wrapper"] == "naive")
            & (success_df["kernel"] == "scaled_dot_prod_attention")
        ]
        if not baseline.empty:
            baseline_t = float(baseline.iloc[0]["wall_time_s"])
            speedup_rows = []
            for _, row in success_df.iterrows():
                speedup_rows.append(
                    {
                        "label": f"{row['wrapper']} | {row['kernel']}",
                        "speedup": baseline_t / float(row["wall_time_s"]),
                    }
                )
            speedup_df = pd.DataFrame(speedup_rows).sort_values("speedup", ascending=False)
            axes[1].barh(speedup_df["label"], speedup_df["speedup"])
            axes[1].set_title("Speedup vs naive + scaled_dot_prod_attention")
            axes[1].set_xlabel("Speedup (x)")
            axes[1].grid(True, axis="x", alpha=0.25)
        else:
            axes[1].text(0.05, 0.5, "Baseline row missing", transform=axes[1].transAxes)
            axes[1].set_axis_off()

        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    summary_rows = []
    for _, row in success_df.sort_values("wall_time_s").head(5).iterrows():
        summary_rows.append(
            {
                "wrapper": row["wrapper"],
                "kernel": row["kernel"],
                "wall_time_s": round(float(row["wall_time_s"]), 4),
                "world_size": int(row["world_size"]),
            }
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# LM Matrix Benchmark\n\n")
        f.write(f"Rows: {len(df)}\n\n")
        if summary_rows:
            f.write("## Top 5 Fastest Runs\n\n")
            f.write(_rows_to_markdown(summary_rows, ["wrapper", "kernel", "wall_time_s", "world_size"]))
            f.write("\n")
        else:
            f.write("No successful benchmark rows were produced.\n")

    return {
        "csv": csv_path,
        "png": png_path,
        "md": md_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LM training matrix (4 kernels x 2 wrappers)")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE (supports dotted keys).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lm_matrix"),
        help="Directory for CSV/PNG/Markdown outputs.",
    )
    return parser


def run_lm_matrix_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path]]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_lm_matrix.py is CUDA-only.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("benchmark_lm_matrix.py requires at least 2 CUDA devices.")

    base_config = json.loads(args.config.read_text())
    config = apply_overrides(base_config, args.override)
    assets = prepare_dataset_assets(config)
    training_cfg = dict(config.get("training", {}))

    rows = []
    for kernel_name in ATTENTION_KERNELS.keys():
        canonical_kernel, _ = resolve_attention_kernel(kernel_name)
        for wrapper in WRAPPERS:
            row = {
                "wrapper": wrapper,
                "kernel": canonical_kernel,
                "status": "failed",
                "world_size": torch.cuda.device_count(),
                "wall_time_s": float("nan"),
                "error": "",
            }
            ddp_args = _to_ddp_namespace(training_cfg, assets, canonical_kernel)
            try:
                if wrapper == "naive":
                    result = run_naive_ddp_training(ddp_args)
                else:
                    result = run_flashddp_training(ddp_args)
                row["wall_time_s"] = float(result["wall_time_s"])
                row["world_size"] = int(result["world_size"])
                row["status"] = "success"
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                row["error"] = str(exc)
            rows.append(row)

    df = pd.DataFrame(rows)
    artifacts = write_artifacts(df, args.output_dir)
    return df, artifacts


def main() -> None:
    args = build_parser().parse_args()
    df, artifacts = run_lm_matrix_benchmark(args)

    print("LM matrix benchmark complete")
    print(df)
    print("Artifacts:")
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
