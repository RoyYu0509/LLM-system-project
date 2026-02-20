from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNELS

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


DEFAULT_TIERS = [
    {"tier": "Small", "batch": 1, "heads": 8, "q_n": 128, "k_n": 256, "d": 64},
    {"tier": "Medium", "batch": 1, "heads": 8, "q_n": 256, "k_n": 512, "d": 64},
    {"tier": "Large", "batch": 1, "heads": 8, "q_n": 512, "k_n": 1024, "d": 64},
    {"tier": "XL", "batch": 1, "heads": 8, "q_n": 1024, "k_n": 2048, "d": 64},
    {"tier": "XXL", "batch": 1, "heads": 4, "q_n": 2048, "k_n": 4096, "d": 64},
]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def parse_tier_spec(spec: str) -> dict[str, Any]:
    """Parse custom tier spec format: NAME:BATCH:HEADS:Q_N:K_N:D."""
    parts = spec.split(":")
    if len(parts) != 6:
        raise ValueError(
            f"Invalid --tier '{spec}'. Expected NAME:BATCH:HEADS:Q_N:K_N:D"
        )
    name, batch, heads, q_n, k_n, d = parts
    return {
        "tier": name,
        "batch": int(batch),
        "heads": int(heads),
        "q_n": int(q_n),
        "k_n": int(k_n),
        "d": int(d),
    }


def validate_tiers(tiers: list[dict[str, Any]]) -> None:
    for tier in tiers:
        for key in ("batch", "heads", "q_n", "k_n", "d"):
            value = int(tier[key])
            if not is_power_of_two(value):
                raise ValueError(
                    f"Tier '{tier['tier']}' has non power-of-two {key}={value}."
                )


def build_patterns_for_tier(tier: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "pattern": "square",
            "q_n": int(tier["q_n"]),
            "k_n": int(tier["q_n"]),
        },
        {
            "pattern": "rect_qk",
            "q_n": int(tier["q_n"]),
            "k_n": int(tier["k_n"]),
        },
        {
            "pattern": "rect_kq",
            "q_n": int(tier["k_n"]),
            "k_n": int(tier["q_n"]),
        },
    ]


def benchmark_forward_only(
    attention_fn,
    batch: int,
    heads: int,
    q_n: int,
    k_n: int,
    d: int,
    dtype: torch.dtype,
    warmup_iters: int,
    bench_iters: int,
    is_causal: bool,
) -> float:
    shape_q = (batch * heads, q_n, d)
    shape_kv = (batch * heads, k_n, d)

    q = torch.randn(shape_q, device="cuda", dtype=dtype)
    k = torch.randn(shape_kv, device="cuda", dtype=dtype)
    v = torch.randn(shape_kv, device="cuda", dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = attention_fn(q, k, v, is_causal)
        torch.cuda.synchronize()

        times_ms = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(bench_iters):
            start.record()
            _ = attention_fn(q, k, v, is_causal)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    del q, k, v
    torch.cuda.empty_cache()
    gc.collect()

    return float(sum(times_ms) / len(times_ms))


def _rows_to_markdown(rows: list[dict[str, Any]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep, *body])


def write_artifacts(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "attention_sweep_results.csv"
    png_path = output_dir / "attention_sweep_plot.png"
    md_path = output_dir / "attention_sweep_summary.md"

    df.to_csv(csv_path, index=False)

    success_df = df[df["status"] == "success"].copy()
    if not success_df.empty:
        tier_order = list(dict.fromkeys(success_df["tier"].tolist()))
        patterns = ["square", "rect_qk", "rect_kq"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, pattern in zip(axes, patterns):
            sub = success_df[success_df["pattern"] == pattern]
            pivot = sub.pivot(index="tier", columns="kernel", values="forward_ms")
            pivot = pivot.reindex(tier_order)
            for kernel in pivot.columns:
                ax.plot(pivot.index, pivot[kernel], marker="o", label=kernel)
            ax.set_title(pattern)
            ax.set_xlabel("Tier")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel("Forward Time (ms, log scale)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    summary_rows = []
    for (tier, pattern), group in success_df.groupby(["tier", "pattern"]):
        fastest = group.loc[group["forward_ms"].idxmin()]
        summary_rows.append(
            {
                "tier": tier,
                "pattern": pattern,
                "fastest_kernel": fastest["kernel"],
                "forward_ms": round(float(fastest["forward_ms"]), 4),
            }
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Attention Sweep Benchmark\n\n")
        f.write(f"Rows: {len(df)}\n\n")
        if summary_rows:
            f.write("## Fastest Kernel By Tier/Pattern\n\n")
            f.write(_rows_to_markdown(summary_rows, ["tier", "pattern", "fastest_kernel", "forward_ms"]))
            f.write("\n")
        else:
            f.write("No successful benchmark rows were produced.\n")

    return {
        "csv": csv_path,
        "png": png_path,
        "md": md_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUDA attention forward-only sweep benchmark")
    parser.add_argument("--DTYPE", type=str, default="float16", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--WARMUP_ITERS", type=int, default=20)
    parser.add_argument("--BENCH_ITERS", type=int, default=100)
    parser.add_argument("--IS_CAUSAL", action="store_true")
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        help="Custom tier in format NAME:BATCH:HEADS:Q_N:K_N:D",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/attention_sweep"),
        help="Directory for CSV/PNG/Markdown outputs.",
    )
    return parser


def run_attention_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path]]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_attention_sweep.py is CUDA-only.")

    tiers = [parse_tier_spec(spec) for spec in args.tier] if args.tier else list(DEFAULT_TIERS)
    validate_tiers(tiers)

    dtype = DTYPE_DICT[args.DTYPE]
    rows = []

    for tier in tiers:
        tier_patterns = build_patterns_for_tier(tier)
        for pattern in tier_patterns:
            for kernel_name, kernel_fn in ATTENTION_KERNELS.items():
                row = {
                    "tier": tier["tier"],
                    "pattern": pattern["pattern"],
                    "kernel": kernel_name,
                    "batch": tier["batch"],
                    "heads": tier["heads"],
                    "q_n": pattern["q_n"],
                    "k_n": pattern["k_n"],
                    "d": tier["d"],
                    "dtype": args.DTYPE,
                    "is_causal": bool(args.IS_CAUSAL),
                }
                try:
                    ms = benchmark_forward_only(
                        attention_fn=kernel_fn,
                        batch=tier["batch"],
                        heads=tier["heads"],
                        q_n=pattern["q_n"],
                        k_n=pattern["k_n"],
                        d=tier["d"],
                        dtype=dtype,
                        warmup_iters=args.WARMUP_ITERS,
                        bench_iters=args.BENCH_ITERS,
                        is_causal=bool(args.IS_CAUSAL),
                    )
                    row["forward_ms"] = ms
                    row["status"] = "success"
                    row["error"] = ""
                except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                    row["forward_ms"] = float("nan")
                    row["status"] = "failed"
                    row["error"] = str(exc)
                rows.append(row)

    df = pd.DataFrame(rows)
    artifacts = write_artifacts(df, args.output_dir)
    return df, artifacts


def main() -> None:
    args = build_parser().parse_args()
    df, artifacts = run_attention_sweep(args)

    print("Attention sweep complete")
    print(df)
    print("Artifacts:")
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
