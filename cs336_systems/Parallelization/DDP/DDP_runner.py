import argparse
import time

import torch
import torch.multiprocessing as mp

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    attention_kernel_choices,
    resolve_attention_kernel,
)
from cs336_systems.Parallelization.DDP.naiveDDP import naive_LLM_DDP
from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TransformerLM with naive DDP gradient all-reduce"
    )

    # Data and Training Hyperparameters
    parser.add_argument("--EPOCHES", type=int, default=10)
    parser.add_argument("--WARMUP_EPOCHS", type=int, default=5)
    parser.add_argument("--EVAL_INTERVAL", type=int, default=5)
    parser.add_argument("--TR_BAT_SIZE", type=int, default=8)
    parser.add_argument("--VAL_BAT_SIZE", type=int, default=8)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--VAL_PATH", type=str, required=True)
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))

    # Model Hyperparameters
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=256)
    parser.add_argument("--PRINT_EVERY", type=int, default=1)
    parser.add_argument("--VOCAB_SIZE", type=int, default=10000)
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0)
    parser.add_argument("--NUM_LAYERS", type=int, default=16)
    parser.add_argument("--D_MODEL", type=int, default=256)
    parser.add_argument("--NUM_HEADS", type=int, default=8)
    parser.add_argument("--D_FF", type=int, default=256 * 4)
    parser.add_argument(
        "--ATTN_KERNEL",
        type=str,
        default="scaled_dot_prod_attention",
        choices=attention_kernel_choices(include_aliases=True),
    )

    # Optimizer Hyperparameters
    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--BETA1", type=float, default=0.9)
    parser.add_argument("--BETA2", type=float, default=0.999)
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8)
    return parser


def run_naive_ddp_training(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training.")

    dtype = DTYPE_DICT[args.DTYPE]
    canonical_kernel_name, attention_fn = resolve_attention_kernel(args.ATTN_KERNEL)
    backend = "nccl"
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices detected.")

    model_kwargs = dict(
        vocab_size=args.VOCAB_SIZE,
        context_length=args.CONTEXT_LENGTH,
        num_layers=args.NUM_LAYERS,
        d_model=args.D_MODEL,
        heads_num=args.NUM_HEADS,
        d_ff=args.D_FF,
        theta=args.ROPE_THETA,
        device="cpu",
        dtype=dtype,
        attention_fn=attention_fn,
    )
    model = TransformerLM(**model_kwargs)

    optim_kwargs = dict(
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY,
        betas=(args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    val_dataset = TokenStreamDataset(args.VAL_PATH, args.CONTEXT_LENGTH)
    tr_dataset = TokenStreamDataset(args.TRAIN_PATH, args.CONTEXT_LENGTH)

    t0 = time.perf_counter()
    mp.spawn(
        fn=naive_LLM_DDP,
        args=(
            world_size,
            tr_dataset,
            val_dataset,
            model,
            optim_kwargs,
            cross_entropy,
            args.EPOCHES,
            args.EVAL_INTERVAL,
            args.TR_BAT_SIZE,
            args.VAL_BAT_SIZE,
            backend,
            args.PRINT_EVERY,
            args.WARMUP_EPOCHS,
        ),
        nprocs=world_size,
        join=True,
    )
    wall_time_s = time.perf_counter() - t0

    return {
        "wrapper": "naive",
        "attention_kernel": canonical_kernel_name,
        "world_size": world_size,
        "wall_time_s": wall_time_s,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_naive_ddp_training(args)
    print(f"DDP run summary: {result}")


if __name__ == "__main__":
    main()
