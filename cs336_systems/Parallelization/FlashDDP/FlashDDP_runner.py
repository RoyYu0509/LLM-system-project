import argparse
import os
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import tqdm

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    attention_kernel_choices,
    resolve_attention_kernel,
)
from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset
from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def set_dist_env(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parallel_train(
    rank: int,
    parallel_wrapper: torch.nn.Module,
    world_size: int,
    tr_dataset: Dataset,
    val_dataset: Dataset,
    model: torch.nn.Module,
    optimizer_args: dict,
    loss_fn: Callable,
    epochs: int,
    eval_interval: int,
    tr_batch_size: int,
    val_batch_size: int,
    backend: str,
    print_every=None,
    time_warmup_ep=None,
    bucket_size_mb=1,
):
    set_dist_env(rank, world_size, backend=backend)

    if backend != "nccl":
        raise NotImplementedError("This function is only implemented for GPU training with NCCL backend.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tr_sampler = DistributedSampler(
        dataset=tr_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=tr_batch_size,
        sampler=tr_sampler,
        shuffle=False,
        num_workers=2,
    )

    val_sampler = DistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
    )

    model = model.to(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.train()
    model = parallel_wrapper(model, bucket_size_mb=bucket_size_mb)

    optimizer = AdamW(model.parameters(), **optimizer_args)

    total_avg_epoch_time = 0.0
    counted_epochs = 0

    for epoch in range(epochs):
        tr_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        acc_loss = 0.0

        _synchronize_if_cuda(device)
        t0 = time.perf_counter()

        for i, (bat_x, bat_y_ref) in enumerate(
            tqdm.tqdm(tr_loader, desc=f"Rank {rank} Epoch {epoch + 1}", unit="batch")
        ):
            bat_x = bat_x.to(device, non_blocking=True)
            bat_y_ref = bat_y_ref.to(device, non_blocking=True)

            # Preserve param.grad -> grad_buffer views used by DDPOverlapBucketed.
            optimizer.zero_grad(set_to_none=False)
            bat_y_pred = model(bat_x)
            loss = loss_fn(bat_y_pred, bat_y_ref)
            loss.backward()

            model.finish_gradient_synchromnization()

            optimizer.step()
            acc_loss += loss.item()

            if print_every is not None and (i + 1) % print_every == 0 and rank == 0:
                print(f"Epoch {epoch + 1} batch {i + 1} loss={loss.item():.4f}")

            if i + 1 == 100:
                break

        _synchronize_if_cuda(device)
        epoch_time = time.perf_counter() - t0

        if time_warmup_ep is None or epoch >= time_warmup_ep:
            timing = torch.tensor([epoch_time], device=device)
            dist.all_reduce(timing, op=dist.ReduceOp.SUM)
            timing /= world_size
            avg_epoch_time = timing[0].item()
            total_avg_epoch_time += avg_epoch_time
            counted_epochs += 1

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_bat_x, val_bat_y_ref in tqdm.tqdm(
                    val_loader,
                    desc=f"Rank {rank} Evaluating Epoch {epoch + 1}",
                    unit="batch",
                ):
                    val_bat_x = val_bat_x.to(device, non_blocking=True)
                    val_bat_y_ref = val_bat_y_ref.to(device, non_blocking=True)
                    val_bat_y_pred = model(val_bat_x)
                    val_loss += loss_fn(val_bat_y_pred, val_bat_y_ref).item()

            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)

            if rank == 0:
                print(
                    f"Epoch {epoch + 1:04d} | epoch_time={epoch_time:.4f}s | "
                    f"loss_sum={acc_loss:.4f} | val_loss={avg_val_loss:.4f}"
                )
            model.train()

    if rank == 0 and counted_epochs > 0:
        print("\n=== FlashDDP Timing Summary ===")
        print(f"Average total time per profiled epoch: {total_avg_epoch_time / counted_epochs:.4f}s")

    dist.barrier()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TransformerLM with overlapped bucketed DDP"
    )

    # Data and Training Hyperparameters
    parser.add_argument("--EPOCHES", type=int, default=5)
    parser.add_argument("--WARMUP_EPOCHS", type=int, default=2)
    parser.add_argument("--EVAL_INTERVAL", type=int, default=100)
    parser.add_argument("--TR_BAT_SIZE", type=int, default=8)
    parser.add_argument("--VAL_BAT_SIZE", type=int, default=8)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--VAL_PATH", type=str, required=True)
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--BUCKET_SIZE_MB", type=int, default=1)

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


def run_flashddp_training(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

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
        fn=parallel_train,
        args=(
            DDPOverlapBucketed,
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
            args.BUCKET_SIZE_MB,
        ),
        nprocs=world_size,
        join=True,
    )
    wall_time_s = time.perf_counter() - t0

    return {
        "wrapper": "flashddp",
        "attention_kernel": canonical_kernel_name,
        "world_size": world_size,
        "wall_time_s": wall_time_s,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_flashddp_training(args)
    print(f"FlashDDP run summary: {result}")


if __name__ == "__main__":
    main()
