
# Implementation Plan: CUDA-Only Pipeline + Benchmarks + `checkpointing_every`
## Repo Overview
CS336_Basics — core LM + training primitives
- BPE/tokenizer training pipeline
- Built a Transformer LM module (TransformerLM) that wires together embedding, multiple transformer blocks, final norm, and LM head projection.
- Implemented a pre-norm Transformer block
- Implemented attention stack pieces
- Implemented core training

CS336_Systems — FlashAttention2 + bucketed/overlapped DDP
- FlashAttention-2 (PyTorch) implementation exposed via an autograd Function adapter path.
- FlashAttention-2 (Triton) implementation (kernel code + autotune configs present).
- Vectorized Torch attention implementation
- Naïve DDP (all-reduce per-parameter gradients).
- Naïve Bucketed DDP (Bucketed all parameters and send them at once).
- Bucketed + overlapped DDP container class: you have a DDPOverlapBucketed implementation that buckets params, registers per-param grad hooks, and async all-reduces buckets.

------------

Above is a list of the things I have implemented in the current project repo. Now I want to conclude this project by generate a clear visualization to show what i have done so far, ie (show the optimizations). Some of them I have already drafted, but I need you to reimplement in a clearer way and produce good visualization.

## Summary
Implement the approved plan directly with concise, script-first code, and add explicit `checkpointing_every` checkpoint cadence support in the training path.  
No new complex modules; only focused scripts and small helper functions.

## Files To Add
1. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/experiments/run_pipeline.py`
2. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/experiments/benchmark_lm_matrix.py`
3. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/experiments/benchmark_attention_sweep.py`
4. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/experiments/default_pipeline_config.json`

## Files To Modify
1. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336-basics/cs336_basics/lm_trainer.py`
2. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336-basics/cs336_basics/trainer.py`
3. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336-basics/cs336_basics/transfromer/scaled_dot_prod_attention.py`
4. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/Parallelization/DDP/DDP_runner.py`
5. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336_systems/Parallelization/FlashDDP/FlashDDP_runner.py`
6. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/README.md`
7. `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336-basics/run.md`

## Public Interfaces
1. `run_pipeline.py` uses a training config file, supports `dataset_mode` (`tinystories/local_text/local_npy/auto`), `attention_kernel` {scaled_dot_prod_attention, FlashAttention-2 PyTorch, Vectorized Torch, FlashAttention-2 Triton}, and `ddp_wrapper` (`none/naive/flashddp`).
2. CUDA-only execution for all new scripts.
3. Distributed run only when trainnig config specified `ddp_wrapper` and `torch.cuda.device_count() > 1`.
4. Benchmark outputs are local CSV + PNG + Markdown only.
5. To sum up, the new script run the entire pipeline (downloading the TinyStory Dataset, tokenizer training, dataset building, and LM training) with different options to switch between different Attention Kernels and DDP training wrappers.

## `checkpointing_every` Specification
1. Add `checkpointing_every` as first-class training hyperparameter in config and CLI.
2. In `/Users/UTMC_PC23/Desktop/assignment2-systems-main/cs336-basics/cs336_basics/lm_trainer.py`, add `CHECKPOINTING_EVERY` argument and use it as the checkpoint cadence.
3. Keep backward compatibility with existing `SAVE_INTERVAL`.
4. Resolution rule: if `CHECKPOINTING_EVERY` is set, it overrides `SAVE_INTERVAL`.
5. Always save final checkpoint at last iteration regardless of cadence.
6. `checkpointing_every <= 0` means disable periodic checkpoints and only save final checkpoint.

## Benchmark Scope
1. `benchmark_lm_matrix.py`: full LM training benchmark for 4 kernels × 2 wrappers (`naive`, `flashddp`).
2. `benchmark_attention_sweep.py`: pure attention forward-only benchmark from Small to XXL.
3. Attention sweep includes both square and rectangular `(Q_N, K_N)` patterns.
4. All tier dimensions must be powers of 2; invalid custom tiers fail fast.

## Test Cases
1. `checkpointing_every` cadence test verifies checkpoint save steps and final save behavior.
2. Backward compatibility test for `SAVE_INTERVAL`.
3. Config override precedence test (`--override` over config file).
4. Dataset routing tests for `local_text`, `local_npy`, and `tinystories`.
5. Kernel/wrapper selection tests.
6. Attention sweep power-of-2 validation tests.
7. Benchmark output schema tests for CSV and artifact presence.

## Assumptions
1. No CPU/MPS fallback paths are required.
2. WandB remains required for pipeline training runs.
3. Benchmarks run locally and produce visualization artifacts under an artifacts directory.