# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## Experiments Pipeline (CUDA-only)

The repo now includes a config-driven pipeline and benchmark scripts under
`cs336_systems/experiments`.

### 1. End-to-end pipeline

Runs data routing (`tinystories/local_text/local_npy/auto`), tokenizer/data
build (when needed), and LM training with kernel/wrapper selection:

```sh
uv run python cs336_systems/experiments/run_pipeline.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --override training.attention_kernel=\"FlashAttention-2 Triton\" \
  --override training.ddp_wrapper=\"none\" \
  --override training.CHECKPOINTING_EVERY=200
```

Notes:
- `training.CHECKPOINTING_EVERY` overrides `training.SAVE_INTERVAL`.
- `CHECKPOINTING_EVERY <= 0` disables periodic checkpoints and still saves the final checkpoint.
- Distributed training is used only when `ddp_wrapper` is `naive` or `flashddp` and `torch.cuda.device_count() > 1`.

### 2. LM matrix benchmark (4 kernels x 2 wrappers)

Produces local `CSV + PNG + Markdown` artifacts:

```sh
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --output-dir artifacts/lm_matrix
```

### 3. Attention forward-only sweep benchmark

Benchmarks square and rectangular `(Q_N, K_N)` tiers from Small to XXL:

```sh
uv run python cs336_systems/experiments/benchmark_attention_sweep.py \
  --DTYPE float16 \
  --output-dir artifacts/attention_sweep
```

Custom tiers use `NAME:BATCH:HEADS:Q_N:K_N:D` and all dimensions must be powers of 2.




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

1). A training script on the raw plaint LLM, with training, validation loss plot recorded on WanDB. With a final loss.
2). A compare plot to compare the forward pass time of different Attention Kernels.
3). A compare plot to compare the forward pass time of the LLM model with different Attention Kernel.
4). A compare plot on the training time when using plain LLM, then plus different Attention Kernels, then plus different DDP training warpper (with bucketing and overlapping comm and comp).
