import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def softmax(input: torch.Tensor, axis: int = -1):
    """
    Compute softmax along a specified axis, with numerical stability.

    Parameters:
        input: Tensor of any shape
        axis: Dimension along which to apply softmax
    """
    # Subtract max for numerical stability
    max_val, _ = torch.max(input, dim=axis, keepdim=True) # Keep dim for Broadcasting
    input_stable = input - max_val

    # Compute exp and normalize
    exp_x = torch.exp(input_stable)
    sum_exp = torch.sum(exp_x, dim=axis, keepdim=True)
    softmax_output = exp_x / sum_exp

    return softmax_output


@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(query, key, value, is_causal: bool = False):
    """
    Return an output with the shape (batch_size, seq_len, d_model)

    Accepts 3D packed format:
    key:        (batch, seq_k, d_model)
    query:      (batch, seq_q, d_model)
    value:      (batch, seq_v, d_model)
    is_causal:  bool

    Return:
        - attention: Float[Tensor, "batch seq_q d_model"]
    """
    # Compute Normalized QtK & Apply causal mask
    with nvtx.range("Compute QK^t"):
        norm_qk: Float[torch.Tensor, "... seq_q, seq_k"]
        norm_qk = einsum(key, query, "... seq_k d_k, ... seq_q d_k -> ... seq_q seq_k") / torch.sqrt(torch.tensor(key.shape[-1], device=query.device))
        # Triangular Causal Mask
        if is_causal:
            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_q, seq_k, device=query.device, dtype=torch.bool))
            norm_qk = norm_qk.masked_fill(~causal_mask, torch.finfo(norm_qk.dtype).min)
    
    # Softmax
    with nvtx.range("Computing softmax"):
        softmax_qk: Float[torch.Tensor, "... seq_q seq_k"]
        softmax_qk = softmax(norm_qk, axis=-1)

    # Attention
    with nvtx.range("Multiplied V"):
        attention: Float[Tensor, "... seq_q d_v"]
        attention = einsum(softmax_qk, value, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

    return attention


####################################################################
# Add Different FlashAttention Kernels 
####################################################################

from cs336_systems.FlashAttention.flash_attention_torch_vectorized import vectorized_attn_torch_fn


@nvtx.range("Vectorized-Attention-Torch")
def vectorized_attention_torch(query, key, value, is_causal: bool = False):
    return vectorized_attn_torch_fn(query, key, value, is_causal)  # Positional args only


try:
    from cs336_systems.FlashAttention.flash_attention_triton import flash_attn_triton_fn
    _TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local Triton install
    flash_attn_triton_fn = None
    _TRITON_IMPORT_ERROR = exc


@nvtx.range("FlashAttention-MyTriton")
def flash_attention_my_triton(query, key, value, is_causal: bool = False):
    if flash_attn_triton_fn is None:
        raise RuntimeError(
            "FlashAttention-2 Triton kernel is unavailable in this environment."
        ) from _TRITON_IMPORT_ERROR
    return flash_attn_triton_fn(query, key, value, is_causal)  # Positional args only


from cs336_systems.FlashAttention.flash_attention_torch_naive import flash_attn_torch_fn


@nvtx.range("FlashAttention-PyTorch")
def flash_attention_pytorch(query, key, value, is_causal: bool = False):
    return flash_attn_torch_fn(query, key, value, is_causal)


ATTENTION_KERNELS = {
    "scaled_dot_prod_attention": scaled_dot_product_attention,
    "FlashAttention-2 PyTorch": flash_attention_pytorch,
    "Vectorized Torch": vectorized_attention_torch,
    "FlashAttention-2 Triton": flash_attention_my_triton,
}


# Keep backward compatibility with existing scripts.
ATTENTION_KERNEL_ALIASES = {
    "naive attention": "scaled_dot_prod_attention",
    "scaled_dot_product_attention": "scaled_dot_prod_attention",
    "comptorch": "Vectorized Torch",
    "mytriton": "FlashAttention-2 Triton",
    "flashattention-2 pytorch": "FlashAttention-2 PyTorch",
    "flashattention-2 triton": "FlashAttention-2 Triton",
    "vectortorch": "Vectorized Torch",
    "vectorized_torch": "Vectorized Torch",
    "vectorized torch": "Vectorized Torch",
}


def resolve_attention_kernel(kernel_name: str):
    """Return (canonical_kernel_name, callable_attention_fn)."""
    if kernel_name in ATTENTION_KERNELS:
        return kernel_name, ATTENTION_KERNELS[kernel_name]

    normalized = kernel_name.strip().lower()
    canonical = ATTENTION_KERNEL_ALIASES.get(normalized)
    if canonical is None:
        supported = sorted(
            list(ATTENTION_KERNELS.keys())
            + ["Naive Attention", "CompTorch", "MyTriton"]
        )
        raise ValueError(
            f"Unknown attention kernel '{kernel_name}'. Supported values: {supported}"
        )
    return canonical, ATTENTION_KERNELS[canonical]


def attention_kernel_choices(include_aliases: bool = True) -> list[str]:
    choices = list(ATTENTION_KERNELS.keys())
    if include_aliases:
        choices += ["Naive Attention", "CompTorch", "MyTriton"]
    return choices
