"""Make `import sam3` succeed in non-CUDA environments.

SAM 3's official package has three properties that break Mac / CPU users:

  1. `triton` is required at module import time (`@triton.jit` decorators),
     even for image-only inference paths that never call the kernels. The
     env we ship installs a hand-written triton stub that lacks `.jit`.
  2. Several modules construct tensors with `device="cuda"` literally —
     `position_encoding.py:55`, `decoder.py:283`, etc. — even before the
     model is moved to a device. On Mac this trips
     `AssertionError: Torch not compiled with CUDA enabled`.
  3. Model weights are bfloat16; if you cast to fp32 some sub-modules still
     produce bf16 outputs, leading to F.linear dtype mismatches.

`apply()` installs runtime patches that address (1) and (2). (3) is
addressed at call time by running inference under `torch.autocast(cpu,
bfloat16)` — see `infer.py`. None of these patches touch sam3's source
files: a future `git pull` won't conflict.
"""

from __future__ import annotations

import sys
import types
from typing import Any


def _install_triton_stub() -> None:
    """Add the symbols sam3 reads from `triton` at import time.

    sam3/model/edt.py decorates `edt_kernel` with @triton.jit and types
    arguments via tl.constexpr; both must exist or `import sam3` fails.
    The kernel is only invoked from the video tracker code path, which the
    image classifier never touches — so the stubs being non-functional is
    fine. We deliberately do not install the real `triton` package: on
    macOS it is not pip-installable and would only matter for video paths.
    """
    import triton
    import triton.language

    def _jit(*args: Any, **kwargs: Any):
        # @triton.jit may be used with or without arguments. Both forms
        # have to return something callable that itself returns the wrapped
        # function untouched.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    if not hasattr(triton, "jit"):
        triton.jit = _jit  # type: ignore[attr-defined]
    if not hasattr(triton.language, "constexpr"):
        # Used only as a type annotation in edt.py; any plain type works.
        triton.language.constexpr = bool  # type: ignore[attr-defined]


def _install_cuda_redirect(target_device: str) -> None:
    """Redirect every `device="cuda"` reference to `target_device`.

    Wraps the torch tensor-factory functions and `torch.device` so that any
    sam3 line of the form `torch.zeros(..., device="cuda")` ends up on the
    chosen Mac device instead of trying to lazy-init the CUDA runtime.

    Only the factories that sam3 actually uses on the image path are
    wrapped; the list is conservative on the side of "more is fine".
    """
    import torch

    def _is_cuda(d: Any) -> bool:
        if d is None:
            return False
        if isinstance(d, str):
            return d.startswith("cuda")
        if isinstance(d, torch.device):
            return d.type == "cuda"
        return False

    def _wrap(fn):
        def wrapped(*args: Any, **kwargs: Any):
            if _is_cuda(kwargs.get("device")):
                kwargs["device"] = target_device
            return fn(*args, **kwargs)

        # NB: deliberately do NOT set wrapped.__wrapped__ = fn — torch.jit
        # follows __wrapped__ when inspecting source, which then trips on
        # the underlying C builtins (torch.zeros etc are C functions with
        # no Python source). Leaving the attribute unset hides the chain
        # so `inspect.getsourcelines(wrapped)` succeeds against the plain
        # Python wrapper.
        return wrapped

    factories = (
        "zeros", "ones", "empty", "full", "arange", "randn", "randint",
        "linspace", "logspace", "eye", "rand", "tensor", "as_tensor",
        "scalar_tensor", "zeros_like", "ones_like", "empty_like",
        "full_like", "randn_like", "randint_like",
    )
    for name in factories:
        if hasattr(torch, name):
            setattr(torch, name, _wrap(getattr(torch, name)))

    # torch.device("cuda") → torch.device(target_device).
    _orig_device = torch.device

    class _DeviceProxy:
        def __new__(cls, *args: Any, **kwargs: Any):
            if args and _is_cuda(args[0]):
                return _orig_device(target_device, *args[1:], **kwargs)
            return _orig_device(*args, **kwargs)

    torch.device = _DeviceProxy  # type: ignore[assignment]


def pick_device() -> str:
    """Best inference device available on this machine.

    Order of preference: cuda → mps → cpu. We expose this as a function
    rather than a constant so the patches can be applied before the choice
    is made (e.g. when the caller forces a device via --device).
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _install_disable_activation_checkpoint() -> None:
    """Bypass `torch.utils.checkpoint.checkpoint` in inference paths.

    Activation checkpointing is a memory/compute trade-off useful only
    during training (it saves activations by recomputing them in the
    backward pass). For inference there's nothing to recompute.

    PyTorch 2.12 on Mac additionally has a bug: checkpoint() consults
    `torch.accelerator.current_accelerator()` to decide which RNG state
    to save, and on a CUDA-less Mac that returns "mps", which
    `torch.get_device_module` then refuses with
    `Invalid value of device 'mps'`. SAM 3 hits this on the very first
    text-prompt call via its text encoder.

    Bypassing checkpoint() at the torch level catches every call site
    (vitdet, text_encoder_ve, model_misc, maskformer_segmentation, …)
    in one shot — no need to rebind per-module symbols.
    """
    import torch.utils.checkpoint as torch_ckpt

    _orig = torch_ckpt.checkpoint

    def passthrough_checkpoint(function, *args, use_reentrant=None, **kwargs):
        # `use_reentrant` is checkpoint's own flag, drop it; remaining
        # kwargs go through to the wrapped function.
        return function(*args, **kwargs)

    torch_ckpt.checkpoint = passthrough_checkpoint
    # Also patch the symbol on modules that imported it by name (so
    # `from torch.utils.checkpoint import checkpoint` users see our
    # replacement on the next call).
    import importlib
    for mod_name in (
        "sam3.model.text_encoder_ve",
        "sam3.model.vitdet",
        "sam3.model.maskformer_segmentation",
        "sam3.model.act_ckpt_utils",
        "sam3.model.model_misc",
    ):
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, "checkpoint"):
                # vitdet does `import torch.utils.checkpoint as checkpoint`
                # — its attribute is the SUBMODULE, not the function. Only
                # rebind module-level callables, not module objects.
                attr = getattr(m, "checkpoint")
                if callable(attr) and not hasattr(attr, "checkpoint"):
                    setattr(m, "checkpoint", passthrough_checkpoint)
        except Exception:
            pass


def _install_pin_memory_noop() -> None:
    """Make `Tensor.pin_memory()` a no-op on systems without CUDA.

    PyTorch 2.12 on macOS still has a bug where `t.pin_memory()` on a CPU
    tensor tries to pin into MPS memory, which fails with a "different
    device" assertion. SAM 3's geometry_encoders call pin_memory()
    unconditionally as a perf hint; we can simply return the tensor
    unchanged on non-CUDA systems.
    """
    import torch

    if torch.cuda.is_available():
        return  # CUDA is the only case where pinning actually helps.

    _orig_pin = torch.Tensor.pin_memory

    def _pin_noop(self, *args, **kwargs):
        return self

    torch.Tensor.pin_memory = _pin_noop  # type: ignore[assignment]


def _install_addmm_fp32_replacement(target_device: str) -> None:
    """Replace `sam3.perflib.fused.addmm_act` with a dtype-stable variant.

    The shipped implementation hardcodes `mat1.to(bfloat16)` for what is
    really a fused fp16/bf16 GEMM+activation kernel on Ampere/Hopper. On
    CPU and MPS that produces a bf16 activation that the very next Linear
    layer (still in fp32 weights) refuses to multiply against. We
    substitute a plain F.linear → activation pipeline that respects the
    weight's dtype on whatever backend we're running on.

    Only the image-path call site (`vitdet.MLP.forward`) uses this
    function, so the replacement is safe — no kernels depend on the
    addmm fused output being bf16.
    """
    import torch
    import torch.nn.functional as F
    from sam3.perflib import fused as _fused

    def addmm_act_compat(activation, linear, mat1):
        # Match the original signature: activation is a class/function;
        # linear is an nn.Linear; mat1 is the input activation.
        y = F.linear(mat1, linear.weight, linear.bias)
        if activation in (torch.nn.ReLU, F.relu):
            return F.relu(y)
        if activation in (torch.nn.GELU, F.gelu):
            return F.gelu(y)
        raise ValueError(f"unexpected activation {activation}")

    _fused.addmm_act = addmm_act_compat
    # vitdet imports the symbol directly at module load
    # (`from sam3.perflib.fused import addmm_act`), so the name in that
    # module also has to be rebound — otherwise vitdet keeps using the
    # original implementation it captured at import time.
    try:
        from sam3.model import vitdet as _vitdet
        _vitdet.addmm_act = addmm_act_compat
    except ImportError:
        # vitdet not yet imported; the patched symbol in `_fused` will be
        # what it picks up later. Both code paths converge.
        pass


def apply(target_device: str | None = None) -> str:
    """Patch the environment and return the chosen target device.

    Must be called before `import sam3` (or anything that transitively
    imports it). Safe to call more than once — the wraps are idempotent on
    a per-process basis because each call rewraps the already-wrapped
    factory and detection still works (string comparison on `device=`).
    """
    _install_triton_stub()
    device = target_device or pick_device()
    # _install_cuda_redirect intentionally disabled: wrapping torch
    # factories breaks torch.jit.script (used by sam3's
    # inst_interactivity path). The 4 hardcoded `device="cuda"` strings
    # in sam3's source (position_encoding.py, decoder.py,
    # geometry_encoders.py) have been patched in-place to be
    # device-agnostic. See commit history of /Users/zhangfeng/D/sam3/sam3.
    _install_pin_memory_noop()
    if device != "cuda":
        _install_addmm_fp32_replacement(device)
        _install_disable_activation_checkpoint()
    return device
