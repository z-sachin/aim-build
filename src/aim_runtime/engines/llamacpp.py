# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""llama.cpp engine (``llama-server``, ZenDNN backend) for EPYC.

``llama-server -m`` needs one ``.gguf`` file. The cached model directory holds
either GGUF files or a Hugging Face checkpoint, which is converted to GGUF once
on first launch.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from aim_common.object_model import Precision
from aim_runtime.cpu_detector import CpuDetector
from aim_runtime.engines.base import BaseEngine
from aim_runtime.engines.engine_args_models import EngineArgsFormat, EngineArgsModel

if TYPE_CHECKING:
    from aim_runtime.model_cache_resolver import ResolvedModelPath
    from aim_runtime.object_model import Profile

logger = logging.getLogger(__name__)

# From the same llama.cpp commit as llama-server (docker/Dockerfile.aim-epyc-base).
CONVERTER_PATH = "/opt/llamacpp/convert/convert_hf_to_gguf.py"

# GGUF type the converter writes for each profile precision.
GGUF_OUTTYPE_BY_PRECISION: dict[Precision, str] = {
    Precision.BF16: "bf16",
    Precision.FP16: "f16",
    Precision.FP32: "f32",
    Precision.INT8: "q8_0",
}

KvCacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]

# Bytes per element for each KV cache type (block-quantized types: block bytes / 32).
KV_CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "q4_1": 20 / 32,
    "iq4_nl": 18 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
}

# llama.cpp pads the context to a multiple of this.
CTX_SIZE_ALIGNMENT = 256


class LlamaCppEngineArgsModel(EngineArgsModel):
    """Pydantic model for ``llama-server`` CLI arguments.

    Pydantic-only validation (no Python CLI parser to delegate to). Fields
    mirror the llama.cpp commit pinned in ``docker/Dockerfile.aim-epyc-base``;
    unlisted flags pass through.
    """

    # Serving
    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    timeout: int | None = None
    metrics: bool | None = None
    embedding: bool | None = None
    reranking: bool | None = None
    alias: str | None = None
    jinja: bool | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None

    # AIM-only, not llama-server flags: the .gguf file to load, and the KV cache
    # budget in GiB (as VLLM_CPU_KVCACHE_SPACE) that ctx-size is derived from.
    gguf_filename: str | None = None
    kv_cache_space: float | None = None

    # Context / batching
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    n_predict: int | None = None
    parallel: int | None = None
    cont_batching: bool | None = None
    kv_unified: bool | None = None
    cache_reuse: int | None = None
    cache_type_k: KvCacheType | None = None
    cache_type_v: KvCacheType | None = None
    flash_attn: Literal["on", "off", "auto"] | None = None

    # CPU / memory placement
    threads: int | None = None
    threads_batch: int | None = None
    cpu_mask: str | None = None
    cpu_range: str | None = None
    cpu_strict: Literal[0, 1] | None = None
    numa: Literal["distribute", "isolate", "numactl"] | None = None
    load_mode: Literal["auto", "none", "mmap", "mlock", "mmap+mlock"] | None = None

    # RoPE
    rope_scaling: Literal["none", "linear", "yarn"] | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    # LoRA
    lora: str | None = None


class LlamaCppEngine(BaseEngine):
    """llama.cpp-backed serving engine (``llama-server``)."""

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = LlamaCppEngineArgsModel
    ARGS_FORMAT: ClassVar[EngineArgsFormat] = EngineArgsFormat.STANDARD

    def apply_engine_defaults(self, engine_args: dict[str, Any], served_model_names: list[str]) -> None:
        """One thread per available core, pinned when the cpuset is known (as for vLLM).

        Otherwise llama-server sizes its pool from the host's core count. Explicit
        values from the profile or ``AIM_ENGINE_ARGS`` win.
        """
        detector = CpuDetector()
        if "threads" not in engine_args:
            engine_args["threads"] = detector.available_cores

        pinned = any(key in engine_args for key in ("cpu-mask", "cpu_mask", "cpu-range", "cpu_range"))
        if not pinned and detector.cpuset_bind:
            cpus = CpuDetector._parse_cpuset_to_set(detector.cpuset_bind)
            engine_args["cpu-mask"] = format(sum(1 << cpu for cpu in cpus), "x")
            if "cpu-strict" not in engine_args and "cpu_strict" not in engine_args:
                engine_args["cpu-strict"] = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # (checkpoint dir, GGUF outtype, target file) still to be converted.
        self._pending_conversion: tuple[str, str, str] | None = None

    def resolve_model_path(self, resolved: "ResolvedModelPath", engine_args: dict[str, Any], profile: "Profile") -> str:
        """Resolve the cached model directory to the ``.gguf`` file to load.

        In order: ``gguf_filename`` (popped from ``engine_args``); for a Hugging
        Face checkpoint, ``<model>-<outtype>.gguf`` for the profile's precision;
        otherwise the directory's single ``.gguf`` file. ``kv_cache_space`` (also
        popped) then sets ``ctx-size``, unless ``ctx-size`` is given explicitly.

        Raises:
            ValueError: if the model is not in the cache, the file cannot be
                determined unambiguously, or the KV cache size cannot be derived.
        """
        gguf_filename = _pop_arg(engine_args, "gguf_filename")
        kv_cache_space = _pop_arg(engine_args, "kv_cache_space")

        model_path = self._resolve_gguf_path(resolved, gguf_filename, profile)

        if kv_cache_space is not None:
            if _get_arg(engine_args, "ctx_size") is not None:
                logger.info("ctx-size is set explicitly; ignoring kv_cache_space")
            else:
                engine_args["ctx-size"] = self._ctx_size_for_kv_space(
                    float(kv_cache_space), model_path, resolved.path, engine_args
                )
        return model_path

    def _resolve_gguf_path(
        self, resolved: "ResolvedModelPath", gguf_filename: Optional[str], profile: "Profile"
    ) -> str:
        """Return the ``.gguf`` file to load; see :meth:`resolve_model_path`."""
        if not resolved.is_local_dir:
            raise ValueError(
                f"Model '{resolved.model_id}' is not in the AIM model cache; llama-server needs a local "
                f"model. Run `./entrypoint.py download-to-cache --model-id hf://{resolved.model_id}` "
                f"or mount the model directory at <AIM_CACHE_PATH>/{resolved.model_id}."
            )

        if gguf_filename:
            candidate = os.path.join(resolved.path, gguf_filename)
            if not os.path.isfile(candidate):
                raise ValueError(f"gguf_filename '{gguf_filename}' not found under '{resolved.path}'.")
            return candidate

        if self._is_hf_checkpoint(resolved.path):
            return self._converted_gguf_path(resolved.path, profile)

        matches = sorted(glob.glob(os.path.join(resolved.path, "*.gguf")))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"No .gguf file or Hugging Face checkpoint found under '{resolved.path}'.")
        raise ValueError(
            f"Multiple .gguf files found under '{resolved.path}': {[os.path.basename(m) for m in matches]}. "
            "Set `gguf_filename` in engine_args to pick one."
        )

    def prepare_model(self) -> None:
        """Convert the Hugging Face checkpoint to GGUF if needed.

        The result is written next to the checkpoint and reused on later
        launches; it is renamed into place only when complete.

        Raises:
            RuntimeError: if the cache directory is not writable or the
                conversion fails.
        """
        if self._pending_conversion is None:
            return
        checkpoint_dir, outtype, target = self._pending_conversion
        if not os.access(checkpoint_dir, os.W_OK):
            raise RuntimeError(
                f"Cannot convert '{checkpoint_dir}' to GGUF: the directory is not writable. Mount the "
                f"model cache read-write for the first launch, or provide a .gguf file in it."
            )

        partial = f"{target}.partial"
        command = [sys.executable, CONVERTER_PATH, checkpoint_dir, "--outtype", outtype, "--outfile", partial]
        logger.info(f"Converting Hugging Face checkpoint to GGUF ({outtype}): {target}")
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            if os.path.exists(partial):
                os.remove(partial)
            raise RuntimeError(f"GGUF conversion of '{checkpoint_dir}' failed: {e}") from e
        os.replace(partial, target)
        self._pending_conversion = None
        logger.info(f"GGUF conversion complete: {target}")

    @staticmethod
    def _ctx_size_for_kv_space(
        kv_cache_space_gib: float, model_path: str, model_dir: str, engine_args: dict[str, Any]
    ) -> int:
        """Number of context tokens whose KV cache fits in ``kv_cache_space_gib``.

        Uses the GGUF metadata, or ``config.json`` for a checkpoint not yet
        converted. Every layer is counted as full attention, so hybrid and
        sliding-window models get a conservative (smaller) context.
        """
        k_bytes = KV_CACHE_TYPE_BYTES[_get_arg(engine_args, "cache_type_k") or "f16"]
        v_bytes = KV_CACHE_TYPE_BYTES[_get_arg(engine_args, "cache_type_v") or "f16"]
        if os.path.isfile(model_path):
            layers = _kv_layers_from_gguf(model_path)
        else:
            layers = _kv_layers_from_hf_config(os.path.join(model_dir, "config.json"))

        bytes_per_token = sum(n_head_kv * (k_len * k_bytes + v_len * v_bytes) for n_head_kv, k_len, v_len in layers)
        if bytes_per_token <= 0:
            raise ValueError(f"Cannot derive the KV cache size per token for '{model_path}'; set ctx-size instead.")
        ctx_size = int(kv_cache_space_gib * 1024**3 // bytes_per_token) // CTX_SIZE_ALIGNMENT * CTX_SIZE_ALIGNMENT
        if ctx_size < CTX_SIZE_ALIGNMENT:
            raise ValueError(f"kv_cache_space of {kv_cache_space_gib} GiB is too small for '{model_path}'.")
        logger.info(
            f"KV cache: {kv_cache_space_gib} GiB at {bytes_per_token / 1024:.1f} KiB/token -> ctx-size {ctx_size}"
        )
        return ctx_size

    @staticmethod
    def _is_hf_checkpoint(path: str) -> bool:
        """Whether ``path`` holds a Hugging Face checkpoint in a format the converter reads."""
        return os.path.isfile(os.path.join(path, "config.json")) and bool(
            glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "pytorch_model*.bin"))
        )

    def _converted_gguf_path(self, checkpoint_dir: str, profile: "Profile") -> str:
        """Return the GGUF converted from ``checkpoint_dir``, scheduling the conversion if it is missing."""
        precision = profile.metadata.precision
        outtype = GGUF_OUTTYPE_BY_PRECISION.get(precision)
        if outtype is None:
            supported = ", ".join(p.value for p in GGUF_OUTTYPE_BY_PRECISION)
            raise ValueError(
                f"Cannot convert a Hugging Face checkpoint to GGUF for precision '{precision}'. "
                f"Supported: {supported}; or provide a .gguf file and set `gguf_filename`."
            )
        target = os.path.join(checkpoint_dir, f"{os.path.basename(checkpoint_dir)}-{outtype}.gguf")
        if not os.path.isfile(target):
            self._pending_conversion = (checkpoint_dir, outtype, target)
        return target


def _get_arg(engine_args: dict[str, Any], name: str) -> Any:
    """Value of an engine argument given in either snake_case or kebab-case."""
    return engine_args.get(name, engine_args.get(name.replace("_", "-")))


def _pop_arg(engine_args: dict[str, Any], name: str) -> Any:
    """Remove an engine argument in either spelling and return its value."""
    snake, kebab = engine_args.pop(name, None), engine_args.pop(name.replace("_", "-"), None)
    return snake if snake is not None else kebab


def _kv_layers_from_gguf(path: str) -> list[tuple[int, int, int]]:
    """(KV heads, key length, value length) per layer, from GGUF metadata."""
    from gguf import GGUFReader

    fields = GGUFReader(path).fields

    def value(key: str, default: Any = None) -> Any:
        field = fields.get(key)
        return default if field is None else field.contents()

    arch = value("general.architecture")
    n_layer = value(f"{arch}.block_count")
    n_head = value(f"{arch}.attention.head_count")
    n_head_kv = value(f"{arch}.attention.head_count_kv", n_head)
    head_dim = value(f"{arch}.embedding_length", 0) // (max(n_head) if isinstance(n_head, list) else n_head or 1)
    k_len = value(f"{arch}.attention.key_length", head_dim)
    v_len = value(f"{arch}.attention.value_length", head_dim)
    heads_per_layer = n_head_kv if isinstance(n_head_kv, list) else [n_head_kv] * n_layer
    return [(heads, k_len, v_len) for heads in heads_per_layer]


def _kv_layers_from_hf_config(path: str) -> list[tuple[int, int, int]]:
    """(KV heads, key length, value length) per layer, from a Hugging Face ``config.json``."""
    with open(path) as f:
        config = json.load(f)
    config = config.get("text_config", config)
    n_head = config["num_attention_heads"]
    n_head_kv = config.get("num_key_value_heads") or n_head
    head_dim = config.get("head_dim") or config["hidden_size"] // n_head
    return [(n_head_kv, head_dim, head_dim)] * config["num_hidden_layers"]
