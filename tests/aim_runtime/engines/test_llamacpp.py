# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for the llama.cpp engine: argument validation and .gguf file selection."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aim_common import Engine, Precision
from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines import LlamaCppEngine, LlamaCppEngineArgsModel, build_engine
from aim_runtime.engines.llamacpp import CONVERTER_PATH
from aim_runtime.model_cache_resolver import ResolvedModelPath

MODEL_ID = "org/model-GGUF"


def _engine() -> LlamaCppEngine:
    config = AIMConfig(model_id=MODEL_ID, engine=Engine.LLAMACPP)
    engine_config = EngineConfig(engine=Engine.LLAMACPP, launch="llama-server", model_arg="-m")
    return build_engine(config, engine_config)


def _cached(path) -> ResolvedModelPath:
    return ResolvedModelPath(path=str(path), is_local_dir=True, model_id=MODEL_ID)


def _profile(precision: Precision = Precision.BF16):
    return SimpleNamespace(metadata=SimpleNamespace(precision=precision))


def _hf_checkpoint(path):
    (path / "config.json").write_text("{}")
    (path / "model-00001-of-00001.safetensors").touch()
    return path


class TestLlamaCppEngineArgsModel:
    def test_profile_like_kebab_keys_accepted(self):
        LlamaCppEngineArgsModel.model_validate(
            {"host": "0.0.0.0", "ctx-size": 8192, "threads": 96, "load-mode": "none", "parallel": 4}
        )

    def test_snake_case_keys_accepted(self):
        LlamaCppEngineArgsModel.model_validate({"ctx_size": 8192, "load_mode": "mmap"})

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            LlamaCppEngineArgsModel.model_validate({"threads": "many"})

    @pytest.mark.parametrize("field,value", [("load-mode", "no-mmap"), ("numa", "bogus"), ("flash-attn", "yes")])
    def test_invalid_choice_rejected(self, field, value):
        with pytest.raises(ValidationError):
            LlamaCppEngineArgsModel.model_validate({field: value})

    def test_unknown_flag_passes_through(self):
        LlamaCppEngineArgsModel.model_validate({"some-new-flag": 1})


class TestResolveModelPath:
    def test_not_cached_raises(self):
        resolved = ResolvedModelPath(path=MODEL_ID, is_local_dir=False, model_id=MODEL_ID)
        with pytest.raises(ValueError, match="not in the AIM model cache"):
            _engine().resolve_model_path(resolved, {}, _profile())

    def test_single_gguf_selected(self, tmp_path):
        (tmp_path / "model-q8_0.gguf").touch()
        assert _engine().resolve_model_path(_cached(tmp_path), {}, _profile()) == str(tmp_path / "model-q8_0.gguf")

    def test_no_gguf_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .gguf file"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile())

    def test_multiple_gguf_without_hint_raises(self, tmp_path):
        (tmp_path / "model-bf16.gguf").touch()
        (tmp_path / "model-q8_0.gguf").touch()
        with pytest.raises(ValueError, match="Multiple .gguf files"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile())

    @pytest.mark.parametrize("key", ["gguf_filename", "gguf-filename"])
    def test_gguf_filename_picks_file_and_is_stripped(self, tmp_path, key):
        (tmp_path / "model-bf16.gguf").touch()
        (tmp_path / "model-q8_0.gguf").touch()
        engine_args = {key: "model-q8_0.gguf", "threads": 8}
        assert _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile()) == str(
            tmp_path / "model-q8_0.gguf"
        )
        assert engine_args == {"threads": 8}

    def test_gguf_filename_missing_raises(self, tmp_path):
        (tmp_path / "model-bf16.gguf").touch()
        with pytest.raises(ValueError, match="not found"):
            _engine().resolve_model_path(_cached(tmp_path), {"gguf_filename": "other.gguf"}, _profile())


class TestApplyEngineDefaults:
    @pytest.fixture
    def cores(self, monkeypatch):
        def _set(available: int, cpuset: str | None):
            monkeypatch.setattr("aim_runtime.engines.llamacpp.CpuDetector.available_cores", available)
            monkeypatch.setattr("aim_runtime.engines.llamacpp.CpuDetector.cpuset_bind", cpuset)

        return _set

    def test_threads_and_pinning_follow_cpuset(self, cores):
        cores(4, "0-1,4-5")
        engine_args: dict = {}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"threads": 4, "cpu-mask": "33", "cpu-strict": 1}

    def test_no_pinning_without_known_cpuset(self, cores):
        cores(8, None)
        engine_args: dict = {}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"threads": 8}

    def test_explicit_values_kept(self, cores):
        cores(96, "0-95")
        engine_args = {"threads": 48, "cpu-range": "0-47"}
        _engine().apply_engine_defaults(engine_args, [MODEL_ID])
        assert engine_args == {"threads": 48, "cpu-range": "0-47"}


class TestHfCheckpointConversion:
    def test_resolves_to_converted_file_named_by_precision(self, tmp_path):
        _hf_checkpoint(tmp_path)
        engine = _engine()
        path = engine.resolve_model_path(_cached(tmp_path), {}, _profile(Precision.INT8))
        assert path == str(tmp_path / f"{tmp_path.name}-q8_0.gguf")

    def test_existing_converted_file_is_reused(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)
        (tmp_path / f"{tmp_path.name}-bf16.gguf").touch()
        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", pytest.fail)
        engine = _engine()
        engine.resolve_model_path(_cached(tmp_path), {}, _profile())
        engine.prepare_model()

    def test_prepare_model_runs_converter_once(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)
        calls = []

        def fake_run(command, check):
            calls.append(command)
            open(command[command.index("--outfile") + 1], "w").close()

        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", fake_run)
        engine = _engine()
        path = engine.resolve_model_path(_cached(tmp_path), {}, _profile())
        engine.prepare_model()
        engine.prepare_model()

        assert len(calls) == 1
        assert calls[0][1:5] == [CONVERTER_PATH, str(tmp_path), "--outtype", "bf16"]
        assert (tmp_path / path).is_file()
        assert not list(tmp_path.glob("*.partial"))

    def test_failed_conversion_leaves_no_file(self, tmp_path, monkeypatch):
        _hf_checkpoint(tmp_path)

        def fake_run(command, check):
            open(command[command.index("--outfile") + 1], "w").close()
            raise subprocess.CalledProcessError(1, command)

        monkeypatch.setattr("aim_runtime.engines.llamacpp.subprocess.run", fake_run)
        engine = _engine()
        engine.resolve_model_path(_cached(tmp_path), {}, _profile())
        with pytest.raises(RuntimeError, match="conversion"):
            engine.prepare_model()
        assert not list(tmp_path.glob("*.gguf*"))

    def test_pytorch_bin_checkpoint_is_converted(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "pytorch_model.bin").touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {}, _profile())
        assert path == str(tmp_path / f"{tmp_path.name}-bf16.gguf")

    def test_unsupported_precision_raises(self, tmp_path):
        _hf_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="precision"):
            _engine().resolve_model_path(_cached(tmp_path), {}, _profile(Precision.FP8))

    def test_gguf_filename_wins_over_checkpoint(self, tmp_path):
        _hf_checkpoint(tmp_path)
        (tmp_path / "custom.gguf").touch()
        path = _engine().resolve_model_path(_cached(tmp_path), {"gguf_filename": "custom.gguf"}, _profile())
        assert path == str(tmp_path / "custom.gguf")


LLAMA_8B_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
}


class TestKvCacheSpace:
    def _checkpoint(self, path, config=LLAMA_8B_CONFIG):
        _hf_checkpoint(path)
        (path / "config.json").write_text(json.dumps(config))
        return path

    def test_ctx_size_from_hf_config(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv-cache-space": 80, "parallel": 128}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        # 32 layers * 8 KV heads * (128 + 128) * 2 bytes = 128 KiB per token.
        assert engine_args == {"ctx-size": 655360, "parallel": 128}

    def test_quantized_kv_cache_fits_more_tokens(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv_cache_space": 80, "cache-type-k": "q8_0", "cache-type-v": "q8_0"}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args["ctx-size"] == 1233408

    def test_explicit_ctx_size_wins(self, tmp_path):
        self._checkpoint(tmp_path)
        engine_args = {"kv-cache-space": 80, "ctx-size": 8192}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == {"ctx-size": 8192}

    def test_ctx_size_from_gguf_metadata(self, tmp_path):
        gguf = pytest.importorskip("gguf")
        writer = gguf.GGUFWriter(str(tmp_path / "model.gguf"), "llama")
        writer.add_block_count(32)
        writer.add_embedding_length(4096)
        writer.add_head_count(32)
        writer.add_head_count_kv(8)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        engine_args = {"kv-cache-space": 1}
        _engine().resolve_model_path(_cached(tmp_path), engine_args, _profile())
        assert engine_args == {"ctx-size": 8192}

    def test_budget_too_small_raises(self, tmp_path):
        self._checkpoint(tmp_path)
        with pytest.raises(ValueError, match="too small"):
            _engine().resolve_model_path(_cached(tmp_path), {"kv-cache-space": 0.001}, _profile())
