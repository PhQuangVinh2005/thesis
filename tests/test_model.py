"""
Tests for model layer — pre-inference sanity checks.

Run these BEFORE starting a long batch to catch config/model errors early:
    python -m pytest tests/test_model.py -v
"""

import pytest
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Config Parsing Tests (no GPU needed) ───


class TestModelConfigs:
    """Verify all model YAML configs are parseable and have required keys."""

    CONFIGS_DIR = PROJECT_ROOT / "configs" / "models"
    REQUIRED_KEYS = {"backend", "model_name", "generation"}
    REQUIRED_GENERATION_KEYS = {"max_tokens", "temperature", "top_p"}

    @pytest.fixture
    def all_config_paths(self):
        return sorted(self.CONFIGS_DIR.glob("*.yaml"))

    def test_configs_exist(self, all_config_paths):
        """At least one model config exists."""
        assert len(all_config_paths) > 0, f"No YAML configs found in {self.CONFIGS_DIR}"

    @pytest.mark.parametrize("config_name", [
        "qwen3_5_2b.yaml",
        "qwen3_5_4b.yaml",
        "qwen3_5_9b.yaml",
        "biomistral_7b.yaml",
        "biomistral_7b_slerp.yaml",
    ])
    def test_config_has_required_keys(self, config_name):
        """Each config has backend, model_name, and generation section."""
        from src.utils.io import load_yaml

        path = self.CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        config = load_yaml(str(path))

        for key in self.REQUIRED_KEYS:
            assert key in config, f"Missing required key '{key}' in {config_name}"

        gen = config["generation"]
        for key in self.REQUIRED_GENERATION_KEYS:
            assert key in gen, f"Missing generation.{key} in {config_name}"

    @pytest.mark.parametrize("config_name,expected_backend", [
        ("qwen3_5_2b.yaml", "ollama"),
        ("qwen3_5_4b.yaml", "ollama"),
        ("qwen3_5_9b.yaml", "ollama"),
        ("biomistral_7b.yaml", "transformers"),
        ("biomistral_7b_slerp.yaml", "transformers"),
    ])
    def test_backend_is_correct(self, config_name, expected_backend):
        """Each config uses the expected backend."""
        from src.utils.io import load_yaml

        path = self.CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        config = load_yaml(str(path))
        assert config["backend"] == expected_backend, (
            f"{config_name} has backend='{config['backend']}', expected '{expected_backend}'"
        )

    @pytest.mark.parametrize("config_name", [
        "biomistral_7b.yaml",
        "biomistral_7b_slerp.yaml",
    ])
    def test_biomistral_has_8bit(self, config_name):
        """BioMistral models should use 8-bit quantization."""
        from src.utils.io import load_yaml

        path = self.CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        config = load_yaml(str(path))
        assert config.get("quantization") == "8bit", (
            f"{config_name} should have quantization='8bit' for 16GB VRAM"
        )

    @pytest.mark.parametrize("config_name", [
        "qwen3_5_2b.yaml",
        "qwen3_5_4b.yaml",
        "qwen3_5_9b.yaml",
    ])
    def test_ollama_has_num_ctx(self, config_name):
        """Ollama configs must have num_ctx in model_params."""
        from src.utils.io import load_yaml

        path = self.CONFIGS_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        config = load_yaml(str(path))
        assert config["backend"] == "ollama"
        assert "num_ctx" in config.get("model_params", {}), (
            f"{config_name} missing model_params.num_ctx"
        )


# ─── Factory Tests (no GPU needed) ───


class TestModelFactory:
    """Test that ModelFactory correctly parses configs."""

    def test_available_backends(self):
        from src.models.factory import ModelFactory
        backends = ModelFactory.available_backends()
        assert "transformers" in backends
        assert "ollama" in backends

    def test_unknown_backend_raises(self):
        from src.models.factory import ModelFactory
        with pytest.raises(ValueError, match="Unknown backend"):
            ModelFactory.create({"backend": "nonexistent", "model_name": "test"})

    def test_factory_does_not_mutate_config(self):
        """Factory should not modify the original config dict."""
        pytest.importorskip("torch")
        from src.models.factory import ModelFactory

        original = {
            "backend": "transformers",
            "model_name": "test/model",
            "generation": {"max_tokens": 100, "temperature": 0.1, "top_p": 0.9},
            "model_params": {"max_model_len": 2048},
        }
        original_copy = {
            "backend": "transformers",
            "model_name": "test/model",
            "generation": {"max_tokens": 100, "temperature": 0.1, "top_p": 0.9},
            "model_params": {"max_model_len": 2048},
        }

        # This will fail at model load (no real model), but config should be untouched
        try:
            ModelFactory.create(original)
        except Exception:
            pass

        assert original == original_copy, "Factory mutated the input config"


# ─── Model Class Tests (no GPU needed for import/structure) ───


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTransformersModelStructure:
    """Test TransformersModel class structure without loading a real model."""

    def test_class_exists(self):
        from src.models.hf_model import TransformersModel
        assert TransformersModel is not None

    def test_inherits_base_llm(self):
        from src.models.hf_model import TransformersModel
        from src.models.base import BaseLLM
        assert issubclass(TransformersModel, BaseLLM)

    def test_has_cleanup_method(self):
        from src.models.hf_model import TransformersModel
        assert hasattr(TransformersModel, "cleanup")
        assert callable(getattr(TransformersModel, "cleanup"))

    def test_has_required_methods(self):
        from src.models.hf_model import TransformersModel
        for method_name in ("generate", "batch_generate", "get_model_info", "cleanup"):
            assert hasattr(TransformersModel, method_name), f"Missing method: {method_name}"


class TestOllamaModelStructure:
    """Test OllamaModel class structure without connecting to Ollama."""

    def test_class_exists(self):
        from src.models.ollama_model import OllamaModel
        assert OllamaModel is not None

    def test_inherits_base_llm(self):
        from src.models.ollama_model import OllamaModel
        from src.models.base import BaseLLM
        assert issubclass(OllamaModel, BaseLLM)

    def test_has_required_methods(self):
        from src.models.ollama_model import OllamaModel
        for method_name in ("generate", "batch_generate", "get_model_info", "cleanup"):
            assert hasattr(OllamaModel, method_name), f"Missing method: {method_name}"


# ─── Experiment Config Tests ───


class TestExperimentConfigs:
    """Verify experiment configs reference valid model/prompt configs."""

    EXPERIMENT_DIR = PROJECT_ROOT / "configs" / "experiment"

    @pytest.mark.parametrize("config_name", [
        "qwen3_5_2b.yaml",
        "qwen3_5_4b.yaml",
        "qwen3_5_9b.yaml",
        "biomistral7b.yaml",
        "biomistral7b_slerp.yaml",
    ])
    def test_experiment_references_exist(self, config_name):
        """Experiment config references valid model and prompt configs."""
        from src.utils.io import load_yaml

        path = self.EXPERIMENT_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")

        config = load_yaml(str(path))
        configs_dir = path.parent.parent

        # Check model config exists
        model_path = configs_dir / config["model_config"]
        assert model_path.exists(), f"Model config not found: {model_path}"

        # Check prompt config exists
        prompt_path = configs_dir / config["prompt_config"]
        assert prompt_path.exists(), f"Prompt config not found: {prompt_path}"


# ─── Smoke Test (needs GPU — skipped if unavailable) ───


class TestModelSmoke:
    """Smoke test that loads the smallest model. Skip if no GPU."""

    @pytest.fixture
    def gpu_available(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("No GPU available — skipping smoke test")

    @pytest.mark.slow
    def test_load_and_generate_one_token(self, gpu_available):
        """Load Qwen3.5-2B and generate 1 token as sanity check."""
        from src.models.hf_model import TransformersModel

        model = TransformersModel(
            model_name="Qwen/Qwen3.5-2B",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            torch_dtype="float16",
            device="auto",
        )
        try:
            result = model.generate("Hello")
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            model.cleanup()
