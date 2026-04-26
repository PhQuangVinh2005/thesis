"""Factory for creating LLM instances from YAML config."""

from typing import Dict, Any

from .base import BaseLLM

# Lazy imports to avoid loading torch at module level
_MODEL_REGISTRY: Dict[str, Any] = {
    "transformers": lambda: __import__(
        "src.models.hf_model", fromlist=["TransformersModel"]
    ).TransformersModel,
    "ollama": lambda: __import__(
        "src.models.ollama_model", fromlist=["OllamaModel"]
    ).OllamaModel,
}


class ModelFactory:
    """Create LLM instances from YAML config dict."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseLLM:
        config = dict(config)
        backend = config.pop("backend", "transformers")

        if backend not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(_MODEL_REGISTRY.keys())}"
            )

        generation = config.pop("generation", {})
        model_params = config.pop("model_params", {})
        if "max_model_len" in model_params:
            config["max_model_len"] = model_params["max_model_len"]
        if "num_ctx" in model_params:
            config["num_ctx"] = model_params["num_ctx"]

        model_cls = _MODEL_REGISTRY[backend]()
        return model_cls(**config, **generation)

    @staticmethod
    def available_backends() -> list:
        return list(_MODEL_REGISTRY.keys())
