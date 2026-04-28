"""Ollama LLM implementation — communicates via REST API."""

import logging
from typing import List, Dict, Any

import requests

from .base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300


class OllamaModel(BaseLLM):
    """LLM wrapper using Ollama REST API for local inference."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        num_ctx: int = 8192,
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.base_url = base_url.rstrip("/")

        self._verify_connection()
        self._warmup()

    def _verify_connection(self) -> None:
        """Check that Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            logger.info(f"Ollama reachable at {self.base_url}")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama not reachable at {self.base_url}. Run: ollama serve"
            )

    def _warmup(self) -> None:
        """Preload model into GPU (keep_alive=-1 prevents auto-unload)."""
        logger.info(f"Preloading: {self.model_name} (keep_alive=-1)")
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "keep_alive": -1,
                    "think": False,
                    "options": {"num_predict": 1, "num_ctx": self.num_ctx},
                },
                timeout=_DEFAULT_TIMEOUT,
            )
            if resp.status_code == 404:
                raise FileNotFoundError(
                    f"Model '{self.model_name}' not found. Run: ollama pull {self.model_name}"
                )
            resp.raise_for_status()
            logger.info(f"OllamaModel ready: {self.model_name} | num_ctx={self.num_ctx}")
        except requests.ConnectionError:
            raise ConnectionError("Lost connection to Ollama during warmup.")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response from a prompt."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise TimeoutError(f"Ollama timed out after {_DEFAULT_TIMEOUT}s")
        except requests.ConnectionError:
            raise ConnectionError("Lost connection to Ollama during generation.")

        return resp.json().get("response", "").strip()

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts sequentially."""
        results = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(self.generate(prompt, **kwargs))
            except Exception as e:
                logger.error(f"Generation failed for prompt {i}: {e}")
                results.append(f"[ERROR] {e}")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "backend": "ollama",
            "base_url": self.base_url,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def cleanup(self) -> None:
        """Unload model from Ollama GPU memory (keep_alive=0)."""
        logger.info(f"Unloading: {self.model_name}")
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Failed to unload: {e}")
