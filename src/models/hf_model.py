"""HuggingFace Transformers LLM implementation."""

import gc
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base import BaseLLM

logger = logging.getLogger(__name__)


class TransformersModel(BaseLLM):
    """LLM wrapper using HuggingFace Transformers for local inference."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        torch_dtype: str = "float16",
        device: str = "auto",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_model_len = max_model_len
        self.quantization = quantization
        self._device_str = device

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit quantization enabled (nf4 + double quant)")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("8-bit quantization enabled")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Flash Attention 3 auto-detection
        self._attn_impl = "eager"
        try:
            import flash_attn_3  # noqa: F401
            self._attn_impl = "flash_attention_2"
            logger.info("Flash Attention 3 detected")
        except ImportError:
            pass

        logger.info(f"Loading model: {model_name} (dtype={torch_dtype}, quant={quantization}, attn={self._attn_impl})")
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": device,
            "trust_remote_code": True,
            "attn_implementation": self._attn_impl,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            # bitsandbytes 8-bit requires float16 internally
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["dtype"] = resolved_dtype

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()

        logger.info(f"TransformersModel ready: {model_name} | dtype={self.model.dtype}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response from a prompt."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Truncate if exceeding max_model_len
        if self.max_model_len and input_len > self.max_model_len - max_tokens:
            allowed = max(1, self.max_model_len - max_tokens)
            inputs["input_ids"] = inputs["input_ids"][:, -allowed:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -allowed:]
            logger.warning(f"Input truncated: {input_len} → {allowed} tokens")
            input_len = allowed

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
            "backend": "transformers",
            "torch_dtype": str(self.model.dtype),
            "quantization": self.quantization,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_model_len": self.max_model_len,
        }

    def cleanup(self) -> None:
        """Release GPU memory."""
        logger.info(f"Cleaning up: {self.model_name}")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("GPU memory released")
