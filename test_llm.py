import os
# Force vLLM V0 engine to avoid V1 memory pre-allocation OOMs
os.environ["VLLM_USE_V1"] = "0"
# Prevent memory fragmentation on limited VRAM cards
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_DISABLE_FLASHINFER"] = "1"

# Monkey-patch ModelConfig BEFORE importing vllm LLM
from vllm.config import ModelConfig
original_init = ModelConfig.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if hasattr(self, "hf_config") and self.hf_config is not None:
        self.hf_config.tie_word_embeddings = True
        if hasattr(self, "hf_text_config") and self.hf_text_config is not None:
            self.hf_text_config.tie_word_embeddings = True
            text_config_keys = [
                "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "hidden_act",
                "max_position_embeddings", "rms_norm_eps", "head_dim",
                "rope_parameters", "layer_types", "pad_token_id", "bos_token_id",
                "eos_token_id", "tie_word_embeddings"
            ]
            for key in text_config_keys:
                if hasattr(self.hf_config, key):
                    val = getattr(self.hf_config, key)
                    setattr(self.hf_text_config, key, val)
            print(f"[PATCH] Sync'd flat config keys to hf_text_config. hidden_size={self.hf_text_config.hidden_size}, intermediate_size={self.hf_text_config.intermediate_size}")

ModelConfig.__init__ = patched_init

from vllm import LLM, SamplingParams

def main():
    model_path = "models/qwen35_4b_base_dpo_10_merged"
    print(f"Loading LLM with model: {model_path} in vLLM mode")
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=6144,
            gpu_memory_utilization=0.80,
            disable_log_stats=True,
            enforce_eager=True,
            enable_prefix_caching=True,
            block_size=64,
            mamba_block_size=64,
        )
        print("LLM successfully loaded!")
        
        # Test generation
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
        )
        outputs = llm.generate("Patient was admitted with severe chest pain.", sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
