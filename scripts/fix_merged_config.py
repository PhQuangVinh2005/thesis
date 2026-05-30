#!/usr/bin/env python3
"""Restore and prepare Qwen3.5 merged model configs for vLLM inference.

Patches Qwen3.5-4B text configs to match vLLM nightly type registries:
  - Changes model_type from 'qwen3_5_text' to 'qwen3_5' to trigger Qwen3_5Config loading.
  - Ensures architectures is correctly set to Qwen3_5ForCausalLM for text causal generation.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fix_config(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    logger.info(f"Patching configuration: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # 1. Patch model_type to 'qwen3_5' so vLLM resolves it to vLLM's Qwen3_5Config
    old_type = config.get("model_type")
    if old_type == "qwen3_5_text":
        config["model_type"] = "qwen3_5"
        logger.info("  ✓ Changed model_type: 'qwen3_5_text' -> 'qwen3_5'")
    else:
        logger.info(f"  • model_type is already: '{old_type}'")

    # 2. Validate/force Qwen3_5ForCausalLM as the primary architecture
    archs = config.get("architectures", [])
    if "Qwen3_5ForCausalLM" not in archs:
        config["architectures"] = ["Qwen3_5ForCausalLM"]
        logger.info(f"  ✓ Updated architectures to: {config['architectures']}")
    else:
        logger.info(f"  • architectures already correct: {archs}")

    # Write patched config back
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("✓ Patch successfully saved!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix Qwen3.5 merged config for vLLM compatibility.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the merged model directory to patch.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    logger.info("=" * 60)
    logger.info(f"Fixing merged config in: {model_dir}")
    logger.info("=" * 60)

    if fix_config(model_dir):
        logger.info("✓ Model config is fully compatible with vLLM nightly.")
    else:
        logger.error("✗ Failed to patch config.")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
