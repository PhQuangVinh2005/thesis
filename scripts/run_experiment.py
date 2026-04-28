#!/usr/bin/env python3
"""
Unified experiment runner.

Handles single-model or all-models execution with HuggingFace Transformers.
Models are loaded directly onto GPU — no server management needed.

Usage:
    # Single model (baseline):
    python scripts/run_experiment.py --config configs/experiment/biomistral7b.yaml

    # All baseline models (auto cleanup GPU between models):
    python scripts/run_experiment.py --all

    # All models for a specific technique:
    python scripts/run_experiment.py --technique fewshot_1
    python scripts/run_experiment.py --technique fewshot_5
    python scripts/run_experiment.py --technique fewshot_10

    # Dry-run (no model loaded):
    python scripts/run_experiment.py --all --dry-run
    python scripts/run_experiment.py --technique fewshot_1 --dry-run

    # Debug with limited samples:
    python scripts/run_experiment.py --config configs/experiment/qwen3_5_2b.yaml --max-samples 5

    # Specific range only:
    python scripts/run_experiment.py --config configs/experiment/biomistral7b.yaml --range 0_1k
"""

# ── Config ─────────────────────────────────────────────────────────────
# Order: lighter models first for faster feedback
ALL_CONFIGS = [
    "configs/experiment/qwen3_5_2b.yaml",
    "configs/experiment/qwen3_5_4b.yaml",
    "configs/experiment/qwen3_5_9b.yaml",
    "configs/experiment/biomistral7b.yaml",
    "configs/experiment/biomistral7b_slerp.yaml",
]

# Technique-specific config directories
TECHNIQUE_CONFIGS = {
    "fewshot_1": [
        "configs/experiment/fewshot_1/qwen3_5_2b.yaml",
        "configs/experiment/fewshot_1/qwen3_5_4b.yaml",
        "configs/experiment/fewshot_1/qwen3_5_9b.yaml",
        "configs/experiment/fewshot_1/biomistral7b.yaml",
        "configs/experiment/fewshot_1/biomistral7b_slerp.yaml",
    ],
    "fewshot_5": [
        "configs/experiment/fewshot_5/qwen3_5_2b.yaml",
        "configs/experiment/fewshot_5/qwen3_5_4b.yaml",
        "configs/experiment/fewshot_5/qwen3_5_9b.yaml",
        "configs/experiment/fewshot_5/biomistral7b.yaml",
        "configs/experiment/fewshot_5/biomistral7b_slerp.yaml",
    ],
    "fewshot_10": [
        "configs/experiment/fewshot_10/qwen3_5_2b.yaml",
        "configs/experiment/fewshot_10/qwen3_5_4b.yaml",
        "configs/experiment/fewshot_10/qwen3_5_9b.yaml",
        "configs/experiment/fewshot_10/biomistral7b.yaml",
        "configs/experiment/fewshot_10/biomistral7b_slerp.yaml",
    ],
    "cove": [
        "configs/experiment/cove/qwen3_5_2b.yaml",
        "configs/experiment/cove/qwen3_5_4b.yaml",
        "configs/experiment/cove/qwen3_5_9b.yaml",
        "configs/experiment/cove/biomistral7b.yaml",
        "configs/experiment/cove/biomistral7b_slerp.yaml",
    ],
}
# ───────────────────────────────────────────────────────────────────────

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mimic_bhc import MIMICBHCLoader
from src.models.factory import ModelFactory
from src.prompts.templates import PromptTemplate
from src.pipelines.summarizer import SummarizationPipeline
from src.techniques.baseline import BaselineTechnique
from src.techniques.fewshot import FewShotTechnique
from src.techniques.cove import CoVeTechnique
from src.utils.io import load_yaml, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── GPU Cleanup ───────────────────────────────────────────────────────

def cleanup_model(model) -> None:
    """Release GPU memory after finishing with a model."""
    if model is not None and hasattr(model, "cleanup"):
        model.cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Technique Factory ─────────────────────────────────────────────────

def create_technique(config: dict, prompt_template: PromptTemplate):
    """Create technique from experiment config.

    If config has a 'technique' section, create the specified technique.
    Otherwise, return BaselineTechnique (backward compatible).
    """
    tech_cfg = config.get("technique")
    if tech_cfg is None:
        return BaselineTechnique()

    name = tech_cfg["name"]
    if name == "fewshot":
        examples_file = str(PROJECT_ROOT / tech_cfg["examples_file"])
        indices = tech_cfg["indices"]
        instruction = prompt_template.template.split("{context}")[0].strip()
        return FewShotTechnique(
            examples_file=examples_file,
            indices=indices,
            instruction=instruction,
        )
    elif name == "cove":
        configs_dir = str(PROJECT_ROOT / "configs")
        plan_template = PromptTemplate.from_yaml(
            _resolve_path(configs_dir, tech_cfg["plan_prompt"])
        )
        verify_refine_template = PromptTemplate.from_yaml(
            _resolve_path(configs_dir, tech_cfg["verify_refine_prompt"])
        )
        return CoVeTechnique(
            plan_template=plan_template,
            verify_refine_template=verify_refine_template,
            n_questions=tech_cfg.get("n_questions", 5),
        )
    else:
        raise ValueError(f"Unknown technique: {name}")


# ── Experiment Logic ──────────────────────────────────────────────────

def run_single_range(
    range_id: str,
    config: dict,
    model,
    pipeline: SummarizationPipeline,
    prompt_template: PromptTemplate,
    data_dir: str,
    args,
) -> None:
    """Run experiment for a single token range."""
    data_cfg = config.get("data", {})
    sample_size = data_cfg.get("sample_size")
    sample_seed = data_cfg.get("sample_seed", 42)

    # Load data
    loader = MIMICBHCLoader(data_dir)
    all_samples = loader.load(range_id=range_id)
    logger.info(f"Loaded {len(all_samples)} samples from range_{range_id}")

    # Sample if configured
    if sample_size:
        samples = loader.sample(n=sample_size, seed=sample_seed)
        logger.info(f"Sampled {len(samples)} from {len(all_samples)} (seed={sample_seed})")
    else:
        samples = all_samples

    # CLI --max-samples override
    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {len(samples)} samples (--max-samples)")

    # Dry-run: print sample prompts and exit
    if args.dry_run:
        for i, sample in enumerate(samples[:3]):
            prompt = pipeline.format_prompt(sample)
            print(f"\n{'='*60}")
            print(f"[range_{range_id}] Sample {i+1}: {sample.sample_id}")
            print(f"Context: {len(sample.context)} chars | Target: {len(sample.labeled_summary or '')} chars")
            print(f"{'='*60}")
            print(f"PROMPT (first 500 chars):\n{prompt[:500]}...")
        logger.info(f"[DRY RUN] range_{range_id}: {len(samples)} samples would be processed.")
        return

    # Output path
    output_cfg = config.get("output", {})
    output_dir = PROJECT_ROOT / output_cfg.get("dir", "outputs/") / f"range_{range_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = str(output_dir / output_cfg.get("predictions_file", "predictions.jsonl"))

    logger.info(f"Output: {predictions_file}")

    # Run summarization
    results = pipeline.run(
        samples=samples,
        output_path=predictions_file,
        save_every=config.get("generation", {}).get("save_every", 10),
    )

    # Save experiment metadata
    meta = {
        "experiment_name": config["experiment_name"],
        "model": model.get_model_info(),
        "prompt": prompt_template.to_dict(),
        "data": {
            "range": range_id,
            "total_loaded": len(all_samples),
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "actual_processed": len(results),
        },
        "timestamp": datetime.now().isoformat(),
    }
    save_json(meta, str(output_dir / "experiment_meta.json"))

    errors = sum(1 for r in results if r.predicted_summary and r.predicted_summary.startswith("[ERROR]"))
    logger.info(f"[range_{range_id}] Done! {len(results)} generated, {errors} errors.")


def run_single_config(config_path: str, args):
    """Run all ranges for a single experiment config. Returns model for cleanup."""
    config = load_yaml(config_path)
    configs_dir = str(PROJECT_ROOT / "configs")
    logger.info(f"Experiment: {config['experiment_name']}")

    # Load model config
    model_config_path = _resolve_path(configs_dir, config["model_config"])
    model_config = load_yaml(model_config_path)
    logger.info(f"Model: {model_config['model_name']}")

    # Load prompt template
    prompt_config_path = _resolve_path(configs_dir, config["prompt_config"])
    prompt_template = PromptTemplate.from_yaml(prompt_config_path)
    logger.info(f"Prompt: {prompt_template.name}")

    # Resolve ranges
    data_cfg = config.get("data", {})
    ranges = [args.range] if args.range else data_cfg.get("ranges", ["0_1k"])
    data_dir = str(PROJECT_ROOT / data_cfg.get("data_dir", "data/processed/mimic_iv_bhc"))

    # Create technique from config (baseline if no technique section)
    technique = create_technique(config, prompt_template)
    logger.info(f"Technique: {technique}")

    # Initialize model (shared across ranges)
    if args.dry_run:
        model = None
        pipeline = SummarizationPipeline(
            model=None, prompt_template=prompt_template, technique=technique,
        )
        logger.info("=== DRY RUN MODE ===")
    else:
        model = ModelFactory.create(model_config)
        pipeline = SummarizationPipeline(
            model=model, prompt_template=prompt_template, technique=technique,
        )

    # Loop over ranges
    for i, range_id in enumerate(ranges):
        logger.info(f"\n{'='*60}")
        logger.info(f"Range {i+1}/{len(ranges)}: {range_id}")
        logger.info(f"{'='*60}")
        run_single_range(range_id, config, model, pipeline, prompt_template, data_dir, args)

    return model


def _resolve_path(base_dir: str, relative_path: str) -> str:
    """Resolve config path relative to configs/ directory."""
    path = Path(base_dir) / relative_path
    if not path.exists():
        path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {relative_path} (tried {path})")
    return str(path)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run summarization experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Single experiment config YAML")
    group.add_argument("--all", action="store_true", help="Run ALL baseline model configs sequentially")
    group.add_argument(
        "--technique", type=str, choices=list(TECHNIQUE_CONFIGS.keys()),
        help="Run ALL models for a specific technique (e.g., fewshot_1)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per range (debug)")
    parser.add_argument("--range", type=str, default=None, choices=["0_1k", "1k_2k", "2k_4k"], help="Run only this range")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling model")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        model = run_single_config(args.config, args)
        cleanup_model(model)
    else:
        # Determine which config list to run
        if args.technique:
            configs = TECHNIQUE_CONFIGS[args.technique]
            label = f"technique={args.technique}"
        else:
            configs = ALL_CONFIGS
            label = "baseline (all)"

        logger.info(f"{'='*60}")
        logger.info(f"Running {label}: {len(configs)} models")
        logger.info(f"{'='*60}")

        failed = []
        for idx, config_path in enumerate(configs):
            config = load_yaml(config_path)
            logger.info(f"\n{'='*60}")
            logger.info(f"Model {idx+1}/{len(configs)}: {config.get('experiment_name', config_path)}")
            logger.info(f"{'='*60}")

            model = None
            try:
                model = run_single_config(config_path, args)
                logger.info(f"✓ {config.get('experiment_name')} complete!")
            except Exception as e:
                logger.error(f"✗ {config.get('experiment_name')} FAILED: {e}")
                failed.append(config.get("experiment_name", config_path))
            finally:
                cleanup_model(model)

        if failed:
            logger.warning(f"\n⚠ {len(failed)} model(s) failed: {failed}")
        logger.info(f"\n✓ Done! {len(configs) - len(failed)}/{len(configs)} models succeeded.")


if __name__ == "__main__":
    main()
