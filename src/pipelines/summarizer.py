"""Summarization pipeline — orchestrates prompt → technique → prediction."""

import json
import time
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..data.schema import EvalSample
from ..models.base import BaseLLM
from ..prompts.templates import PromptTemplate
from ..techniques.base import BaseTechnique
from ..techniques.baseline import BaselineTechnique

logger = logging.getLogger(__name__)


class SummarizationPipeline:
    """Generate summaries with checkpoint/resume support."""

    def __init__(
        self,
        model: BaseLLM,
        prompt_template: PromptTemplate,
        technique: Optional[BaseTechnique] = None,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.technique = technique or BaselineTechnique()

    def format_prompt(self, sample: EvalSample) -> str:
        prompt = self.prompt_template.format(context=sample.context)
        sample.instruction = self.prompt_template.template
        return prompt

    def summarize_single(self, sample: EvalSample) -> EvalSample:
        prompt = self.format_prompt(sample)
        sample.predicted_summary = self.technique.generate(
            self.model, prompt, sample.context,
        )
        if hasattr(self.technique, "last_prompt_instruction"):
            sample.instruction = self.technique.last_prompt_instruction
        return sample

    def run(
        self,
        samples: List[EvalSample],
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        save_every: int = 10,
    ) -> List[EvalSample]:
        if max_samples is not None:
            samples = samples[:max_samples]

        # Resume support — skip completed samples
        completed_ids = set()
        existing_results = []
        if output_path and Path(output_path).exists():
            with open(output_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get("predicted_summary"):
                        completed_ids.add(record["sample_id"])
                        existing_results.append(record)
            if completed_ids:
                logger.info(f"Resuming: {len(completed_ids)} already done")

        pending = [s for s in samples if s.sample_id not in completed_ids]
        logger.info(
            f"Summarization: {len(pending)} pending, "
            f"{len(completed_ids)} done, {len(samples)} total "
            f"(technique={self.technique.name})"
        )

        results: List[EvalSample] = []
        total_time = 0.0
        for i, sample in enumerate(tqdm(pending, desc="Generating")):
            t0 = time.time()
            try:
                results.append(self.summarize_single(sample))
            except Exception as e:
                logger.error(f"Failed on {sample.sample_id}: {e}")
                sample.predicted_summary = f"[ERROR] {e}"
                results.append(sample)

            elapsed = time.time() - t0
            total_time += elapsed
            sample.metadata["inference_time_s"] = round(elapsed, 2)
            logger.info(f"{sample.sample_id}: {elapsed:.1f}s (avg {total_time/(i+1):.1f}s)")

            if output_path and (i + 1) % save_every == 0:
                self._save_results(existing_results, results, output_path)

        if output_path:
            self._save_results(existing_results, results, output_path)
            logger.info(f"Saved {len(results)} results to {output_path}")

        return results

    def _save_results(self, existing: list, new: List[EvalSample], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in existing:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            for sample in new:
                record = {
                    "sample_id": sample.sample_id,
                    "context": sample.context,
                    "predicted_summary": sample.predicted_summary,
                    "labeled_summary": sample.labeled_summary,
                    "instruction": sample.instruction,
                    **sample.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
