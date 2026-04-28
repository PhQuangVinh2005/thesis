"""Chain-of-Verification (CoVe) technique for hallucination reduction.

Implements the "Factor + Revise" variant from Dhuliawala et al. (2023).
3 LLM calls per sample:
  1. DRAFT   — baseline generation (may contain hallucinations)
  2. PLAN    — generate verification questions from draft (sees draft)
  3. VERIFY+REFINE — answer questions from source only (no draft), then rewrite

Option C: plan with draft visible, verify without draft visible.
"""

import logging
import re

from ..models.base import BaseLLM
from ..prompts.templates import PromptTemplate
from .base import BaseTechnique

logger = logging.getLogger(__name__)


class CoVeTechnique(BaseTechnique):
    """Chain-of-Verification: draft → plan → verify+refine.

    Args:
        plan_template: Prompt for generating verification questions (sees draft).
        verify_refine_template: Prompt for answering questions + rewriting (no draft).
        n_questions: Number of verification questions to generate.
    """

    def __init__(
        self,
        plan_template: PromptTemplate,
        verify_refine_template: PromptTemplate,
        n_questions: int = 5,
    ):
        super().__init__(name="cove")
        self.plan_template = plan_template
        self.verify_refine_template = verify_refine_template
        self.n_questions = n_questions
        logger.info(f"CoVeTechnique: n_questions={n_questions}")

    def generate(self, model: BaseLLM, prompt: str, context: str) -> str:
        """Execute the 3-step CoVe pipeline."""
        # Step 1: DRAFT — use pipeline's pre-formatted baseline prompt
        logger.info("CoVe step 1/3: drafting initial summary")
        draft = model.generate(prompt)
        logger.info(f"Draft: {len(draft)} chars")

        # Step 2: PLAN — generate verification questions (model sees draft + context)
        logger.info("CoVe step 2/3: planning verification questions")
        plan_prompt = self.plan_template.format(
            draft=draft,
            context=context,
            n_questions=self.n_questions,
        )
        questions = model.generate(plan_prompt)
        logger.info(f"Questions: {len(questions)} chars")

        # Step 3: VERIFY + REFINE — answer questions from source, then rewrite
        # The model does NOT see the draft (Option C)
        logger.info("CoVe step 3/3: verifying + refining")
        verify_prompt = self.verify_refine_template.format(
            context=context,
            questions=questions,
        )
        raw_output = model.generate(verify_prompt)

        # Extract the summary from the combined verify+refine output
        final_summary = self._extract_summary(raw_output)
        logger.info(f"Final summary: {len(final_summary)} chars")

        # Store metadata for logging
        self.last_prompt_instruction = (
            f"CoVe (n_questions={self.n_questions}): "
            f"draft({len(draft)}c) → plan({len(questions)}c) → verify+refine({len(final_summary)}c)"
        )
        self.last_draft = draft
        self.last_questions = questions
        self.last_verification = raw_output

        return final_summary

    def _extract_summary(self, raw_output: str) -> str:
        """Extract the corrected summary from verify+refine output.

        Tries section markers first, then heuristic fallbacks.
        Logs a warning if extraction quality is uncertain.
        """
        # Primary: look for explicit section markers (handles markdown variants)
        markers = [
            r"#+\s*PART\s*2[^a-zA-Z]*(?:CORRECTED|VERIFIED|REVISED)?\s*(?:SUMMARY)?[^a-zA-Z]*\n+",
            r"\*{0,2}(?:Verified|Corrected|Final|Revised)\s*[Ss]ummary\s*\*{0,2}\s*[:：]\s*",
            r"(?:Verified|Corrected|Final|Revised)\s*[Ss]ummary\s*\*{0,2}\s*[:：]?\s*\*{0,2}\s*\n",
            r"CORRECTED\s+SUMMARY\s*\*{0,2}\s*\n",
            r"PART\s*2\s*[:：—–-]\s*",
            r"Brief\s+Hospital\s+Course\s*[:：]\s*",
        ]
        for pattern in markers:
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                summary = raw_output[match.end():].strip()
                if len(summary) > 50:
                    return summary

        # Detect verification-only output (model never wrote a summary)
        verification_indicators = [
            r"^\s*\d+\.\s*(CONFIRMED|CONTRADICTED|UNVERIFIABLE)",
            r"^\s*1\.\s*\*\*",
        ]
        is_verification_only = any(
            re.match(p, raw_output.strip(), re.IGNORECASE)
            for p in verification_indicators
        )
        if is_verification_only:
            logger.warning(
                "Model output contains only verification Q&A — no summary section found. "
                "Returning raw output as fallback."
            )
            return raw_output.strip()

        # Fallback: if output is reasonably short, assume it's just the summary
        if len(raw_output) < 3000:
            return raw_output.strip()

        # Last resort: take everything after the last double newline
        parts = raw_output.rsplit("\n\n", 1)
        if len(parts) == 2 and len(parts[1]) > 50:
            return parts[1].strip()

        logger.warning("Could not cleanly extract summary from verify+refine output")
        return raw_output.strip()
