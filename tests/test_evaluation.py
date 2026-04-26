"""Tests for evaluation metrics."""

import pytest
from src.evaluation.base import BaseMetric

try:
    from src.evaluation.completeness.rouge import ROUGEMetric
    from src.evaluation.completeness.bleu import BLEUMetric
    HAS_EVAL_DEPS = True
except ImportError:
    HAS_EVAL_DEPS = False
    ROUGEMetric = None  # type: ignore
    BLEUMetric = None   # type: ignore


# ─── Sample data ───

PREDICTION = (
    "Patient is a 65-year-old male admitted for chest pain. "
    "Diagnosed with acute myocardial infarction. Started on aspirin and heparin."
)
REFERENCE = (
    "A 65-year-old male presented with chest pain. He was diagnosed with "
    "acute myocardial infarction and treated with aspirin, heparin, and nitroglycerin."
)
CONTEXT = (
    "HISTORY OF PRESENT ILLNESS: 65-year-old male presents to ED with severe "
    "substernal chest pain radiating to left arm. ECG shows ST elevation in leads "
    "II, III, aVF. Troponin I elevated at 2.5 ng/mL. Diagnosis: STEMI. "
    "Started on aspirin 325mg, heparin drip, nitroglycerin."
)


class TestBaseMetric:
    def test_cannot_instantiate_abstract(self):
        """BaseMetric is abstract and should not be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric(name="test")


# ─── Completeness tests (P vs L) ───


@pytest.mark.skipif(not HAS_EVAL_DEPS, reason="rouge-score not installed")
class TestROUGEMetric:
    def test_requires_reference(self):
        metric = ROUGEMetric()
        with pytest.raises(ValueError):
            metric.compute(prediction="test summary", reference=None)

    def test_compute_returns_9_keys(self):
        metric = ROUGEMetric()
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        expected_keys = {
            "rouge1_precision", "rouge1_recall", "rouge1_f",
            "rouge2_precision", "rouge2_recall", "rouge2_f",
            "rougeL_precision", "rougeL_recall", "rougeL_f",
        }
        assert set(scores.keys()) == expected_keys

    def test_scores_in_range(self):
        metric = ROUGEMetric()
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"

    def test_identical_texts_high_score(self):
        metric = ROUGEMetric()
        scores = metric.compute(prediction=REFERENCE, reference=REFERENCE)
        assert scores["rouge1_f"] > 0.99
        assert scores["rougeL_f"] > 0.99


@pytest.mark.skipif(not HAS_EVAL_DEPS, reason="sacrebleu not installed")
class TestBLEUMetric:
    def test_requires_reference(self):
        metric = BLEUMetric()
        with pytest.raises(ValueError):
            metric.compute(prediction="test", reference=None)

    def test_compute_returns_bleu_score(self):
        metric = BLEUMetric()
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        assert "bleu_score" in scores

    def test_score_in_range(self):
        metric = BLEUMetric()
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        assert 0.0 <= scores["bleu_score"] <= 100.0

    def test_identical_texts_high_score(self):
        metric = BLEUMetric()
        scores = metric.compute(prediction=REFERENCE, reference=REFERENCE)
        assert scores["bleu_score"] > 90.0


class TestBERTScoreMetric:
    """BERTScore tests — only run if bert-score is installed."""

    @pytest.fixture
    def metric(self):
        try:
            from src.evaluation.completeness.bert_score import BERTScoreMetric
            return BERTScoreMetric()
        except ImportError:
            pytest.skip("bert-score not installed")

    def test_requires_reference(self, metric):
        with pytest.raises(ValueError):
            metric.compute(prediction="test", reference=None)

    def test_compute_returns_3_keys(self, metric):
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        assert "bertscore_precision" in scores
        assert "bertscore_recall" in scores
        assert "bertscore_f1" in scores

    def test_scores_in_range(self, metric):
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


class TestMEDCONMetric:
    """MEDCON tests — only run if scispacy + en_core_sci_lg are installed."""

    @pytest.fixture
    def metric(self):
        try:
            from src.evaluation.completeness.medcon import MEDCONMetric
            m = MEDCONMetric()
            m._get_nlp()
            return m
        except (ImportError, OSError):
            pytest.skip("scispacy or en_core_sci_lg not installed")

    def test_requires_reference(self, metric):
        with pytest.raises(ValueError):
            metric.compute(prediction="test", reference=None)

    def test_compute_returns_3_keys(self, metric):
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        assert "medcon_precision" in scores
        assert "medcon_recall" in scores
        assert "medcon_f1" in scores

    def test_scores_in_range(self, metric):
        scores = metric.compute(prediction=PREDICTION, reference=REFERENCE)
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"

    def test_identical_texts_perfect_score(self, metric):
        scores = metric.compute(prediction=REFERENCE, reference=REFERENCE)
        assert scores["medcon_f1"] == 1.0


# ─── Faithfulness tests (P vs C) ───


class TestFENICEMetric:
    """FENICE tests — only run if FENICE is installed."""

    @pytest.fixture
    def metric(self):
        try:
            import fenice  # noqa: F401
            from src.evaluation.faithfulness.fenice import FENICEMetric
            return FENICEMetric()
        except ImportError:
            pytest.skip("FENICE not installed")

    def test_requires_context(self, metric):
        with pytest.raises(ValueError):
            metric.compute(prediction="test", context=None)

    def test_compute_returns_score(self, metric):
        scores = metric.compute(prediction=PREDICTION, context=CONTEXT)
        assert "fenice_score" in scores

    def test_score_in_range(self, metric):
        scores = metric.compute(prediction=PREDICTION, context=CONTEXT)
        assert 0.0 <= scores["fenice_score"] <= 1.0


class TestAlignScoreMetric:
    """AlignScore tests — only run if alignscore is installed."""

    @pytest.fixture
    def metric(self):
        try:
            from src.evaluation.faithfulness.alignscore import AlignScoreMetric
            return AlignScoreMetric()
        except ImportError:
            pytest.skip("AlignScore not installed")

    def test_requires_context(self, metric):
        with pytest.raises(ValueError):
            metric.compute(prediction="test", context=None)






# ─── Pipeline integration test ───


@pytest.mark.skipif(not HAS_EVAL_DEPS, reason="evaluation deps not installed")
class TestEvaluationPipeline:
    """Integration test for EvaluationPipeline with lightweight metrics."""

    def test_run_completeness(self, tmp_path):
        from src.data.schema import EvalSample
        from src.pipelines.evaluator import EvaluationPipeline

        samples = [
            EvalSample(
                sample_id="test-1",
                context=CONTEXT,
                predicted_summary=PREDICTION,
                labeled_summary=REFERENCE,
            ),
            EvalSample(
                sample_id="test-2",
                context="Another medical record",
                predicted_summary="Patient admitted for surgery.",
                labeled_summary="Patient underwent surgical procedure.",
            ),
        ]

        metrics = [ROUGEMetric(), BLEUMetric()]
        pipeline = EvaluationPipeline(metrics=metrics)

        result = pipeline.run(
            samples=samples,
            output_dir=str(tmp_path),
            model_name="test-model",
            range_id="0_1k",
        )

        assert result.num_samples == 2
        assert "rouge1_f" in result.scores
        assert "bleu_score" in result.scores

        # Check output files
        assert (tmp_path / "eval_scores.jsonl").exists()
        assert (tmp_path / "eval_summary.json").exists()

        # Verify eval_summary.json structure
        import json
        with open(tmp_path / "eval_summary.json") as f:
            summary = json.load(f)
        assert summary["model"] == "test-model"
        assert summary["range"] == "0_1k"
        assert summary["num_samples"] == 2
        assert "time_taken_seconds" in summary

    def test_run_with_custom_filenames(self, tmp_path):
        """Test that phase-specific filenames work."""
        from src.data.schema import EvalSample
        from src.pipelines.evaluator import EvaluationPipeline

        samples = [
            EvalSample(
                sample_id="test-1",
                context=CONTEXT,
                predicted_summary=PREDICTION,
                labeled_summary=REFERENCE,
            ),
        ]

        pipeline = EvaluationPipeline(metrics=[ROUGEMetric()])
        pipeline.run(
            samples=samples,
            output_dir=str(tmp_path),
            scores_filename="faith_scores.jsonl",
            summary_filename="faith_summary.json",
        )

        assert (tmp_path / "faith_scores.jsonl").exists()
        assert (tmp_path / "faith_summary.json").exists()
