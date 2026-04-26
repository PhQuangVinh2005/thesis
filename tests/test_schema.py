"""Tests for data schemas."""

import pytest
from src.data.schema import MedicalRecord, EvalSample, ExperimentResult


class TestMedicalRecord:
    def test_create_record(self):
        record = MedicalRecord(record_id="test_001", context="Patient admitted for...")
        assert record.record_id == "test_001"
        assert record.context == "Patient admitted for..."
        assert record.metadata == {}

    def test_create_record_with_metadata(self):
        record = MedicalRecord(
            record_id="test_002",
            context="...",
            metadata={"subject_id": 12345, "hadm_id": 67890},
        )
        assert record.metadata["subject_id"] == 12345


class TestEvalSample:
    def test_create_sample(self):
        sample = EvalSample(sample_id="s1", context="Patient was admitted...")
        assert sample.sample_id == "s1"
        assert not sample.has_prediction
        assert not sample.has_ground_truth

    def test_sample_with_prediction(self):
        sample = EvalSample(
            sample_id="s2",
            context="...",
            predicted_summary="Summary text",
        )
        assert sample.has_prediction
        assert not sample.has_ground_truth

    def test_sample_complete(self):
        sample = EvalSample(
            sample_id="s3",
            context="Full record...",
            instruction="Summarize...",
            predicted_summary="AI summary",
            labeled_summary="Doctor summary",
        )
        assert sample.has_prediction
        assert sample.has_ground_truth


class TestExperimentResult:
    def test_create_result(self):
        result = ExperimentResult(
            experiment_name="baseline_v1",
            model_name="BioMistral-7B",
            prompt_name="baseline",
        )
        assert result.experiment_name == "baseline_v1"
        assert result.scores == {}
        assert result.num_samples == 0
