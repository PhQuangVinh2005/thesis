"""Tests for data loaders."""

import pytest
from src.data.base import BaseDataLoader


class TestBaseDataLoader:
    def test_cannot_instantiate_abstract(self):
        """BaseDataLoader is abstract and should not be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataLoader(data_dir="test/")

    def test_repr(self):
        """Concrete subclass should have readable repr."""
        # TODO: Test with concrete implementation after dataset finalization
        pass
