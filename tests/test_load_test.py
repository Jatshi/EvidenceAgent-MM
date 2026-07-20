from __future__ import annotations

import pytest

from evidenceagent_mm.evaluation import percentile


def test_load_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_load_percentile_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1.0], 101)
