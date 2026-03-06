from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adele_plot import DEFAULT_CAPABILITIES, prepare_pie_matrix


def test_prepare_pie_matrix_uses_default_capability_intersection() -> None:
    df = pd.DataFrame(
        {
            "AS": [0, 1, 5, 6],
            "KNn": [1, 1, 4, 5],
            "OTHER": [9, 9, 9, 9],
        }
    )
    data = prepare_pie_matrix(df)
    assert data.capabilities == ("AS", "KNn")
    assert data.matrix.shape == (6, 2)
    assert data.matrix[0, 0] == 1.0
    assert data.matrix[1, 0] == 1.0
    assert data.matrix[5, 0] == 2.0
    assert data.matrix[1, 1] == 2.0
    assert data.matrix[4, 1] == 1.0
    assert data.matrix[5, 1] == 1.0


def test_prepare_pie_matrix_matrix_mode() -> None:
    matrix = np.ones((6, 3), dtype=float)
    data = prepare_pie_matrix(
        matrix,
        capabilities=("AS", "KNn", "VO"),
        levels=("0", "1", "2", "3", "4", "5+"),
    )
    assert data.matrix.shape == (6, 3)
    assert data.capabilities == ("AS", "KNn", "VO")


def test_prepare_pie_matrix_matrix_default_capabilities_slice() -> None:
    matrix = np.zeros((6, 4), dtype=float)
    data = prepare_pie_matrix(matrix)
    assert data.capabilities == DEFAULT_CAPABILITIES[:4]


def test_prepare_pie_matrix_empty_intersection_raises() -> None:
    df = pd.DataFrame({"X": [0, 1], "Y": [2, 3]})
    with pytest.raises(ValueError, match="No capability columns found"):
        prepare_pie_matrix(df)


def test_prepare_pie_matrix_custom_capability_override() -> None:
    df = pd.DataFrame({"X": [0, 1, 2], "Y": [3, 4, 5]})
    data = prepare_pie_matrix(df, capabilities=("Y",))
    assert data.capabilities == ("Y",)
    assert data.matrix.shape == (6, 1)


def test_prepare_pie_matrix_invalid_level_raises() -> None:
    df = pd.DataFrame({"AS": [0, "bad", 2]})
    with pytest.raises(ValueError, match="Unsupported demand level"):
        prepare_pie_matrix(df)
