from __future__ import annotations

import numpy as np

from adele_plot.pie import compute_alpha


def test_compute_alpha_bounds() -> None:
    matrix = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    alpha = compute_alpha(matrix, percentile=(0, 100), gamma=1.0, alpha_range=(0.0, 1.0))
    assert alpha.shape == matrix.shape
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= 1.0)


def test_compute_alpha_constant_matrix() -> None:
    matrix = np.full((2, 3), 5.0)
    alpha = compute_alpha(matrix, percentile=(5, 95), gamma=0.6, alpha_range=(0.0, 0.95))
    assert np.all(alpha == 0.0)
