from __future__ import annotations

import matplotlib
import numpy as np

from adele_plot import DEFAULT_CAPABILITIES, plot_response_pie

matplotlib.use("Agg")


def test_plot_response_pie_smoke() -> None:
    matrix = np.arange(108, dtype=float).reshape(6, 18)
    result = plot_response_pie(matrix, capabilities=DEFAULT_CAPABILITIES, title="Smoke")
    assert result.matrix.shape == (6, 18)
    assert result.alpha.shape == (6, 18)
    assert result.ax.get_title() == "Smoke"
