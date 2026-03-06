from __future__ import annotations

from dataclasses import dataclass

import matplotlib.axes
import matplotlib.figure
import numpy as np


@dataclass(frozen=True)
class PieMatrix:
    matrix: np.ndarray
    capabilities: tuple[str, ...]
    levels: tuple[str, ...]


@dataclass(frozen=True)
class PlotResult:
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    matrix: np.ndarray
    alpha: np.ndarray
