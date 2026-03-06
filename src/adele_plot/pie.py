from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_hex

from adele_plot.types import PieMatrix, PlotResult
from adele_plot.validation import (
    normalize_demand_level,
    resolve_dataframe_capabilities,
    validate_levels,
    validate_non_negative,
)

DEFAULT_CAPABILITIES: tuple[str, ...] = (
    "AS",
    "CEc",
    "CEe",
    "CL",
    "MCr",
    "MCt",
    "MCu",
    "MS",
    "QLl",
    "QLq",
    "SNs",
    "KNa",
    "KNc",
    "KNf",
    "KNn",
    "KNs",
    "AT",
    "VO",
)
DEFAULT_LEVELS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5+")
START_ANGLE = 90.0
RING_WIDTH = 0.16666
SLICE_EDGE_COLOR = "white"
RING_EDGE_COLOR = "black"
EDGE_LINEWIDTH = 0.3
FG_COLOR = (0.122, 0.467, 0.706)
BG_COLOR = (1.0, 1.0, 1.0)
ALPHA_PERCENTILE = (5.0, 95.0)
ALPHA_GAMMA = 0.6
ALPHA_RANGE = (0.0, 0.95)
TITLE_PAD = 30.0
LABEL_RADIUS = 1.12
LABEL_FONTSIZE = 10.0
RING_FONTSIZE = 8.0
RING_LABEL_X_OFFSET = -0.03
AXIS_LIMIT = 1.2
COLORBAR_GAMMA = 0.6

PlotInput: TypeAlias = PieMatrix | pd.DataFrame | np.ndarray


def compute_alpha(
    matrix: np.ndarray,
    *,
    percentile: tuple[float, float] = (5.0, 95.0),
    gamma: float = 0.6,
    alpha_range: tuple[float, float] = (0.0, 0.95),
) -> np.ndarray:
    p_low, p_high = np.percentile(matrix.reshape(-1), percentile)

    denominator = p_high - p_low
    if denominator <= 1e-12:
        return np.full_like(matrix, alpha_range[0], dtype=float)

    x = np.clip((matrix.reshape(-1) - p_low) / denominator, 0.0, 1.0)
    a_min, a_max = alpha_range
    alpha = a_min + (a_max - a_min) * (x**gamma)
    return alpha.reshape(matrix.shape)


def alpha_to_hex(
    alpha: float,
    *,
    fg: tuple[float, float, float] = FG_COLOR,
    bg: tuple[float, float, float] = BG_COLOR,
) -> str:
    composite = tuple((1.0 - alpha) * b + alpha * f for f, b in zip(fg, bg))
    return to_hex(composite)


def prepare_pie_matrix(
    data: pd.DataFrame | np.ndarray,
    *,
    capabilities: Sequence[str] | None = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
) -> PieMatrix:
    levels = validate_levels(levels)
    if isinstance(data, pd.DataFrame):
        resolved_capabilities = resolve_dataframe_capabilities(
            data,
            requested_capabilities=capabilities,
            default_capabilities=DEFAULT_CAPABILITIES,
        )
        matrix = np.zeros((len(levels), len(resolved_capabilities)), dtype=float)
        level_to_index = {level: idx for idx, level in enumerate(levels)}
        for capability_idx, capability in enumerate(resolved_capabilities):
            series = data[capability].dropna()
            for value in series:
                normalized_level = normalize_demand_level(value)
                if normalized_level not in level_to_index:
                    supported_levels = ", ".join(levels)
                    raise ValueError(
                        f"Unsupported demand level '{normalized_level}' in column '{capability}'. "
                        f"Supported levels: {supported_levels}"
                    )
                matrix[level_to_index[normalized_level], capability_idx] += 1.0
        return PieMatrix(
            matrix=matrix,
            capabilities=resolved_capabilities,
            levels=levels,
        )

    cast_matrix = np.asarray(data, dtype=float)
    if cast_matrix.ndim != 2:
        raise ValueError("Matrix input must be a 2D array.")
    if cast_matrix.shape[0] != len(levels):
        raise ValueError(
            f"Matrix row count must equal number of levels ({len(levels)}). "
            f"Got {cast_matrix.shape[0]}."
        )
    if capabilities is None:
        if cast_matrix.shape[1] <= len(DEFAULT_CAPABILITIES):
            resolved_capabilities = DEFAULT_CAPABILITIES[: cast_matrix.shape[1]]
        else:
            raise ValueError(
                "Matrix has more columns than DEFAULT_CAPABILITIES. "
                "Pass capability names explicitly via `capabilities`."
            )
    else:
        resolved_capabilities = tuple(capabilities)
    if len(resolved_capabilities) != cast_matrix.shape[1]:
        raise ValueError(
            "Number of capability names must match matrix columns. "
            f"Got {len(resolved_capabilities)} names for {cast_matrix.shape[1]} columns."
        )
    validate_non_negative(cast_matrix)
    return PieMatrix(
        matrix=cast_matrix,
        capabilities=resolved_capabilities,
        levels=levels,
    )


def plot_response_pie(
    data: PlotInput,
    *,
    capabilities: Sequence[str] | None = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
    ax=None,
    fig=None,
    add_colorbar: bool = False,
    title: str | None = None,
) -> PlotResult:
    resolved = (
        data
        if isinstance(data, PieMatrix)
        else prepare_pie_matrix(
            data,
            capabilities=capabilities,
            levels=levels,
        )
    )

    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    matrix = resolved.matrix
    alpha = compute_alpha(
        matrix,
        percentile=ALPHA_PERCENTILE,
        gamma=ALPHA_GAMMA,
        alpha_range=ALPHA_RANGE,
    )

    n_levels, n_caps = alpha.shape
    wedgeprops = dict(
        width=RING_WIDTH,
        edgecolor=SLICE_EDGE_COLOR,
        linewidth=EDGE_LINEWIDTH,
    )

    for ring_idx, level_idx in enumerate(range(n_levels - 1, -1, -1)):
        radius = 1.0 - ring_idx * RING_WIDTH
        colors = [alpha_to_hex(a, fg=FG_COLOR, bg=BG_COLOR) for a in alpha[level_idx]]
        ax.pie(
            [1] * n_caps,
            radius=radius,
            colors=colors,
            wedgeprops=wedgeprops,
            startangle=START_ANGLE,
            counterclock=False,
        )

    boundary_radii = [1.0 - idx * RING_WIDTH for idx in range(n_levels + 1)]
    for radius in boundary_radii:
        if radius <= 0:
            continue
        circle = plt.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            edgecolor=RING_EDGE_COLOR,
            linewidth=EDGE_LINEWIDTH,
            zorder=5,
        )
        ax.add_artist(circle)

    labels = list(resolved.capabilities)
    sector_width = 360.0 / len(labels)
    theta = START_ANGLE - (np.arange(len(labels)) + 0.5) * sector_width
    for ang_deg, label in zip(theta, labels):
        ang_rad = np.deg2rad(ang_deg)
        x = LABEL_RADIUS * np.cos(ang_rad)
        y = LABEL_RADIUS * np.sin(ang_rad)
        ax.text(
            x,
            y,
            label,
            va="center",
            ha="center",
            rotation=0.0,
            rotation_mode="anchor",
            fontsize=LABEL_FONTSIZE,
            color="black",
        )

    if labels:
        ring_texts = tuple(reversed(resolved.levels))
        for idx, ring_text in enumerate(ring_texts):
            radius = 1.0 - (idx + 0.5) * RING_WIDTH
            x = RING_LABEL_X_OFFSET
            y = radius
            ax.text(
                x,
                y,
                ring_text,
                ha="right",
                va="center",
                fontsize=RING_FONTSIZE,
            )

    ax.set(aspect="equal")
    if title:
        ax.set_title(title, pad=TITLE_PAD)
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)

    if add_colorbar:
        cmap = LinearSegmentedColormap.from_list("white_to_blue", [BG_COLOR, FG_COLOR])
        p_low, p_high = np.percentile(matrix.reshape(-1), ALPHA_PERCENTILE)
        if abs(p_high - p_low) <= 1e-12:
            p_high = p_low + 1.0
        norm = PowerNorm(gamma=COLORBAR_GAMMA, vmin=p_low, vmax=p_high)
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            pad=0.06,
            fraction=0.05,
            shrink=0.90,
        )
        colorbar.set_label("Count per (demand, level)")

    return PlotResult(fig=fig, ax=ax, matrix=matrix, alpha=alpha)
