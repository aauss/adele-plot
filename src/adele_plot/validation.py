from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def normalize_demand_level(value: object) -> str:
    """Normalize scalar demand levels into the canonical labels."""
    if isinstance(value, (int, np.integer)):
        as_int = int(value)
        return "5+" if as_int >= 5 else str(as_int)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            as_int = int(value)
            return "5+" if as_int >= 5 else str(as_int)
        raise ValueError(f"Invalid non-integer demand level: {value}")
    text = str(value).strip()
    if text.isdigit():
        as_int = int(text)
        return "5+" if as_int >= 5 else str(as_int)
    if text in {"5+", "5 +"}:
        return "5+"
    return text


def resolve_dataframe_capabilities(
    df: pd.DataFrame,
    *,
    requested_capabilities: Sequence[str] | None,
    default_capabilities: tuple[str, ...],
) -> tuple[str, ...]:
    if requested_capabilities is not None:
        missing = [
            column for column in requested_capabilities if column not in df.columns
        ]
        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(
                f"Requested capability columns are missing: {missing_joined}"
            )
        return tuple(requested_capabilities)

    chosen = tuple(
        capability for capability in default_capabilities if capability in df.columns
    )
    if not chosen:
        raise ValueError(
            "No capability columns found in DataFrame. "
            "Expected at least one default capability column."
        )
    return chosen


def validate_non_negative(matrix: np.ndarray) -> None:
    if np.any(matrix < 0):
        raise ValueError("Matrix values must be non-negative.")


def validate_levels(levels: Sequence[str]) -> tuple[str, ...]:
    if not levels:
        raise ValueError("Levels cannot be empty.")
    return tuple(levels)
