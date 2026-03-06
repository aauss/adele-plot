# adele-plot

Reusable plotting package for ADeLe radial response plots.

## Install

```bash
pip install adele-plot
```

## Input format

`plot_response_pie(...)` accepts either:

- wide `pandas.DataFrame` where each capability is a column and each cell is a demand level, or
- a pre-aggregated matrix with shape `(6, n_capabilities)` (`levels x capabilities`).

## Quickstart

```python
from adele_plot import DEFAULT_CAPABILITIES, plot_response_pie

result = plot_response_pie(df, title="ADeLe Response Pie", add_colorbar=True)
```

Default capabilities are:

```python
DEFAULT_CAPABILITIES = (
    "AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl",
    "QLq", "SNs", "KNa", "KNc", "KNf", "KNn", "KNs", "AT", "VO",
)
```

For DataFrame input, if `capabilities` is not provided, the function automatically uses
the intersection of DataFrame columns and `DEFAULT_CAPABILITIES` in that exact order.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m build
twine check dist/*
```

