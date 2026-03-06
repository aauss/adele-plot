from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adele_plot import DEFAULT_CAPABILITIES, plot_response_pie


def main() -> None:
    capabilities = DEFAULT_CAPABILITIES

    rng = np.random.default_rng(seed=7)
    rows = []
    for _ in range(120):
        row = {}
        for capability in capabilities:
            row[capability] = int(rng.integers(0, 8))
        rows.append(row)
    df = pd.DataFrame(rows, columns=capabilities)

    result = plot_response_pie(df, title="ADeLe Response Pie", add_colorbar=True)
    result.fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
