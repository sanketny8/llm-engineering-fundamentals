from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotStyle:
    title: str
    xlabel: str
    ylabel: str


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)


