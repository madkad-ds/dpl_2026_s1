"""
Presentation style: one palette, slide-sized figures, and an export that keeps
text out of the image.

Conventions the figures follow:
  - 2023 in grey-blue, 2024 in magenta; everything that is not a season or the
    quantity being emphasised is in greys.
  - Filled markers clear the null, hollow markers lean (the interval reaches the null).
  - No titles or captions inside the image. Every figure function accepts a title
    for notebook viewing, but the export writes the image without it and a JSON
    beside it carrying the title, caption and any other text, so slide text is
    edited in the slide software.
  - Figures are drawn at the physical size they will occupy on a 16:9 slide, so
    point sizes are slide point sizes and nothing is scaled afterwards.
Exports: vector PDF (Keynote and PowerPoint place it cleanly) and a 300 dpi PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

SEASON = {"earlier": "#7f8c9a", "later": "#d6336c"}
INK = "#1f2937"
GREY = "#b8bec6"
GREY_DARK = "#4b5563"
GREY_TEXT = "#6a6a6a"
LIGHT = "#e6e8eb"
PALE = "#f6f7f9"
ACCENT_PALE = "#fbe3ec"
GREYS = ["#c9cdd2", "#dfe2e6", "#a7aeb6", "#eceef1"]

# Physical sizes in inches for a 13.33 x 7.5 inch (16:9) slide.
SIZES = {"half": (6.2, 4.6), "half_tall": (6.2, 5.4), "third": (4.0, 4.6), "full": (12.5, 5.2), "wide": (8.5, 4.6), "small": (6.2, 2.6)}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa0a6",
        "axes.labelsize": 11, "axes.labelcolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "xtick.labelsize": 10, "ytick.labelsize": 10.5, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "legend.fontsize": 10,
        "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def new_figure(size: str = "half", **kwargs):
    apply_style()
    w, h = SIZES[size]
    fig, ax = plt.subplots(figsize=(w, h), **kwargs)
    return fig, ax


def tidy(ax):
    ax.tick_params(length=0)
    ax.grid(False)


def season_colour(season: int, earlier: int) -> str:
    return SEASON["earlier"] if season == earlier else SEASON["later"]


def export(fig, name: str, folder: Path, text: dict | None = None, title: str | None = None) -> dict:
    """Write name.pdf and name.png without any title, and name.text.json with the
    figure's text. If the figure was drawn with a title for notebook viewing, it is
    removed before export."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_title("")
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    payload = {"figure": name, "title": title, **(text or {})}
    (folder / f"{name}.text.json").write_text(json.dumps(payload, indent=2))
    return payload
