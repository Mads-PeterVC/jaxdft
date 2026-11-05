"""One altair theme for every chart in the docs.

Colours come from a validated categorical palette (CVD-checked adjacent pairs),
assigned in fixed order and never cycled. Use ``SERIES[i]`` by role, e.g. the
computed result is always ``SERIES[0]`` and its reference ``SERIES[1]``.

altair is a docs-only dependency, so it is imported lazily here and nothing in
the physics imports this module.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

WIDTH = 640
HEIGHT = 360


def theme() -> dict:
    """Vega-Lite config: thin marks, recessive axes and grid, text in text colours."""
    axis = {
        "domainColor": TEXT_SECONDARY,
        "tickColor": TEXT_SECONDARY,
        "gridColor": GRID,
        "labelColor": TEXT_SECONDARY,
        "titleColor": TEXT_PRIMARY,
        "labelFontSize": 12,
        "titleFontSize": 13,
        "titleFontWeight": "normal",
    }
    return {
        "config": {
            "background": SURFACE,
            "view": {"stroke": None, "continuousWidth": WIDTH, "continuousHeight": HEIGHT},
            "font": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
            "axis": axis,
            "legend": {"labelColor": TEXT_SECONDARY, "titleColor": TEXT_PRIMARY, "orient": "top"},
            "title": {"color": TEXT_PRIMARY, "fontSize": 14, "fontWeight": "normal", "anchor": "start"},
            "range": {"category": list(SERIES)},
            "line": {"strokeWidth": 2},
            "point": {"size": 64, "filled": True},
        }
    }


def enable() -> None:
    """Register and enable the jaxdft altair theme. Call once at the top of a document."""
    import altair as alt

    alt.theme.register("jaxdft", enable=True)(theme)
    # Band structures are a few thousand points; inline them rather than erroring.
    alt.data_transformers.disable_max_rows()
