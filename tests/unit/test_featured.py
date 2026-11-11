"""Publishing featured figures and numbers for the documentation site."""

import json

import altair as alt
import pandas as pd
import pytest

from jaxdft.plotting import featured


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "_quarto.yml").write_text("project:\n  type: website\n")
    document_dir = tmp_path / "validation"
    document_dir.mkdir()
    monkeypatch.chdir(document_dir)
    return tmp_path


def test_publish_writes_figure_and_headline_to_project_results(project):
    chart = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line().encode(x="x", y="y")
    path = featured.publish(chart, "9z", caption="A line.", headline=[("slope", "1.0")])

    assert path == project / "_results" / "milestone_9z.json"
    stored = json.loads(path.read_text())
    assert stored["milestone"] == "9z"
    assert stored["caption"] == "A line."
    assert stored["headline"] == [{"label": "slope", "value": "1.0"}]
    assert 'id="featured-9z"' in stored["html"]
    assert "<html" not in stored["html"]  # a fragment, to be placed inside a page


def test_publish_numbers_round_trips(project):
    path = featured.publish_numbers("stats", {"commits": 3, "unit": "Å"})
    assert json.loads(path.read_text()) == {"commits": 3, "unit": "Å"}


def test_results_dir_requires_a_quarto_project(tmp_path):
    with pytest.raises(FileNotFoundError):
        featured.results_dir(tmp_path)
