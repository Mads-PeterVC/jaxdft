"""Quarto post-render script: build the milestone ladder on the front page.

Reads ``milestones.yml`` (what the milestones are) and ``_results/`` (the
figure and headline numbers each validation document published during the
render) and writes the ladder into the rendered ``index.html``. It runs no
calculation. Running it twice gives the same page.

A full render fails if a completed milestone's validation document was
rendered but published no figure, or if a linked document does not exist.
"""

import datetime
import html
import json
import os
import re
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent
SITE = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", DOCS / "_site"))
if not SITE.is_absolute():
    SITE = DOCS / SITE
RESULTS = DOCS / "_results"
STATS_PAGE = "engineering/project_stats.html"
STATUS_LABEL = {"done": "Done", "in-progress": "In progress"}


def fail(message):
    sys.exit(f"front_page.py: {message}")


def load_milestones():
    milestones = yaml.safe_load((DOCS / "milestones.yml").read_text())
    for m in milestones:
        missing = {"id", "title", "capability", "status"} - m.keys()
        if missing:
            fail(f"milestone {m.get('id', '?')} lacks {sorted(missing)}")
        if m["status"] not in STATUS_LABEL:
            fail(f"milestone {m['id']}: unknown status {m['status']!r}")
        if m["status"] == "done" and not isinstance(
            m.get("reached"), datetime.date
        ):
            fail(f"milestone {m['id']} is done but has no 'reached' date")
    return milestones


def rendered_this_run():
    """Output files of this render, as paths relative to the site."""
    listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
    return {
        Path(line).resolve().relative_to(SITE.resolve()).as_posix()
        for line in listed.splitlines()
        if line.strip() and Path(line).resolve().is_relative_to(SITE.resolve())
    }


def page(source):
    """Site path of the page rendered from a document path in milestones.yml."""
    if not (DOCS / source).exists():
        fail(f"{source} is listed in milestones.yml but does not exist")
    return Path(source).with_suffix(".html").as_posix()


def figure(m, rendered):
    result = RESULTS / f"milestone_{m['id']}.json"
    if not result.exists():
        if page(m["validation"]) in rendered:
            fail(
                f"{m['validation']} was rendered but published no figure for "
                f"milestone {m['id']} (jaxdft.plotting.featured.publish)"
            )
        return (
            '<p class="milestone-pending">Render the validation document to '
            "see its figure here.</p>"
        )
    data = json.loads(result.read_text())
    numbers = "".join(
        f"<div><dt>{html.escape(h['label'])}</dt>"
        f"<dd>{html.escape(h['value'])}</dd></div>"
        for h in data["headline"]
    )
    return (
        f'<figure class="milestone-figure">{data["html"]}'
        f"<figcaption>{html.escape(data['caption'])}</figcaption></figure>"
        + (f'<dl class="milestone-headline">{numbers}</dl>' if numbers else "")
    )


def entry(m, rendered):
    status = m["status"]
    when = (
        f'<time datetime="{m["reached"].isoformat()}">'
        f"{m['reached'].strftime('%-d %B %Y')}</time>"
        if status == "done"
        else ""
    )
    links = [
        f'<a href="{page(m[kind])}">{label}</a>'
        for kind, label in [("theory", "Theory"), ("validation", "Validation")]
        if kind in m
    ]
    body = figure(m, rendered) if "validation" in m else ""
    return (
        f'<li class="milestone milestone-{status}">'
        '<div class="milestone-text">'
        f'<p class="milestone-meta"><span class="milestone-status">'
        f"{STATUS_LABEL[status]}</span>{when}</p>"
        f"<h3>{html.escape(m['title'])}</h3>"
        f"<p>{html.escape(m['capability'])}</p>"
        + (
            f'<p class="milestone-links">{" · ".join(links)}</p>'
            if links
            else ""
        )
        + f'</div><div class="milestone-body">{body}</div></li>'
    )


def effort_strip():
    link = f'<a href="{STATS_PAGE}">Progress in numbers</a>'
    stats = RESULTS / "project_stats.json"
    if not (SITE / STATS_PAGE).exists():
        return ""
    if not stats.exists():
        return link
    numbers = " · ".join(
        f"{html.escape(str(item['value']))} {html.escape(item['label'])}"
        for item in json.loads(stats.read_text())["strip"]
    )
    return f"{numbers} · {link}"


def fill(text, name, content):
    pattern = re.compile(
        rf'(<div id="{name}">).*?(</div><!--/{name}-->)', re.DOTALL
    )
    if not pattern.search(text):
        fail(f"index.html has no '{name}' placeholder")
    return pattern.sub(lambda match: match[1] + content + match[2], text)


def main():
    index = SITE / "index.html"
    if not index.exists():
        return
    rendered = rendered_this_run()
    ladder = "".join(entry(m, rendered) for m in load_milestones())
    text = index.read_text()
    text = fill(
        text, "milestone-ladder", f'<ol class="milestones">{ladder}</ol>'
    )
    text = fill(text, "effort-strip", effort_strip())
    index.write_text(text)


main()
