"""Excerpts of the development conversations, shown under docs/conversations/.

Three jobs, one per way of calling this file:

``conversations.py`` (no arguments; Quarto post-render script)
    Fill every ``<div class="transcript" data-excerpt="NAME">`` in the rendered
    conversation pages from ``conversations/data/NAME.json``. Running it twice
    gives the same page.

``conversations.py outline SESSION``
    List what the user wrote in a Claude Code session, with the line index of
    each message, to choose the ``lines`` of a new excerpt.

``conversations.py extract [NAME ...] [--force]``
    Read the excerpts listed in ``conversations/excerpts.yml`` out of the
    Claude Code session logs and store them as ``conversations/data/NAME.json``.
    Without names it extracts the excerpts that have no data file yet. An
    existing data file is only replaced with ``--force``.

The session logs live outside the repository, under ``~/.claude/projects/``,
and Claude Code deletes them after ``cleanupPeriodDays``. The data files are
the lasting record, so they are tracked by git.

An excerpt keeps what was said and a one-line summary of each tool call. It
leaves out tool output, the model's private reasoning, subagent transcripts
and harness messages. Message text is stored verbatim, except that paths on
the development machine are shortened (see ``REDACTIONS``).
"""

import argparse
import datetime
import html
import json
import os
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

DOCS = Path(__file__).resolve().parent.parent
SECTION = DOCS / "conversations"
DATA = SECTION / "data"
SPEC = SECTION / "excerpts.yml"

# Applied to all stored text. Order matters: most specific first.
REDACTIONS = [
    (re.compile(r"(?:/private)?/tmp/claude-\d+/[^/\s]+/[0-9a-f-]+/"), "/tmp/"),
    (re.compile(r"/Users/[^/\s]+/\.claude/projects/[^/\s]+/"), "~/.claude/"),
    (re.compile(re.escape(str(DOCS.parent)) + "/?"), ""),
    (re.compile(r"/Users/[^/\s]+/"), "~/"),
]
SUMMARY_KEYS = ("description", "file_path", "query", "skill", "url", "command")
SUMMARY_LENGTH = 120


def fail(message):
    sys.exit(f"conversations.py: {message}")


# --- reading session logs ---------------------------------------------------


def redact(text):
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def default_logs():
    """Where Claude Code keeps the session logs of this repository."""
    name = re.sub(r"[^A-Za-z0-9]", "-", str(DOCS.parent))
    return Path.home() / ".claude" / "projects" / name


def read_session(path):
    """The records of a session log as ``(line index, record)`` pairs."""
    records = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        try:
            records.append((index, json.loads(line)))
        except json.JSONDecodeError:
            continue
    return records


def blocks(record):
    content = record.get("message", {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content or []


def is_human(record):
    return (
        record.get("type") == "user"
        and (record.get("origin") or {}).get("kind") == "human"
    )


def tool_summary(tool_input):
    for key in SUMMARY_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            line = redact(value.strip().splitlines()[0])
            if len(line) > SUMMARY_LENGTH:
                line = line[: SUMMARY_LENGTH - 1] + "…"
            return line
    return ""


def question_event(time, tool_input):
    return {
        "kind": "questions",
        "time": time,
        "answered": False,
        "questions": [
            {
                "question": redact(q["question"]),
                "options": [redact(o["label"]) for o in q.get("options", [])],
                "answer": None,
            }
            for q in tool_input.get("questions", [])
        ],
    }


def extract(records, first, last):
    """The events of an excerpt: records ``first`` to ``last``, inclusive.

    Events are dicts with a ``kind`` (user, claude, tool, questions or
    compaction) and a ``time`` (ISO 8601, UTC), in the order they happened.
    """
    events = []
    asked = {}
    for index, record in records:
        if not first <= index <= last or record.get("isSidechain"):
            continue
        time = record.get("timestamp")
        if record.get("isCompactSummary"):
            events.append({"kind": "compaction", "time": time})
        elif is_human(record):
            text = "\n".join(
                b["text"] for b in blocks(record) if b.get("type") == "text"
            )
            event = {"kind": "user", "time": time, "text": redact(text.strip())}
            if record.get("promptSource") == "suggestion_accepted":
                event["suggested"] = True
            events.append(event)
        elif record.get("type") == "user":
            answers = record.get("toolUseResult")
            for block in blocks(record):
                event = asked.get(block.get("tool_use_id"))
                if event is None or not isinstance(answers, dict):
                    continue
                event["answered"] = True
                for q in event["questions"]:
                    answer = answers.get("answers", {}).get(q["question"])
                    q["answer"] = None if answer is None else redact(answer)
        elif record.get("type") == "assistant":
            model = record["message"].get("model")
            for block in blocks(record):
                if block.get("type") == "text" and block["text"].strip():
                    events.append({
                        "kind": "claude",
                        "time": time,
                        "model": model,
                        "text": redact(block["text"].strip()),
                    })
                elif block.get("type") != "tool_use":
                    continue
                elif block["name"] == "AskUserQuestion":
                    asked[block["id"]] = question_event(time, block["input"])
                    events.append(asked[block["id"]])
                else:
                    events.append({
                        "kind": "tool",
                        "time": time,
                        "name": block["name"],
                        "summary": tool_summary(block.get("input", {})),
                    })
    return events


def load_spec():
    spec = yaml.safe_load(SPEC.read_text())
    for entry in spec["excerpts"]:
        missing = {"name", "session", "lines"} - entry.keys()
        if missing:
            fail(f"excerpt {entry.get('name', '?')} lacks {sorted(missing)}")
    return spec


def command_extract(names, force, logs):
    spec = load_spec()
    known = {entry["name"]: entry for entry in spec["excerpts"]}
    for name in names:
        if name not in known:
            fail(f"{name} is not listed in {SPEC.name}")
    DATA.mkdir(exist_ok=True)
    for name, entry in known.items():
        target = DATA / f"{name}.json"
        if names and name not in names:
            continue
        if target.exists() and not force:
            if names:
                fail(f"{target.name} exists; pass --force to replace it")
            continue
        log = logs / f"{entry['session']}.jsonl"
        if not log.exists():
            fail(f"no session log {log} (deleted, or on another machine?)")
        first, last = entry["lines"]
        events = extract(read_session(log), first, last)
        if not any(event["kind"] == "user" for event in events):
            fail(f"{name}: lines {first}-{last} hold no user message")
        excerpt = {
            "name": name,
            "session": entry["session"],
            "lines": [first, last],
            "user": spec["user"],
            "timezone": spec["timezone"],
            "events": events,
        }
        target.write_text(json.dumps(excerpt, ensure_ascii=False, indent=1))
        print(f"{target.relative_to(DOCS)}: {len(events)} events")


def command_outline(session, logs):
    matches = sorted(logs.glob(f"{session}*.jsonl"))
    if len(matches) != 1:
        fail(f"{len(matches)} session logs match {session!r} in {logs}")
    for index, record in read_session(matches[0]):
        if is_human(record):
            text = " ".join(
                b["text"] for b in blocks(record) if b.get("type") == "text"
            )
            when = record.get("timestamp", "")[:16]
            print(f"{index:>6}  {when}  {' '.join(text.split())[:100]}")


# --- rendering --------------------------------------------------------------


def markdown():
    """A GitHub-flavoured Markdown renderer, as the terminal shows replies."""
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    def math(content, options):
        # Delimiters MathJax picks up; pandoc writes its own math the same way.
        if options["display_mode"]:
            return rf"\[{html.escape(content)}\]"
        return rf"\({html.escape(content)}\)"

    def heading_open(self, tokens, idx, options, env):
        # Not real headings: they must stay out of the page outline.
        return f'<p class="t-heading t-{tokens[idx].tag}">'

    def table_open(self, tokens, idx, options, env):
        return '<table class="table table-sm">'

    md = MarkdownIt("commonmark", {"html": False})
    md.enable(["table", "strikethrough"])
    md.use(dollarmath_plugin, allow_space=False, renderer=math)
    md.add_render_rule("heading_open", heading_open)
    md.add_render_rule("heading_close", lambda *args: "</p>")
    md.add_render_rule("table_open", table_open)
    return md


def local(time, zone):
    return datetime.datetime.fromisoformat(time).astimezone(zone)


def duration(start, end):
    minutes = round((end - start).total_seconds() / 60)
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def plural(count, noun):
    return f"{count} {noun}" + ("" if count == 1 else "s")


def render_turn(who, css, when, body, note=""):
    note = f' <span class="t-note">{note}</span>' if note else ""
    return (
        f'<div class="t-turn t-{css}"><p class="t-who">{html.escape(who)}'
        f' <time datetime="{when.isoformat()}">{when:%H:%M}</time>{note}</p>'
        f'<div class="t-body">{body}</div></div>'
    )


def render_work(md, items, start, end):
    """Collapsed record of what Claude did between two messages."""
    rows = []
    for item in items:
        if item["kind"] == "tool":
            rows.append(
                f'<li class="t-tool"><code>{html.escape(item["name"])}</code> '
                f"{html.escape(item['summary'])}</li>"
            )
        else:
            rows.append(f"<li>{md.renderInline(item['text'])}</li>")
    calls = sum(item["kind"] == "tool" for item in items)
    took = duration(start, end)
    return (
        f'<details class="t-work"><summary>Claude worked for {took}: '
        f"{plural(calls, 'tool call')}</summary><ul>{''.join(rows)}</ul>"
        "</details>"
    )


def picked(option, answer):
    """How ``option`` appears in an answer, or None if it was not chosen.

    Several chosen options are joined by ", "; one that itself contains a
    comma is quoted.
    """
    for form in (f'"{option}"', option):
        if answer is not None and (
            answer == form
            or answer.startswith(f"{form}, ")
            or answer.endswith(f", {form}")
            or f", {form}, " in answer
        ):
            return form
    return None


def render_questions(md, event, when):
    rows = []
    for q in event["questions"]:
        options = ""
        written = q["answer"] or ""
        for option in q["options"]:
            form = picked(option, q["answer"])
            options += (
                f'<li class="{"t-chosen" if form else ""}">'
                f"{html.escape(option)}</li>"
            )
            if form:
                written = written.replace(form, "", 1)
        written = written.strip(", ")
        if written:
            written = f'<p class="t-answer">Wrote: {html.escape(written)}</p>'
        rows.append(
            f'<li><p class="t-question">{md.renderInline(q["question"])}</p>'
            f"<ul>{options}</ul>{written}</li>"
        )
    note = "" if event["answered"] else "left unanswered"
    return render_turn(
        "Claude asked",
        "questions",
        when,
        f"<ol>{''.join(rows)}</ol>",
        note,
    )


def render(excerpt):
    """The HTML of an excerpt.

    Claude's work between two messages (tool calls and the short notes around
    them) is folded into one collapsed block; what Claude wrote after its last
    tool call is shown in full, as are the user's messages.
    """
    md = markdown()
    zone = ZoneInfo(excerpt["timezone"])
    events = excerpt["events"]
    models = sorted({e["model"] for e in events if e["kind"] == "claude"})
    meta = (
        '<p class="t-meta">Claude Code session '
        f"<code>{html.escape(excerpt['session'][:8])}</code>, model "
        f"<code>{html.escape(', '.join(models))}</code>. Times are "
        f"{html.escape(excerpt['timezone'])}.</p>"
    )
    out = [meta]
    day = None
    previous = local(events[0]["time"], zone)
    run = []

    def stamp(time):
        nonlocal day
        when = local(time, zone)
        if when.date() != day:
            day = when.date()
            out.append(f'<p class="t-day">{when:%A %-d %B %Y}</p>')
        return when

    def flush():
        nonlocal previous
        tools = [i for i, item in enumerate(run) if item["kind"] == "tool"]
        split = tools[-1] + 1 if tools else 0
        if split:
            stamp(run[0]["time"])
            # Until the reply that follows, or the last call if none does.
            end = local(run[min(split, len(run) - 1)]["time"], zone)
            out.append(render_work(md, run[:split], previous, end))
        for item in run[split:]:
            when = stamp(item["time"])
            out.append(
                render_turn("Claude", "claude", when, md.render(item["text"]))
            )
        if run:
            previous = local(run[-1]["time"], zone)
        run.clear()

    for event in events:
        if event["kind"] in ("claude", "tool"):
            run.append(event)
            continue
        flush()
        when = stamp(event["time"])
        if event["kind"] == "user":
            note = (
                "accepted a suggested reply" if event.get("suggested") else ""
            )
            body = html.escape(event["text"])
            out.append(render_turn(excerpt["user"], "user", when, body, note))
        elif event["kind"] == "questions":
            out.append(render_questions(md, event, when))
        elif event["kind"] == "compaction":
            out.append(
                '<p class="t-gap">The session reached the limit of the '
                "model's context here. Claude Code replaced the earlier "
                "conversation with a summary and the work continued.</p>"
            )
        previous = when
    flush()
    return "".join(out)


PLACEHOLDER = re.compile(
    r'(<div class="transcript" data-excerpt="([^"]+)">).*?'
    r"(</div><!--/transcript-->)",
    re.DOTALL,
)


def fill(text):
    def replace(match):
        source = DATA / f"{match[2]}.json"
        if not source.exists():
            fail(f"no excerpt {source.relative_to(DOCS)}; run 'extract' first")
        return match[1] + render(json.loads(source.read_text())) + match[3]

    return PLACEHOLDER.sub(replace, text)


def post_render():
    site = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", DOCS / "_site"))
    if not site.is_absolute():
        site = DOCS / site
    for source in sorted(SECTION.glob("*.qmd")):
        page = site / "conversations" / f"{source.stem}.html"
        if not page.exists():
            continue
        text = page.read_text()
        if "data-excerpt=" in source.read_text() and not PLACEHOLDER.search(
            text
        ):
            fail(f"{page.name} has no transcript placeholder to fill")
        page.write_text(fill(text))


def main():
    if len(sys.argv) == 1:
        return post_render()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs", type=Path, default=default_logs())
    commands = parser.add_subparsers(dest="command", required=True)
    outline = commands.add_parser("outline")
    outline.add_argument("session", help="session id, or its first characters")
    extraction = commands.add_parser("extract")
    extraction.add_argument("names", nargs="*")
    extraction.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "outline":
        command_outline(args.session, args.logs)
    else:
        command_extract(args.names, args.force, args.logs)


if __name__ == "__main__":
    main()
