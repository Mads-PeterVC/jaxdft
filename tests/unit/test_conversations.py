"""Conversation excerpts for the documentation site (docs/_tools/conversations.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"


@pytest.fixture(scope="module")
def conversations():
    spec = importlib.util.spec_from_file_location(
        "conversations", DOCS / "_tools" / "conversations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def user(text, time, **extra):
    return {
        "type": "user",
        "timestamp": time,
        "origin": {"kind": "human"},
        "message": {"content": text},
        **extra,
    }


def assistant(time, *blocks):
    return {
        "type": "assistant",
        "timestamp": time,
        "message": {"model": "model-x", "content": list(blocks)},
    }


def text(value):
    return {"type": "text", "text": value}


def tool(name, tool_id="t", **tool_input):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


QUESTION = {
    "question": "Which?",
    "options": [{"label": "A (Recommended)"}, {"label": "B, with a comma"}],
}

SESSION = [
    {"type": "mode"},
    user("Please do it.", "2026-11-05T07:00:00Z"),
    assistant(
        "2026-11-05T07:00:05Z",
        {"type": "thinking", "thinking": "private"},
        text("Looking first."),
    ),
    assistant(
        "2026-11-05T07:00:06Z",
        tool("Bash", description="List files", command="ls /Users/someone/x"),
    ),
    {
        "type": "user",
        "timestamp": "2026-11-05T07:00:07Z",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "secret output"}
            ]
        },
    },
    assistant(
        "2026-11-05T07:00:08Z",
        tool("Read", file_path=str(DOCS.parent / "src/jaxdft/units.py")),
    ),
    {
        "type": "assistant",
        "isSidechain": True,
        "timestamp": "2026-11-05T07:00:09Z",
        "message": {"content": [text("subagent chatter")]},
    },
    assistant(
        "2026-11-05T07:03:00Z",
        text(
            "## Done\n\nIt costs $E = mc^2$ and `x < y`.\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        ),
    ),
    assistant(
        "2026-11-05T07:03:01Z", tool("AskUserQuestion", "q", questions=[QUESTION])
    ),
    {
        "type": "user",
        "timestamp": "2026-11-05T07:04:00Z",
        "toolUseResult": {"answers": {"Which?": '"B, with a comma", my own idea'}},
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "q", "content": "answered"}
            ]
        },
    },
    {
        "type": "user",
        "timestamp": "2026-11-05T23:30:00Z",
        "isCompactSummary": True,
        "message": {"content": "This session is being continued..."},
    },
    {
        "type": "user",
        "timestamp": "2026-11-05T23:30:01Z",
        "origin": {"kind": "task-notification"},
        "message": {"content": "<task-notification>"},
    },
    user(
        "Yes <b>go</b>, see /Users/someone/notes.md",
        "2026-11-05T23:31:00Z",
        promptSource="suggestion_accepted",
    ),
    assistant("2026-11-05T23:31:30Z", text("Final answer.")),
    user("Outside the excerpt.", "2026-11-06T09:00:00Z"),
]
RECORDS = list(enumerate(SESSION))


def excerpt(conversations, first=1, last=13):
    return {
        "name": "x",
        "session": "0123456789abcdef",
        "lines": [first, last],
        "user": "Ada",
        "timezone": "Europe/Copenhagen",
        "events": conversations.extract(RECORDS, first, last),
    }


def test_extract_keeps_messages_and_tool_summaries_in_order(conversations):
    events = conversations.extract(RECORDS, 1, 13)
    assert [e["kind"] for e in events] == [
        "user",
        "claude",
        "tool",
        "tool",
        "claude",
        "questions",
        "compaction",
        "user",
        "claude",
    ]
    assert events[2] == {
        "kind": "tool",
        "time": "2026-11-05T07:00:06Z",
        "name": "Bash",
        "summary": "List files",
    }
    assert events[1]["model"] == "model-x"
    assert events[7]["suggested"] is True and "suggested" not in events[0]


def test_extract_leaves_out_tool_output_reasoning_subagents_and_harness_messages(
    conversations,
):
    stored = json.dumps(conversations.extract(RECORDS, 0, len(SESSION)))
    for hidden in [
        "secret output",
        "private",
        "subagent chatter",
        "task-notification",
        "being continued",
    ]:
        assert hidden not in stored


def test_extract_respects_the_line_range(conversations):
    assert [
        e["text"] for e in conversations.extract(RECORDS, 12, 14) if e["kind"] == "user"
    ] == [
        "Yes <b>go</b>, see ~/notes.md",
        "Outside the excerpt.",
    ]


def test_extract_shortens_machine_paths(conversations):
    events = conversations.extract(RECORDS, 1, 13)
    assert events[3]["summary"] == "src/jaxdft/units.py"
    assert "/Users/" not in json.dumps(events)
    assert (
        conversations.redact(
            "/private/tmp/claude-501/-Users-a-b/0a1b-2c/scratchpad/rs.py"
        )
        == "/tmp/scratchpad/rs.py"
    )


def test_extract_attaches_answers_to_questions(conversations):
    asked = conversations.extract(RECORDS, 1, 13)[5]
    assert asked["answered"] is True
    assert asked["questions"] == [
        {
            "question": "Which?",
            "options": ["A (Recommended)", "B, with a comma"],
            "answer": '"B, with a comma", my own idea',
        }
    ]
    unanswered = conversations.extract(RECORDS[:9], 1, 13)[5]
    assert (
        unanswered["answered"] is False and unanswered["questions"][0]["answer"] is None
    )


def test_render_folds_work_and_shows_replies_in_full(conversations):
    page = conversations.render(excerpt(conversations))
    assert "Claude worked for 3 min: 2 tool calls" in page
    work = page[page.index("<details") : page.index("</details>")]
    assert "Looking first." in work and "<code>Bash</code> List files" in work
    assert "Final answer." in page and "Final answer." not in work
    assert page.count('class="t-day"') == 2  # 23:31 UTC is past midnight in Copenhagen
    assert "08:00" in page and "00:31" in page


def test_render_escapes_user_text_and_marks_suggestions(conversations):
    page = conversations.render(excerpt(conversations))
    assert "Yes &lt;b&gt;go&lt;/b&gt;" in page and "<b>go</b>" not in page
    assert "accepted a suggested reply" in page


def test_render_handles_markdown_math_and_headings(conversations):
    page = conversations.render(excerpt(conversations))
    assert r"\(E = mc^2\)" in page
    assert "<code>x &lt; y</code>" in page
    assert '<table class="table table-sm">' in page
    assert '<p class="t-heading t-h2">Done</p>' in page and "<h2" not in page


def test_render_marks_chosen_options_and_written_answers(conversations):
    page = conversations.render(excerpt(conversations))
    assert '<li class="t-chosen">B, with a comma</li>' in page
    assert '<li class="">A (Recommended)</li>' in page
    assert "Wrote: my own idea" in page


def test_fill_is_idempotent(conversations, tmp_path, monkeypatch):
    monkeypatch.setattr(conversations, "DATA", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps(excerpt(conversations)))
    page = '<p>Intro</p><div class="transcript" data-excerpt="x"></div><!--/transcript--><p>After</p>'
    once = conversations.fill(page)
    assert (
        "Final answer." in once
        and once.startswith("<p>Intro</p>")
        and once.endswith("<p>After</p>")
    )
    assert conversations.fill(once) == once


def test_every_listed_excerpt_is_stored_and_shown_on_a_page(conversations):
    pages = "".join(p.read_text() for p in conversations.SECTION.glob("*.qmd"))
    for entry in conversations.load_spec()["excerpts"]:
        stored = json.loads((conversations.DATA / f"{entry['name']}.json").read_text())
        assert (
            stored["lines"] == entry["lines"] and stored["session"] == entry["session"]
        )
        assert f'data-excerpt="{entry["name"]}"' in pages
        assert conversations.render(stored)
