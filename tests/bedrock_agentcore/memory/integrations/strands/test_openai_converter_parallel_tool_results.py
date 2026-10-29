"""Regression coverage for parallel and structured OpenAI converter tool results (#723)."""

import json

from strands.types.session import SessionMessage

from bedrock_agentcore.memory.integrations.strands.converters.openai import OpenAIConverseConverter


def _tool_result(tool_id: str, content: list[dict], status: str = "success") -> dict:
    return {"toolResult": {"toolUseId": tool_id, "content": content, "status": status}}


def _payloads(message: dict) -> list[tuple[str, str]]:
    return OpenAIConverseConverter.message_to_payload(SessionMessage(message=message, message_id=0))


def _event(payloads: list[tuple[str, str]]) -> dict:
    return {
        "payload": [{"conversational": {"content": {"text": text}, "role": role.upper()}} for text, role in payloads]
    }


def _restored(events: list[dict]) -> list[dict]:
    return [message.message for message in OpenAIConverseConverter.events_to_messages(events)]


def test_parallel_tool_results_each_create_an_openai_tool_payload():
    original = {
        "role": "user",
        "content": [
            _tool_result("t1", [{"text": "sunny"}]),
            _tool_result("t2", [{"json": {"temp_c": 21}}]),
        ],
    }

    payloads = _payloads(original)

    assert [role for _, role in payloads] == ["tool", "tool"]
    assert [json.loads(text)["tool_call_id"] for text, _ in payloads] == ["t1", "t2"]


def test_parallel_tool_results_round_trip_in_one_event():
    original = {
        "role": "user",
        "content": [
            _tool_result("t1", [{"text": "sunny"}]),
            _tool_result("t2", [{"json": {"temp_c": 21}}]),
        ],
    }

    assert _restored([_event(_payloads(original))]) == [original]


def test_parallel_tool_results_round_trip_across_newest_first_events():
    original = {
        "role": "user",
        "content": [
            _tool_result("t1", [{"text": "sunny"}]),
            _tool_result("t2", [{"json": {"temp_c": 21}}]),
        ],
    }

    # Memory's list_events API returns newest events first.
    events = [_event([payload]) for payload in reversed(_payloads(original))]
    assert _restored(events) == [original]


def test_json_only_tool_result_remains_structured_after_restore():
    original = {"role": "user", "content": [_tool_result("t1", [{"json": {"temp_c": 21}}])]}

    payloads = _payloads(original)

    assert json.loads(payloads[0][0])["content"] == '{"temp_c": 21}'
    assert _restored([_event(payloads)]) == [original]


def test_tool_result_preserves_mixed_text_json_and_status():
    original = {
        "role": "user",
        "content": [
            _tool_result(
                "t1",
                [{"text": "before"}, {"json": {"ok": True}}, {"text": "after"}],
                status="error",
            )
        ],
    }

    payloads = _payloads(original)

    assert json.loads(payloads[0][0])["content"] == 'before\n{"ok": true}\nafter'
    assert _restored([_event(payloads)]) == [original]


def test_parallel_results_keep_assistant_tool_use_pairing():
    assistant = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "t1", "name": "weather", "input": {}}},
            {"toolUse": {"toolUseId": "t2", "name": "temperature", "input": {}}},
        ],
    }
    user = {
        "role": "user",
        "content": [
            _tool_result("t1", [{"text": "sunny"}]),
            _tool_result("t2", [{"json": {"temp_c": 21}}]),
        ],
    }

    # newest first: tool results followed by the older assistant call event.
    assert _restored([_event(_payloads(user)), _event(_payloads(assistant))]) == [assistant, user]


def test_legacy_openai_tool_payload_without_extension_still_restores():
    legacy = {
        "role": "tool",
        "tool_call_id": "legacy",
        "content": "done",
        "status": "success",
    }

    assert _restored([_event([(json.dumps(legacy), "tool")])]) == [
        {"role": "user", "content": [_tool_result("legacy", [{"text": "done"}])]}
    ]


def test_non_tool_message_separates_adjacent_tool_result_groups():
    first = {"role": "tool", "tool_call_id": "t1", "content": "one"}
    ordinary = {"role": "user", "content": "separate turn"}
    second = {"role": "tool", "tool_call_id": "t2", "content": "two"}
    events = [_event([(json.dumps(item), item["role"])]) for item in (second, ordinary, first)]

    assert _restored(events) == [
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "one"}]}}]},
        {"role": "user", "content": [{"text": "separate turn"}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "content": [{"text": "two"}]}}]},
    ]


def test_parallel_results_round_trip_through_blob_payloads():
    original = {
        "role": "user",
        "content": [
            _tool_result("t1", [{"json": {"x": 1}}]),
            _tool_result("t2", [{"text": "ok"}]),
        ],
    }
    event = {"payload": [{"blob": json.dumps(payload)} for payload in _payloads(original)]}

    assert _restored([event]) == [original]
