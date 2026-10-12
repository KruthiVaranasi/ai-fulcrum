from savings import build_requests, parse_export, tokens

CHATGPT = [{
    "title": "t",
    "mapping": {
        "a": {"message": {"author": {"role": "system"}, "content": {"parts": [""]}, "create_time": 1}},
        "b": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hello there"]}, "create_time": 1759700000}},
        "c": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Hi! How can I help?"]}, "create_time": 1759700001}},
        "d": {"message": {"author": {"role": "user"}, "content": {"parts": ["Explain RAG"]}, "create_time": 1759700002}},
        "e": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["RAG retrieves documents..."]}, "create_time": 1759700003}},
    },
}]

CLAUDE = [{
    "uuid": "x",
    "chat_messages": [
        {"sender": "human", "text": "What is a token?", "created_at": "2026-09-01T10:00:00Z"},
        {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "A chunk of text."}], "created_at": "2026-09-01T10:00:02Z"},
    ],
}]


def test_parse_chatgpt_orders_turns_and_skips_system():
    convs = parse_export(CHATGPT)
    assert [r for r, _, _ in convs[0]] == ["user", "assistant", "user", "assistant"]


def test_parse_claude_reads_content_blocks():
    convs = parse_export(CLAUDE)
    assert convs[0][1][1] == "A chunk of text."


def test_history_is_billed_on_later_turns():
    reqs = build_requests(parse_export(CHATGPT))
    assert len(reqs) == 2
    first_in, second_in = reqs[0][1], reqs[1][1]
    assert first_in == tokens("Hello there")
    assert second_in == tokens("Hello there") + tokens("Hi! How can I help?") + tokens("Explain RAG")
    assert reqs[0][3] == "2025-10"
