"""Generate examples/sample_conversations.json: a small, entirely synthetic ChatGPT-format export
for trying savings.py without your own data."""

import json
import random
from pathlib import Path

random.seed(1)

CHATS = [
    ["Rewrite this to sound more professional: 'hey, can you send me the report by tmrw?'"],
    ["Summarise this paragraph in two lines: " + "Large language models are trained on vast text corpora " * 12],
    ["What's the difference between a list and a tuple in Python?", "When should I use a tuple then?"],
    ["Write a SQL query for 7-day retention by signup week from users(user_id, signup_date) and events(user_id, event_date)."],
    ["A shop sells pens at ₹12 each and gives 1 free for every 5 bought. How much do 23 pens cost? Explain step by step."],
    ["Draft a polite email declining a meeting invite."],
    ["Explain the precision-recall trade-off with a fraud detection example.", "How would I pick a threshold?", "And if false positives cost ₹200 and misses ₹5,000?"],
    ["Under Indian contract law, is an agreement with a minor void or voidable? Cite the section."],
    ["Translate to Hindi: 'The meeting has been moved to Friday.'"],
    ["Design an AI feature for a food delivery app for groups ordering together. Include metrics and risks."],
    ["Give me 5 names for a photo search app."],
    ["Prove that the square root of 2 is irrational."],
    ["Fix this Python error: TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
    ["What is RAG? Keep it short."],
    ["Compare RICE and Kano for prioritisation, with when to use each."],
]

REPLY_WORDS = "This is a synthetic assistant reply used only to estimate output length. "


def make():
    convs, t = [], 1_759_300_000  # late Sep / Oct 2025
    for _ in range(4):  # repeat to mimic a month of use
        for chat in CHATS:
            mapping, i = {}, 0
            for prompt in chat:
                t += random.randint(600, 20_000)
                mapping[f"u{i}"] = {"message": {"author": {"role": "user"}, "content": {"parts": [prompt]}, "create_time": t}}
                reply = REPLY_WORDS * random.randint(3, 40)
                mapping[f"a{i}"] = {"message": {"author": {"role": "assistant"}, "content": {"parts": [reply]}, "create_time": t + 5}}
                i += 1
            convs.append({"title": chat[0][:40], "mapping": mapping})
    return convs


if __name__ == "__main__":
    out = Path(__file__).with_name("sample_conversations.json")
    out.write_text(json.dumps(make(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out}")
