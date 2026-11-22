# How to grade (about 2–3 hours; split it into sittings if you like)

Open `grading_sheet.csv` in Google Sheets (File → Import) or Excel. Turn on **Wrap text** for the whole sheet so long answers are readable.

Each row is one request with two answers, **A** and **B**. They come from two different AI models, in a random order, with no names. Grade every row on three columns:

| Column | What to write | When |
|---|---|---|
| `A_meets_need (yes/no)` | `yes` or `no` | `yes` if answer A does what the user asked, with no significant errors and no important omissions |
| `B_meets_need (yes/no)` | `yes` or `no` | same question for answer B |
| `materially_better (A/B/neither)` | `A`, `B` or `neither` | `A` or `B` only if a typical user with this request would clearly notice and care about the difference; otherwise `neither` |

The rules, the same ones the AI judge followed:
- Minor style differences, wording, slightly less detail or a missing nice-to-have do **not** fail an answer.
- Factual errors, wrong conclusions, ignoring the request, or missing key points that matter **do** fail it.
- Don't reward length or formatting by itself. Padding the user didn't ask for is a flaw.
- `what_a_good_answer_needs` shows the criteria the judge also saw. Use it as a guide, not a checklist.

Write a few words in `notes` when you're unsure. Leave a row blank to skip it.

**Don't open `results/human_review_key.json`.** It says which answer came from which model, and seeing it would bias the grading.

When you're done, save the file as CSV with the same name, and I'll run:

```bash
python -m hybrid.human_review score
```
