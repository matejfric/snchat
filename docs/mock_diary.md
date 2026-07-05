# Mock diary fixture

A deterministic synthetic Czech diary (`tests/mock_diary.py`) that makes the
regression prompts in [error_modes.md §6](error_modes.md#6-example-prompts-regression-reference)
testable with **known ground truth** — both automatically (routing/retrieval) and
manually (answer quality, which needs a live LLM).

## Building & ingesting

```bash
uv run python tests/mock_diary.py            # writes ./mock_diary.zip (gitignored)
uv run streamlit run app.py                  # upload the ZIP in the sidebar → Process
```

124 entries (2025-01 → 2026-06), one per day used, ~34 tagged. Content is fully
deterministic — no randomness — so re-generated ZIPs are byte-identical and answers
are comparable across runs. **Note:** ingesting replaces `diary_vector_db/`; re-upload
your real backup afterwards.

## What the content encodes

| Theme | Entries | Exercises |
|---|---|---|
| Skiing: `lyže` + `skialp`, distinct 2025 vs 2026 story lines | 11 (6 in January) | multilingual 2-tag mapping (§2.1), fetch-all by count (§2.3), follow-up scope anchoring (§3.1) |
| Climbing `lezení`, 12 sessions, 5b→6b progression | 12 | recent-N dates-first (§2.4), progression summaries |
| Running `běh`, distances stated (5 + 10 + 21.1 + 8 + 12 km) | 5 | Czech query "kolik jsem toho naběhal?", `recent=1` |
| Game of Thrones: 12 scattered mentions — EN name, `GoT` acronym, declined Czech (`Hře/Hru/Hry o trůny`), one **long** entry with a buried verbatim mention | 12 | keyword/lexical branch (§2.6), the WRatio length-ratio regression |
| The Witcher books: `Zaklínač/Zaklínače/Zaklínači` + EN show name | 4 | second entity, Czech declension |
| Keyword look-alikes: "**got**ická katedrála" (Kutná Hora), "I for**got** to reply" | 2 | whole-word acronym boundary — must NOT match |
| Thesis/exams `diplomka`/`státnice` + an untagged 2026 entry, wording "úzkost"/"nervózní" | 5 | thematic question routes semantically, `keywords` stays empty |
| Pálava trip `turistika` on **2025-05-18** | 1 | exact-date point lookup (§6.4) |
| Untagged mundane filler (deterministic rotation, no trigger words) | 72 | year 2025 exceeds `FETCH_ALL_MAX` entries **and** `SINGLE_PASS_BUDGET` chars → similarity fallback vs fetch-all + map-reduce |
| Edge-case notes: no date prefix, `2025-02-30`, deleted item, non-Note item, empty text | 5 | data contract (§5) — all must be skipped (the empty-text one parses but isn't indexed) |

Ground-truth constants (`GOT_DATES`, `CLIMBING_DATES`, `JANUARY_SKIING_DATES`, …) are
exported by `tests/mock_diary.py` and derived from the entry lists, so they can't
drift from the content.

## Automated coverage

- `tests/test_mock_diary.py` — ZIP → `parse_standard_notes()` ground truth: counts,
  chronology, tag mapping, contract violations skipped, and a keyword sweep over
  **every** entry (exactly the GoT/Witcher ground-truth dates match; the look-alikes
  and filler don't).
- `tests/test_mock_retrieval.py` — `retrieve()` on a real in-memory Chroma with fake
  embeddings: fetch-all vs top-K by count, recent-N, exact date, fuzzy entity lookup,
  `$contains` tag filters, empty scopes.

What can't be automated offline stays manual: answer wording, semantic similarity
ranking (fake embeddings return arbitrary neighbors), and the extraction LLM's routing.

## Manual answer key (live app)

| Prompt | Expected routing | Expected answer content |
|---|---|---|
| "summarize my skiing" | tags `lyže`+`skialp`, breadth=all | 11 entries; 2025 = old skis/broken pole, 2026 = new Atomic skis, avalanche course |
| "what did I do skiing in January?" | + `month=1`, fetch-all | all **6** January entries (3× 2025, 3× 2026), not top-K |
| "how was skiing in 2025?" then "and in 2026?" | follow-up keeps tags, year flips | 2nd answer framed as **2026** (new skis, Praděd, Stubai) — no 2025 drift |
| "summarize my last 10 climbing sessions" | `recent=10`, tag `lezení` | the 10 newest: 2025-04-02 … 2026-06-10 (excludes 2025-02-12, 2025-03-05) |
| "how is my climbing going?" | tag `lezení`, breadth=all | 5b (2025-02) → first 6a (2025-04) → 6a+ → first 6b (2026-03) → 6b outdoors (2026-06) |
| "my last run" | `recent=1`, tag `běh` | 2026-05-30: 12 km in new shoes |
| "kolik jsem toho naběhal?" | tag `běh` (Czech) | distances from 5 entries, ~56 km total |
| "what did I do on 2025-05-18?" | exact date | Pálava hike, Děvín, wine tasting in Pavlov |
| "what were my impressions of Game of Thrones?" / "did I enjoy GoT?" | `keywords` ≈ [GoT, Game of Thrones, Hra o trůny] | all **12** mentions incl. declined Czech + the long 2025-07-15 entry; verdict "8/10, weak ending" (2025-10-12); **no** Kutná Hora / "forgot" entries |
| "what did I think of the Witcher books?" | `keywords` ≈ [Zaklínač, The Witcher] | 4 mentions, books > show |
| "when did I feel anxious?" | thematic → `keywords` empty, semantic path | 2025-04-28 (thesis), 2025-05-25 (exams), 2026-01-10 (new job) |
| "summarize my whole diary" | no filter → top-K + narrowing tip caption | deliberate limitation (§2.5) |
| "summarize my year 2025" | `year=2025`, breadth=all | fetch-all >50 entries, >45k chars → **map-reduce** path |
| "what was I up to today last year?" | `year`=today−1 + today's month/day | depends on run date — filler covers 2025 every ~6 days, so expect a hit or an honest "no entries" |
