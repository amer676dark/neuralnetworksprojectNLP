"""
Keyword Spotting — detect specific keywords in transcribed Arabic speech.

Approach (text-based, post-ASR):
  Transcribe audio with Whisper/Wav2Vec/SeamlessM4T → search transcript for keywords.

Why text-based instead of acoustic keyword spotting:
  - Works with any Arabic dialect Whisper supports (98 languages)
  - Re-uses the same ASR pipeline we already trained
  - No separate model to train
  - Robust to misspellings via fuzzy matching

For the project requirements, keywords like "طوارئ" (emergency), "موعد نهائي" (deadline),
"امتحان" (exam) are detected with optional fuzzy matching (Levenshtein) to handle ASR errors.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


DEFAULT_KEYWORDS = [
    "طوارئ",       # emergency
    "موعد نهائي",  # deadline
    "امتحان",      # exam
    "اجتماع",      # meeting
    "مساعدة",      # help
    "خطر",         # danger
    "emergency", "deadline", "exam", "meeting", "help",
]


@dataclass
class KeywordHit:
    keyword: str
    matched_text: str
    position: int      # character index in the transcript
    confidence: float  # 1.0 = exact match, lower = fuzzy match


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev[j + 1] + 1
            deletions  = curr[j] + 1
            subs       = prev[j] + (ca != cb)
            curr.append(min(insertions, deletions, subs))
        prev = curr
    return prev[-1]


def _normalize_arabic(text: str) -> str:
    """Light Arabic normalization for matching: strip diacritics, unify alif forms."""
    # Remove tashkeel (diacritics)
    text = re.sub(r"[ً-ٰٟ]", "", text)
    # Unify alif forms
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    # Unify ya / alef-maksura
    text = text.replace("ى", "ي")
    # Unify ta-marbuta
    text = text.replace("ة", "ه")
    return text


def spot_keywords(
    transcript: str,
    keywords: List[str] = None,
    fuzzy: bool = True,
    fuzzy_max_distance: int = 1,
) -> List[KeywordHit]:
    """
    Find keyword occurrences in a transcript.

    - Exact match first (case-insensitive, Arabic-normalized).
    - If fuzzy=True, also accepts matches within `fuzzy_max_distance` edits.
    """
    if not transcript:
        return []
    keywords = keywords or DEFAULT_KEYWORDS

    norm_transcript = _normalize_arabic(transcript.lower())
    hits: List[KeywordHit] = []

    for kw in keywords:
        norm_kw = _normalize_arabic(kw.lower())
        # Exact substring match
        for m in re.finditer(re.escape(norm_kw), norm_transcript):
            hits.append(KeywordHit(
                keyword=kw,
                matched_text=transcript[m.start():m.end()],
                position=m.start(),
                confidence=1.0,
            ))
        # Fuzzy word-level match (only for keywords without spaces — single tokens)
        if fuzzy and " " not in norm_kw and len(norm_kw) >= 4:
            for word_match in re.finditer(r"\S+", norm_transcript):
                word = word_match.group(0)
                if word == norm_kw or len(word) < len(norm_kw) - fuzzy_max_distance:
                    continue
                dist = _levenshtein(word, norm_kw)
                if 0 < dist <= fuzzy_max_distance:
                    hits.append(KeywordHit(
                        keyword=kw,
                        matched_text=transcript[word_match.start():word_match.end()],
                        position=word_match.start(),
                        confidence=1.0 - (dist / max(len(norm_kw), 1)),
                    ))

    # Deduplicate by (keyword, position)
    seen = set()
    deduped = []
    for h in sorted(hits, key=lambda x: (x.position, -x.confidence)):
        key = (h.keyword, h.position)
        if key not in seen:
            seen.add(key)
            deduped.append(h)
    return deduped


def format_hits_table(hits: List[KeywordHit]) -> str:
    """Pretty-print hits as a Markdown table."""
    if not hits:
        return "_No keywords detected._"
    rows = ["| Keyword | Matched Text | Position | Confidence |",
            "|---|---|---|---|"]
    for h in hits:
        rows.append(f"| {h.keyword} | {h.matched_text} | {h.position} | {h.confidence:.2f} |")
    return "\n".join(rows)
