from __future__ import annotations

import re


# 唤醒词别名表（双向）
# key 和 value 都是拼音字串（已去声调、小写）
WAKE_WORD_ALIASES = {
    "hicolmo": ["haikemu"],
    "haikemu": ["hicolmo"],
}


def _to_pinyin_str(text: str) -> str:
    """Convert text to concatenated pinyin string (lowercase, no tones)."""
    try:
        from pypinyin import lazy_pinyin, Style
        syllables = lazy_pinyin(text or "", style=Style.TONE3)
        joined = ''.join(syllables).lower()
        return re.sub(r'[0-9]', '', joined)
    except ImportError:
        return text.lower()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def phoneme_similarity(text_a: str, text_b: str) -> float:
    """Similarity based on pinyin-concatenated edit distance.、

    If text_b is a wake word with aliases, computes similarity against all
    aliases and returns the maximum score.
    """
    py_a = _to_pinyin_str(text_a)
    py_b = _to_pinyin_str(text_b)
    if not py_a or not py_b:
        return -1000000.0

    # 展开 text_b 的别名（如果是唤醒词）
    aliases = [py_b]
    if py_b in WAKE_WORD_ALIASES:
        aliases.extend(WAKE_WORD_ALIASES[py_b])

    # 对所有别名计算相似度，取最高值（最小负距离）
    best_score = float('-inf')
    for alias in aliases:
        if not alias:
            continue
        dist = _levenshtein(py_a, alias)
        score = - dist
        best_score = max(best_score, score)

    return best_score
