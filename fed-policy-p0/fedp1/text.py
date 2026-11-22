"""P1 美联储沟通文本信号：鹰鸽词典、声明措辞 diff、演讲加权。"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z']+")
NEGATIONS = {"not", "no", "never", "without", "hardly", "barely", "unlikely"}
MAX_PHRASE_TOKENS = 8

# 正权重 = 鹰派，负权重 = 鸽派。短语按最长匹配优先。
LEXICON: dict[str, float] = {
    # 通胀与政策立场
    "inflation remains elevated": 2.0,
    "inflation remains too high": 2.6,
    "inflation is too high": 2.6,
    "inflation remains above": 1.2,
    "elevated inflation": 1.5,
    "inflation pressures": 0.8,
    "inflation pressure": 0.8,
    "inflation pressures have persisted": 1.6,
    "inflation has persisted": 1.0,
    "upside risks to inflation": 1.5,
    "inflation has disappointed": 1.8,
    "progress on lowering inflation has disappointed": 2.2,
    "price stability": 0.8,
    "deliver price stability": 1.3,
    "restrictive": 0.9,
    "restrictive policy": 1.1,
    "further tightening": 2.0,
    "additional firming": 2.0,
    "further policy firming": 2.0,
    "raise the target range": 2.6,
    "increase the target range": 2.6,
    "preferred to raise": 2.2,
    "raise rates": 2.0,
    "higher for longer": 1.5,
    "sooner rather than later": 1.5,
    "premature": 1.0,
    "persistent": 0.8,
    "broadening": 0.8,
    "vigilant": 1.0,
    "overheat": 1.5,
    # 增长与就业（强 = 鹰派）
    "strong pace": 1.0,
    "expanding at a strong pace": 1.2,
    "solid pace": 0.6,
    "robust": 0.6,
    "resilient": 0.5,
    "strong": 0.4,
    "job gains have kept pace": 0.6,
    "labor market remains tight": 1.5,
    "tight labor market": 1.2,
    # 鸽派
    "downside risks to employment": -1.5,
    "labor market has cooled": -1.5,
    "labor market cooling": -1.3,
    "labor market has softened": -1.3,
    "cooled": -0.8,
    "cooling": -0.8,
    "slowing": -0.6,
    "moderating": -0.6,
    "softening": -0.8,
    "weakening": -0.8,
    "deteriorating": -1.0,
    "cut the target range": -2.6,
    "lower the target range": -2.6,
    "preferred to lower": -2.2,
    "preferred to cut": -2.2,
    "cut rates": -2.0,
    "policy easing": -1.2,
    "easing": -1.0,
    "support employment": -1.2,
    "maximum employment": -0.4,
    "patient": -1.0,
    "wait and see": -1.0,
    "balanced risks": -0.8,
    "two-sided risks": -0.6,
    "normalize": -0.6,
    "accommodative": -1.0,
    "headwinds": -0.8,
    "uncertainty": -0.2,
    "slowdown": -0.8,
}

STATEMENT_WEIGHT = 0.35
DIFF_WEIGHT = 0.35
SPEECH_WEIGHT = 0.30

ROLE_WEIGHTS = {
    "chair": 3.0,
    "vice_chair": 2.5,
    "governor": 2.0,
    "president": 1.5,
    "other": 1.0,
}

ROLE_LABELS = {
    "chair": "主席",
    "vice_chair": "副主席",
    "governor": "理事",
    "president": "地区联储主席",
    "other": "其他",
}


@dataclass
class LexScore:
    score: float
    raw: float
    total_abs: float
    hawk_hits: int
    dove_hits: int
    words: int
    terms: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "raw": round(self.raw, 4),
            "total_abs": round(self.total_abs, 4),
            "hawk_hits": self.hawk_hits,
            "dove_hits": self.dove_hits,
            "words": self.words,
            "terms": self.terms,
        }


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score_text(text: str, max_terms: int = 12) -> LexScore:
    tokens = tokenize(text)
    lexicon = {tuple(phrase.split()): weight for phrase, weight in LEXICON.items()}
    max_len = min(MAX_PHRASE_TOKENS, max((len(k) for k in lexicon), default=1))
    raw = 0.0
    total_abs = 0.0
    hawk_hits = 0
    dove_hits = 0
    terms: list[dict] = []
    index = 0
    while index < len(tokens):
        matched = False
        for size in range(max_len, 0, -1):
            if index + size > len(tokens):
                continue
            key = tuple(tokens[index : index + size])
            weight = lexicon.get(key)
            if weight is None:
                continue
            phrase = " ".join(key)
            protected = any(token in NEGATIONS for token in key)
            negated = (
                not protected
                and any(token in NEGATIONS for token in tokens[max(0, index - 4) : index])
            )
            effective = -weight * 0.7 if negated else weight
            raw += effective
            total_abs += abs(effective)
            if effective > 0:
                hawk_hits += 1
            elif effective < 0:
                dove_hits += 1
            terms.append(
                {
                    "phrase": phrase,
                    "weight": round(effective, 4),
                    "negated": negated,
                    "position": index,
                }
            )
            index += size
            matched = True
            break
        if not matched:
            index += 1

    score = _clip(raw / (total_abs + 2.0)) if total_abs else 0.0
    terms.sort(key=lambda row: abs(row["weight"]), reverse=True)
    return LexScore(
        score=score,
        raw=raw,
        total_abs=total_abs,
        hawk_hits=hawk_hits,
        dove_hits=dove_hits,
        words=len(tokens),
        terms=terms[:max_terms],
    )


def _sentences(text: str) -> list[str]:
    # 保护 "Lisa D. Cook" 这类姓名缩写，避免在 "D." 处断句。
    protected = re.sub(r"\b([A-Z])\.", r"\1<dot>", text)
    return [
        sentence.replace("<dot>", ".").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
        if sentence.strip()
    ]


def statement_lexicon_text(statement: dict) -> str:
    """声明正文去掉政策动作与异议句，避免与结构化评分重复计数。"""
    kept = []
    for sentence in _sentences(" ".join(statement.get("paragraphs", []))):
        low = sentence.lower()
        if "decided to" in low and "target range" in low:
            continue
        if "voting against" in low or "preferred to" in low:
            continue
        kept.append(sentence)
    return " ".join(kept)


def action_score(action: str | None) -> float:
    return {
        "raise": 1.0,
        "increase": 1.0,
        "maintain": 0.0,
        "lower": -1.0,
        "cut": -1.0,
    }.get(action or "", 0.0)


def score_statement(statement: dict) -> dict:
    lex = score_text(statement_lexicon_text(statement))
    act = action_score(statement.get("action"))
    dissent_balance = int(statement.get("hike_dissents", 0)) - int(
        statement.get("cut_dissents", 0)
    )
    dissent = _clip(dissent_balance * 0.25)
    score = _clip(0.6 * lex.score + 0.2 * act + 0.2 * dissent)
    return {
        "date": statement.get("date"),
        "url": statement.get("url"),
        "action": statement.get("action"),
        "vote": statement.get("vote"),
        "hike_dissents": statement.get("hike_dissents", 0),
        "cut_dissents": statement.get("cut_dissents", 0),
        "lexicon": lex.to_dict(),
        "action_score": round(act, 4),
        "dissent_score": round(dissent, 4),
        "score": round(score, 4),
    }


def score_statement_diff(current: dict, previous: dict) -> dict:
    previous_sentences = _sentences(" ".join(previous.get("paragraphs", [])))
    current_sentences = _sentences(" ".join(current.get("paragraphs", [])))
    matcher = difflib.SequenceMatcher(
        None, [s.lower() for s in previous_sentences], [s.lower() for s in current_sentences],
        autojunk=False,
    )
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(current_sentences[j1:j2])
        if tag in ("delete", "replace"):
            removed.extend(previous_sentences[i1:i2])
    added_score = score_text(" ".join(added)) if added else LexScore(0.0, 0.0, 0.0, 0, 0, 0)
    removed_score = (
        score_text(" ".join(removed)) if removed else LexScore(0.0, 0.0, 0.0, 0, 0, 0)
    )
    diff = _clip(added_score.score - removed_score.score)
    return {
        "previous_date": previous.get("date"),
        "added": added,
        "removed": removed,
        "added_score": round(added_score.score, 4),
        "removed_score": round(removed_score.score, 4),
        "added_terms": added_score.terms,
        "removed_terms": removed_score.terms,
        "diff_score": round(diff, 4),
    }


def recency_weight(days: int) -> float:
    if days <= 7:
        return 1.0
    if days <= 14:
        return 0.8
    if days <= 30:
        return 0.6
    if days <= 45:
        return 0.4
    return 0.2


def score_speeches(speeches: list[dict], as_of: date) -> dict:
    items = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for speech in speeches:
        day = date.fromisoformat(speech["date"])
        days = max((as_of - day).days, 0)
        role = speech.get("role", "other")
        role_weight = ROLE_WEIGHTS.get(role, 1.0)
        recency = recency_weight(days)
        weight = role_weight * recency
        lex = score_text(" ".join(speech.get("paragraphs", [])))
        weighted_sum += weight * lex.score
        weight_sum += weight
        items.append(
            {
                "date": speech["date"],
                "speaker": speech.get("speaker"),
                "role": role,
                "role_label": ROLE_LABELS.get(role, role),
                "title": speech.get("title"),
                "url": speech.get("url"),
                "score": round(lex.score, 4),
                "role_weight": role_weight,
                "recency_weight": recency,
                "weight": round(weight, 4),
                "lexicon": lex.to_dict(),
            }
        )
    items.sort(key=lambda row: row["date"], reverse=True)
    return {
        "count": len(items),
        "weighted_score": round(weighted_sum / weight_sum, 4) if weight_sum else 0.0,
        "items": items,
    }


def build_text_signal(text_dir: str | Path, as_of: date) -> dict:
    base = Path(text_dir)
    statements_payload = json.loads(
        (base / "fomc_statements.json").read_text(encoding="utf-8")
    )
    speeches_payload = json.loads((base / "fomc_speeches.json").read_text(encoding="utf-8"))

    statements = sorted(
        [
            row
            for row in statements_payload.get("statements", [])
            if date.fromisoformat(row["date"]) <= as_of
        ],
        key=lambda row: row["date"],
    )
    if not statements:
        raise ValueError("没有可用的 FOMC 声明文本")
    current = statements[-1]
    previous = statements[-2] if len(statements) > 1 else None

    statement = score_statement(current)
    diff = score_statement_diff(current, previous) if previous else {
        "previous_date": None,
        "added": [],
        "removed": [],
        "added_score": 0.0,
        "removed_score": 0.0,
        "added_terms": [],
        "removed_terms": [],
        "diff_score": 0.0,
    }
    speeches = score_speeches(speeches_payload.get("speeches", []), as_of)

    text_score = _clip(
        STATEMENT_WEIGHT * statement["score"]
        + DIFF_WEIGHT * diff["diff_score"]
        + SPEECH_WEIGHT * speeches["weighted_score"]
    )
    return {
        "statement": statement,
        "statement_diff": diff,
        "speeches": {
            "window_days": speeches_payload.get("window_days"),
            "count": speeches["count"],
            "weighted_score": speeches["weighted_score"],
            "items": speeches["items"],
        },
        "components": {
            "statement": statement["score"],
            "diff": diff["diff_score"],
            "speeches": speeches["weighted_score"],
        },
        "weights": {
            "statement": STATEMENT_WEIGHT,
            "diff": DIFF_WEIGHT,
            "speeches": SPEECH_WEIGHT,
        },
        "text_score": round(text_score, 4),
        "sources": {
            "statements": statements_payload.get("source"),
            "speeches": speeches_payload.get("source"),
            "retrieved_at": statements_payload.get("retrieved_at"),
        },
    }
