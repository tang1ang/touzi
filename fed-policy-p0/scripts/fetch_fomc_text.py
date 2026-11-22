#!/usr/bin/env python3
"""抓取 FOMC 声明与近期演讲文本，落盘为可复现的 JSON。

用法:
    python3 scripts/fetch_fomc_text.py --as-of 2026-11-22
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.federalreserve.gov"
STATEMENT_URL = BASE + "/newsevents/pressreleases/monetary{date}a.htm"
SPEECH_LIST_URL = BASE + "/newsevents/speech/{year}-speeches.htm"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FedPolicyRadar/0.1)"}

POLICY_KEYWORDS = [
    "monetary policy",
    "federal funds",
    "policy rate",
    "interest rate",
    "inflation",
    "price stability",
    "labor market",
    "unemployment",
    "employment",
    "economic outlook",
    "economic activity",
    "economy",
    "macroeconomy",
]


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def content_html(page: str) -> str:
    marker = 'class="col-xs-12 col-sm-8 col-md-8"'
    index = page.find(marker)
    if index < 0:
        raise ValueError("未找到正文容器")
    return page[index:]


def paragraphs(page: str) -> list[str]:
    seg = content_html(page)
    seg = re.sub(r"<script.*?</script>", "", seg, flags=re.S)
    seg = re.sub(r"<style.*?</style>", "", seg, flags=re.S)
    out: list[str] = []
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", seg, re.S):
        text = clean(raw)
        if not text:
            continue
        if "For media inquiries" in text or "Implementation Note" in text:
            break
        if text.startswith("Board of Governors") or "20th Street" in text:
            continue
        if text.startswith("Last Update"):
            continue
        if len(text) < 40:
            continue
        out.append(text)
    return out


def statement_paragraphs(page: str) -> list[str]:
    rows = paragraphs(page)
    cleaned: list[str] = []
    for text in rows:
        if text.startswith("For release at"):
            index = text.find("The Federal Open Market Committee")
            if index < 0:
                continue
            text = text[index:]
        cleaned.append(text)
    return cleaned


def parse_statement(page: str, day: date, url: str) -> dict:
    rows = statement_paragraphs(page)
    joined = " ".join(rows)
    vote = re.search(r"release by a\s+(\d+)\s*[\u2013-]\s*(\d+)\s+vote", joined)
    action = re.search(
        r"decided to\s+(raise|increase|maintain|lower|cut)\s+the target range", joined
    )
    no_votes = int(vote.group(2)) if vote else 0
    prefers_hike = bool(re.search(r"preferred to\s+(raise|increase)\b", joined))
    prefers_cut = bool(re.search(r"preferred to\s+(lower|cut)\b", joined))
    if prefers_hike and not prefers_cut:
        hike_dissents, cut_dissents = no_votes, 0
    elif prefers_cut and not prefers_hike:
        hike_dissents, cut_dissents = 0, no_votes
    else:
        hike_dissents = len(re.findall(r"preferred to\s+(raise|increase)\b", joined))
        cut_dissents = len(re.findall(r"preferred to\s+(lower|cut)\b", joined))
    prefix = re.compile(
        r"^The Federal Open Market Committee approved the following statement "
        r"for release by a [^:]+:\s*"
    )
    rows = [cleaned for row in rows if (cleaned := prefix.sub("", row))]
    return {
        "date": day.isoformat(),
        "url": url,
        "vote": f"{vote.group(1)}-{vote.group(2)}" if vote else None,
        "action": action.group(1) if action else None,
        "hike_dissents": hike_dissents,
        "cut_dissents": cut_dissents,
        "paragraphs": rows,
    }


def parse_speech_list(page: str) -> list[dict]:
    pattern = re.compile(
        r"<time>(\d{2})/(\d{2})/(\d{4})</time>.*?"
        r'<a href="(/newsevents/speech/[^"]+\.htm)"><em>(.*?)</em></a>.*?'
        r'<p class="news__speaker">(.*?)</p>',
        re.S,
    )
    items: list[dict] = []
    seen: set[str] = set()
    for month, day, year, path, title, speaker in pattern.findall(page):
        url = BASE + path
        if url in seen:
            continue
        seen.add(url)
        items.append(
            {
                "date": date(int(year), int(month), int(day)).isoformat(),
                "url": url,
                "title": clean(title),
                "speaker": clean(speaker),
                "role": classify_role(clean(speaker)),
            }
        )
    return items


def classify_role(speaker: str) -> str:
    low = speaker.lower()
    if "vice chair" in low:
        return "vice_chair"
    if "chairman" in low or low.startswith("chair"):
        return "chair"
    if "governor" in low:
        return "governor"
    if "president" in low:
        return "president"
    return "other"


def relevance(text: str) -> int:
    low = text.lower()
    return sum(low.count(keyword) for keyword in POLICY_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取 FOMC 声明与演讲文本")
    parser.add_argument("--as-of", help="基准日 YYYY-MM-DD，默认上海时区当天")
    parser.add_argument("--statements", type=int, default=3)
    parser.add_argument("--speech-days", type=int, default=45)
    parser.add_argument("--max-speeches", type=int, default=10)
    parser.add_argument("--out-dir", default=str(ROOT / "data/text"))
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(ZoneInfo("Asia/Shanghai")).date()
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    calendar = json.loads((ROOT / "data/fomc_calendar.json").read_text(encoding="utf-8"))
    meetings = [
        row
        for row in calendar["meetings"]
        if row.get("type", "policy") == "policy"
        and date.fromisoformat(row["decision_date"]) <= as_of
    ]
    meetings.sort(key=lambda row: row["decision_date"])
    selected = meetings[-args.statements :]

    statements: list[dict] = []
    for row in selected:
        day = date.fromisoformat(row["decision_date"])
        url = STATEMENT_URL.format(date=day.strftime("%Y%m%d"))
        page = fetch(url)
        statements.append(parse_statement(page, day, url))
        print(f"statement {day} ok ({len(statements[-1]['paragraphs'])} 段)")

    statements_payload = {
        "source": "Federal Reserve FOMC statements",
        "retrieved_at": retrieved_at,
        "as_of": as_of.isoformat(),
        "statements": statements,
    }
    (out_dir / "fomc_statements.json").write_text(
        json.dumps(statements_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    list_page = fetch(SPEECH_LIST_URL.format(year=as_of.year))
    entries = parse_speech_list(list_page)
    start = as_of - timedelta(days=args.speech_days)
    candidates = [
        item
        for item in entries
        if start <= date.fromisoformat(item["date"]) <= as_of
    ]
    candidates.sort(key=lambda item: item["date"], reverse=True)

    speeches: list[dict] = []
    for item in candidates:
        if len(speeches) >= args.max_speeches:
            break
        page = fetch(item["url"])
        rows = paragraphs(page)
        body = " ".join(rows)
        title_relevance = relevance(item["title"])
        score_relevance = relevance(body)
        if title_relevance == 0 or score_relevance < 3:
            continue
        speeches.append({**item, "relevance": score_relevance, "paragraphs": rows})
        print(f"speech {item['date']} {item['speaker']} score_relevance={score_relevance}")

    speeches_payload = {
        "source": "Federal Reserve speeches",
        "retrieved_at": retrieved_at,
        "as_of": as_of.isoformat(),
        "window_days": args.speech_days,
        "list_url": SPEECH_LIST_URL.format(year=as_of.year),
        "speeches": speeches,
    }
    (out_dir / "fomc_speeches.json").write_text(
        json.dumps(speeches_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"out={out_dir} statements={len(statements)} speeches={len(speeches)}")


if __name__ == "__main__":
    main()
