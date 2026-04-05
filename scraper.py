"""
GDPR article scraper for the RegComplianceEnv project.

Uses Firecrawl to batch-scrape key GDPR articles from gdpr-info.eu,
extracts structured data, and caches results to ``data/gdpr_cache.json``.
Falls back to ``STATIC_GDPR_DATA`` when no cache is available so that
inference never fails due to missing network data.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firecrawl import Firecrawl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_FILE = DATA_DIR / "gdpr_cache.json"

GDPR_URLS: list[str] = [
    "https://gdpr-info.eu/art-5-gdpr/",
    "https://gdpr-info.eu/art-6-gdpr/",
    "https://gdpr-info.eu/art-7-gdpr/",
    "https://gdpr-info.eu/art-12-gdpr/",
    "https://gdpr-info.eu/art-13-gdpr/",
    "https://gdpr-info.eu/art-14-gdpr/",
    "https://gdpr-info.eu/art-17-gdpr/",
]

# Mapping from URL slug → human-friendly article number
_SLUG_TO_NUM: dict[str, str] = {
    "art-5": "5",
    "art-6": "6",
    "art-7": "7",
    "art-12": "12",
    "art-13": "13",
    "art-14": "14",
    "art-17": "17",
}


# ---------------------------------------------------------------------------
# Static fallback data (real GDPR text excerpts)
# ---------------------------------------------------------------------------

STATIC_GDPR_DATA: dict[str, dict[str, Any]] = {
    "5": {
        "article_number": "5",
        "title": "Principles relating to processing of personal data",
        "full_text": (
            "1. Personal data shall be:\n"
            "(a) processed lawfully, fairly and in a transparent manner in relation "
            "to the data subject ('lawfulness, fairness and transparency');\n"
            "(b) collected for specified, explicit and legitimate purposes and not "
            "further processed in a manner that is incompatible with those purposes; "
            "further processing for archiving purposes in the public interest, "
            "scientific or historical research purposes or statistical purposes "
            "shall, in accordance with Article 89(1), not be considered to be "
            "incompatible with the initial purposes ('purpose limitation');\n"
            "(c) adequate, relevant and limited to what is necessary in relation to "
            "the purposes for which they are processed ('data minimisation');\n"
            "(d) accurate and, where necessary, kept up to date; every reasonable "
            "step must be taken to ensure that personal data that are inaccurate, "
            "having regard to the purposes for which they are processed, are erased "
            "or rectified without delay ('accuracy');\n"
            "(e) kept in a form which permits identification of data subjects for no "
            "longer than is necessary for the purposes for which the personal data "
            "are processed; personal data may be stored for longer periods insofar "
            "as the personal data will be processed solely for archiving purposes in "
            "the public interest, scientific or historical research purposes or "
            "statistical purposes in accordance with Article 89(1) subject to "
            "implementation of the appropriate technical and organisational measures "
            "required by this Regulation in order to safeguard the rights and "
            "freedoms of the data subject ('storage limitation');\n"
            "(f) processed in a manner that ensures appropriate security of the "
            "personal data, including protection against unauthorised or unlawful "
            "processing and against accidental loss, destruction or damage, using "
            "appropriate technical or organisational measures ('integrity and "
            "confidentiality').\n"
            "2. The controller shall be responsible for, and be able to demonstrate "
            "compliance with, paragraph 1 ('accountability')."
        ),
        "key_obligations": [
            "Lawfulness, fairness and transparency",
            "Purpose limitation",
            "Data minimisation",
            "Accuracy",
            "Storage limitation",
            "Integrity and confidentiality",
            "Accountability",
        ],
    },
    "6": {
        "article_number": "6",
        "title": "Lawfulness of processing",
        "full_text": (
            "1. Processing shall be lawful only if and to the extent that at least "
            "one of the following applies:\n"
            "(a) the data subject has given consent to the processing of his or her "
            "personal data for one or more specific purposes;\n"
            "(b) processing is necessary for the performance of a contract to which "
            "the data subject is party or in order to take steps at the request of "
            "the data subject prior to entering into a contract;\n"
            "(c) processing is necessary for compliance with a legal obligation to "
            "which the controller is subject;\n"
            "(d) processing is necessary in order to protect the vital interests of "
            "the data subject or of another natural person;\n"
            "(e) processing is necessary for the performance of a task carried out "
            "in the public interest or in the exercise of official authority vested "
            "in the controller;\n"
            "(f) processing is necessary for the purposes of the legitimate "
            "interests pursued by the controller or by a third party, except where "
            "such interests are overridden by the interests or fundamental rights "
            "and freedoms of the data subject which require protection of personal "
            "data, in particular where the data subject is a child.\n"
            "Point (f) of the first subparagraph shall not apply to processing "
            "carried out by public authorities in the performance of their tasks.\n"
            "2. Member States may maintain or introduce more specific provisions to "
            "adapt the application of the rules of this Regulation with regard to "
            "processing for compliance with points (c) and (e) of paragraph 1 by "
            "determining more precisely specific requirements for the processing and "
            "other measures to ensure lawful and fair processing including for other "
            "specific processing situations as provided for in Chapter IX.\n"
            "3. The basis for the processing referred to in point (c) and (e) of "
            "paragraph 1 shall be laid down by: (a) Union law; or (b) Member State "
            "law to which the controller is subject.\n"
            "4. Where the processing for a purpose other than that for which the "
            "personal data have been collected is not based on the data subject's "
            "consent or on a Union or Member State law, the controller shall, in "
            "order to ascertain whether processing for another purpose is compatible "
            "with the purpose for which the personal data are initially collected, "
            "take into account, inter alia: (a) any link between the purposes; "
            "(b) the context; (c) the nature of the data; (d) the possible "
            "consequences; (e) the existence of appropriate safeguards."
        ),
        "key_obligations": [
            "Obtain valid legal basis before processing",
            "Consent must be specific, informed, and unambiguous",
            "Contractual necessity as lawful basis",
            "Legal obligation compliance",
            "Vital interests protection",
            "Public interest or official authority",
            "Legitimate interests balancing test",
            "Purpose compatibility assessment for further processing",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helper: extract article number from a gdpr-info.eu URL slug
# ---------------------------------------------------------------------------

def _article_number_from_url(url: str) -> str:
    """Extract the article number from a gdpr-info.eu URL.

    >>> _article_number_from_url("https://gdpr-info.eu/art-13-gdpr/")
    '13'
    """
    for slug, num in _SLUG_TO_NUM.items():
        if slug in url:
            return num
    # Fallback: try to pull digits from the URL
    match = re.search(r"art-(\d+)", url)
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------------------
# Helper: parse scraped markdown into structured fields
# ---------------------------------------------------------------------------

def _parse_article_markdown(markdown: str, article_number: str) -> dict[str, Any]:
    """Parse scraped markdown content into a structured article dict.

    Extracts a title, full text, and key obligations from the raw markdown.
    """
    lines = markdown.strip().splitlines()

    # Title: first heading or first non-empty line
    title = f"Article {article_number}"
    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            break

    # Key obligations: lines that look like enumerated points
    obligations: list[str] = []
    for line in lines:
        cleaned = line.strip()
        # Match lines starting with (a), (b), bullet, or numbered list markers
        if re.match(r"^(\([a-z]\)|\d+\.|[-•*])\s+", cleaned):
            # Trim the marker
            text = re.sub(r"^(\([a-z]\)|\d+\.|[-•*])\s+", "", cleaned)
            if len(text) > 15:  # skip very short fragments
                obligations.append(text[:200])  # cap length

    return {
        "article_number": article_number,
        "title": title,
        "full_text": markdown.strip(),
        "key_obligations": obligations or [f"See full text of Article {article_number}"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_gdpr_articles() -> dict[str, Any]:
    """Batch-scrape key GDPR articles from gdpr-info.eu using Firecrawl.

    Results are saved as ``data/gdpr_cache.json`` with a UTC timestamp.
    Returns the full cache dict keyed by article number.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is not set. "
            "Set it in your environment or .env file."
        )

    client = Firecrawl(api_key=api_key)

    # Batch scrape with polling
    job = client.batch_scrape(
        GDPR_URLS,
        formats=["markdown"],
        poll_interval=2,
        wait_timeout=120,
    )

    # Build structured cache
    articles: dict[str, Any] = {}

    for item in job.data:
        url = getattr(item, "metadata", {}).get("sourceURL", "") or ""
        markdown = getattr(item, "markdown", "") or ""

        art_num = _article_number_from_url(url)
        if art_num == "unknown" and not url:
            continue

        articles[art_num] = _parse_article_markdown(markdown, art_num)

    cache_payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "articles": articles,
    }

    # Persist to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return articles


def load_gdpr_cache() -> dict[str, Any]:
    """Load cached GDPR articles from disk, falling back to static data.

    Returns a dict keyed by article number, e.g. ``{"5": {...}, "6": {...}}``.
    """
    if CACHE_FILE.exists():
        try:
            payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            articles = payload.get("articles", {})
            if articles:
                return articles
        except (json.JSONDecodeError, KeyError):
            pass  # fall through to static data

    # Fallback: return a copy so callers can't mutate the module constant
    return {k: dict(v) for k, v in STATIC_GDPR_DATA.items()}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scraping GDPR articles from gdpr-info.eu …")
    result = scrape_gdpr_articles()
    print(f"✓ Successfully scraped {len(result)} articles → {CACHE_FILE}")
    for num, data in sorted(result.items(), key=lambda x: int(x[0])):
        print(f"  • Article {num}: {data['title']}")
