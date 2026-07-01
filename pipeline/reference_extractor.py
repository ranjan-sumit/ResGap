"""
Stage: Reference Extractor
Extracts cited references from uploaded PDFs, then enriches them
with metadata from Semantic Scholar, CrossRef and OpenAlex.

Solves the 50+ paper problem:
  Uploaded PDFs     → Tier 1 (full processing)
  Cited references  → Tier 2 (abstract + params, no PDF needed)
  Refs of refs      → Tier 3 (title/year only, gap signal)

No Apify needed — three free academic APIs cover everything.
"""
import re
import json
import time
import html
import requests
import urllib.parse
from typing import Optional

# ── API constants ──────────────────────────────────────────────────────────────
SS_SEARCH   = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_PAPER    = "https://api.semanticscholar.org/graph/v1/paper/{id}"
SS_REFS     = "https://api.semanticscholar.org/graph/v1/paper/{id}/references"
CROSSREF    = "https://api.crossref.org/works/{doi}"
OPENALEX    = "https://api.openalex.org/works"
UNPAYWALL   = "https://api.unpaywall.org/v2/{doi}"

SS_FIELDS   = "title,year,abstract,authors,externalIds,citationCount,referenceCount"
_HEADERS    = {"User-Agent": "ResearchGapAnalyzer/1.0 (academic research tool)"}
_TIMEOUT    = 8


# ── Reference text parsing ─────────────────────────────────────────────────────

DOI_RE    = re.compile(r'10\.\d{4,9}/[^\s\n,;\"\'<>]+', re.IGNORECASE)
ARXIV_RE  = re.compile(r'arxiv[:\s]+(\d{4}\.\d{4,5})', re.IGNORECASE)
YEAR_RE   = re.compile(r'\b(19|20)\d{2}\b')


def _extract_dois_from_text(text: str) -> list[str]:
    """Pull all DOIs from raw text using regex."""
    raw = DOI_RE.findall(text)
    # Clean trailing punctuation that crept in
    cleaned = []
    for doi in raw:
        doi = doi.rstrip('.,;)>]"\'')
        if len(doi) > 8:
            cleaned.append(doi.lower())
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


def _extract_arxiv_ids(text: str) -> list[str]:
    return list(dict.fromkeys(ARXIV_RE.findall(text)))


def _split_reference_entries(refs_text: str) -> list[str]:
    """
    Split a References section into individual citation strings.
    Handles numbered [1], (1), 1. styles and author-year styles.
    """
    refs_text = re.sub(r'\s+', ' ', refs_text or '').strip()
    if not refs_text:
        return []

    # Try numbered patterns first. PDF extraction often collapses references
    # onto one line, so do not require a leading newline.
    numbered = re.split(r'(?:^|\s)(?:\[\d+\]|\(\d+\)|\d+\.)\s+', refs_text)
    if len(numbered) > 3:
        return [r.strip() for r in numbered if len(r.strip()) > 20]

    # Fall back: split where a likely year ends one citation and a capitalized
    # author/title begins the next.
    entries = re.split(r'(?<=\b(?:19|20)\d{2}[.;])\s+(?=[A-Z][A-Za-z\-]+[,.;])', refs_text)
    entries = [e.strip() for e in entries if len(e.strip()) > 20]
    return entries


def _entry_to_reference(entry: str) -> dict:
    """Best-effort citation parser used when DOI/LLM parsing misses a ref."""
    clean = re.sub(r'\s+', ' ', entry or '').strip()
    clean = re.sub(r'^(?:\[\d+\]|\(\d+\)|\d+\.)\s+', '', clean)
    year_match = YEAR_RE.search(clean)
    year = int(year_match.group(0)) if year_match else None
    doi_match = DOI_RE.search(clean)
    doi = doi_match.group(0).rstrip('.,;)').lower() if doi_match else None

    title = ""
    if year_match:
        after_year = clean[year_match.end():].strip(" .;:-")
        before_year = clean[:year_match.start()].strip(" .;:-")
        # Vancouver/MDPI style often has authors first, then title, then journal/year.
        parts = [p.strip() for p in re.split(r'\.\s+', before_year) if p.strip()]
        if len(parts) >= 2:
            title = parts[-1]
        elif after_year:
            title = after_year.split(". ")[0].strip()
        else:
            title = before_year
    else:
        parts = [p.strip() for p in re.split(r'\.\s+', clean) if p.strip()]
        title = parts[1] if len(parts) > 1 else clean[:180]

    if not title or len(title) < 8:
        title = clean[:180]

    return {
        "title": html.unescape(title[:240]),
        "authors": [],
        "year": year,
        "doi": doi,
        "journal": None,
        "arxiv_id": None,
        "raw_reference": clean[:1200],
    }


def parse_references_section(paper: dict) -> dict:
    """
    Extract raw reference text and initial DOIs from a parsed paper dict.
    Returns {'raw_text': str, 'dois': list, 'arxiv_ids': list, 'entries': list}
    """
    sections = paper.get("sections", {})

    # Find the references section. Some PDFs produce several table/section
    # headers containing "References"; prefer the longest candidate.
    ref_candidates = []
    for sec_name, sec_text in sections.items():
        if re.search(r'references?|bibliography', sec_name.lower()):
            ref_candidates.append(str(sec_text))
    ref_text = max(ref_candidates, key=len) if ref_candidates else ""

    if not ref_text:
        # Try end of full_text (references usually at the end)
        full = paper.get("full_text", "")
        idx  = max(
            full.lower().rfind("references"),
            full.lower().rfind("bibliography"),
        )
        if idx > 0:
            ref_text = full[idx:][:25000]

    return {
        "raw_text":  ref_text[:25000],
        "dois":      _extract_dois_from_text(ref_text),
        "arxiv_ids": _extract_arxiv_ids(ref_text),
        "entries":   _split_reference_entries(ref_text),
    }


# ── LLM-assisted reference parsing ────────────────────────────────────────────

REF_PARSE_PROMPT = """You are a bibliographic parser. Extract structured references 
from this References section text.

For each reference return:
{
  "title": "exact paper title",
  "authors": ["Last FM", "Last FM"],
  "year": 2023,
  "doi": "10.xxxx/xxx or null",
  "journal": "journal name or null",
  "arxiv_id": "XXXX.XXXXX or null"
}

Return ONLY a JSON array. Include up to 30 references.
If title is unclear, skip that entry."""


def parse_references_with_llm(ref_text: str, client) -> list[dict]:
    """Use LLM to parse messy reference text into structured entries."""
    if not ref_text or len(ref_text) < 50:
        return []
    raw = client.chat_json(
        REF_PARSE_PROMPT,
        f"References section:\n\n{ref_text[:6000]}",
        max_tokens=4000,
    )
    result = _safe_parse_list(raw)
    return [r for r in result if isinstance(r, dict) and r.get("title")]


def _safe_parse_list(raw: str) -> list:
    raw = raw.strip()
    for attempt in [
        lambda s: s,
        lambda s: re.sub(r'^```(?:json)?\s*|\s*```$', '', s).strip(),
        lambda s: re.sub(r',\s*([}\]])', r'\1', s),
    ]:
        try:
            result = json.loads(attempt(raw))
            if isinstance(result, list):
                return result
        except Exception:
            pass
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return []


# ── Academic API lookups ───────────────────────────────────────────────────────

def _ss_search_by_title(title: str) -> Optional[dict]:
    """Search Semantic Scholar by title. Returns best match or None."""
    try:
        params = {"query": title[:200], "limit": 3, "fields": SS_FIELDS}
        r = requests.get(SS_SEARCH, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            items = r.json().get("data", [])
            for item in items:
                if item.get("title") and item.get("abstract"):
                    return item
    except Exception:
        pass
    return None


def _ss_by_doi(doi: str) -> Optional[dict]:
    try:
        url = SS_PAPER.format(id=urllib.parse.quote(f"DOI:{doi}", safe=""))
        r   = requests.get(url, params={"fields": SS_FIELDS},
                           headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _ss_by_arxiv(arxiv_id: str) -> Optional[dict]:
    try:
        url = SS_PAPER.format(id=urllib.parse.quote(f"ARXIV:{arxiv_id}", safe=""))
        r   = requests.get(url, params={"fields": SS_FIELDS},
                           headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _crossref_by_doi(doi: str) -> Optional[dict]:
    try:
        r = requests.get(
            CROSSREF.format(doi=urllib.parse.quote(doi, safe="")),
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            msg = r.json().get("message", {})
            return {
                "title":    (msg.get("title") or [""])[0],
                "year":     (msg.get("published", {}).get("date-parts") or [[None]])[0][0],
                "abstract": msg.get("abstract", ""),
                "doi":      doi,
                "venue":    ((msg.get("container-title") or [""])[0]
                             if isinstance(msg.get("container-title"), list) else msg.get("container-title", "")),
                "authors":  _crossref_authors(msg),
                "citations": msg.get("is-referenced-by-count", 0),
                "url":      msg.get("URL") or f"https://doi.org/{doi}",
                "source_api": "crossref",
            }
    except Exception:
        pass
    return None


def _crossref_authors(item: dict) -> list[str]:
    authors = []
    for a in item.get("author", [])[:6]:
        name = f"{a.get('given','')} {a.get('family','')}".strip()
        if name:
            authors.append(name)
    return authors


def _normalise_authors(authors) -> list[str]:
    names = []
    for a in authors or []:
        if isinstance(a, dict):
            name = a.get("name") or f"{a.get('given','')} {a.get('family','')}".strip()
        else:
            name = str(a)
        if name:
            names.append(name)
    return names[:6]


def _crossref_search(title: str) -> Optional[dict]:
    try:
        params = {
            "query": title[:200],
            "rows": 1,
            "select": "DOI,title,author,container-title,published,abstract,URL,is-referenced-by-count",
        }
        r = requests.get("https://api.crossref.org/works",
                         params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                doi = item.get("DOI", "")
                return {
                    "title":    ((item.get("title") or [""])[0]
                                 if isinstance(item.get("title"), list) else item.get("title", "")),
                    "year":     (item.get("published", {}).get("date-parts") or [[None]])[0][0],
                    "abstract": re.sub(r'<[^>]+>', '', item.get("abstract", "") or ""),
                    "doi":      doi,
                    "venue":    ((item.get("container-title") or [""])[0]
                                 if isinstance(item.get("container-title"), list) else item.get("container-title", "")),
                    "authors":  _crossref_authors(item),
                    "citations": item.get("is-referenced-by-count", 0),
                    "url":      item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                    "source_api": "crossref_search",
                }
    except Exception:
        pass
    return None


def _unpaywall_by_doi(doi: str) -> Optional[str]:
    try:
        url = UNPAYWALL.format(doi=urllib.parse.quote(doi, safe=""))
        r = requests.get(url, params={"email": "research@example.com"},
                         headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if data.get("is_oa"):
                loc = data.get("best_oa_location") or {}
                return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        pass
    return None


def _pubmed_search(doi: str = "", title: str = "") -> Optional[dict]:
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        term = f"{doi}[doi]" if doi else title[:180]
        if not term:
            return None
        r = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": 1, "retmode": "json"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        pmid = ids[0]
        r2 = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "json"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if r2.status_code == 200:
            summ = r2.json().get("result", {}).get(pmid, {})
            return {
                "title": summ.get("title", ""),
                "year": (summ.get("pubdate", "") or "")[:4],
                "abstract": "",
                "doi": doi,
                "venue": summ.get("source", ""),
                "authors": [a.get("name", "") for a in summ.get("authors", [])[:6]],
                "citations": 0,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_api": "pubmed",
            }
    except Exception:
        pass
    return None


def _openalex_by_title(title: str) -> Optional[dict]:
    try:
        params = {
            "search":  title[:150],
            "select":  "title,publication_year,abstract_inverted_index,doi",
            "per-page": 1,
        }
        r = requests.get(OPENALEX, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                item = results[0]
                # Reconstruct abstract from inverted index
                inv = item.get("abstract_inverted_index") or {}
                words = sorted(
                    ((pos, word) for word, positions in inv.items() for pos in positions),
                    key=lambda x: x[0],
                )
                abstract = " ".join(w for _, w in words[:200]) if words else ""
                return {
                    "title":    item.get("title", title),
                    "year":     item.get("publication_year"),
                    "abstract": abstract,
                    "doi":      (item.get("doi") or "").replace("https://doi.org/", ""),
                    "venue":    "",
                    "authors":  [],
                    "citations": 0,
                    "url":      item.get("doi") or "",
                    "source_api": "openalex",
                }
    except Exception:
        pass
    return None


# ── Enrichment ────────────────────────────────────────────────────────────────

def enrich_reference(ref: dict, delay: float = 0.3) -> dict:
    """
    Try multiple APIs to get title + abstract for one reference.
    Priority: Semantic Scholar (DOI) → SS (arxiv) → CrossRef → OpenAlex → SS (title)
    """
    doi      = (ref.get("doi") or "").strip().lower()
    arxiv_id = (ref.get("arxiv_id") or "").strip()
    title    = (ref.get("title") or "").strip()

    meta = None

    if doi:
        meta = _ss_by_doi(doi)
        time.sleep(delay)

    if not meta and arxiv_id:
        meta = _ss_by_arxiv(arxiv_id)
        time.sleep(delay)

    if not meta and doi:
        meta = _crossref_by_doi(doi)
        time.sleep(delay)

    if not meta and doi:
        meta = _pubmed_search(doi=doi)
        time.sleep(delay)

    if not meta and title and len(title) > 10:
        meta = _openalex_by_title(title)
        time.sleep(delay)

    if not meta and title and len(title) > 10:
        meta = _crossref_search(title)
        time.sleep(delay)

    if not meta and title and len(title) > 10:
        meta = _ss_search_by_title(title)
        time.sleep(delay)

    if not meta and title and len(title) > 10:
        meta = _pubmed_search(title=title)
        time.sleep(delay)

    if meta:
        resolved_doi = doi or (meta.get("externalIds") or {}).get("DOI", "") or meta.get("doi", "")
        open_access_url = None
        if resolved_doi:
            open_access_url = _unpaywall_by_doi(resolved_doi)
        # Normalise to our schema
        return {
            "title":       meta.get("title") or title,
            "year":        meta.get("year") or meta.get("publication_year") or ref.get("year"),
            "abstract":    (meta.get("abstract") or "")[:1500],
            "doi":         resolved_doi,
            "arxiv_id":    arxiv_id,
            "authors":     _normalise_authors(meta.get("authors") or ref.get("authors", [])),
            "venue":       meta.get("venue", ""),
            "citations":   meta.get("citationCount", meta.get("citations", 0)),
            "ss_paper_id": meta.get("paperId", ""),
            "url":         meta.get("url") or (f"https://doi.org/{resolved_doi}" if resolved_doi else ""),
            "open_access_url": open_access_url or meta.get("open_access_url"),
            "tier":        2,
            "source":      meta.get("source_api", "api_lookup"),
        }

    # Fallback: minimal record from what we parsed
    return {
        "title":    title or "Unknown",
        "year":     ref.get("year"),
        "abstract": "",
        "doi":      doi,
        "arxiv_id": arxiv_id,
        "authors":  ref.get("authors", []),
        "citations": 0,
        "tier":     3,   # Tier 3 — title only, no abstract
        "source":   "parsed_only",
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def extract_and_enrich_references(
    papers: list[dict],
    client,
    max_refs_per_paper: int = 30,
    progress_callback=None,
    return_diagnostics: bool = False,
) -> list[dict]:
    """
    Full reference extraction pipeline for a list of uploaded papers.

    Returns list of enriched reference dicts (Tier 2 and 3).
    Tier 1 papers (uploaded) are NOT included in the output.

    progress_callback: optional fn(current, total, message)
    """
    all_refs  = []
    seen_dois = set()
    seen_titles: set[str] = set()
    diagnostics = {
        "papers": [],
        "raw_entries": 0,
        "structured_entries": 0,
        "fallback_entries": 0,
        "doi_entries": 0,
        "enriched": 0,
        "title_only": 0,
        "duplicates_skipped": 0,
        "errors": [],
    }

    def _norm(value) -> str:
        return str(value or "").strip().lower()

    for paper_idx, paper in enumerate(papers):
        filename = paper.get("filename", f"Paper {paper_idx+1}")
        if progress_callback:
            progress_callback(paper_idx + 1, len(papers),
                              f"Parsing references from {filename}…")

        # Step 1: regex extraction (fast, no API)
        parsed = parse_references_section(paper)
        raw_entries = parsed["entries"]
        diagnostics["raw_entries"] += len(raw_entries)

        # Step 2: LLM parsing for structured fields
        if client:
            try:
                structured = parse_references_with_llm(parsed["raw_text"], client)
            except Exception as e:
                structured = []
                diagnostics["errors"].append(f"{filename}: LLM reference parse failed: {e}")
        else:
            structured = []
        diagnostics["structured_entries"] += len(structured)

        # Merge: structured entries + DOI-only entries + raw-entry fallback.
        combined: list[dict] = list(structured)
        for doi in parsed["dois"]:
            if not any(_norm(e.get("doi")) == doi for e in combined):
                combined.append({"doi": doi, "title": "", "year": None, "authors": []})
                diagnostics["doi_entries"] += 1

        for entry in raw_entries:
            fallback = _entry_to_reference(entry)
            title_key = _norm(fallback.get("title"))[:60]
            doi_key = _norm(fallback.get("doi"))
            if doi_key and any(_norm(e.get("doi")) == doi_key for e in combined):
                continue
            if title_key and len(title_key) > 10 and any(_norm(e.get("title"))[:60] == title_key for e in combined):
                continue
            combined.append(fallback)
            diagnostics["fallback_entries"] += 1

        diagnostics["papers"].append({
            "filename": filename,
            "raw_entries": len(raw_entries),
            "structured_entries": len(structured),
            "combined_entries": len(combined),
            "dois": len(parsed["dois"]),
        })

        # Step 3: Enrich up to max_refs_per_paper per paper
        refs_to_process = combined if max_refs_per_paper is None else combined[:max_refs_per_paper]
        for i, ref in enumerate(refs_to_process):
            doi_key   = _norm(ref.get("doi"))
            title_key = _norm(ref.get("title"))[:60]

            # Skip duplicates across papers
            if doi_key and doi_key in seen_dois:
                diagnostics["duplicates_skipped"] += 1
                continue
            if title_key and len(title_key) > 10 and title_key in seen_titles:
                diagnostics["duplicates_skipped"] += 1
                continue

            if progress_callback:
                progress_callback(
                    paper_idx + 1, len(papers),
                    f"Enriching ref {i+1}/{len(refs_to_process)} from {filename}...",
                )

            try:
                enriched = enrich_reference(ref, delay=0.25)
            except Exception as e:
                diagnostics["errors"].append(f"{filename}: enrichment failed for {ref.get('title','Unknown')[:80]}: {e}")
                enriched = {
                    **ref,
                    "abstract": "",
                    "citations": 0,
                    "tier": 3,
                    "source": "enrichment_failed",
                }
            enriched["found_in_paper"] = filename
            enriched.setdefault("raw_reference", ref.get("raw_reference", ""))

            if doi_key:
                seen_dois.add(doi_key)
            if title_key and len(title_key) > 10:
                seen_titles.add(title_key)

            if enriched.get("tier") == 2 and enriched.get("abstract"):
                diagnostics["enriched"] += 1
            else:
                diagnostics["title_only"] += 1
            all_refs.append(enriched)

    if return_diagnostics:
        return all_refs, diagnostics
    return all_refs


def summarise_reference_corpus(refs: list[dict]) -> dict:
    """
    Summary stats for display in the UI.

    Handles mixed year formats from different metadata sources:
    - int: 2010
    - str: "2010"
    - partial date strings: "2010 Jan", "2010-05-01"
    - None / invalid values
    """
    refs = refs or []

    tier2 = [r for r in refs if r.get("tier") == 2]
    tier3 = [r for r in refs if r.get("tier") == 3]
    with_abstract = [r for r in refs if r.get("abstract")]

    years = []
    for r in refs:
        raw_year = r.get("year")
        if raw_year is None:
            continue

        try:
            year = int(str(raw_year).strip()[:4])
        except (TypeError, ValueError):
            continue

        if 1800 <= year <= 2100:
            years.append(year)

    return {
        "total": len(refs),
        "tier2_abstract": len(tier2),
        "tier3_title": len(tier3),
        "with_abstract": len(with_abstract),
        "year_range": f"{min(years)}–{max(years)}" if years else "N/A",
    }
