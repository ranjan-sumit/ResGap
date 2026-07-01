"""
Stage 6: Academic Search Validator
Validates each identified gap against Semantic Scholar and arXiv APIs.
Determines if the gap is: Open / Partially Addressed / Already Solved.
"""
import requests
import urllib.parse
import time
import re
import json   # B3 fix — moved from inside _assess_coverage to module level

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_BASE = "http://export.arxiv.org/api/query"

FIELDS = "title,year,abstract,citationCount,authors"


def _search_semantic_scholar(query: str, limit: int = 5) -> list:
    """Search Semantic Scholar for papers related to a gap."""
    params = {
        "query": query,
        "limit": limit,
        "fields": FIELDS,
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_BASE, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            papers = data.get("data", [])
            results = []
            for p in papers:
                results.append({
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "abstract": (p.get("abstract") or "")[:300],
                    "citations": p.get("citationCount", 0),
                    "authors": [a.get("name", "") for a in (p.get("authors") or [])[:3]],
                    "source": "Semantic Scholar",
                })
            return results
    except Exception:
        pass
    return []


def _search_arxiv(query: str, limit: int = 3) -> list:
    """Search arXiv as a backup academic source."""
    params = {
        "search_query": f"all:{urllib.parse.quote(query)}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_BASE, params=params, timeout=10)
        if resp.status_code == 200:
            content = resp.text
            entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
            results = []
            for entry in entries:
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                year_match = re.search(r'<published>(\d{4})', entry)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                if title_match:
                    results.append({
                        "title": title_match.group(1).strip().replace("\n", " "),
                        "year": int(year_match.group(1)) if year_match else None,
                        "abstract": (summary_match.group(1).strip()[:300] if summary_match else ""),
                        "citations": None,
                        "source": "arXiv",
                    })
            return results
    except Exception:
        pass
    return []


def _build_search_query(gap: dict) -> str:
    """Build a focused academic search query from a gap description."""
    title = gap.get("title", "")
    missing = gap.get("missing_connection", "")
    description = gap.get("description", "")[:150]

    # Use title + missing connection for best query
    if missing:
        return f"{title} {missing}"[:120]
    return f"{title} {description}"[:120]


def _assess_coverage(gap: dict, found_papers: list, client) -> dict:
    """
    Use LLM to assess if the found papers actually address the gap.
    Returns status + reasoning.
    """
    if not found_papers:
        return {"status": "open", "reasoning": "No relevant papers found in academic search."}

    prompt = f"""Gap: {gap.get('title')}
Description: {gap.get('description')}

Papers found in academic search:
{json.dumps([{'title': p['title'], 'year': p['year'], 'abstract': p['abstract']} for p in found_papers[:5]], indent=2)}

Does the gap appear to be:
- "open": Not addressed by any of these papers
- "partial": Partially addressed but significant work remains
- "solved": Already thoroughly addressed

Return ONLY JSON: {{"status": "open|partial|solved", "reasoning": "1 sentence explanation"}}"""

    try:
        # B3 fix: client is passed in — use it directly, no re-import needed
        raw   = client.chat_json("You are a research gap validator.", prompt, max_tokens=200)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return result
    except Exception:
        pass

    # Fallback: if we found recent papers, call it partial
    recent = [p for p in found_papers if p.get("year") and p["year"] >= 2022]
    if len(recent) >= 2:
        return {"status": "partial", "reasoning": "Related recent work found but gap may still exist."}
    return {"status": "open", "reasoning": "No directly relevant recent papers found."}


def validate_gaps(gaps: list, client, domain_config: dict = None) -> list:
    """
    Validate all gaps against academic databases.
    Updates each gap with: validation_status, existing_papers, reasoning.
    """
    boost_terms = []
    if domain_config:
        from domain_config import get_search_boost_terms
        boost_terms = get_search_boost_terms(domain_config)

    validated = []
    for gap in gaps:
        query = _build_search_query(gap)

        # Boost query with 1-2 domain-specific terms for better precision
        if boost_terms:
            # Pick the most relevant boost term based on gap title
            gap_lower = gap.get("title", "").lower()
            relevant_boosts = [t for t in boost_terms if any(w in gap_lower for w in t.split())]
            boost = relevant_boosts[0] if relevant_boosts else boost_terms[0]
            query = f"{query} {boost}"
            query = query[:150]  # keep query length reasonable

        # Search both sources
        ss_papers = _search_semantic_scholar(query, limit=5)
        time.sleep(0.5)
        arxiv_papers = _search_arxiv(query, limit=3)

        all_papers = ss_papers + arxiv_papers

        assessment = _assess_coverage(gap, all_papers, client)

        gap["validation_status"] = assessment.get("status", "open")
        gap["validation_reasoning"] = assessment.get("reasoning", "")
        gap["existing_papers"] = all_papers[:5]
        gap["search_query_used"] = query

        validated.append(gap)
        time.sleep(0.3)

    return validated
