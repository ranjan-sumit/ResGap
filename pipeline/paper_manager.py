"""
Paper Tier Manager — handles 50+ paper corpora gracefully.

Tier 1 (uploaded PDFs)      → full text + all pipeline stages
Tier 2 (cited references)   → abstract only → parameter extraction only
Tier 3 (distant references) → title + year only → gap signal / index

The LLM sees Tier 1 full wiki + Tier 2 abstract summaries.
Mathematical modelling uses parameters from both Tier 1 and Tier 2.
"""
import json
import re


# ── Tier assignment ────────────────────────────────────────────────────────────

TIER_LABELS = {
    1: ("Uploaded PDF",   "#58a6ff", "Full text processed"),
    2: ("Cited Paper",    "#3fb950", "Abstract + parameters"),
    3: ("Distant Ref",   "#8b949e", "Title/year only"),
}


def assign_tiers(uploaded_papers: list, reference_papers: list) -> list:
    """
    Merge uploaded and reference papers into one corpus with tier labels.
    Uploaded papers are always Tier 1.
    Reference papers keep their tier (2 or 3) from the extractor.
    Returns a flat list sorted: Tier 1 first, then Tier 2, then Tier 3.
    """
    corpus = []

    for p in uploaded_papers:
        p_copy = dict(p)
        p_copy["tier"]       = 1
        p_copy["tier_label"] = TIER_LABELS[1][0]
        p_copy["tier_color"] = TIER_LABELS[1][1]
        corpus.append(p_copy)

    for ref in reference_papers:
        tier      = ref.get("tier", 2)
        ref_copy  = dict(ref)
        ref_copy["tier"]       = tier
        ref_copy["tier_label"] = TIER_LABELS[tier][0]
        ref_copy["tier_color"] = TIER_LABELS[tier][1]
        corpus.append(ref_copy)

    corpus.sort(key=lambda x: x.get("tier", 9))
    return corpus


# ── Tier 2 wiki compilation (abstract-only) ───────────────────────────────────

TIER2_WIKI_PROMPT = """You are a research analyst. Given only the TITLE and ABSTRACT
of a research paper (no full text available), extract what you can.

Be conservative — only extract what is explicitly stated in the abstract.
Do not infer or extrapolate.

Return ONLY JSON:
{
  "title": "paper title",
  "year": <int or null>,
  "domain": "research domain",
  "contributions": ["1-2 contributions visible in abstract"],
  "methods": ["methods mentioned"],
  "datasets": ["datasets mentioned"],
  "key_findings": ["specific findings with numbers if stated"],
  "limitations": ["any limitations mentioned"],
  "key_concepts": ["important terms"],
  "confidence": "abstract_only"
}"""


def compile_tier2_wiki(ref: dict, client) -> dict:
    """
    Compile a lightweight wiki page for a Tier 2 paper (abstract only).
    Much faster and cheaper than full wiki compilation.
    """
    title    = ref.get("title", "Unknown Paper")
    abstract = ref.get("abstract", "")
    year     = ref.get("year", "")

    if not abstract or len(abstract) < 50:
        # Tier 3 fallback — no abstract available
        return {
            "title":       title,
            "year":        year,
            "domain":      "Unknown",
            "contributions": [],
            "methods":     [],
            "datasets":    [],
            "key_findings": [],
            "limitations": [],
            "key_concepts": [],
            "confidence":  "title_only",
            "source_file": ref.get("doi") or title[:40],
            "tier":        3,
        }

    prompt = f"Title: {title}\nYear: {year}\n\nAbstract:\n{abstract}"
    raw    = client.chat_json(TIER2_WIKI_PROMPT, prompt, max_tokens=1000)

    result = _safe_parse(raw)
    if not isinstance(result, dict):
        result = {}

    result["source_file"] = ref.get("doi") or title[:40]
    result["tier"]        = 2
    result.setdefault("title", title)
    result.setdefault("year", year)
    return result


def _safe_parse(raw: str) -> dict:
    raw = raw.strip()
    for attempt in [
        lambda s: s,
        lambda s: re.sub(r'^```(?:json)?\s*|\s*```$', '', s).strip(),
        lambda s: re.sub(r',\s*([}\]])', r'\1', s),
    ]:
        try:
            r = json.loads(attempt(raw))
            if isinstance(r, dict):
                return r
        except Exception:
            pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


def compile_tier2_batch(refs: list, client,
                        progress_callback=None) -> list:
    """
    Compile lightweight wiki pages for all Tier 2 references.
    Tier 3 refs (no abstract) get a stub wiki.
    Returns list of wiki dicts.
    """
    wikis = []
    tier2 = [r for r in refs if r.get("tier") == 2 and r.get("abstract")]
    tier3 = [r for r in refs if r.get("tier") == 3 or not r.get("abstract")]

    for i, ref in enumerate(tier2):
        if progress_callback:
            progress_callback(i + 1, len(tier2),
                              f"Compiling Tier 2 wiki {i+1}/{len(tier2)}: {ref.get('title','')[:40]}…")
        wiki = compile_tier2_wiki(ref, client)
        wikis.append(wiki)

    for ref in tier3:
        wikis.append({
            "title":       ref.get("title", "Unknown"),
            "year":        ref.get("year"),
            "domain":      "Unknown",
            "contributions": [],
            "methods":     [],
            "key_findings": [],
            "limitations": [],
            "key_concepts": [],
            "confidence":  "title_only",
            "source_file": ref.get("doi") or ref.get("title", "")[:40],
            "tier":        3,
        })

    return wikis


# ── Parameter extraction from Tier 2 ─────────────────────────────────────────

TIER2_PARAM_PROMPT = """You are a biostatistician. Extract quantitative parameters
from this paper's ABSTRACT ONLY. Be conservative — only extract numbers explicitly stated.

Return ONLY a JSON array of parameters. Same schema as Tier 1 extraction.
If nothing quantitative is stated, return [].

Parameter schema:
{
  "name": "parameter_name",
  "category": "prevalence|incidence|odds_ratio|relative_risk|hazard_ratio|efficacy|survival|other",
  "value": <float>,
  "ci_lower": <float or null>,
  "ci_upper": <float or null>,
  "unit": "percent|ratio|per_year|dimensionless",
  "population": "population described",
  "condition": "condition",
  "source_paper": "paper title",
  "source_section": "Abstract",
  "confidence": "medium",
  "is_derived": false,
  "notes": "abstract-only extraction"
}"""


def extract_tier2_parameters(refs: list, client,
                             progress_callback=None) -> list:
    """
    Extract parameters from Tier 2 paper abstracts.
    These feed into the simulation alongside Tier 1 params.
    """
    all_params = []
    eligible   = [r for r in refs if r.get("tier") == 2 and r.get("abstract")]

    for i, ref in enumerate(eligible):
        if progress_callback:
            progress_callback(i + 1, len(eligible),
                              f"Extracting Tier 2 params {i+1}/{len(eligible)}…")

        title    = ref.get("title", "Unknown")
        abstract = ref.get("abstract", "")
        prompt   = f"Title: {title}\n\nAbstract:\n{abstract}"

        raw    = client.chat_json(TIER2_PARAM_PROMPT, prompt, max_tokens=1500)
        params = _safe_parse_list(raw)

        for p in params:
            if isinstance(p, dict) and p.get("value") is not None:
                p["source_paper"] = p.get("source_paper") or title
                p["tier"]         = 2
                all_params.append(p)

    return all_params


def _safe_parse_list(raw: str) -> list:
    raw = raw.strip()
    for attempt in [
        lambda s: s,
        lambda s: re.sub(r'^```(?:json)?\s*|\s*```$', '', s).strip(),
        lambda s: re.sub(r',\s*([}\]])', r'\1', s),
    ]:
        try:
            r = json.loads(attempt(raw))
            if isinstance(r, list):
                return r
        except Exception:
            pass
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group())
            if isinstance(r, list):
                return r
        except Exception:
            pass
    return []


# ── Corpus statistics ─────────────────────────────────────────────────────────

def corpus_stats(corpus: list) -> dict:
    tier_counts = {1: 0, 2: 0, 3: 0}
    for p in corpus:
        tier_counts[p.get("tier", 3)] += 1
    return {
        "total":   len(corpus),
        "tier1":   tier_counts[1],
        "tier2":   tier_counts[2],
        "tier3":   tier_counts[3],
        "has_abstracts": tier_counts[1] + tier_counts[2],
    }
