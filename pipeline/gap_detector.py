"""
Stage 5: Gap Detector Agent
Queries the knowledge graph + wiki with domain context
to identify citation-grounded research gaps.
"""
import json
from domain_config import inject_domain_into_gap_prompt


GAP_DETECTION_PROMPT = """You are an expert research gap analyst. Your task is to identify genuine, specific, and actionable research gaps from a set of analyzed papers.

Researcher Context:
- Domain: {domain}
- Research Interest: {interest}
- Gap Type Focus: {gap_type}

You have access to:
1. Compiled wiki pages from {n_papers} papers
2. Knowledge graph with entities and communities
3. Orphan concepts (mentioned once, never built upon)
4. Missing bridges (limitations in one paper that could be solved by another)

A GOOD research gap must be:
- Specific and actionable (not "more research needed")
- Supported by evidence from the papers
- Not already the main contribution of one of the papers
- Feasible (something a PhD student could actually work on)

Return ONLY a JSON array of gaps:
[
  {{
    "title": "Short descriptive title of the gap",
    "description": "2-3 sentence specific description of what is missing and why it matters",
    "gap_type": "methodology|application|dataset|evaluation|theory|benchmark",
    "confidence": "high|medium|low",
    "evidence": [
      "Paper X (Section: Limitations): exact quote or paraphrase",
      "Paper Y (Section: Future Work): exact quote or paraphrase"
    ],
    "supporting_papers": ["paper title 1", "paper title 2"],
    "missing_connection": "What concept/method/application combination is absent",
    "why_important": "Why filling this gap would matter to the field"
  }}
]"""


def detect_gaps(wiki: dict, graph: dict, context: dict, client, domain_config: dict = None) -> list:
    """
    Main gap detection: uses wiki + graph + domain context to find research gaps.
    Every gap is grounded with citations.
    """
    pages = wiki.get("pages", [])
    communities = graph.get("communities", [])
    orphans = graph.get("orphan_concepts", [])
    bridges = graph.get("missing_bridges", [])
    cross_links = wiki.get("cross_links", {})

    # Build comprehensive context for the detector
    wiki_summary = []
    for p in pages:
        wiki_summary.append({
            "title": p.get("title", p.get("source_file", "Unknown")),
            "contributions": p.get("contributions", []),
            "methods": p.get("methods", []),
            "limitations": p.get("limitations", []),
            "future_work": p.get("future_work", []),
            "evaluated_on": p.get("evaluated_on", []),
            "comparison_gap": p.get("comparison_gap", ""),
        })

    community_summary = [
        {
            "theme": c.get("theme", ""),
            "summary": c.get("summary", ""),
            "gap_signal": c.get("gap_signal", ""),
        }
        for c in communities
    ]

    orphan_signals = [o.get("signal", "") for o in orphans[:10]]
    bridge_signals = [b.get("description", "") for b in bridges[:10]]
    conflicts = cross_links.get("conflicts", [])
    complementary = cross_links.get("complementary_pairs", [])

    prompt_system = inject_domain_into_gap_prompt(
        GAP_DETECTION_PROMPT.format(
            domain=context.get("domain", "General AI/ML"),
            interest=context.get("interest", "General research"),
            gap_type=context.get("gap_type", "Any"),
            n_papers=len(pages),
        ),
        domain_config or {}
    )

    user_prompt = f"""WIKI PAGES (per-paper compiled knowledge):
{json.dumps(wiki_summary, indent=2)}

KNOWLEDGE GRAPH COMMUNITIES:
{json.dumps(community_summary, indent=2)}

ORPHAN CONCEPTS (mentioned but not built upon):
{json.dumps(orphan_signals, indent=2)}

MISSING BRIDGES (limitations that could be solved cross-paper):
{json.dumps(bridge_signals, indent=2)}

COMPLEMENTARY PAPER PAIRS:
{json.dumps(complementary, indent=2)}

CONFLICTS BETWEEN PAPERS:
{json.dumps(conflicts, indent=2)}

Based on all this, identify 4-7 specific, evidence-grounded research gaps.
Prioritize gaps aligned with the researcher's domain: {context.get('domain', 'General AI/ML')}"""

    raw = client.chat_json(prompt_system, user_prompt, max_tokens=4000)

    try:
        gaps = json.loads(raw)
        if not isinstance(gaps, list):
            gaps = []
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                gaps = json.loads(match.group())
            except Exception:
                gaps = []
        else:
            gaps = []

    # Attach validation placeholder
    for g in gaps:
        g["validation_status"] = "pending"
        g["existing_papers"] = []

    return gaps
