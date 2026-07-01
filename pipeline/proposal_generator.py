"""
Stage 7: Research Proposal Generator
Generates 3-5 concrete, actionable research paper proposals
from validated gaps. Each proposal is citation-grounded.
"""
import json
from domain_config import inject_domain_into_proposal_prompt


PROPOSAL_SYSTEM_PROMPT = """You are a senior research advisor helping generate concrete, fundable research paper proposals.

Each proposal must be:
- Specific and actionable (a PhD student could implement this)
- Grounded in the analyzed papers (cite them)
- Filling a validated open or partial gap
- Novel but realistic in scope

Return ONLY a JSON array:
[
  {{
    "title": "Proposed paper title (specific, academic-style)",
    "problem_statement": "2-3 sentences: what exact problem does this paper solve and why does it matter?",
    "methodology": "Specific technical approach: what methods, models, data, experiments would be used",
    "novelty": "What makes this different from existing work (reference the gap evidence)",
    "builds_on": ["List of source paper titles this directly builds upon"],
    "addresses_gap": "Which gap (by title) this proposal addresses",
    "expected_contribution": "What the field gains if this paper succeeds",
    "suggested_experiments": ["3-4 specific experiments to validate the approach"],
    "potential_datasets": ["Datasets that could be used"],
    "confidence": "high|medium|low",
    "effort_estimate": "short (1-3 months) | medium (3-6 months) | long (6-12 months)",
    "risk": "low|medium|high — technical risk assessment"
  }}
]"""


def generate_proposals(gaps: list, wiki: dict, context: dict, client, domain_config: dict = None) -> list:
    """
    Generate research proposals from validated open/partial gaps.
    """
    # Only generate proposals for open or partial gaps
    actionable_gaps = [
        g for g in gaps
        if g.get("validation_status") in ["open", "partial"]
    ]

    if not actionable_gaps:
        actionable_gaps = gaps  # Use all if all are marked solved

    # Compact wiki context
    pages = wiki.get("pages", [])
    paper_summaries = [
        {
            "title": p.get("title", p.get("source_file", "Unknown")),
            "methods": p.get("methods", []),
            "datasets": p.get("datasets", []),
            "contributions": p.get("contributions", []),
        }
        for p in pages
    ]

    # Build gap summaries for proposal generation
    gap_summaries = [
        {
            "title": g.get("title"),
            "description": g.get("description"),
            "evidence": g.get("evidence", []),
            "gap_type": g.get("gap_type"),
            "validation_status": g.get("validation_status"),
            "missing_connection": g.get("missing_connection"),
            "why_important": g.get("why_important"),
        }
        for g in actionable_gaps[:6]  # Cap at 6 gaps to keep prompt manageable
    ]

    user_prompt = f"""Research Context:
- Domain: {context.get('domain', 'General AI/ML')}
- Researcher Interest: {context.get('interest', 'General')}

Source Papers Available to Build On:
{json.dumps(paper_summaries, indent=2)}

Validated Research Gaps:
{json.dumps(gap_summaries, indent=2)}

Generate 3-5 concrete research paper proposals. 
Prioritize gaps marked as 'open' over 'partial'.
Make proposals specific enough that a researcher could begin working on them immediately."""

    raw = client.chat_json(
        inject_domain_into_proposal_prompt(PROPOSAL_SYSTEM_PROMPT, domain_config or {}),
        user_prompt, max_tokens=5000
    )

    try:
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            proposals = []
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                proposals = json.loads(match.group())
            except Exception:
                proposals = []
        else:
            proposals = []

    return proposals
