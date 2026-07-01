"""
Stage 8B: Cross-Paper Discovery Engine
The core value-add: finds what NO individual paper said,
by combining unique contributions across all uploaded papers.

Produces novel, testable hypotheses with full evidence chains.
"""
import json
import re

FINGERPRINT_PROMPT = """You are a research analyst. For this single research paper,
identify what it UNIQUELY contributes that is NOT commonly known.

Focus on:
- The specific numerical findings that are distinctive
- The specific population or context that makes this paper unique
- The specific limitation or gap this paper explicitly leaves open
- Any unexpected or counter-intuitive result

Return ONLY JSON:
{
  "paper_title": "title",
  "unique_contribution": "One specific sentence — what this paper adds that others likely don't",
  "distinctive_findings": ["2-3 specific quantitative or qualitative findings unique to this paper"],
  "open_problems": ["problems this paper acknowledges but does not solve"],
  "methods_not_tried": ["methods or approaches this paper explicitly did NOT test"],
  "population_gap": "patient group or setting this paper did NOT cover",
  "strongest_evidence": "The single most reliable number or finding from this paper"
}"""

COMBINATION_PROMPT = """You are a research synthesis expert.
Given unique contributions from {n} papers, identify what emerges from COMBINING them
that NO SINGLE PAPER addressed.

Think like this:
- Paper A shows X works in population P
- Paper B shows biomarker B predicts condition C
- COMBINATION: Test X in population P stratified by biomarker B — never done before

For each combination, ask: "Has any paper in this set already addressed this?"
If yes, skip it. Only report GENUINELY NEW combinations.

Return ONLY JSON array:
[
  {{
    "combination_type": "method_plus_population|biomarker_plus_treatment|cross_domain|dataset_gap|methodology_fusion",
    "papers_involved": ["paper title 1", "paper title 2"],
    "what_each_paper_contributes": {{"paper1": "contribution", "paper2": "contribution"}},
    "novel_combination": "Specific description of the untested combination",
    "why_nobody_has_done_this": "Honest assessment of why this gap exists",
    "evidence_strength": "strong|moderate|weak",
    "feasibility": "high|medium|low"
  }}
]"""

HYPOTHESIS_PROMPT = """You are a senior researcher generating novel, testable hypotheses
from cross-paper combinations for the {domain} domain.

Each hypothesis must be:
- SPECIFIC (not vague — name the intervention, population, outcome)
- TESTABLE (describable as a study design)
- NOVEL (not addressed by any of the input papers)
- GROUNDED (supported by evidence from multiple papers)

Return ONLY JSON array:
[
  {{
    "hypothesis_id": "H1",
    "title": "Short, specific hypothesis title",
    "claim": "Precise testable statement: [Intervention X] in [Population Y] will [Outcome Z] by [estimate] compared to [comparator]",
    "evidence_chain": [
      "Paper A (section): finding that supports this",
      "Paper B (section): finding that supports this"
    ],
    "mechanism": "Biological or clinical mechanism that makes this plausible",
    "validation_study": "Specific study design that would test this: RCT|cohort|case-control with N~X patients",
    "clinical_opportunity": "How this changes clinical practice if true",
    "commercial_opportunity": "Potential product, service, or patent opportunity",
    "confidence": "high|medium|low",
    "confidence_reasoning": "Why this confidence level",
    "novelty_score": <1-10>,
    "impact_score": <1-10>,
    "papers_combined": ["paper titles used to derive this hypothesis"]
  }}
]"""


def _robust_parse(raw: str, expected: type = dict):
    if not raw:
        return {} if expected == dict else []
    for strategy in [
        lambda s: s,
        lambda s: re.sub(r'^```(?:json)?\s*|\s*```$', '', s).strip(),
        lambda s: re.sub(r',\s*([}\]])', r'\1', s),
    ]:
        try:
            result = json.loads(strategy(raw))
            if isinstance(result, expected):
                return result
        except Exception:
            pass
    for pattern, exp in [(r'\{.*\}', dict), (r'\[.*\]', list)]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, expected):
                    return result
            except Exception:
                pass
    return {} if expected == dict else []


def extract_paper_fingerprints(wiki: dict, client, domain_config: dict = None) -> list:
    """Extract what each paper uniquely contributes."""
    pages     = wiki.get("pages", [])
    fingerprints = []

    for page in pages:
        title   = page.get("title", page.get("source_file", "Unknown"))
        context = json.dumps({
            "title":        title,
            "contributions": page.get("contributions", []),
            "methods":       page.get("methods", []),
            "key_findings":  page.get("key_findings", []),
            "limitations":   page.get("limitations", []),
            "future_work":   page.get("future_work", []),
            "datasets":      page.get("datasets", []),
            "evaluated_on":  page.get("evaluated_on", []),
        }, indent=2)

        raw = client.chat_json(
            FINGERPRINT_PROMPT,
            f"Extract the unique fingerprint of this paper:\n\n{context}",
            max_tokens=1500,
        )
        fp = _robust_parse(raw, dict)
        if not fp:
            fp = {"paper_title": title, "unique_contribution": "Could not extract"}
        fingerprints.append(fp)

    return fingerprints


def generate_cross_paper_combinations(fingerprints: list, wiki: dict, client) -> list:
    """Find what emerges from combining papers that no single paper addressed."""
    if len(fingerprints) < 2:
        return []

    n = len(fingerprints)
    prompt = COMBINATION_PROMPT.format(n=n)

    user_prompt = (
        f"Paper fingerprints from {n} research papers:\n\n"
        f"{json.dumps(fingerprints, indent=2)}\n\n"
        "Identify all novel combinations that emerge from these papers together."
    )

    raw = client.chat_json(prompt, user_prompt, max_tokens=4000)
    combinations = _robust_parse(raw, list)
    return combinations if isinstance(combinations, list) else []


def generate_novel_hypotheses(
    combinations: list,
    fingerprints: list,
    params: list,
    wiki: dict,
    context: dict,
    client,
) -> list:
    """Generate specific testable hypotheses from cross-paper combinations."""
    domain = context.get("domain", "Healthcare AI")
    prompt = HYPOTHESIS_PROMPT.format(domain=domain)

    # Build rich context for hypothesis generation
    param_summary = [
        {k: p.get(k) for k in ["name", "value", "category", "condition", "source_paper"]}
        for p in params[:15]
    ]

    user_prompt = (
        f"Domain: {domain}\n"
        f"Researcher interest: {context.get('interest', 'General')}\n\n"
        f"Cross-paper combinations identified:\n{json.dumps(combinations, indent=2)}\n\n"
        f"Paper fingerprints:\n{json.dumps(fingerprints, indent=2)}\n\n"
        f"Key parameters extracted:\n{json.dumps(param_summary, indent=2)}\n\n"
        "Generate 3-5 novel, testable hypotheses. Be specific — name exact interventions, "
        "populations, and outcomes. These should be things no individual paper stated."
    )

    raw = client.chat_json(prompt, user_prompt, max_tokens=5000)
    hypotheses = _robust_parse(raw, list)
    return hypotheses if isinstance(hypotheses, list) else []


def run_discovery_engine(
    wiki: dict,
    params: list,
    context: dict,
    client,
    domain_config: dict = None,
) -> dict:
    """
    Main entry point for Stage 8B.
    Returns fingerprints, combinations, and novel hypotheses.
    """
    fingerprints  = extract_paper_fingerprints(wiki, client, domain_config)
    combinations  = generate_cross_paper_combinations(fingerprints, wiki, client)
    hypotheses    = generate_novel_hypotheses(
        combinations, fingerprints, params, wiki, context, client
    )

    return {
        "fingerprints":  fingerprints,
        "combinations":  combinations,
        "hypotheses":    hypotheses,
        "stats": {
            "papers_fingerprinted": len(fingerprints),
            "combinations_found":   len(combinations),
            "hypotheses_generated": len(hypotheses),
        },
    }
