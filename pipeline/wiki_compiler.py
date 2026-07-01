"""
Stage 3: LLM Wiki Compiler — Karpathy Pattern
Compiles raw paper content into a structured, LLM-maintained wiki.
Instead of naive RAG (query → retrieve chunks → answer),
this compiles knowledge once at ingestion time.
"""
import json
from pipeline.pdf_parser import get_smart_excerpt
from pipeline.page_index import tree_to_summary
from domain_config import inject_domain_into_wiki_prompt


WIKI_SYSTEM_PROMPT = """You are a research analyst compiling a structured knowledge wiki from a research paper.

Your task is to extract and organize the key information into a structured wiki page.
Be precise, specific, and evidence-based. Do not generalize or hallucinate.

Return ONLY a JSON object with this exact structure:
{
  "title": "Full paper title or best guess from content",
  "year": "Publication year if found, else null",
  "domain": "Primary research domain (e.g., NLP, Computer Vision, Healthcare AI)",
  "contributions": ["List of 3-5 specific technical contributions"],
  "methods": ["List of methods, models, algorithms used"],
  "datasets": ["List of datasets used for evaluation"],
  "key_findings": ["List of 3-5 specific quantitative or qualitative findings"],
  "limitations": ["List of EXPLICITLY stated limitations — be specific, use paper's own words"],
  "future_work": ["List of future work directions mentioned by authors"],
  "key_concepts": ["List of 8-12 important technical terms, acronyms, concepts"],
  "evaluated_on": ["Tasks or benchmarks the paper evaluates on"],
  "comparison_gap": "Any methods the paper acknowledges it did NOT compare against"
}"""


CROSS_LINK_SYSTEM_PROMPT = """You are a research analyst identifying connections between research papers.

Given wiki pages from multiple papers, identify:
1. Shared concepts, methods, or datasets across papers
2. Where one paper's limitation could be addressed by another paper's method
3. Where papers work on complementary problems
4. Contradictions or conflicting findings between papers

Return ONLY a JSON object:
{
  "shared_concepts": [{"concept": "name", "papers": ["paper1", "paper2"], "context": "how they relate"}],
  "limitation_bridges": [{"limitation_in": "paper name", "limitation": "text", "potential_solution_in": "paper name", "solution": "text"}],
  "complementary_pairs": [{"paper1": "name", "paper2": "name", "relationship": "explanation"}],
  "conflicts": [{"paper1": "name", "paper2": "name", "conflict": "explanation"}]
}"""


def compile_wiki_page(paper: dict, tree, client, domain_config: dict = None) -> dict:
    """
    Compile a wiki page for one paper using LLM.
    The wiki is the compiled, structured version — not raw text.
    """
    # Build a smart excerpt prioritizing key sections
    excerpt = get_smart_excerpt(paper, max_chars=10000)
    tree_outline = tree_to_summary(tree, max_chars=1500)

    user_prompt = f"""Paper filename: {paper['filename']}

Document Structure (PageIndex tree):
{tree_outline}

Paper Content (prioritizing Limitations, Future Work, Results, Abstract):
{excerpt}

Compile a structured wiki page from this paper."""

    raw = client.chat_json(
        inject_domain_into_wiki_prompt(WIKI_SYSTEM_PROMPT, domain_config or {}),
        user_prompt, max_tokens=2500
    )

    try:
        wiki = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON if wrapped in text
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                wiki = json.loads(match.group())
            except Exception:
                wiki = {"title": paper['filename'], "raw_response": raw}
        else:
            wiki = {"title": paper['filename'], "raw_response": raw}

    # Attach source reference
    wiki["source_file"] = paper["filename"]
    wiki["char_count"] = paper.get("char_count", 0)
    return wiki


def compile_cross_links(wiki_pages: list, client) -> dict:
    """
    Identify cross-paper links and relationships.
    This is where gaps start to emerge — connections that don't exist yet.
    """
    # Build a compact summary of all wikis for the cross-link prompt
    wiki_summaries = []
    for w in wiki_pages:
        summary = {
            "title": w.get("title", w.get("source_file", "Unknown")),
            "contributions": w.get("contributions", []),
            "methods": w.get("methods", []),
            "limitations": w.get("limitations", []),
            "future_work": w.get("future_work", []),
            "key_concepts": w.get("key_concepts", []),
        }
        wiki_summaries.append(summary)

    user_prompt = f"""Here are wiki pages from {len(wiki_pages)} research papers:

{json.dumps(wiki_summaries, indent=2)}

Identify connections, bridges, and gaps between these papers."""

    raw = client.chat_json(CROSS_LINK_SYSTEM_PROMPT, user_prompt, max_tokens=3000)

    try:
        cross_links = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                cross_links = json.loads(match.group())
            except Exception:
                cross_links = {}
        else:
            cross_links = {}

    return cross_links


def build_wiki(papers: list, trees: list, client, domain_config: dict = None) -> dict:
    """
    Build the full wiki: individual pages + cross-paper links.
    Returns: {pages: [...], cross_links: {...}, index: {...}}
    """
    pages = []
    for paper, tree in zip(papers, trees):
        wiki_page = compile_wiki_page(paper, tree, client, domain_config=domain_config)
        pages.append(wiki_page)

    cross_links = compile_cross_links(pages, client)

    # Build the index (table of contents)
    index = {
        "total_papers": len(pages),
        "domains": list(set(p.get("domain", "Unknown") for p in pages)),
        "all_concepts": [],
        "all_methods": [],
        "all_limitations": [],
        "all_future_work": [],
    }
    for p in pages:
        index["all_concepts"].extend(p.get("key_concepts", []))
        index["all_methods"].extend(p.get("methods", []))
        index["all_limitations"].extend(p.get("limitations", []))
        index["all_future_work"].extend(p.get("future_work", []))

    # Deduplicate
    for key in ["all_concepts", "all_methods", "all_limitations", "all_future_work"]:
        seen = set()
        deduped = []
        for item in index[key]:
            if item.lower() not in seen:
                seen.add(item.lower())
                deduped.append(item)
        index[key] = deduped

    return {
        "pages": pages,
        "cross_links": cross_links,
        "index": index,
    }
