"""
Stage 2: PageIndex Tree Builder
Converts extracted paper content into a hierarchical tree structure,
simulating expert navigation of long documents without chunking.
"""
import re


# Tree node structure
def make_node(title: str, content: str = "", level: int = 0) -> dict:
    return {
        "title": title,
        "content": content.strip(),
        "level": level,
        "children": [],
        "word_count": len(content.split()),
    }


# Canonical section hierarchy for research papers
SECTION_HIERARCHY = {
    "abstract": 0,
    "introduction": 1,
    "related work": 2,
    "background": 2,
    "literature review": 2,
    "methodology": 3,
    "method": 3,
    "approach": 3,
    "proposed": 3,
    "experiment": 4,
    "evaluation": 4,
    "results": 4,
    "discussion": 5,
    "limitation": 6,
    "conclusion": 7,
    "future work": 7,
    "summary": 7,
    "references": 8,
}


def _normalize_section(title: str) -> str:
    """Normalize section title to canonical form."""
    t = title.strip().lower()
    for canonical in SECTION_HIERARCHY:
        if canonical in t:
            return canonical
    return t


def _get_level(title: str) -> int:
    """Get hierarchical level for a section."""
    norm = _normalize_section(title)
    return SECTION_HIERARCHY.get(norm, 3)


def build_tree(paper: dict) -> dict:
    """
    Build a hierarchical tree index from the parsed paper.
    The tree mirrors how a human expert navigates a paper:
    root → major sections → subsections → key content nodes
    """
    sections = paper.get("sections", {})
    filename = paper.get("filename", "Paper")

    root = make_node(title=filename, level=-1)

    # Sort sections by their canonical order
    section_items = [(k, v) for k, v in sections.items() if k != "_header" and v.strip()]
    section_items.sort(key=lambda x: _get_level(x[0]))

    for sec_name, sec_content in section_items:
        level = _get_level(sec_name)
        node = make_node(title=sec_name, content=sec_content, level=level)

        # Build sub-nodes for long sections (split by paragraphs)
        if len(sec_content) > 1500:
            paragraphs = [p.strip() for p in sec_content.split("\n\n") if len(p.strip()) > 100]
            for i, para in enumerate(paragraphs[:8]):
                sub_node = make_node(
                    title=f"{sec_name} — para {i+1}",
                    content=para,
                    level=level + 1,
                )
                node["children"].append(sub_node)

        root["children"].append(node)

    # Add metadata nodes
    tables = paper.get("tables", [])
    if tables:
        table_node = make_node(
            title="Tables",
            content=f"{len(tables)} table(s) extracted",
            level=8,
        )
        for t in tables:
            child = make_node(
                title=f"Table (page {t['page']})",
                content=t["content"][:800],
                level=9,
            )
            table_node["children"].append(child)
        root["children"].append(table_node)

    figures = paper.get("figures", [])
    if figures:
        fig_node = make_node(title="Figures (Vision)", content="", level=8)
        for f in figures:
            child = make_node(
                title=f"Figure (page {f['page']})",
                content=f["description"],
                level=9,
            )
            fig_node["children"].append(child)
        root["children"].append(fig_node)

    return root


def tree_to_summary(tree: dict, max_chars: int = 3000) -> str:
    """
    Flatten a tree into a readable outline string for use in prompts.
    Respects max_chars limit.
    """
    lines = []
    used = 0

    def _walk(node: dict, depth: int = 0):
        nonlocal used
        indent = "  " * depth
        header = f"{indent}## {node['title']} ({node['word_count']} words)"
        if node["content"]:
            preview = node["content"][:200].replace("\n", " ")
            line = f"{header}\n{indent}   {preview}...\n"
        else:
            line = f"{header}\n"

        if used + len(line) < max_chars:
            lines.append(line)
            used += len(line)
            for child in node["children"]:
                _walk(child, depth + 1)

    _walk(tree)
    return "\n".join(lines)


def get_section_content(tree: dict, target_section: str) -> str:
    """Navigate tree to find content of a specific section."""
    target_norm = target_section.strip().lower()

    def _search(node: dict) -> str | None:
        if target_norm in node["title"].lower():
            return node["content"]
        for child in node["children"]:
            result = _search(child)
            if result:
                return result
        return None

    return _search(tree) or ""
