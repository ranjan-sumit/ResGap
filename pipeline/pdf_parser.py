"""
Stage 1: Multimodal PDF Parser
Extracts text by section, tables, and optionally figures via GPT-4o Vision.
"""
import io
import re
import pdfplumber

# PyMuPDF — needed for page-to-image conversion (vision analysis).
# If import fails, vision features are disabled but text/table extraction still works.
# Fix: pip uninstall fitz frontend PyMuPDF -y && pip install PyMuPDF
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Section headers commonly found in research papers
SECTION_PATTERNS = [
    r"abstract",
    r"introduction",
    r"related work",
    r"background",
    r"literature review",
    r"methodology|method|approach|proposed",
    r"experiment|evaluation|results",
    r"discussion",
    r"limitation",
    r"conclusion|future work|summary",
    r"references|bibliography",
]


def _detect_section(text: str) -> str | None:
    """Return a normalized section name if the line looks like a section header."""
    clean = text.strip().lower()
    if len(clean) > 80 or len(clean) < 2:
        return None
    for pattern in SECTION_PATTERNS:
        if re.search(pattern, clean):
            # Return clean title-cased version
            return text.strip()
    return None


def _extract_text_by_section(pdf_path: str) -> dict:
    """
    Use pdfplumber to extract text, grouping content by detected section headers.
    Returns dict: {section_name: text_content}
    """
    sections = {"_header": ""}
    current_section = "_header"
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            for line in page_text.split("\n"):
                sec = _detect_section(line)
                if sec:
                    current_section = sec
                    if current_section not in sections:
                        sections[current_section] = ""
                else:
                    sections[current_section] = sections.get(current_section, "") + line + "\n"
                full_text.append(line)

    return {
        "sections": sections,
        "full_text": "\n".join(full_text),
        "char_count": len("\n".join(full_text)),
    }


def _extract_tables(pdf_path: str) -> list:
    """Extract tables from PDF using pdfplumber."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            for tbl in page_tables:
                if tbl and len(tbl) > 1:
                    # Convert to readable markdown-style string
                    rows = []
                    for row in tbl:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    tables.append({
                        "page": page_num,
                        "content": "\n".join(rows),
                        "rows": len(tbl),
                        "cols": len(tbl[0]) if tbl else 0,
                    })
    return tables


def _get_figure_pages(pdf_path: str) -> list:
    """
    Detect pages that are likely figure-heavy (low text density).
    Returns list of page indices (0-based).
    """
    figure_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Heuristic: page has images but few words → figure page
            words = len(text.split())
            has_image_ref = bool(re.search(r"figure|fig\.|chart|graph|diagram", text.lower()))
            if words < 80 or (has_image_ref and words < 200):
                figure_pages.append(i)
    return figure_pages[:5]  # Cap at 5 pages to control API cost


def _page_to_image_bytes(pdf_path: str, page_index: int, dpi: int = 150) -> bytes:
    """Convert a PDF page to PNG image bytes using PyMuPDF."""
    if not FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF not available. Run: pip uninstall fitz frontend PyMuPDF -y && pip install PyMuPDF")
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _analyze_figures_with_vision(pdf_path: str, client, figure_pages: list) -> list:
    """
    Send figure-heavy pages to GPT-4o Vision for analysis.
    Returns list of figure descriptions.
    """
    descriptions = []
    system = (
        "You are a research paper analyst. Analyze this page from a research paper. "
        "Identify and describe any figures, charts, graphs, or result tables. "
        "Focus on: what is being shown, key values or trends, and any stated limitations or caveats."
    )
    for page_idx in figure_pages:
        try:
            img_bytes = _page_to_image_bytes(pdf_path, page_idx)
            desc = client.vision(
                system=system,
                user_text="Describe the figures and tables on this page in detail.",
                image_bytes=img_bytes,
            )
            descriptions.append({"page": page_idx + 1, "description": desc})
        except Exception as e:
            descriptions.append({"page": page_idx + 1, "description": f"[Vision analysis failed: {e}]"})
    return descriptions


def parse_pdf(pdf_path: str, filename: str, client=None, use_vision: bool = False) -> dict:
    """
    Full PDF parsing pipeline for one paper.
    Returns structured content dict.
    """
    result = {
        "filename": filename,
        "sections": {},
        "full_text": "",
        "tables": [],
        "figures": [],
        "char_count": 0,
    }

    # Text extraction
    text_data = _extract_text_by_section(pdf_path)
    result["sections"] = text_data["sections"]
    result["full_text"] = text_data["full_text"]
    result["char_count"] = text_data["char_count"]

    # Table extraction
    result["tables"] = _extract_tables(pdf_path)

    # Vision-based figure extraction (optional, requires PyMuPDF)
    if use_vision and client and FITZ_AVAILABLE:
        figure_pages = _get_figure_pages(pdf_path)
        if figure_pages:
            result["figures"] = _analyze_figures_with_vision(pdf_path, client, figure_pages)

    return result


def get_smart_excerpt(paper: dict, max_chars: int = 12000) -> str:
    """
    Build a smart excerpt from the paper that prioritizes
    Limitations, Future Work, Results, and Abstract for gap analysis.
    """
    priority_keywords = [
        "limitation", "future work", "conclusion", "discussion",
        "abstract", "result", "experiment", "finding"
    ]
    sections = paper.get("sections", {})
    excerpt_parts = []
    used_chars = 0

    # First: add priority sections
    for sec_name, sec_text in sections.items():
        if any(kw in sec_name.lower() for kw in priority_keywords):
            chunk = f"[{sec_name}]\n{sec_text.strip()}\n\n"
            if used_chars + len(chunk) < max_chars * 0.7:
                excerpt_parts.append(chunk)
                used_chars += len(chunk)

    # Then: fill with other sections
    for sec_name, sec_text in sections.items():
        if not any(kw in sec_name.lower() for kw in priority_keywords):
            chunk = f"[{sec_name}]\n{sec_text.strip()}\n\n"
            if used_chars + len(chunk) < max_chars:
                excerpt_parts.append(chunk)
                used_chars += len(chunk)

    # Add table summaries
    tables = paper.get("tables", [])
    if tables:
        table_summary = f"[TABLES]\n"
        for t in tables[:5]:
            table_summary += f"Table (page {t['page']}, {t['rows']}x{t['cols']}):\n{t['content'][:500]}\n\n"
        if used_chars + len(table_summary) < max_chars:
            excerpt_parts.append(table_summary)

    # Add figure descriptions
    figures = paper.get("figures", [])
    if figures:
        fig_summary = "[FIGURES]\n"
        for f in figures[:3]:
            fig_summary += f"Page {f['page']}: {f['description'][:300]}\n\n"
        if used_chars + len(fig_summary) < max_chars:
            excerpt_parts.append(fig_summary)

    return "".join(excerpt_parts)[:max_chars]
