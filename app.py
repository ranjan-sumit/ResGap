"""
Research Gap Analyzer
An AI-powered tool for identifying research gaps across academic papers.
Architecture: LLM Wiki + PageIndex + LazyGraphRAG + Academic Validation
"""
import streamlit as st
import tempfile
import os
import json
import time
import html
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Gap Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark editorial theme ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .stApp { background-color: #0d1117; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* ── Main header ── */
  .main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
  }
  .sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
  }

  /* ── Stage cards ── */
  .stage-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .stage-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .stage-body { font-size: 0.9rem; color: #c9d1d9; }

  /* ── Gap cards ── */
  .gap-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .gap-card.open { border-left-color: #3fb950; }
  .gap-card.partial { border-left-color: #d29922; }
  .gap-card.solved { border-left-color: #f85149; }
  .gap-card.pending { border-left-color: #8b949e; }
  .gap-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e6edf3;
    margin-bottom: 0.5rem;
  }
  .gap-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    margin-bottom: 0.6rem;
  }
  .gap-desc { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Proposal cards ── */
  .proposal-card {
    background: #0d2136;
    border: 1px solid #1f4f7a;
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
  }
  .proposal-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #79c0ff;
    margin-bottom: 0.8rem;
  }
  .proposal-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
  }
  .proposal-body { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-right: 0.4rem;
  }
  .badge-open { background: #0f3d1a; color: #3fb950; border: 1px solid #238636; }
  .badge-partial { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .badge-solved { background: #3d0f0f; color: #f85149; border: 1px solid #da3633; }
  .badge-high { background: #0f2d3d; color: #79c0ff; border: 1px solid #1f6feb; }
  .badge-medium { background: #2d2500; color: #e3b341; border: 1px solid #9e6a03; }
  .badge-low { background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }

  /* ── Wiki card ── */
  .wiki-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
  }
  .wiki-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #e6edf3;
    margin-bottom: 0.6rem;
  }
  .wiki-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .wiki-item { font-size: 0.83rem; color: #c9d1d9; margin-left: 0.8rem; }

  /* ── Graph stats ── */
  .stat-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    color: #58a6ff;
    display: block;
  }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3) !important;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
  }
  [data-testid="stFileUploader"]:hover { border-color: #58a6ff; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161b22;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #30363d;
    padding: 0 0.5rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    color: #8b949e !important;
    padding: 0.7rem 1.2rem !important;
    text-transform: uppercase !important;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1.5rem;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c9d1d9 !important;
  }
  .streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
  }

  /* ── Progress ── */
  .stProgress > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff) !important; }

  /* ── Divider ── */
  hr { border-color: #30363d !important; }

  /* ── Code ── */
  code { font-family: 'DM Mono', monospace !important; color: #79c0ff !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Enterprise UI overlay — Pfizer-inspired clinical productivity style ──────
st.markdown("""
<style>
  :root {
    --bg: #f5f9ff;
    --surface: #ffffff;
    --surface-muted: #edf6ff;
    --border: #c9dff5;
    --text: #0b1f3a;
    --muted: #55708f;
    --accent: #0066cc;
    --accent-strong: #004b93;
    --accent-soft: #e5f2ff;
    --success: #087f5b;
    --warning: #b7791f;
    --danger: #c81e1e;
  }
  html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0f7ff 100%) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text); }
  [data-testid="stSidebar"] .stMarkdown h3,
  [data-testid="stSidebar"] .stMarkdown h4,
  [data-testid="stSidebar"] .stMarkdown h5 {
    color: var(--muted) !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }
  .main-header {
    color: var(--accent-strong) !important;
    -webkit-text-fill-color: var(--accent-strong) !important;
    background: linear-gradient(90deg, var(--accent-strong), #0093d0) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
  }
  .sub-header {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    margin-bottom: 1.2rem !important;
  }
  .stage-card, .wiki-card, .gap-card, .proposal-card, .ref-card, .stat-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 3px rgba(0, 76, 147, 0.08) !important;
  }
  .stage-title, .wiki-label, .proposal-section, .stat-label, .gap-meta, .ref-meta {
    color: var(--accent-strong) !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
    letter-spacing: 0.04em !important;
  }
  .stage-body, .gap-desc, .proposal-body, .wiki-item, .ref-title {
    color: var(--text) !important;
  }
  .stat-number { color: var(--accent) !important; }
  .stButton > button, .stDownloadButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
    font-size: 0.86rem !important;
    letter-spacing: 0 !important;
    padding: 0.48rem 0.9rem !important;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--accent-strong) !important;
    transform: none !important;
    box-shadow: none !important;
  }
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
  }
  [data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed #5aa9e6 !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
    font-size: 0.78rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
  }
  .badge, .tier-badge {
    border-radius: 999px !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
  }
  .tier-1 { background:#e5f2ff !important; color:#004b93 !important; border:1px solid #7ab8f5 !important; }
  .tier-2 { background:#e7f8f1 !important; color:#087f5b !important; border:1px solid #83d7b1 !important; }
  .tier-3 { background:#f2f5f8 !important; color:#55708f !important; border:1px solid #c9d6e2 !important; }
  .badge-high, .badge-open { background:#e7f8f1 !important; color:#087f5b !important; border-color:#83d7b1 !important; }
  .badge-medium, .badge-partial { background:#fff7e6 !important; color:#b7791f !important; border-color:#f0c36a !important; }
  .badge-low, .badge-solved { background:#fff1f1 !important; color:#c81e1e !important; border-color:#f0a0a0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Extra CSS additions (skeleton, tier badges, run history) ──────────────────
st.markdown("""
<style>
  /* Skeleton loader */
  @keyframes shimmer {
    0%   { background-position: -800px 0; }
    100% { background-position:  800px 0; }
  }
  .skeleton {
    background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%);
    background-size: 800px 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 4px;
    height: 14px;
    margin-bottom: 8px;
  }
  .skeleton-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
  }

  /* Tier badges */
  .tier-1 { background:#0f2d3d; color:#58a6ff; border:1px solid #1f6feb; }
  .tier-2 { background:#0f3d1a; color:#3fb950; border:1px solid #238636; }
  .tier-3 { background:#1c1c1c; color:#8b949e; border:1px solid #30363d; }
  .tier-badge {
    display:inline-block; padding:0.15rem 0.55rem; border-radius:10px;
    font-family:'DM Mono',monospace; font-size:0.65rem; font-weight:500;
    margin-right:0.4rem; vertical-align:middle;
  }

  /* Run history cards */
  .run-card {
    background:#161b22; border:1px solid #30363d; border-radius:6px;
    padding:0.65rem 0.9rem; margin-bottom:0.4rem; cursor:pointer;
    transition:border-color 0.15s;
  }
  .run-card:hover { border-color:#58a6ff; }
  .run-card-title {
    font-size:0.82rem; color:#e6edf3; font-weight:500; margin-bottom:0.2rem;
  }
  .run-card-meta {
    font-family:'DM Mono',monospace; font-size:0.65rem; color:#484f58;
  }

  /* Reference paper list */
  .ref-card {
    background:#161b22; border:1px solid #30363d; border-left:3px solid #30363d;
    border-radius:5px; padding:0.6rem 0.9rem; margin-bottom:0.3rem;
  }
  .ref-card.tier-2-card { border-left-color:#3fb950; }
  .ref-card.tier-3-card { border-left-color:#484f58; }
  .ref-title { font-size:0.85rem; color:#c9d1d9; }
  .ref-meta  { font-family:'DM Mono',monospace; font-size:0.68rem; color:#484f58; margin-top:0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import tempfile, os, json, traceback
from datetime import datetime

from utils.azure_client import AzureOpenAIClient
from utils.export import (
    to_json, to_markdown_report, to_pdf_report,
    generate_report_narrative, to_enterprise_html_report,
    to_enterprise_pdf_report,
)
from utils.database import init_db, save_run, list_runs, load_run, delete_run
from pipeline.pdf_parser import parse_pdf
from pipeline.page_index import build_tree, tree_to_summary
from pipeline.wiki_compiler import build_wiki
from pipeline.graph_builder import build_knowledge_graph
from pipeline.gap_detector import detect_gaps
from pipeline.academic_search import validate_gaps
from pipeline.proposal_generator import generate_proposals
from pipeline.param_extractor import (
    extract_parameters, derive_cross_paper_parameters,
    check_sufficiency, params_to_csv, group_params_by_category,
)
from pipeline.reference_extractor import (
    extract_and_enrich_references, summarise_reference_corpus,
)
from pipeline.paper_manager import (
    assign_tiers, corpus_stats, compile_tier2_batch,
    extract_tier2_parameters, TIER_LABELS,
)
from pipeline.discovery_engine import run_discovery_engine
from pipeline.llm_council import council_select_model, council_synthesise_insights
from pipeline.simulation_engine import run_simulation, auto_select_models
from pipeline.multi_run import run_multi_council
from pipeline.doc_generator import build_zip
from domain_config import get_domain_display_options, parse_domain_selection, get_domain_config

# Initialise DB on startup
init_db()

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "results":        None,
    "running":        False,
    "comp_results":   None,
    "pipeline_trace": [],
    "ref_papers":     [],    # enriched reference paper list
    "ref_diagnostics": {},
    "corpus":         [],    # full corpus (Tier 1 + Tier 2 + Tier 3)
    "loaded_run_id":  None,  # which saved run is currently displayed
    "report_bundle":  None,
    "current_run_name": "",
    "current_run_id": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────
def _badge(text, color="#58a6ff", bg="#0f2d3d", border="#1f6feb"):
    return (f'<span style="display:inline-block;padding:0.15rem 0.5rem;'
            f'border-radius:10px;font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            f'font-weight:500;color:{color};background:{bg};border:1px solid {border};'
            f'margin-right:0.3rem;">{text}</span>')

def _tier_badge(tier: int) -> str:
    label, color, _ = TIER_LABELS.get(tier, ("Unknown","#8b949e",""))
    bgs = {1:"#0f2d3d", 2:"#0f3d1a", 3:"#1c1c1c"}
    borders = {1:"#1f6feb", 2:"#238636", 3:"#30363d"}
    return _badge(f"T{tier} {label}", color, bgs.get(tier,"#1c1c1c"), borders.get(tier,"#30363d"))

def _card(title, body, color="#58a6ff"):
    return (f'<div class="stage-card"><div class="stage-title" style="color:{color};">{title}</div>'
            f'<div class="stage-body">{body}</div></div>')

def _section_label(t):
    st.markdown(f'<div class="wiki-label">{t}</div>', unsafe_allow_html=True)

def _item(t, color="#c9d1d9"):
    st.markdown(f'<div class="wiki-item" style="color:{color};margin-bottom:0.25rem;">{t}</div>',
                unsafe_allow_html=True)

def _skeleton(lines=3):
    html = '<div class="skeleton-card">' + ''.join(
        f'<div class="skeleton" style="width:{w}%;"></div>'
        for w in ([90, 70, 55, 40][:lines])
    ) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _add_trace(stage, inp, out, status="✅"):
    st.session_state.pipeline_trace.append(
        {"stage": stage, "inputs": inp, "outputs": out, "status": status}
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Research Gap Analyzer")
    st.markdown("---")

    # API config
    st.markdown("##### API Configuration")
    st.markdown("""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:6px;
                padding:0.8rem 1rem;margin-bottom:0.8rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Model Stack</div>
      <div style="font-size:0.78rem;color:#e6edf3;font-weight:500;">🧠 Master — gpt-oss-120b</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#484f58;line-height:1.9;">
        Council A1 · gpt-oss-20b<br>Council A2 · llama-3.2-3b<br>
        Council A3 · nemotron-30b<br>Chair · gpt-oss-120b · NVIDIA NIM
      </div>
    </div>""", unsafe_allow_html=True)

    api_key = st.text_input("NVIDIA API Key", type="password",
                             placeholder="nvapi-••••••••••••••••")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">'
                '🔗 <a href="https://build.nvidia.com" target="_blank" style="color:#58a6ff;">'
                'Get key → build.nvidia.com</a></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Domain
    st.markdown("##### Research Domain")
    opts         = get_domain_display_options()
    default_idx  = next((i for i, d in enumerate(opts) if "Healthcare" in d), 0)
    sel_display  = st.selectbox("Select Domain", opts, index=default_idx)
    selected_domain = parse_domain_selection(sel_display)
    domain_cfg   = get_domain_config(selected_domain)

    st.markdown(f"""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:6px;
                padding:0.7rem 0.9rem;margin-top:0.3rem;">
      <div style="font-size:0.78rem;color:#c9d1d9;">{domain_cfg['description']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Research Context")
    interest = st.text_area("Your Specific Interest",
                             placeholder=f"e.g., {domain_cfg.get('gap_examples','')[:80]}...",
                             height=70)
    gap_type = st.selectbox("Gap Type Focus",
                             ["Any","Methodology","Application","Dataset",
                              "Evaluation","Theory","Benchmark"])

    st.markdown("---")
    st.markdown("##### Options")
    use_vision = st.checkbox("🖼️ Vision (figures/tables)", value=False)
    extract_refs = st.checkbox("📚 Extract & enrich references (50+ papers)", value=True,
                               help="Parse cited papers, query Semantic Scholar for abstracts.")
    ref_limit = st.selectbox("Reference limit per paper",
                             ["25", "50", "100", "All"],
                             index=1,
                             help="Higher limits take longer but improve reference coverage.")
    show_prompts = st.checkbox("🔍 Show prompts (explainability)", value=False)
    n_council_runs = st.select_slider("🔄 Council runs", options=[1,2,3], value=2,
                                       help="Run insight council N times and merge findings.")

    st.markdown("---")

    # ── Run History ────────────────────────────────────────────────────────────
    st.markdown("##### 🕑 Run History")
    runs = list_runs(limit=10)
    if not runs:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                    'color:#484f58;">No saved runs yet.</div>', unsafe_allow_html=True)
    else:
        for run in runs:
            ts_short = run["created_at"][:16].replace("T", " ")
            comp_dot = " 🧬" if run["has_comp"] else ""
            if st.button(
                f"📄 {run['name'][:22]}{comp_dot}",
                key=f"load_{run['id']}",
                help=f"{ts_short} · {run['domain']} · {run['n_papers']} papers · {run['n_gaps']} gaps",
            ):
                loaded = load_run(run["id"])
                if loaded.get("results"):
                    st.session_state.results      = loaded["results"]
                    st.session_state.comp_results = loaded.get("comp_results")
                    st.session_state.ref_papers   = loaded["results"].get("ref_papers", [])
                    st.session_state.ref_diagnostics = loaded["results"].get("ref_diagnostics", {})
                    st.session_state.corpus       = loaded["results"].get("corpus", loaded["results"].get("papers", []))
                    st.session_state.loaded_run_id = run["id"]
                    st.session_state.current_run_id = run["id"]
                    st.session_state.current_run_name = run["name"]
                    st.rerun()

            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                f'color:#484f58;margin-top:-0.5rem;margin-bottom:0.4rem;padding-left:0.2rem;">'
                f'{ts_short} · {run["n_papers"]}p · {run["n_gaps"]}g</div>',
                unsafe_allow_html=True,
            )

    if runs:
        if st.button("🗑 Clear all history", key="clear_history"):
            for r in runs:
                delete_run(r["id"])
            st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Research Gap Analyzer</div>', unsafe_allow_html=True)

# Show loaded run banner
if st.session_state.loaded_run_id:
    runs_meta = {r["id"]: r for r in list_runs()}
    run_meta  = runs_meta.get(st.session_state.loaded_run_id, {})
    st.markdown(f"""
    <div style="background:#2d1f00;border:1px solid #9e6a03;border-radius:8px;
                padding:0.7rem 1.2rem;margin-bottom:0.5rem;">
      <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#d29922;">
        📂 Showing saved run: <strong>{run_meta.get('name','')}</strong>
        — {run_meta.get('created_at','')[:16].replace('T',' ')}
      </span>
      <span style="font-size:0.75rem;color:#8b949e;margin-left:1rem;">
        Upload new papers above to run a fresh analysis</span>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="sub-header">Upload papers → AI discovers what the field hasn\'t found yet</div>',
            unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload up to 5 research papers (PDF)",
    type=["pdf"], accept_multiple_files=True,
)
if uploaded_files and len(uploaded_files) > 5:
    st.warning("Maximum 5 papers. Only first 5 will be processed.")
    uploaded_files = uploaded_files[:5]

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 5))
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            st.markdown(
                f'<div class="stage-card" style="text-align:center;padding:0.8rem;">'
                f'<div class="stage-title">{_tier_badge(1)} Paper {i+1}</div>'
                f'<div class="stage-body" style="font-size:0.78rem;word-break:break-word;">{f.name}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#8b949e;">'
                f'{len(f.getvalue())//1024} KB</div></div>',
                unsafe_allow_html=True)

st.markdown("")
run_col, name_col, _ = st.columns([1, 2, 1])
with run_col:
    run_btn = st.button("🚀 Run Analysis", disabled=st.session_state.running)
with name_col:
    run_name = st.text_input("Run name (for history)", value="", placeholder="e.g., Diabetes AI review",
                              label_visibility="collapsed")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    if not api_key:
        st.error("Please enter your NVIDIA API key.")
        st.stop()
    if len(uploaded_files) < 2:
        st.warning("Upload at least 2 papers.")
        st.stop()

    st.session_state.running        = True
    st.session_state.pipeline_trace = []
    st.session_state.loaded_run_id  = None

    client  = AzureOpenAIClient(api_key=api_key.strip())
    context = {
        "domain":   selected_domain,
        "interest": interest or "General research",
        "gap_type": gap_type,
    }
    name = run_name.strip() or f"{selected_domain[:20]} — {datetime.now().strftime('%H:%M')}"

    prog = st.progress(0)
    stat = st.empty()
    def upd(msg, pct):
        prog.progress(pct)
        stat.markdown(_card("Pipeline Running", f"⟳ {msg}"), unsafe_allow_html=True)

    try:
        all_papers, all_trees = [], []

        # Stage 1+2: Parse + PageIndex
        upd("Stage 1 — Parsing PDFs & building PageIndex trees…", 5)
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uf.getvalue()); tmp_path = tmp.name
            paper = parse_pdf(tmp_path, uf.name,
                              client=client if use_vision else None,
                              use_vision=use_vision)
            paper["tier"] = 1
            all_papers.append(paper)
            all_trees.append(build_tree(paper))
            os.unlink(tmp_path)

        _add_trace("Stage 1 — PDF Parse",
                   f"{len(uploaded_files)} PDFs",
                   f"{sum(p.get('char_count',0) for p in all_papers):,} chars extracted")
        prog.progress(12)

        # Stage 1B: Reference extraction (optional)
        ref_papers = []
        ref_diagnostics = {}
        if extract_refs:
            upd("Stage 1B — Extracting & enriching references (50+ paper corpus)…", 14)
            limit_value = None if ref_limit == "All" else int(ref_limit)
            def ref_upd(_current, _total, msg):
                upd(f"Stage 1B — {msg}", 14)
            ref_papers, ref_diagnostics = extract_and_enrich_references(
                all_papers,
                client,
                max_refs_per_paper=limit_value,
                progress_callback=ref_upd,
                return_diagnostics=True,
            )
            st.session_state.ref_papers = ref_papers
            st.session_state.ref_diagnostics = ref_diagnostics
            summary = summarise_reference_corpus(ref_papers)
            _add_trace("Stage 1B — Reference Extraction",
                       f"{len(uploaded_files)} PDFs scanned",
                       f"{summary['total']} refs from {ref_diagnostics.get('raw_entries',0)} raw entries: "
                       f"{summary['tier2_abstract']} with abstract, {summary['tier3_title']} title-only")
        prog.progress(20)

        # Stage 2: Wiki
        upd("Stage 2 — Compiling LLM Wiki pages…", 22)
        wiki = build_wiki(all_papers, all_trees, client, domain_config=domain_cfg)
        _add_trace("Stage 2 — LLM Wiki",
                   f"{len(all_papers)} papers",
                   f"{len(wiki.get('pages',[]))} pages, "
                   f"{len(wiki.get('cross_links',{}).get('shared_concepts',[]))} links")
        prog.progress(34)

        # Stage 3: Knowledge Graph
        upd("Stage 3 — Building knowledge graph (LazyGraphRAG)…", 36)
        graph = build_knowledge_graph(wiki, client, domain_config=domain_cfg)
        _add_trace("Stage 3 — Knowledge Graph",
                   "Wiki pages + cross-links",
                   f"{graph['stats']['entity_count']} entities, "
                   f"{graph['stats']['relationship_count']} relationships")
        prog.progress(48)

        # Stage 4+5: Gaps + Validation
        upd("Stage 4 — Detecting research gaps…", 50)
        gaps = detect_gaps(wiki, graph, context, client, domain_config=domain_cfg)
        upd("Stage 5 — Validating gaps (Semantic Scholar + arXiv)…", 58)
        validated_gaps = validate_gaps(gaps, client, domain_config=domain_cfg)
        _add_trace("Stage 4-5 — Gaps",
                   "Wiki + graph + domain context",
                   f"{len(validated_gaps)} gaps: "
                   f"{sum(1 for g in validated_gaps if g.get('validation_status')=='open')} open")
        prog.progress(66)

        # Stage 6: Proposals
        upd("Stage 6 — Generating proposals…", 68)
        proposals = generate_proposals(validated_gaps, wiki, context, client,
                                       domain_config=domain_cfg)
        _add_trace("Stage 6 — Proposals", f"{len(validated_gaps)} gaps",
                   f"{len(proposals)} proposals")
        prog.progress(75)

        # Build full corpus
        corpus = assign_tiers(all_papers, ref_papers)
        st.session_state.corpus = corpus

        # Store results
        results = {
            "papers": [
                {"filename": p["filename"], "wiki": w,
                 "char_count": p.get("char_count",0),
                 "sections": p.get("sections",{}),
                 "tables": p.get("tables",[]),
                 "tree_summary": tree_to_summary(t, max_chars=2000),
                 "tier": 1}
                for p, w, t in zip(all_papers, wiki["pages"], all_trees)
            ],
            "wiki": wiki, "graph": graph,
            "gaps": validated_gaps, "proposals": proposals,
            "communities": graph.get("communities", []),
            "context": context,
            "corpus_stats": corpus_stats(corpus),
            "ref_summary": summarise_reference_corpus(ref_papers),
            "ref_papers": ref_papers,
            "ref_diagnostics": ref_diagnostics,
            "corpus": corpus,
            "pipeline_trace": st.session_state.pipeline_trace,
        }
        st.session_state.results = results

        # Auto-save to database
        saved_id = save_run(name, results)
        st.session_state.loaded_run_id = None  # fresh run, not a loaded one
        st.session_state.current_run_id = saved_id
        st.session_state.current_run_name = name

        prog.progress(80)
        prog.empty(); stat.empty()
        st.success(f"✅ Done — {len(validated_gaps)} gaps, {len(proposals)} proposals. "
                   f"Saved as '{name}' ({saved_id[:8]}…)")

    except Exception as e:
        prog.empty(); stat.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.running = False

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.results:
    res        = st.session_state.results
    gaps       = res.get("gaps", [])
    proposals  = res.get("proposals", [])
    graph      = res.get("graph", {})
    wiki       = res.get("wiki", {})
    wiki_pages = wiki.get("pages", [])
    context    = res.get("context", {})
    ref_papers = st.session_state.ref_papers or res.get("ref_papers", [])
    ref_diagnostics = st.session_state.ref_diagnostics or res.get("ref_diagnostics", {})
    corpus     = st.session_state.corpus or res.get("papers", [])

    # Domain banner
    res_cfg = get_domain_config(context.get("domain","General AI/ML"))
    st.markdown(f"""
    <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;
                padding:0.8rem 1.2rem;margin:0.8rem 0;">
      <span style="font-size:1.4rem;">{res_cfg['icon']}</span>
      <span style="font-size:0.9rem;color:#e6edf3;font-weight:500;margin-left:0.8rem;">
        {context.get('domain','')} — Domain-Tuned Analysis</span>
    </div>""", unsafe_allow_html=True)

    # Corpus stats bar
    c_stats = res.get("corpus_stats", {})
    ref_sum = res.get("ref_summary", {})
    if c_stats:
        t1,t2,t3,t4 = st.columns(4)
        for col,(val,lbl,color) in zip([t1,t2,t3,t4],[
            (c_stats.get("tier1",0), "Uploaded PDFs", "#58a6ff"),
            (c_stats.get("tier2",0), "Cited (w/ abstract)", "#3fb950"),
            (c_stats.get("tier3",0), "Distant refs", "#8b949e"),
            (ref_sum.get("total",0), "Total corpus", "#d29922"),
        ]):
            with col:
                st.markdown(f'<div class="stat-box"><span class="stat-number" '
                            f'style="color:{color};">{val}</span>'
                            f'<span class="stat-label">{lbl}</span></div>',
                            unsafe_allow_html=True)
        st.markdown("")

    # Summary row
    m1,m2,m3,m4,m5 = st.columns(5)
    for col,(val,lbl) in zip([m1,m2,m3,m4,m5],[
        (len(res.get("papers",[])), "Papers Analysed"),
        (graph.get("stats",{}).get("entity_count",0), "Graph Entities"),
        (graph.get("stats",{}).get("community_count",0), "Communities"),
        (len(gaps), "Gaps Found"),
        (len(proposals), "Proposals"),
    ]):
        with col:
            st.markdown(f'<div class="stat-box"><span class="stat-number">{val}</span>'
                        f'<span class="stat-label">{lbl}</span></div>', unsafe_allow_html=True)

    st.markdown("")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
        "📄 Wiki Pages",
        "📚 Reference Corpus",
        "🕸️ Knowledge Graph",
        "🔍 Research Gaps",
        "💡 Proposals",
        "📊 Computational Lab",
        "🔎 Pipeline Trace",
        "📦 Export",
    ])

    # ── Tab 1: Wiki Pages ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Compiled Wiki Pages")
        for i, paper_res in enumerate(res.get("papers",[])):
            page  = paper_res.get("wiki", {})
            title = page.get("title", paper_res.get("filename", f"Paper {i+1}"))
            with st.expander(f"T1 Uploaded PDF · {title}", expanded=False):
                st.markdown(
                    f'{_tier_badge(1)} '
                    f'<span style="font-weight:700;color:#004b93;">{html.escape(str(title))}</span>',
                    unsafe_allow_html=True,
                )
                c1,c2 = st.columns(2)
                with c1:
                    _section_label("Contributions")
                    for c in page.get("contributions",[])[:5]: _item(f"• {c}")
                    st.markdown("")
                    _section_label("Methods")
                    for m in page.get("methods",[])[:5]: _item(f"• {m}")
                with c2:
                    _section_label("Limitations ⚠️")
                    for lim in page.get("limitations",[])[:4]: _item(f"• {lim}","#f85149")
                    st.markdown("")
                    _section_label("Future Work 🔭")
                    for fw in page.get("future_work",[])[:4]: _item(f"• {fw}","#3fb950")
                tree_sum = paper_res.get("tree_summary","")
                if tree_sum:
                    with st.expander("🌲 PageIndex Tree"):
                        st.code(tree_sum, language=None)
                tables = paper_res.get("tables",[])
                if tables:
                    with st.expander(f"📊 {len(tables)} table(s)"):
                        for t in tables[:3]:
                            st.markdown(f"**Table p.{t['page']}:**")
                            st.code(t["content"][:400])

        cross_links = wiki.get("cross_links",{})
        shared = cross_links.get("shared_concepts",[])
        if shared:
            st.markdown("### Cross-Paper Links")
            for link in shared[:6]:
                st.markdown(_card(
                    "Shared Concept",
                    f"<strong>{link.get('concept','')}</strong> — {link.get('context','')}<br>"
                    f"<span style='font-family:\"DM Mono\",monospace;font-size:0.7rem;color:#58a6ff;'>"
                    f"{', '.join(link.get('papers',[]))}</span>"
                ), unsafe_allow_html=True)

    # ── Tab 2: Reference Corpus ────────────────────────────────────────────────
    with tab2:
        st.markdown("### Reference Corpus")
        if ref_diagnostics:
            d1,d2,d3,d4 = st.columns(4)
            for col,(val,lbl,color) in zip([d1,d2,d3,d4],[
                (ref_diagnostics.get("raw_entries",0), "Raw Citations", "#2563eb"),
                (ref_diagnostics.get("structured_entries",0), "LLM Parsed", "#059669"),
                (ref_diagnostics.get("fallback_entries",0), "Fallback Parsed", "#d97706"),
                (ref_diagnostics.get("duplicates_skipped",0), "Duplicates", "#64748b"),
            ]):
                with col:
                    st.markdown(f'<div class="stat-box"><span class="stat-number" '
                                f'style="font-size:1.3rem;color:{color};">{val}</span>'
                                f'<span class="stat-label">{lbl}</span></div>',
                                unsafe_allow_html=True)
            if ref_diagnostics.get("errors"):
                with st.expander("Reference extraction warnings"):
                    for err in ref_diagnostics.get("errors", [])[:20]:
                        st.warning(err)
            st.markdown("")

        if not ref_papers:
            if not res.get("ref_summary",{}).get("total"):
                st.info("No references are available yet. Enable reference extraction in the sidebar and run the analysis.")
            else:
                st.info("Reference summary exists, but detailed references were not saved in this older run. Re-run once to persist them.")
        else:
            summary = summarise_reference_corpus(ref_papers)
            ra,rb,rc,rd = st.columns(4)
            for col,(val,lbl,color) in zip([ra,rb,rc,rd],[
                (summary["total"],"Total Refs","#58a6ff"),
                (summary["tier2_abstract"],"With Abstract","#3fb950"),
                (summary["tier3_title"],"Title Only","#8b949e"),
                (summary["year_range"],"Year Range","#d29922"),
            ]):
                with col:
                    st.markdown(f'<div class="stat-box"><span class="stat-number" '
                                f'style="font-size:1.3rem;color:{color};">{val}</span>'
                                f'<span class="stat-label">{lbl}</span></div>',
                                unsafe_allow_html=True)
            st.markdown("")

            # Filter
            fc1,fc2,fc3 = st.columns([1, 1, 1])
            with fc1:
                tier_filter = st.selectbox("Filter by tier",
                                           ["All","Tier 2 (with abstract)","Tier 3 (title only)"],
                                           key="ref_tier_filter")
            with fc2:
                search_q = st.text_input("Search references", placeholder="keyword…",
                                          key="ref_search")
            with fc3:
                ref_view = st.selectbox("View references as",
                                        ["Grouped by source paper", "Cards", "Table"],
                                        key="ref_view")

            filtered = ref_papers
            if "Tier 2" in tier_filter:
                filtered = [r for r in ref_papers if r.get("tier")==2]
            elif "Tier 3" in tier_filter:
                filtered = [r for r in ref_papers if r.get("tier")==3]
            if search_q:
                q = search_q.lower()
                filtered = [r for r in filtered
                            if q in ((r.get("title") or "")+" "+(r.get("abstract") or "")).lower()]

            st.markdown(f'<div style="font-size:0.8rem;color:#8b949e;margin-bottom:0.8rem;">'
                        f'Showing {len(filtered)} of {len(ref_papers)} references</div>',
                        unsafe_allow_html=True)

            def _render_ref(ref, idx=None):
                tier    = ref.get("tier",3)
                t_class = "tier-2-card" if tier==2 else "tier-3-card"
                year    = ref.get("year","")
                doi     = ref.get("doi","")
                found_in = ref.get("found_in_paper","")
                title_text = html.escape(str(ref.get("title") or "Unknown"))
                abstract = html.escape(str(ref.get("abstract") or "")[:240])
                raw_reference = ref.get("raw_reference", "")
                source = html.escape(str(ref.get("source") or ""))
                citations = ref.get("citations", 0)
                oa_url = ref.get("open_access_url")
                url = ref.get("url") or (f"https://doi.org/{doi}" if doi else "")
                venue = html.escape(str(ref.get("venue") or ""))
                authors = html.escape(", ".join(ref.get("authors", [])[:4])) if isinstance(ref.get("authors"), list) else ""
                prefix = f"#{idx} " if idx is not None else ""

                st.markdown(
                    f'<div class="ref-card {t_class}">'
                    f'<div class="ref-title">{_tier_badge(tier)} {prefix}{title_text}</div>'
                    f'<div class="ref-meta">{html.escape(str(year))} &nbsp;·&nbsp; {html.escape(str(doi)[:40]) if doi else "no DOI"}'
                    f' &nbsp;·&nbsp; source: {source or "parsed"}'
                    f' &nbsp;·&nbsp; citations: {html.escape(str(citations or 0))}'
                    f' &nbsp;·&nbsp; cited in: {html.escape(str(found_in))}</div>'
                    + (f'<div style="font-size:0.76rem;color:#55708f;margin-top:0.2rem;">{authors}</div>' if authors else '')
                    + (f'<div style="font-size:0.76rem;color:#55708f;margin-top:0.2rem;font-style:italic;">{venue}</div>' if venue else '')
                    + (f'<div style="font-size:0.82rem;color:#344054;margin-top:0.5rem;line-height:1.55;">{abstract}…</div>'
                       if abstract else '')
                    + '</div>',
                    unsafe_allow_html=True,
                )
                link_cols = st.columns([1, 1, 5])
                if url:
                    link_cols[0].markdown(f"[DOI/Source]({url})")
                if oa_url:
                    link_cols[1].markdown(f"[Open PDF]({oa_url})")
                if raw_reference:
                    with st.expander(f"Raw citation: {title_text[:80]}"):
                        st.code(raw_reference, language=None)

            if ref_view == "Table":
                import pandas as pd
                rows = []
                for i, r in enumerate(filtered, 1):
                    rows.append({
                        "#": i,
                        "Title": r.get("title", "Unknown"),
                        "About / Abstract": (r.get("abstract") or r.get("raw_reference") or "")[:300],
                        "Year": r.get("year", ""),
                        "Tier": r.get("tier", ""),
                        "Source API": r.get("source", ""),
                        "Citations": r.get("citations", 0),
                        "Cited In": r.get("found_in_paper", ""),
                        "DOI": r.get("doi", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=520)
            elif ref_view == "Grouped by source paper":
                grouped_refs = {}
                for r in filtered:
                    grouped_refs.setdefault(r.get("found_in_paper") or "Unknown source paper", []).append(r)
                for source_paper, refs_for_paper in grouped_refs.items():
                    with st.expander(f"{source_paper} · {len(refs_for_paper)} fetched references", expanded=False):
                        tier2_count = sum(1 for r in refs_for_paper if r.get("tier") == 2)
                        abstract_count = sum(1 for r in refs_for_paper if r.get("abstract"))
                        st.markdown(
                            f'<div class="stage-card"><div class="stage-title">Fetched From This Uploaded Paper</div>'
                            f'<div class="stage-body">{len(refs_for_paper)} references · {tier2_count} enriched · '
                            f'{abstract_count} with abstract/about text</div></div>',
                            unsafe_allow_html=True,
                        )
                        for i, ref in enumerate(refs_for_paper[:40], 1):
                            _render_ref(ref, idx=i)
                        if len(refs_for_paper) > 40:
                            st.info(f"Showing first 40 of {len(refs_for_paper)} references for this source paper. Use search or table view to inspect the rest.")
            else:
                for i, ref in enumerate(filtered[:80], 1):
                    _render_ref(ref, idx=i)

    # ── Tab 3: Knowledge Graph ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### Knowledge Graph (LazyGraphRAG)")
        stats = graph.get("stats",{})
        graph_diag = graph.get("diagnostics", {})
        if graph_diag.get("used_fallback"):
            st.warning("The LLM graph extraction returned no entities, so a deterministic fallback graph was built from wiki concepts, methods, datasets, limitations, and findings.")
        elif graph_diag:
            st.info(f"LLM graph extraction produced {graph_diag.get('llm_entities', 0)} entities and {graph_diag.get('llm_relationships', 0)} relationships.")
        gc1,gc2,gc3,gc4 = st.columns(4)
        for col,(lbl,val) in zip([gc1,gc2,gc3,gc4],[
            ("Entities",stats.get("entity_count",0)),
            ("Relations",stats.get("relationship_count",0)),
            ("Communities",stats.get("community_count",0)),
            ("Orphan Signals",stats.get("orphan_count",0)),
        ]):
            with col:
                st.markdown(f'<div class="stat-box"><span class="stat-number" '
                            f'style="font-size:1.5rem;">{val}</span>'
                            f'<span class="stat-label">{lbl}</span></div>',
                            unsafe_allow_html=True)
        st.markdown("")

        # Network diagram
        network_data = graph.get("network_data",{})
        nodes = network_data.get("nodes",[])
        edges = network_data.get("edges",[])
        if nodes:
            try:
                import plotly.graph_objects as go, math
                n = len(nodes)
                angles = [2*math.pi*i/n for i in range(n)]
                radius = max(1, n/(2*math.pi))
                px = [radius*math.cos(a) for a in angles]
                py = [radius*math.sin(a) for a in angles]
                ex,ey = [],[]
                for e in edges:
                    si,ti = e["source_idx"],e["target_idx"]
                    if si<n and ti<n:
                        ex+=[px[si],px[ti],None]; ey+=[py[si],py[ti],None]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",
                                         line=dict(width=1,color="#30363d"),hoverinfo="none"))
                fig.add_trace(go.Scatter(x=px,y=py,mode="markers+text",
                    text=[nd["name"][:18] for nd in nodes],textposition="top center",
                    textfont=dict(size=9,color="#c9d1d9"),
                    marker=dict(size=[nd["size"] for nd in nodes],
                                color=[nd["color"] for nd in nodes],
                                line=dict(width=1,color="#0d1117")),
                    hovertext=[f"{nd['name']}<br>Type: {nd['type']}" for nd in nodes],
                    hoverinfo="text"))
                fig.update_layout(showlegend=False,paper_bgcolor="#161b22",
                    plot_bgcolor="#0d1117",font=dict(color="#c9d1d9",size=10),
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    height=430,margin=dict(l=10,r=10,t=30,b=10),
                    title="Knowledge Graph — Entity Network")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("pip install plotly for network diagram")

        for comm in graph.get("communities",[]):
            with st.expander(f"🏘️ {comm.get('theme','Community')}"):
                st.markdown(f'<div class="stage-body">{comm.get("summary","")}</div>',
                            unsafe_allow_html=True)
                if comm.get("gap_signal"):
                    st.markdown(
                        f'<div style="background:#2d1f00;border:1px solid #9e6a03;border-radius:4px;'
                        f'padding:0.5rem 0.8rem;margin-top:0.5rem;font-size:0.82rem;color:#d29922;">'
                        f'⚠️ {comm["gap_signal"]}</div>', unsafe_allow_html=True)

        entities = graph.get("entities",[])
        if entities:
            with st.expander(f"📋 All {len(entities)} entities"):
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "Name":e.get("name",""),"Type":e.get("type",""),
                    "Importance":e.get("importance",""),
                    "Papers":", ".join(e.get("papers",[]))[:60],
                } for e in entities]), use_container_width=True, height=260)

        orphans = graph.get("orphan_concepts",[])
        if orphans:
            st.markdown("#### Orphan Concepts")
            for o in orphans[:8]:
                ent = o.get("entity",{})
                st.markdown(
                    f'<div class="stage-card" style="padding:0.6rem 1rem;margin-bottom:0.3rem;">'
                    f'<span style="font-family:\'DM Mono\',monospace;color:#79c0ff;">{ent.get("name","")}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;color:#484f58;font-size:0.7rem;'
                    f'margin-left:0.6rem;">[{ent.get("type","")}]</span>'
                    f'<div style="font-size:0.78rem;color:#8b949e;">{o.get("signal","")}</div></div>',
                    unsafe_allow_html=True)

    # ── Tab 4: Research Gaps ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### Identified Research Gaps")
        open_n    = sum(1 for g in gaps if g.get("validation_status")=="open")
        partial_n = sum(1 for g in gaps if g.get("validation_status")=="partial")
        solved_n  = sum(1 for g in gaps if g.get("validation_status")=="solved")
        f1,f2,f3 = st.columns(3)
        for col,(lbl,cnt,color) in zip([f1,f2,f3],[
            ("🟢 Open",open_n,"#3fb950"),
            ("🟡 Partial",partial_n,"#d29922"),
            ("🔴 Solved",solved_n,"#f85149"),
        ]):
            with col:
                st.markdown(f'<div class="stat-box"><span class="stat-number" '
                            f'style="color:{color};font-size:1.5rem;">{cnt}</span>'
                            f'<span class="stat-label">{lbl}</span></div>',
                            unsafe_allow_html=True)
        st.markdown("")

        status_badge = {
            "open":    _badge("Open","#3fb950","#0f3d1a","#238636"),
            "partial": _badge("Partial","#d29922","#2d1f00","#9e6a03"),
            "solved":  _badge("Solved","#f85149","#3d0f0f","#da3633"),
            "pending": _badge("Pending","#8b949e","#1c1c1c","#30363d"),
        }
        for gap in gaps:
            status = gap.get("validation_status","pending")
            conf   = gap.get("confidence","medium")
            st.markdown(
                f'<div class="gap-card {status}">'
                f'<div class="gap-title">{gap.get("title","Untitled")}</div>'
                f'<div class="gap-meta">{status_badge.get(status,"")} '
                f'{_badge(conf.title()+" Conf","#79c0ff","#0f2d3d","#1f6feb")}</div>'
                f'<div class="gap-desc">{gap.get("description","")}</div></div>',
                unsafe_allow_html=True)
            with st.expander("Evidence + validation"):
                ec1,ec2 = st.columns(2)
                with ec1:
                    _section_label("Evidence from Papers")
                    for ev in gap.get("evidence",[]): _item(f"📎 {ev}")
                    if gap.get("search_query_used"):
                        st.markdown("")
                        _section_label("Search Query Used")
                        st.code(gap["search_query_used"], language=None)
                with ec2:
                    for ep in gap.get("existing_papers",[])[:4]:
                        _item(f"📄 {ep.get('title','N/A')} ({ep.get('year','')}) [{ep.get('source','')}]")

    # ── Tab 5: Proposals ───────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Research Proposals")
        for i, proposal in enumerate(proposals, 1):
            conf   = proposal.get("confidence","medium")
            effort = proposal.get("effort_estimate","")
            ec     = {"short":"#3fb950","medium":"#d29922","long":"#f85149"}.get(
                (effort.split()[0].lower() if effort else ""), "#8b949e")
            st.markdown(
                f'<div class="proposal-card">'
                f'<div class="proposal-title">#{i}  {proposal.get("title","Untitled")}</div>'
                f'<div>{_badge(conf.title()+" Conf","#79c0ff","#0f2d3d","#1f6feb")}'
                f'{_badge(effort,ec,"#1a2a1a",ec)}</div>'
                f'<div class="proposal-section">Problem Statement</div>'
                f'<div class="proposal-body">{proposal.get("problem_statement","")}</div>'
                f'<div class="proposal-section">Proposed Methodology</div>'
                f'<div class="proposal-body">{proposal.get("methodology","")}</div>'
                f'<div class="proposal-section">Novelty</div>'
                f'<div class="proposal-body">{proposal.get("novelty","")}</div></div>',
                unsafe_allow_html=True)
            with st.expander("Experiments + datasets"):
                e1,e2 = st.columns(2)
                with e1:
                    for exp in proposal.get("suggested_experiments",[]): _item(f"🧪 {exp}")
                with e2:
                    for src in proposal.get("builds_on",[]): _item(f"📎 {src}")

    # ── Tab 6: Computational Lab ───────────────────────────────────────────────
    with tab6:
        st.markdown("### 📊 Computational Lab")
        st.markdown(
            '<div style="font-size:0.82rem;color:#8b949e;margin-bottom:1.5rem;">'
            'Parameter extraction (Tier 1 + Tier 2) → Cross-paper discovery → '
            'LLM Council model selection → Multi-algorithm simulation → '
            'Novel findings → Download package.</div>',
            unsafe_allow_html=True)

        cr_col, ci_col = st.columns([1, 3])
        with cr_col:
            run_comp = st.button("🧬 Run Computational Lab", key="run_comp")
        with ci_col:
            st.markdown(
                '<div style="font-size:0.78rem;color:#8b949e;padding-top:0.5rem;">'
                f'Stages 7–12 · Uses Tier 1 + Tier 2 params · {n_council_runs}× council runs</div>',
                unsafe_allow_html=True)
        default_lab_name = (
            st.session_state.current_run_name
            or context.get("domain")
            or f"Analysis — {datetime.now().strftime('%H:%M')}"
        )
        lab_run_name = st.text_input(
            "Computational Lab run name",
            value=f"{default_lab_name} · Computational Lab",
            key="comp_run_name",
            help="This name is used in run history after the lab output is saved.",
        )

        if run_comp:
            comp_prog = st.progress(0)
            comp_stat = st.empty()
            def cupd(msg, pct):
                comp_prog.progress(pct)
                comp_stat.markdown(_card("Computational Lab", f"⟳ {msg}"),
                                   unsafe_allow_html=True)
            try:
                client = AzureOpenAIClient(api_key=api_key.strip())

                # Stage 8A: Tier 1 parameter extraction
                papers_raw = [
                    {"sections": p.get("sections",{}), "tables": p.get("tables",[]),
                     "filename": p.get("filename","")}
                    for p in res.get("papers",[])
                ]
                cupd("Stage 8A — Extracting Tier 1 parameters…", 8)
                t1_params = extract_parameters(wiki, papers_raw, client)

                # Stage 8A-ext: Tier 2 parameters from reference abstracts
                t2_params = []
                tier2_refs = [r for r in ref_papers if r.get("tier")==2 and r.get("abstract")]
                if tier2_refs:
                    cupd(f"Stage 8A-T2 — Extracting Tier 2 params from {len(tier2_refs)} ref abstracts…", 14)
                    t2_params = extract_tier2_parameters(tier2_refs, client)

                params = t1_params + t2_params

                cupd("Stage 8A — Deriving cross-paper values…", 18)
                derived = derive_cross_paper_parameters(params, client)
                params += derived

                sufficiency = check_sufficiency(params, client)
                csv_data    = params_to_csv(params)
                _add_trace("Stage 8A — Parameters",
                           f"{len(papers_raw)} T1 papers + {len(tier2_refs)} T2 abstracts",
                           f"{len(t1_params)} T1 + {len(t2_params)} T2 + {len(derived)} derived = {len(params)} total")
                comp_prog.progress(25)

                # Stage 8B: Discovery
                cupd("Stage 8B — Cross-paper discovery engine…", 27)
                discovery = run_discovery_engine(wiki, params, context, client, domain_cfg)
                _add_trace("Stage 8B — Discovery",
                           f"{len(wiki.get('pages',[]))} wiki pages",
                           f"{len(discovery.get('hypotheses',[]))} novel hypotheses")
                comp_prog.progress(40)

                # Stage 9: Council model selection
                cupd(f"Stage 9 — LLM Council selecting model ({len(params)} params)…", 42)
                model_council = council_select_model(params, sufficiency, context, client)
                comp_prog.progress(55)

                # Stage 10: Multi-algorithm simulation
                models_auto = auto_select_models(params, sufficiency)
                cupd(f"Stage 10 — Running {len(models_auto)} algorithms: {', '.join(models_auto)}…", 57)
                sim_results = run_simulation(params, model_council, sufficiency)
                _add_trace("Stage 10 — Simulation",
                           f"{len(models_auto)} algorithms selected",
                           f"Models: {sim_results.get('models_run',[])} run")
                comp_prog.progress(72)

                # Stage 11: Multi-run council
                def _cp(run_num, total):
                    cupd(f"Stage 11 — Council run {run_num}/{total}…",
                         74 + int((run_num-1)/total*12))
                cupd(f"Stage 11 — Council synthesising insights ({n_council_runs}× runs)…", 74)
                merged_council = run_multi_council(
                    n_runs=n_council_runs,
                    sim_results=sim_results, discovery=discovery,
                    params=params, gaps=gaps, context=context,
                    client=client, progress_callback=_cp,
                )
                comp_prog.progress(88)

                # Stage 12: Package
                cupd("Stage 12 — Building report package…", 90)
                zip_bytes = build_zip(
                    params_csv=csv_data, sim_results=sim_results,
                    model_council=model_council, insight_council=merged_council,
                    context=context, params=params, sufficiency=sufficiency,
                    discovery=discovery,
                )
                comp_prog.progress(100)

                st.session_state.comp_results = {
                    "params": params, "derived_count": len(derived),
                    "t1_count": len(t1_params), "t2_count": len(t2_params),
                    "sufficiency": sufficiency, "csv_data": csv_data,
                    "discovery": discovery, "model_council": model_council,
                    "sim_results": sim_results, "insight_council": merged_council,
                    "n_runs": n_council_runs, "zip_bytes": zip_bytes,
                }
                # Save comp results to existing run
                if st.session_state.results:
                    try:
                        saved_comp_id = save_run(
                            lab_run_name.strip() or f"{default_lab_name} · Computational Lab",
                            st.session_state.results,
                            st.session_state.comp_results,
                        )
                        st.session_state.current_run_id = saved_comp_id
                        st.session_state.current_run_name = lab_run_name.strip() or f"{default_lab_name} · Computational Lab"
                        st.session_state.loaded_run_id = saved_comp_id
                    except Exception:
                        pass

                comp_prog.empty(); comp_stat.empty()
                st.success(
                    f"✅ Computational Lab saved — {len(params)} params "
                    f"(T1:{len(t1_params)} T2:{len(t2_params)} derived:{len(derived)}), "
                    f"{len(models_auto)} simulations, "
                    f"{len(discovery.get('hypotheses',[]))} hypotheses. "
                    f"Run history will show the 🧬 marker.")

            except Exception as e:
                comp_prog.empty(); comp_stat.empty()
                st.error(f"Error: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

        # ── Comp results display ───────────────────────────────────────────────
        if st.session_state.comp_results:
            cr = st.session_state.comp_results
            params         = cr["params"]
            sufficiency    = cr["sufficiency"]
            discovery      = cr["discovery"]
            model_council  = cr["model_council"]
            sim_results    = cr["sim_results"]
            insight_council = cr["insight_council"]
            n_runs         = cr.get("n_runs", 1)

            st.markdown("---")

            # Data availability
            score      = sufficiency.get("coverage_score",0)
            sc         = "#3fb950" if score>=7 else "#d29922" if score>=4 else "#f85149"
            t1c, t2c   = cr.get("t1_count",0), cr.get("t2_count",0)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {sc};border-radius:8px;
                        padding:1rem 1.2rem;margin-bottom:1.2rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{sc};
                          text-transform:uppercase;letter-spacing:0.1em;">Data Availability</div>
              <div style="font-size:1.2rem;color:{sc};font-family:'DM Mono',monospace;">{score}/10</div>
              <div style="font-size:0.85rem;color:#c9d1d9;margin-top:0.3rem;">
                {len(params)} total parameters — {_tier_badge(1)} {t1c} from uploaded PDFs &nbsp;
                {_tier_badge(2)} {t2c} from cited abstracts &nbsp;
                🔀 {cr.get('derived_count',0)} cross-paper derived
              </div>
              <div style="font-size:0.78rem;color:#8b949e;margin-top:0.3rem;">
                {sufficiency.get('why_score_is_low','')}
              </div>
            </div>""", unsafe_allow_html=True)

            # Parameters grouped
            st.markdown("#### Stage 8 — Extracted Parameters")
            grouped = group_params_by_category(params)
            for cat, cat_params in list(grouped.items())[:10]:
                t1_in_cat = [p for p in cat_params if p.get("tier",1)==1]
                t2_in_cat = [p for p in cat_params if p.get("tier",1)==2]
                label = (f"📊 {cat.replace('_',' ').title()} "
                         f"({len(cat_params)} total: {len(t1_in_cat)} T1 + {len(t2_in_cat)} T2)")
                with st.expander(label):
                    for p in cat_params:
                        ci = (f" (CI: {p['ci_lower']}–{p['ci_upper']})"
                              if p.get("ci_lower") is not None else "")
                        tier_t = p.get("tier",1)
                        derived_t = " 🔀" if p.get("is_derived") else ""
                        conf_c = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(
                            p.get("confidence","medium"),"#8b949e")
                        st.markdown(
                            f'<div class="wiki-item" style="margin-bottom:0.4rem;">'
                            f'{_tier_badge(tier_t)}'
                            f'<span style="color:#79c0ff;">{p.get("name","")}</span> = '
                            f'<span style="color:{conf_c};font-family:\'DM Mono\',monospace;">'
                            f'{p.get("value","")}{ci}</span> {p.get("unit","")}'
                            f'<span style="color:#484f58;font-size:0.7rem;"> '
                            f'[{p.get("source_paper","")[:45]}]{derived_t}</span></div>',
                            unsafe_allow_html=True)

            st.download_button("⬇ Download parameters.csv", data=cr["csv_data"],
                               file_name="parameters.csv", mime="text/csv")

            # Discovery
            st.markdown("#### Stage 8B — Novel Hypotheses")
            for h in discovery.get("hypotheses",[]):
                conf  = h.get("confidence","medium")
                cc    = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(conf,"#8b949e")
                st.markdown(
                    f'<div style="background:#0d2136;border:1px solid #1f4f7a;'
                    f'border-left:4px solid {cc};border-radius:8px;'
                    f'padding:1.2rem 1.5rem;margin-bottom:1rem;">'
                    f'<div style="font-size:1.05rem;color:#e6edf3;font-weight:500;">'
                    f'{h.get("hypothesis_id","H?")}  {h.get("title","")}</div>'
                    f'<div style="font-size:0.87rem;color:#c9d1d9;margin-top:0.4rem;">'
                    f'{h.get("claim","")}</div>'
                    f'<div style="margin-top:0.5rem;">'
                    + _badge(f"Novelty {h.get('novelty_score',0)}/10","#79c0ff","#0f2d3d","#1f6feb")
                    + _badge(conf.title()+" Conf",cc,"#161b22",cc)
                    + '</div></div>',
                    unsafe_allow_html=True)
                with st.expander(f"Evidence + opportunities for {h.get('hypothesis_id','')}"):
                    hc1,hc2 = st.columns(2)
                    with hc1:
                        _section_label("Mechanism"); _item(h.get("mechanism",""))
                        st.markdown("")
                        _section_label("Validation Study"); _item(h.get("validation_study",""),"#79c0ff")
                    with hc2:
                        _section_label("Clinical Opportunity"); _item(h.get("clinical_opportunity",""),"#3fb950")
                        st.markdown("")
                        _section_label("Commercial Opportunity"); _item(h.get("commercial_opportunity",""),"#d29922")

            # Simulation
            st.markdown(f"#### Stage 10 — Simulation: {sim_results.get('model_used','')}")
            models_run = sim_results.get("models_run",[])
            if models_run:
                st.markdown("Algorithms run: " +
                            " ".join(_badge(m) for m in models_run),
                            unsafe_allow_html=True)

            sens = sim_results.get("sensitivity_ranking",[])
            if sens:
                st.markdown("**Top sensitivity drivers (% deviation from central value):**")
                for i,s in enumerate(sens[:6],1):
                    bar = int(min(s.get("importance",0),1.0)*100)
                    swing = s.get("swing_pct", "")
                    label_extra = f" — {swing}% range" if swing else ""
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;">'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#c9d1d9;">'
                        f'{i}. {s["parameter"]}{label_extra}</div>'
                        f'<div style="background:#30363d;border-radius:3px;height:6px;margin-top:2px;">'
                        f'<div style="background:#58a6ff;width:{bar}%;height:6px;border-radius:3px;"></div>'
                        f'</div></div>', unsafe_allow_html=True)

            # Council insights (multi-run merged)
            st.markdown(f"#### Stage 11 — Council Findings ({n_runs}× runs merged)")
            ic = insight_council
            cov = ic.get("coverage_stats",{})
            st.markdown(f"""
            <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;
                        padding:0.9rem 1.2rem;margin-bottom:1.2rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;
                          text-transform:uppercase;letter-spacing:0.1em;">Multi-Run Coverage</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-top:0.5rem;">
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#3fb950;font-family:'DM Mono',monospace;">
                    {cov.get('total_novel_findings',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Novel Findings</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#79c0ff;font-family:'DM Mono',monospace;">
                    {cov.get('total_clinical_insights',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Clinical Insights</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.4rem;color:#d29922;font-family:'DM Mono',monospace;">
                    {cov.get('total_next_steps',0)}</div>
                  <div style="font-size:0.68rem;color:#8b949e;text-transform:uppercase;">Next Steps</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Novel findings
            for g in ic.get("novel_finding_groups",[]):
                nf   = g["best"]
                cc   = g["confidence_color"]
                freq = g["frequency"]
                st.markdown(
                    f'<div style="background:#0a1f0a;border:2px solid {cc};border-radius:10px;'
                    f'padding:1.2rem 1.5rem;margin-bottom:1rem;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:{cc};">🔬 NOVEL FINDING</div>'
                    f'<div>{_badge(g["confidence"],cc,"#161b22",cc)}'
                    f'{_badge(freq,"#8b949e","#1c1c1c","#30363d")}</div></div>'
                    f'<div style="font-size:1.05rem;color:#e6edf3;font-weight:600;margin:0.5rem 0 0.4rem;">'
                    f'{nf.get("title","")}</div>'
                    f'<div style="font-size:0.88rem;color:#c9d1d9;line-height:1.7;">{nf.get("claim","")}</div>'
                    f'<div style="font-size:0.75rem;color:#8b949e;margin-top:0.4rem;font-style:italic;">'
                    f'What\'s new: {nf.get("what_makes_it_new","")}</div></div>',
                    unsafe_allow_html=True)

            # Insights grid
            def _show_groups(groups, icon):
                for g in groups:
                    cc = g["confidence_color"]
                    st.markdown(
                        f'<div style="background:#161b22;border:1px solid #30363d;'
                        f'border-left:3px solid {cc};border-radius:5px;'
                        f'padding:0.6rem 0.9rem;margin-bottom:0.35rem;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<div style="font-size:0.85rem;color:#c9d1d9;">{icon} {g["representative"]}</div>'
                        f'<div>{_badge(g["confidence"],cc,"#161b22",cc)}'
                        f'{_badge(g["frequency"],"#8b949e","#1c1c1c","#30363d")}</div></div></div>',
                        unsafe_allow_html=True)

            ic1,ic2 = st.columns(2)
            with ic1:
                if ic.get("clinical_insight_groups"):
                    st.markdown("##### Clinical Insights")
                    _show_groups(ic["clinical_insight_groups"],"🏥")
                if ic.get("next_step_groups"):
                    st.markdown("##### Next Steps")
                    _show_groups(ic["next_step_groups"],"→")
            with ic2:
                if ic.get("research_insight_groups"):
                    st.markdown("##### Research Insights")
                    _show_groups(ic["research_insight_groups"],"🔬")
                if ic.get("limitation_groups"):
                    st.markdown("##### Limitations")
                    _show_groups(ic["limitation_groups"],"⚠️")

            # Download
            st.markdown("---")
            st.markdown("#### Stage 12 — Download Report Package")
            st.markdown(
                '<div class="stage-card"><div class="stage-title">Contents</div>'
                '<div class="stage-body">📄 parameters.csv · 📊 simulation_results.json · '
                '🏛️ council_debate.md · 💡 insights.md · 📋 full_report.md · 🔬 novel_hypotheses.json'
                '</div></div>', unsafe_allow_html=True)
            st.download_button("⬇ Download ZIP",
                               data=cr["zip_bytes"],
                               file_name="computational_lab_output.zip",
                               mime="application/zip")

    # ── Tab 7: Pipeline Trace ──────────────────────────────────────────────────
    with tab7:
        st.markdown("### 🔎 Pipeline Trace — Explainability View")
        trace = st.session_state.pipeline_trace
        if not trace:
            st.info("Run the main analysis to see the pipeline trace here.")
        else:
            for i, entry in enumerate(trace, 1):
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-left:3px solid #58a6ff;border-radius:6px;'
                    f'padding:0.8rem 1.2rem;margin-bottom:0.5rem;">'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#58a6ff;">'
                    f'{entry.get("status","✅")} STAGE {i} — {entry["stage"]}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">'
                    f'<div><span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">IN: </span>'
                    f'<span style="font-size:0.8rem;color:#c9d1d9;">{entry["inputs"]}</span></div>'
                    f'<div><span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#484f58;">OUT: </span>'
                    f'<span style="font-size:0.8rem;color:#3fb950;">{entry["outputs"]}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

        if show_prompts:
            st.markdown("---")
            st.markdown("#### Active Prompts")
            from pipeline.wiki_compiler import WIKI_SYSTEM_PROMPT
            from pipeline.gap_detector import GAP_DETECTION_PROMPT
            with st.expander("Wiki Compiler Prompt"):
                st.code(WIKI_SYSTEM_PROMPT, language=None)
            with st.expander("Gap Detection Prompt"):
                st.code(GAP_DETECTION_PROMPT, language=None)

    # ── Tab 8: Export ──────────────────────────────────────────────────────────
    with tab8:
        st.markdown("### Export Results")
        st.markdown(
            '<div class="stage-card"><div class="stage-title">Polished Final Report</div>'
            '<div class="stage-body">Generates an enterprise HTML report and matching PDF with cover page, executive summary, corpus overview, references, graph, gaps, validation, proposals, risks, and appendices.</div></div>',
            unsafe_allow_html=True)
        gen_col, mode_col = st.columns([1, 2])
        with gen_col:
            generate_report = st.button("Generate final report", key="generate_final_report")
        with mode_col:
            use_llm_report = st.checkbox("Use LLM narrative polish", value=True,
                                         help="Uses the NVIDIA API key to write concise report narrative from existing structured results only.")

        if generate_report:
            try:
                report_client = AzureOpenAIClient(api_key=api_key.strip()) if (use_llm_report and api_key) else None
                narrative = generate_report_narrative(res, context, client=report_client)
                html_report = to_enterprise_html_report(res, context, narrative=narrative)
                pdf_report = to_enterprise_pdf_report(res, context, narrative=narrative)
                st.session_state.report_bundle = {
                    "html": html_report,
                    "pdf": pdf_report,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "used_llm": bool(report_client),
                }
                st.success("Final report generated.")
            except Exception as e:
                st.error(f"Final report generation failed: {e}")
                with st.expander("Report error details"):
                    st.code(traceback.format_exc())

        if st.session_state.report_bundle:
            bundle = st.session_state.report_bundle
            st.info(f"Report ready. Generated at {bundle.get('generated_at','')} · LLM narrative: {'yes' if bundle.get('used_llm') else 'no'}")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("Download HTML Report",
                                   data=bundle["html"].encode("utf-8"),
                                   file_name="research_gap_final_report.html",
                                   mime="text/html")
            with dl2:
                st.download_button("Download PDF Report",
                                   data=bundle["pdf"],
                                   file_name="research_gap_final_report.pdf",
                                   mime="application/pdf")

        st.markdown("---")
        ex1,ex2,ex3 = st.columns(3)
        with ex1:
            st.markdown('<div class="stage-card"><div class="stage-title">PDF Report</div>'
                        '<div class="stage-body">Legacy quick PDF summary.</div></div>',
                        unsafe_allow_html=True)
            try:
                st.download_button("Download Quick PDF",
                                   data=to_pdf_report(res, context),
                                   file_name="research_gap_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF report generation failed: {e}")
        with ex2:
            st.markdown('<div class="stage-card"><div class="stage-title">Markdown Report</div>'
                        '<div class="stage-body">Full analysis — wiki, gaps, proposals.</div></div>',
                        unsafe_allow_html=True)
            st.download_button("Download Markdown",
                               data=to_markdown_report(res, context),
                               file_name="research_gap_report.md", mime="text/markdown")
        with ex3:
            st.markdown('<div class="stage-card"><div class="stage-title">JSON Export</div>'
                        '<div class="stage-body">Complete structured data.</div></div>',
                        unsafe_allow_html=True)
            st.download_button("Download JSON",
                               data=to_json(res),
                               file_name="research_gap_analysis.json",
                               mime="application/json")
        with st.expander("Raw JSON"):
            st.json(res)
