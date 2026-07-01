from .pdf_parser import parse_pdf, get_smart_excerpt
from .page_index import build_tree, tree_to_summary
from .wiki_compiler import build_wiki
from .graph_builder import build_knowledge_graph
from .gap_detector import detect_gaps
from .academic_search import validate_gaps
from .proposal_generator import generate_proposals
from .param_extractor import (
    extract_parameters, derive_cross_paper_parameters,
    check_sufficiency, params_to_csv, group_params_by_category,
)
from .discovery_engine import run_discovery_engine
from .llm_council import council_select_model, council_synthesise_insights
from .simulation_engine import run_simulation, auto_select_models
from .multi_run import run_multi_council, merge_council_runs
from .reference_extractor import (
    extract_and_enrich_references, parse_references_section,
    summarise_reference_corpus,
)
from .paper_manager import (
    assign_tiers, corpus_stats, compile_tier2_batch,
    extract_tier2_parameters, TIER_LABELS,
)
from .doc_generator import build_zip
