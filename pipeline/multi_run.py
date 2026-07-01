"""
Multi-Run Council — runs the insight council N times and merges results.

Rationale: LLM outputs are non-deterministic. Instead of forcing determinism
(which fails), we run N times and UNION all findings so nothing is missed.

Frequency = confidence:
  Appears in all runs  → Consensus  (highest trust)
  Appears in most runs → Likely     (medium trust)
  Appears in one run   → Unique     (still valuable — don't discard)

The simulation numbers are always identical (numpy seed=42 is fixed).
Only the LLM interpretation stage is run multiple times.
"""
import re
from pipeline.llm_council import council_synthesise_insights


# ── Text similarity (no LLM needed — keyword overlap) ────────────────────────

def _keywords(text: str) -> set:
    """Extract meaningful keywords — words >4 chars, no stopwords."""
    stopwords = {
        "with", "from", "this", "that", "have", "will", "been", "their",
        "which", "would", "could", "should", "about", "these", "those",
        "study", "paper", "research", "patients", "clinical", "analysis",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in stopwords}


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def _confidence_tag(count: int, n_runs: int) -> tuple:
    """Return (label, color) based on how many runs produced this finding.
    B7 fix: n_runs=1 always produced 'Consensus' (1/1=1.0) which is misleading.
    Single-run findings are now tagged 'Single Run' to be honest about confidence."""
    if n_runs == 1:
        return "Single Run", "#8b949e"   # grey — honest about single-run limitation
    ratio = count / n_runs
    if ratio >= 0.85:
        return "Consensus", "#3fb950"    # green
    elif ratio >= 0.5:
        return "Likely",    "#d29922"    # yellow
    else:
        return "Unique",    "#58a6ff"    # blue — still shown, still valuable


# ── Group similar text items ──────────────────────────────────────────────────

def _group_items(items: list, n_runs: int, threshold: float = 0.30) -> list:
    """
    Group text items by keyword similarity.
    Items with >threshold similarity are merged into one group.
    Returns groups sorted by frequency (consensus first).
    """
    groups = []

    for item in items:
        text = item["text"]
        matched = False
        for group in groups:
            if _similarity(text, group["representative"]) > threshold:
                group["count"] += 1
                group["runs"].append(item["run"])
                group["variants"].append(text)
                # Keep the most detailed version as representative
                if len(text) > len(group["representative"]):
                    group["representative"] = text
                matched = True
                break
        if not matched:
            groups.append({
                "representative": text,
                "count":          1,
                "runs":           [item["run"]],
                "variants":       [text],
            })

    # Sort: consensus first, then by count, then alphabetically
    groups.sort(key=lambda g: (-g["count"], g["representative"]))

    for g in groups:
        label, color = _confidence_tag(g["count"], n_runs)
        g["confidence"]       = label
        g["confidence_color"] = color
        g["frequency"]        = f"{g['count']}/{n_runs} runs"

    return groups


def _group_novel_findings(findings: list, n_runs: int) -> list:
    """
    Group novel findings across runs by claim similarity.
    Each group gets a confidence tag. All are shown — nothing discarded.
    """
    groups = []

    for finding in findings:
        claim = finding.get("claim", "")
        matched = False
        for group in groups:
            if _similarity(claim, group["best"]["claim"]) > 0.35:
                group["count"] += 1
                group["runs"].append(finding["run_index"])
                group["all_versions"].append(finding)
                # Keep highest-confidence version
                current_conf = {"high": 3, "medium": 2, "low": 1}
                best_conf    = current_conf.get(group["best"].get("confidence", "low"), 1)
                this_conf    = current_conf.get(finding.get("confidence", "low"), 1)
                if this_conf > best_conf or (
                    this_conf == best_conf and
                    len(finding.get("claim", "")) > len(group["best"].get("claim", ""))
                ):
                    group["best"] = finding
                matched = True
                break
        if not matched:
            groups.append({
                "best":         finding,
                "count":        1,
                "runs":         [finding["run_index"]],
                "all_versions": [finding],
            })

    groups.sort(key=lambda g: (-g["count"], g["best"].get("impact_score", 0)))

    for g in groups:
        label, color = _confidence_tag(g["count"], n_runs)
        g["confidence"]       = label
        g["confidence_color"] = color
        g["frequency"]        = f"{g['count']}/{n_runs} runs"

    return groups


# ── Main multi-run function ───────────────────────────────────────────────────

def run_multi_council(
    n_runs:      int,
    sim_results: dict,
    discovery:   dict,
    params:      list,
    gaps:        list,
    context:     dict,
    client,
    progress_callback=None,
) -> dict:
    """
    Run the insight council N times, merge all findings.

    Args:
        n_runs:            Number of council runs (2 or 3 recommended)
        progress_callback: optional fn(run_number, n_runs) for UI updates
        All other args:    passed directly to council_synthesise_insights

    Returns merged results with frequency-tagged findings.
    """
    all_runs = []

    for i in range(n_runs):
        if progress_callback:
            progress_callback(i + 1, n_runs)

        try:
            result = council_synthesise_insights(
                sim_results=sim_results,
                discovery=discovery,
                params=params,
                gaps=gaps,
                context=context,
                client=client,
            )
            result["run_index"] = i + 1
            all_runs.append(result)
        except Exception as e:
            all_runs.append({
                "run_index":     i + 1,
                "final_decision": {},
                "agents":        [],
                "error":         str(e),
            })

    return merge_council_runs(all_runs)


def merge_council_runs(runs: list) -> dict:
    """
    Merge N council run results into one unified result.
    Every unique finding is preserved. Confidence tagged by frequency.
    """
    n = len(runs)

    # Collect all items across runs
    raw_novel    = []
    raw_clinical = []
    raw_research = []
    raw_steps    = []
    raw_limits   = []
    raw_uncerts  = []
    all_agents   = []
    simulation_interpretations = []

    for run in runs:
        fd  = run.get("final_decision", {})
        nf  = fd.get("novel_finding", {})
        idx = run.get("run_index", 1)

        if nf and nf.get("claim"):
            nf["run_index"] = idx
            raw_novel.append(nf)

        for item in fd.get("clinical_insights", []):
            raw_clinical.append({"text": item, "run": idx})
        for item in fd.get("research_insights", []):
            raw_research.append({"text": item, "run": idx})
        for item in fd.get("actionable_next_steps", []):
            raw_steps.append({"text": item, "run": idx})
        for item in fd.get("limitations", []):
            raw_limits.append({"text": item, "run": idx})
        for item in fd.get("key_uncertainties", []):
            raw_uncerts.append({"text": item, "run": idx})

        interp = fd.get("simulation_interpretation", "")
        if interp:
            simulation_interpretations.append({"text": interp, "run": idx})

        for agent in run.get("agents", []):
            agent["run_index"] = idx
            all_agents.append(agent)

    # Group and tag everything
    novel_groups    = _group_novel_findings(raw_novel, n)
    clinical_groups = _group_items(raw_clinical, n)
    research_groups = _group_items(raw_research, n)
    steps_groups    = _group_items(raw_steps, n)
    limits_groups   = _group_items(raw_limits, n)
    uncert_groups   = _group_items(raw_uncerts, n)

    # Build summary stats
    consensus_count = sum(1 for g in novel_groups if g["confidence"] == "Consensus")
    unique_count    = sum(1 for g in novel_groups if g["confidence"] == "Unique")

    return {
        # Core results
        "novel_finding_groups":   novel_groups,
        "clinical_insight_groups": clinical_groups,
        "research_insight_groups": research_groups,
        "next_step_groups":        steps_groups,
        "limitation_groups":       limits_groups,
        "uncertainty_groups":      uncert_groups,
        "simulation_interpretations": simulation_interpretations,

        # For backward compat — best single novel finding
        "final_decision": {
            "novel_finding":          novel_groups[0]["best"] if novel_groups else {},
            "clinical_insights":      [g["representative"] for g in clinical_groups],
            "research_insights":      [g["representative"] for g in research_groups],
            "actionable_next_steps":  [g["representative"] for g in steps_groups],
            "limitations":            [g["representative"] for g in limits_groups],
            "simulation_interpretation": simulation_interpretations[0]["text"] if simulation_interpretations else "",
        },

        # Metadata
        "run_count":       n,
        "all_runs":        runs,
        "all_agents":      all_agents,
        "coverage_stats": {
            "total_novel_findings":     len(novel_groups),
            "consensus_findings":       consensus_count,
            "unique_findings":          unique_count,
            "total_clinical_insights":  len(clinical_groups),
            "total_research_insights":  len(research_groups),
            "total_next_steps":         len(steps_groups),
        },
    }
