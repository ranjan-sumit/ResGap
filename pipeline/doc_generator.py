"""
Stage 12: Document Generator — updated to include novel findings and multi-algorithm results.
"""
import json, io, zipfile
from datetime import datetime


def build_council_debate_md(model_council: dict, insight_council: dict) -> str:
    lines = ["# LLM Council Debates\n",
             f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
             "## Council 1 — Model Selection\n"]
    for a in model_council.get("agents", []):
        lines += [f"### {a['agent']} ({a['role']}) [{a.get('model','')}]\n", a["response"], "\n"]
    c = model_council.get("consensus", {})
    if c:
        lines += [f"### Consensus\n{c.get('consensus','')}\n"]
        for pt in c.get("key_points", []):
            lines.append(f"- {pt}")
    fd = model_council.get("final_decision", {})
    if fd:
        lines += [f"\n### Final Decision\n**Model:** {fd.get('selected_model','')}\n",
                  f"**Rationale:** {fd.get('rationale','')}\n"]

    lines += ["\n---\n\n## Council 2 — Insight Synthesis\n"]
    for a in insight_council.get("agents", []):
        lines += [f"### {a['agent']} ({a['role']}) [{a.get('model','')}]\n", a["response"], "\n"]

    fi = insight_council.get("final_decision", {})
    nf = fi.get("novel_finding", {})
    if nf:
        lines += ["\n### Novel Finding\n",
                  f"**{nf.get('title','')}**\n\n{nf.get('claim','')}\n\n",
                  f"**What makes it new:** {nf.get('what_makes_it_new','')}\n",
                  f"**Mechanism:** {nf.get('mechanism','')}\n",
                  f"**Validation study:** {nf.get('validation_study','')}\n",
                  f"**Clinical opportunity:** {nf.get('clinical_opportunity','')}\n",
                  f"**Confidence:** {nf.get('confidence','')}\n"]
    return "\n".join(lines)


def build_insights_md(insight_council: dict, discovery: dict) -> str:
    fi  = insight_council.get("final_decision", {})
    nf  = fi.get("novel_finding", {})
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Insights & Novel Findings\n*{now}*\n"]

    if nf:
        lines += ["## 🔬 Novel Finding\n",
                  f"### {nf.get('title','')}\n",
                  f"**Claim:** {nf.get('claim','')}\n",
                  f"**What's New:** {nf.get('what_makes_it_new','')}\n",
                  f"**Mechanism:** {nf.get('mechanism','')}\n",
                  f"**Validation:** {nf.get('validation_study','')}\n",
                  f"**Clinical:** {nf.get('clinical_opportunity','')}\n",
                  f"**Commercial:** {nf.get('commercial_opportunity','')}\n",
                  f"**Confidence:** {nf.get('confidence','')} — {nf.get('confidence_reasoning','')}\n"]

    hypotheses = discovery.get("hypotheses", [])
    if hypotheses:
        lines.append("## Cross-Paper Hypotheses\n")
        for h in hypotheses:
            lines += [f"### {h.get('hypothesis_id','H?')}: {h.get('title','')}\n",
                      f"{h.get('claim','')}\n",
                      f"*Confidence: {h.get('confidence','')} | Novelty: {h.get('novelty_score','')}/10 | Impact: {h.get('impact_score','')}/10*\n"]

    for section, key in [
        ("Clinical Insights", "clinical_insights"),
        ("Research Insights", "research_insights"),
        ("Next Steps", "actionable_next_steps"),
        ("Limitations", "limitations"),
    ]:
        items = fi.get(key, [])
        if items:
            lines.append(f"## {section}\n")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines)


def build_full_report_md(context, params, sufficiency, model_council,
                          sim_results, insight_council, discovery) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fi  = insight_council.get("final_decision", {})
    nf  = fi.get("novel_finding", {})
    fd  = model_council.get("final_decision", {})

    lines = [f"# Computational Modelling Report\n*{now}*\n",
             f"**Domain:** {context.get('domain','')}\n",
             f"**Models Run:** {sim_results.get('model_used','')}\n",
             f"**Parameters:** {len(params)} ({sufficiency.get('coverage_score',0)}/10 coverage)\n",
             "\n---\n"]

    if nf:
        lines += ["## 🔬 NOVEL FINDING\n",
                  f"**{nf.get('title','')}**\n\n{nf.get('claim','')}\n\n",
                  f"Evidence: {'; '.join(nf.get('evidence_chain',[]))}\n",
                  f"Confidence: **{nf.get('confidence','')}** — {nf.get('confidence_reasoning','')}\n",
                  "\n---\n"]

    lines += ["## Model Selection\n",
              f"Selected: {fd.get('selected_model','')}\n",
              f"{fd.get('rationale','')}\n\n",
              "## Headline Results\n"]
    for k, v in list(sim_results.get("headline_numbers", {}).items())[:10]:
        lines.append(f"- **{k}:** {v}")

    sens = sim_results.get("sensitivity_ranking", [])
    if sens:
        lines += ["\n## Sensitivity — Top Drivers\n"]
        for i, s in enumerate(sens[:5], 1):
            lines.append(f"{i}. {s['parameter']} (importance {s['importance']})")

    hypotheses = discovery.get("hypotheses", [])
    if hypotheses:
        lines += ["\n## Cross-Paper Hypotheses\n"]
        for h in hypotheses[:3]:
            lines += [f"### {h.get('hypothesis_id','')}: {h.get('title','')}\n",
                      f"{h.get('claim','')}\n"]

    lines.append("\n---\n*Report by Research Gap Analyzer — Computational Lab*")
    return "\n".join(lines)


def build_zip(params_csv, sim_results, model_council, insight_council,
              context, params, sufficiency, discovery=None) -> bytes:
    discovery = discovery or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("parameters.csv", params_csv)
        zf.writestr("simulation_results.json",
                    json.dumps(sim_results, indent=2, default=str))
        zf.writestr("council_debate.md",
                    build_council_debate_md(model_council, insight_council))
        zf.writestr("insights.md",
                    build_insights_md(insight_council, discovery))
        zf.writestr("full_report.md",
                    build_full_report_md(context, params, sufficiency,
                                         model_council, sim_results,
                                         insight_council, discovery))
        if discovery.get("hypotheses"):
            zf.writestr("novel_hypotheses.json",
                        json.dumps(discovery["hypotheses"], indent=2))
    buf.seek(0)
    return buf.read()
