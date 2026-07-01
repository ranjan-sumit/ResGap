"""
Domain Configuration — Per-domain prompt tuning.
Each domain customizes:
  - Wiki extraction focus
  - Knowledge graph entity types + relationships
  - Gap detection priorities
  - Proposal framing
  - Academic search keywords
"""

DOMAINS = {
    "Healthcare AI": {
        "icon": "🏥",
        "description": "Clinical AI, medical imaging, EHR, diagnostics, drug discovery",
        "wiki_focus": """
Pay special attention to:
- Clinical validation datasets (real-world vs synthetic, sample size, demographics)
- Patient population coverage (age, ethnicity, gender balance)
- Regulatory considerations (FDA approval, CE marking, clinical trials)
- Clinical metrics (sensitivity, specificity, AUC, NNT, clinical utility)
- Deployment setting (ICU, radiology, primary care, low-resource settings)
- Bias and fairness in patient subgroups
- Generalizability across hospital systems / EHR vendors
- Ethical concerns (explainability, consent, liability)
- Whether the model was validated on prospective vs retrospective data
""",
        "entity_types": [
            "disease", "clinical_task", "imaging_modality", "biomarker",
            "patient_population", "ehr_system", "clinical_metric",
            "dataset", "model_architecture", "regulatory_pathway",
            "deployment_setting", "fairness_concern"
        ],
        "relationship_types": [
            "diagnoses", "predicts", "validated_on", "biased_against",
            "generalizes_to", "fails_on", "requires", "approved_for",
            "compared_with", "outperforms", "underperforms_on"
        ],
        "gap_priorities": [
            "Clinical validation on diverse/underrepresented patient populations",
            "Real-world prospective validation beyond benchmark datasets",
            "Multi-site / multi-vendor generalizability",
            "Fairness and bias across demographic subgroups",
            "Low-resource or LMIC (low and middle income country) deployment",
            "Regulatory pathway gaps — validated but not approved",
            "Explainability for clinical decision support",
            "Integration with actual EHR workflows",
            "Rare disease or underserved condition coverage",
        ],
        "proposal_framing": """
Frame proposals as:
- Clinical problems first, technology second
- Include proposed clinical validation study design (prospective/retrospective, sample size estimate)
- Mention IRB/ethics considerations
- Suggest real-world deployment context (which clinical setting, which specialists)
- Reference relevant clinical guidelines or standards (TRIPOD, STARD, CONSORT-AI)
- Consider health equity and access implications
""",
        "search_boost_terms": [
            "clinical validation", "patient outcomes", "EHR", "EMR",
            "medical imaging", "diagnosis", "prognosis", "treatment",
            "FDA", "clinical trial", "healthcare", "hospital"
        ],
        "wiki_entity_examples": "diseases, imaging modalities, clinical tasks, patient demographics, hospital systems, clinical metrics (AUC, sensitivity, specificity)",
        "gap_examples": "No validation on pediatric populations, no multi-site study, no comparison against clinician baseline, limited to single EHR vendor",
    },

    "NLP / LLMs": {
        "icon": "💬",
        "description": "Language models, text generation, understanding, reasoning",
        "wiki_focus": """
Pay special attention to:
- Benchmark datasets used and their known limitations/biases
- Language coverage (English-only vs multilingual)
- Model scale and compute requirements
- Instruction tuning / RLHF approach
- Hallucination and factuality issues
- Long-context handling limitations
- Domain adaptation (general vs specialized)
- Evaluation methodology (automated vs human)
- Safety and alignment considerations
""",
        "entity_types": [
            "language_model", "benchmark", "task", "language",
            "training_technique", "dataset", "evaluation_metric",
            "application_domain", "safety_concern", "compute_requirement"
        ],
        "relationship_types": [
            "evaluates_on", "outperforms", "trained_with", "fine_tuned_on",
            "fails_on", "hallucinates_in", "generalizes_to", "requires",
            "biased_toward", "multilingual_for", "aligned_with"
        ],
        "gap_priorities": [
            "Multilingual and low-resource language coverage",
            "Long-context reasoning beyond current limits",
            "Hallucination reduction in specialized domains",
            "Evaluation beyond standard English benchmarks",
            "Efficient fine-tuning for domain adaptation",
            "Reasoning and multi-step planning gaps",
            "Safety and alignment in agentic settings",
        ],
        "proposal_framing": """
Frame proposals with:
- Specific benchmark or evaluation suite to demonstrate improvement
- Compute budget considerations (parameter count, training FLOPs)
- Ablation study design
- Comparison baselines (which existing models)
- Reproducibility considerations (model release, data availability)
""",
        "search_boost_terms": [
            "large language model", "transformer", "benchmark", "fine-tuning",
            "RLHF", "instruction tuning", "reasoning", "hallucination"
        ],
        "wiki_entity_examples": "language models, benchmarks, tasks, languages, training techniques",
        "gap_examples": "English-only evaluation, no low-resource language support, no long-context benchmark, missing human evaluation",
    },

    "Computer Vision": {
        "icon": "👁️",
        "description": "Image recognition, detection, segmentation, video understanding",
        "wiki_focus": """
Pay special attention to:
- Dataset diversity (domains, lighting, viewpoints, demographic representation)
- Real-world vs controlled setting performance gap
- Robustness to distribution shift, adversarial examples, corruptions
- Computational efficiency (inference speed, model size, edge deployment)
- Few-shot and zero-shot generalization
- Video vs static image limitations
- 3D understanding gaps
- Annotation cost and label efficiency
""",
        "entity_types": [
            "model_architecture", "dataset", "task", "modality",
            "benchmark", "deployment_constraint", "augmentation_technique",
            "evaluation_metric", "application_domain"
        ],
        "relationship_types": [
            "trained_on", "evaluated_on", "outperforms", "fails_on",
            "generalizes_to", "robust_to", "sensitive_to", "requires",
            "deployed_on", "compared_with"
        ],
        "gap_priorities": [
            "Distribution shift and domain generalization",
            "Few-shot learning in novel visual domains",
            "Edge device deployment constraints",
            "3D scene understanding",
            "Video temporal reasoning",
            "Annotation-efficient methods",
            "Demographic and geographic dataset bias",
        ],
        "proposal_framing": """
Frame proposals with:
- Specific dataset splits and evaluation protocol
- Inference speed / FLOPs / parameter count targets
- Hardware deployment target (GPU, edge device, mobile)
- Ablation study on key architectural decisions
""",
        "search_boost_terms": [
            "image classification", "object detection", "segmentation",
            "vision transformer", "dataset", "benchmark", "robustness", "domain shift"
        ],
        "wiki_entity_examples": "architectures, datasets, tasks (detection, segmentation), metrics (mAP, IoU), deployment settings",
        "gap_examples": "No evaluation on out-of-distribution data, single dataset benchmark, no edge deployment study",
    },

    "Cybersecurity AI": {
        "icon": "🔐",
        "description": "Threat detection, malware analysis, intrusion detection, SOC automation",
        "wiki_focus": """
Pay special attention to:
- Attack types covered and notably missing
- Dataset recency and whether it reflects current threat landscape
- Adversarial robustness — can the model be evaded?
- False positive rates in operational settings
- Real-world deployment vs lab evaluation gap
- Zero-day and novel threat detection capability
- Explainability for SOC analyst workflows
- Privacy-preserving constraints (federated learning needs)
- MITRE ATT&CK coverage
""",
        "entity_types": [
            "attack_type", "malware_family", "detection_method", "dataset",
            "network_protocol", "deployment_setting", "evasion_technique",
            "mitre_technique", "false_positive_rate", "explainability_method"
        ],
        "relationship_types": [
            "detects", "evades", "trained_on", "fails_against",
            "generalizes_to", "maps_to_mitre", "deployed_in",
            "generates_alert_for", "compared_with", "outperforms"
        ],
        "gap_priorities": [
            "Novel/zero-day attack detection beyond seen malware families",
            "Adversarial evasion robustness",
            "Low false positive rate in production SOC environments",
            "Dataset recency — models trained on stale threat data",
            "MITRE ATT&CK coverage gaps",
            "Explainability for analyst-in-the-loop workflows",
            "Cross-environment generalization (cloud, OT/ICS, mobile)",
            "Federated/privacy-preserving threat intelligence sharing",
        ],
        "proposal_framing": """
Frame proposals with:
- Specific attack scenario and threat model
- Dataset recency and provenance (public vs enterprise)
- MITRE ATT&CK technique coverage
- False positive rate target for operational viability
- Red team / adversarial evaluation plan
- SOC workflow integration considerations
""",
        "search_boost_terms": [
            "intrusion detection", "malware", "threat detection", "anomaly detection",
            "MITRE ATT&CK", "SOC", "network security", "adversarial"
        ],
        "wiki_entity_examples": "attack types, malware families, detection methods, MITRE techniques, datasets, SOC workflows",
        "gap_examples": "No adversarial evaluation, stale dataset (pre-2022), no zero-day testing, single environment (only network, no endpoint)",
    },

    "Federated Learning": {
        "icon": "🌐",
        "description": "Privacy-preserving distributed ML, edge intelligence",
        "wiki_focus": """
Pay special attention to:
- Communication efficiency (rounds to convergence, bandwidth)
- Non-IID data heterogeneity handling
- Privacy guarantees (differential privacy epsilon values)
- Client drop-out and system heterogeneity
- Aggregation strategy limitations
- Convergence guarantees and theoretical bounds
- Fairness across clients with imbalanced data
- Threat model (poisoning attacks, model inversion)
""",
        "entity_types": [
            "aggregation_algorithm", "privacy_mechanism", "dataset",
            "communication_strategy", "client_type", "attack_type",
            "convergence_metric", "fairness_metric", "deployment_scenario"
        ],
        "relationship_types": [
            "aggregates", "protects_against", "vulnerable_to", "converges_under",
            "evaluated_on", "outperforms", "requires", "handles_heterogeneity"
        ],
        "gap_priorities": [
            "Extreme non-IID data distribution handling",
            "Communication efficiency at large client scale",
            "Combined privacy + fairness guarantees",
            "Cross-silo vs cross-device generalization",
            "Byzantine robustness under realistic attack models",
            "Convergence guarantees for heterogeneous systems",
        ],
        "proposal_framing": """
Frame proposals with:
- Theoretical convergence analysis
- Privacy budget (epsilon, delta) reporting
- Communication cost analysis
- Benchmark against FedAvg and recent baselines
- System heterogeneity simulation design
""",
        "search_boost_terms": [
            "federated learning", "differential privacy", "non-IID",
            "communication efficiency", "aggregation", "privacy", "distributed"
        ],
        "wiki_entity_examples": "aggregation algorithms, privacy mechanisms, datasets, communication strategies, attack types",
        "gap_examples": "No non-IID evaluation, no communication cost analysis, missing privacy-utility tradeoff study",
    },

    "Reinforcement Learning": {
        "icon": "🎮",
        "description": "RL algorithms, multi-agent, offline RL, real-world deployment",
        "wiki_focus": """
Pay special attention to:
- Sample efficiency (training steps to convergence)
- Transfer across environments
- Sim-to-real gap
- Reward shaping and specification choices
- Multi-agent coordination challenges
- Safety constraints during exploration
- Offline RL — dataset quality and coverage assumptions
- Benchmark environment limitations vs real-world applicability
""",
        "entity_types": [
            "algorithm", "environment", "reward_function", "policy",
            "exploration_strategy", "benchmark", "application_domain",
            "safety_constraint", "transfer_method"
        ],
        "relationship_types": [
            "trained_in", "evaluated_on", "transfers_to", "fails_in",
            "outperforms", "requires", "unsafe_in", "compared_with"
        ],
        "gap_priorities": [
            "Sample efficiency for real-world applications",
            "Safe exploration under hard constraints",
            "Sim-to-real transfer gaps",
            "Offline RL dataset coverage assumptions",
            "Multi-agent scalability beyond toy settings",
            "Reward misspecification robustness",
        ],
        "proposal_framing": """
Frame proposals with:
- Environment and benchmark specification
- Sample efficiency measurement protocol
- Safety constraint formalization
- Comparison against on-policy and off-policy baselines
- Real-world deployment feasibility analysis
""",
        "search_boost_terms": [
            "reinforcement learning", "policy gradient", "Q-learning",
            "multi-agent", "offline RL", "reward", "exploration", "sample efficiency"
        ],
        "wiki_entity_examples": "algorithms, environments, reward functions, policies, benchmarks",
        "gap_examples": "Only evaluated in simulation, no safety constraint analysis, limited to single-agent setting",
    },

    "General AI/ML": {
        "icon": "🤖",
        "description": "General machine learning, deep learning, any AI domain",
        "wiki_focus": """
Pay special attention to:
- Dataset limitations and potential biases
- Generalization beyond training distribution
- Computational efficiency and scalability
- Comparison with relevant baselines
- Ablation studies and their completeness
- Reproducibility (code, data, hyperparameters)
- Theoretical justification vs empirical results
- Failure modes and edge cases
""",
        "entity_types": [
            "model", "dataset", "task", "metric", "method",
            "application", "limitation", "baseline", "training_technique"
        ],
        "relationship_types": [
            "outperforms", "trained_on", "evaluated_on", "extends",
            "compared_with", "fails_on", "requires", "generalizes_to"
        ],
        "gap_priorities": [
            "Dataset diversity and bias",
            "Out-of-distribution generalization",
            "Computational efficiency",
            "Reproducibility",
            "Missing ablation studies",
            "Underexplored application domains",
        ],
        "proposal_framing": """
Frame proposals with:
- Clear problem formulation
- Experimental design with appropriate baselines
- Dataset and evaluation protocol
- Reproducibility plan
""",
        "search_boost_terms": [
            "machine learning", "deep learning", "neural network",
            "benchmark", "dataset", "generalization", "evaluation"
        ],
        "wiki_entity_examples": "models, datasets, tasks, metrics, methods, applications",
        "gap_examples": "Single dataset evaluation, no ablation study, missing comparison baselines",
    },
}


def get_domain_config(domain_name: str) -> dict:
    """Return config for a domain, falling back to General AI/ML."""
    return DOMAINS.get(domain_name, DOMAINS["General AI/ML"])


def get_domain_names() -> list:
    """Return list of available domain names."""
    return list(DOMAINS.keys())


def get_domain_display_options() -> list:
    """Return display strings with icons for selectbox."""
    return [f"{v['icon']} {k}" for k, v in DOMAINS.items()]


def parse_domain_selection(display_str: str) -> str:
    """Extract domain name from display string (strips icon)."""
    for name in DOMAINS:
        if name in display_str:
            return name
    return "General AI/ML"


def inject_domain_into_wiki_prompt(base_prompt: str, domain_config: dict) -> str:
    """Inject domain-specific focus into wiki compilation prompt."""
    focus = domain_config.get("wiki_focus", "")
    entity_examples = domain_config.get("wiki_entity_examples", "")
    return base_prompt + f"""

DOMAIN-SPECIFIC FOCUS:
{focus}

For 'key_concepts', prioritize: {entity_examples}
"""


def inject_domain_into_gap_prompt(base_prompt: str, domain_config: dict) -> str:
    """Inject domain-specific gap priorities into gap detection prompt."""
    priorities = domain_config.get("gap_priorities", [])
    priorities_str = "\n".join(f"  - {p}" for p in priorities)
    return base_prompt + f"""

DOMAIN GAP PRIORITIES — prioritize gaps in these categories:
{priorities_str}

Gap examples from this domain: {domain_config.get('gap_examples', '')}
"""


def inject_domain_into_proposal_prompt(base_prompt: str, domain_config: dict) -> str:
    """Inject domain-specific proposal framing."""
    framing = domain_config.get("proposal_framing", "")
    return base_prompt + f"""

DOMAIN PROPOSAL FRAMING:
{framing}
"""


def inject_domain_into_graph_prompt(base_prompt: str, domain_config: dict) -> str:
    """Inject domain-specific entity and relationship types for graph building."""
    entity_types = ", ".join(domain_config.get("entity_types", []))
    rel_types = ", ".join(domain_config.get("relationship_types", []))
    return base_prompt + f"""

DOMAIN-SPECIFIC ENTITY TYPES to prioritize: {entity_types}
DOMAIN-SPECIFIC RELATIONSHIP TYPES to use: {rel_types}
"""


def get_search_boost_terms(domain_config: dict) -> list:
    """Return domain-specific terms to boost academic search queries."""
    return domain_config.get("search_boost_terms", [])
