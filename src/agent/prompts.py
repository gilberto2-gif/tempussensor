"""Prompt templates for each of the 9 agent phases.

Each prompt is designed for the Claude API and carries the scientific
context needed for that specific phase of the AgentOS v4 NEURAL loop.
"""

SYSTEM_PROMPT = """You are TempusSensor, an autonomous AI agent specialized in optimizing \
discrete time crystal (DTC) sensors for early biomagnetic anomaly detection.

You operate on this scientific foundation:
- Paper source: "Discrete time crystal acts as a usable sensor for weak magnetic oscillations" \
(confidence: 0.52) — FIRST practical use of time crystals for measuring weak magnetic fields.
- Crypto-quantum framework: Bennett/Brassard BB84 (confidence: 0.74) for data integrity.
- Problem: SQUIDs and optical pumping sensors need cryogenics or strict lab conditions. \
DTC sensors could operate at more accessible temperatures.
- Target patients: 50M epilepsy + 55M Alzheimer worldwide.

You MUST:
1. Propagate confidence levels (0-1) on EVERY assertion.
2. Mark anything with confidence < 0.7 as "LOW CONFIDENCE".
3. Classify evidence as: DATO_EXPERIMENTAL | RESULTADO_SIMULACION | INFERENCIA | EXTRAPOLACION | SIN_SOPORTE.
4. Reject SIN_SOPORTE assertions entirely.
5. Filter papers post-2023 ONLY.
6. Evaluate ethics in every significant decision: BENEFICENCIA, NO_MALEFICENCIA, JUSTICIA, TRANSPARENCIA."""


# Phase 1: PERCIBIR
PERCIBIR_PROMPT = """Analyze these newly discovered papers about discrete time crystals and magnetic sensing.

Papers:
{papers_json}

For each paper, extract:
1. Classification: EXPERIMENTAL | TEORICO | REVIEW | SIMULACION
2. Key parameters: material, temperature_K, sensitivity_pT/√Hz, frequency_Hz, SNR, coherence_T2, n_qubits, lattice_geometry, environmental_conditions
3. Reproducibility score (0-1): Are methods detailed enough to replicate?
4. Novelty score (0-1): How new is the approach?
5. Biosensing relevance (0-1): Direct relevance to biomagnetic detection?

Focus on papers relevant to 1-100 Hz range (brain alpha/beta waves).
Return JSON array with each paper's extracted data."""


# Phase 2: MODELAR
MODELAR_PROMPT = """Given the current knowledge graph state and new data, update the DTC sensor knowledge model.

Current graph summary:
{graph_summary}

New results:
{new_results}

Identify:
1. New material×technique combinations to add to the graph
2. Status updates for existing combinations: PROBADO → update metrics, PREDICHO → if validated
3. White spaces: promising unexplored combinations (material × technique × conditions)
4. Competing technologies (SQUID, OPM) — update their status

Return JSON with:
- new_nodes: [{type, name, properties}]
- new_relationships: [{from, to, type, properties}]
- white_spaces: [{material, technique, reason, estimated_priority}]
- status_updates: [{node, field, old_value, new_value}]"""


# Phase 3: PLANIFICAR
PLANIFICAR_PROMPT = """Generate testable hypotheses based on current knowledge and white spaces.

Knowledge state:
{knowledge_state}

White spaces identified:
{white_spaces}

Clinical targets:
- MEG: < 10 fT/√Hz, 1-100 Hz
- MCG: < 50 fT/√Hz, 0.1-50 Hz
- Biomarkers: < 1 pT/√Hz, 0.01-10 Hz
- Temperature: > 250K (near ambient)
- Cost: < $50,000 USD

Generate hypotheses with structure:
"Si [action] entonces [result] porque [mechanism]"

Classify each as:
- INCREMENTAL: small improvement on known approach
- COMBINATORIA: novel combination of known elements
- RADICAL: fundamentally new approach

Rank by: (confidence × impact) / testing_cost

Also identify "moonshot" hypotheses: low confidence but transformative impact.

Return JSON array of hypotheses."""


# Phase 4: SENTIR
SENTIR_PROMPT = """Calculate urgency metrics for the current research priorities.

Current best DTC sensitivity: {best_sensitivity_pT} pT/√Hz
Clinical target: {clinical_target_pT} pT/√Hz
Gap factor: {gap_factor}x

Global context:
- Epilepsy patients: 50M worldwide, many without access to MEG
- Alzheimer patients: 55M, early detection critical for intervention
- Current MEG installations: ~200 globally at ~$2M each
- Target: accessible sensor at < $50K

Failed hypotheses this cycle: {failed_hypotheses}
Successful results: {successes}

Calculate:
1. DALYs (Disability-Adjusted Life Years) potentially avoidable per year of delay
2. Access gap: current MEG coverage vs need
3. Cost reduction impact per breakthrough tier
4. Motivational reframing: how failed hypotheses reduce search space

Return JSON with urgency_score (0-1), dalys_avoidable, access_gap_analysis, motivation_context."""


# Phase 5: DECIDIR
DECIDIR_PROMPT = """Based on simulation results and Pareto analysis, decide the optimal DTC configuration.

Simulation results:
{simulation_results}

Pareto frontier:
{pareto_data}

Robustness analysis:
{robustness_data}

For the top 3 Pareto-optimal configurations:
1. Evaluate robustness to ±5% parameter variations
2. Identify bifurcation points (where small changes cause large sensitivity jumps)
3. Assess manufacturability and scalability
4. Compare to clinical requirements

Select the BEST configuration and justify the decision.

Return JSON with:
- selected_config: {{material, parameters, predicted_sensitivity, confidence}}
- alternatives: [top 3 with trade-off analysis]
- bifurcation_warnings: [parameters near critical points]
- clinical_gap_analysis: {{target, current, gap, timeline_estimate}}"""


# Phase 6: GOBERNAR
GOBERNAR_PROMPT = """Audit the traceability and integrity of all assertions made in this cycle.

Assertions to audit:
{assertions}

For each assertion, classify as:
- DATO_EXPERIMENTAL: directly from published experimental data
- RESULTADO_SIMULACION: from our PINN/GNN simulations
- INFERENCIA: logical deduction from multiple sources
- EXTRAPOLACION: extending beyond available data
- SIN_SOPORTE: no traceable evidence (REJECT THESE)

Calculate integrity score (0-1) for each.

Risk assessment in 4 dimensions:
1. EPISTEMICO: how certain are we of the underlying science?
2. RECURSOS: what resources are at stake?
3. REPUTACION: could this damage scientific credibility?
4. ETICO: could this lead to patient harm?

Return JSON with audit trail and risk scores."""


# Phase 7: EJECUTAR
EJECUTAR_PROMPT = """Generate a detailed experimental protocol based on the selected configuration.

Selected configuration:
{config}

Simulation predictions:
{predictions}

Clinical target: Detect {target_field_pT} pT fields at {target_freq_hz} Hz

Generate a complete protocol including:
1. Materials list with specifications and suppliers
2. Equipment setup (detailed)
3. Step-by-step procedure
4. Success metrics: SNR > 3 at target = SUCCESS, SNR > 3 at 4x target = PARTIAL
5. Safety considerations
6. Estimated timeline and cost
7. All hyperparameters and physical assumptions (TRANSPARENCY requirement)

Also assess: could this protocol generate false positives that harm patients?

Return JSON protocol with all fields."""


# Phase 8: RECORDAR
RECORDAR_PROMPT = """Summarize this agent cycle for memory storage.

Cycle number: {cycle_number}
Phase results:
{phase_results}

Generate 4 types of memory entries:

1. EPISODIC: What happened this cycle, what was the key decision, what lesson was learned?
2. SEMANTIC: What new knowledge should be added to the graph?
3. PROCEDURAL: Did we discover a better way to do something? (ingest papers, configure sims, etc.)
4. PROSPECTIVE: What should we do next? Triggers:
   - Temporal: "Check arXiv in 7 days for paper X response"
   - Event: "When paper Y is published, re-evaluate hypothesis Z"
   - Result: "If next simulation shows SNR > 5, escalate to protocol generation"

Return JSON with memory entries for each type."""


# Phase 9: REFLEXIONAR
REFLEXIONAR_PROMPT = """Self-evaluate the agent's performance and calibration.

Prediction history:
{predictions}

Resolved predictions:
{resolved}

Current cycle metrics:
{metrics}

Perform:
1. Accuracy analysis: what fraction of predictions were correct?
2. Calibration (ECE - Expected Calibration Error): are our confidence scores well-calibrated?
3. Bias detection:
   - Confirmation bias: are we only seeking evidence that supports our hypotheses?
   - Overconfidence: are our confidence scores systematically too high?
   - Material bias: are we unfairly favoring one material system?
4. Proactive initiatives:
   - Valuable white space to explore?
   - Prediction verifiable soon?
   - Model blind spot detected?

Classify each initiative as: EXPLORAR | EXPLOTAR | MANTENER | COMUNICAR

Ethics check (4 principles):
1. BENEFICENCIA: Are we advancing toward low-cost accessible MEG?
2. NO_MALEFICENCIA: Could any of our recommendations lead to false positives/negatives?
3. JUSTICIA: Is our design maintaining low costs or trending toward elitism?
4. TRANSPARENCIA: Is everything reproducible and auditable?

Return JSON with self-evaluation, biases detected, initiatives, and ethics assessment."""
