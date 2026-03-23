"""Streamlit dashboard for TempusSensor.

Displays: knowledge graph, hypotheses, simulations, protocols,
integrity verification, and confidence metrics.
"""

import json

import httpx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://api:8000/api/v1"

st.set_page_config(
    page_title="TempusSensor",
    page_icon="🔬",
    layout="wide",
)


def api_get(endpoint: str, params: dict | None = None):
    """Helper to call the FastAPI backend."""
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, params: dict | None = None, json_body: dict | None = None):
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", params=params, json=json_body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ===========================================================================
# Sidebar
# ===========================================================================

st.sidebar.title("TempusSensor")
st.sidebar.markdown("**DTC Sensor Optimization Agent**")

page = st.sidebar.radio(
    "Navigation",
    ["Status", "Papers", "Hypotheses", "Simulations", "Protocols", "Integrity", "Explorer"],
)

# Clinical requirements reference
with st.sidebar.expander("Clinical Targets"):
    st.markdown("""
    | Target | Sensitivity | Freq |
    |--------|-----------|------|
    | MEG | <10 fT/√Hz | 1-100 Hz |
    | MCG | <50 fT/√Hz | 0.1-50 Hz |
    | Biomarkers | <1 pT/√Hz | 0.01-10 Hz |

    **Constraints:** >250K, <1L, <5kg, <$50K
    """)

# Confidence legend
with st.sidebar.expander("Confidence Legend"):
    st.markdown("""
    - 🟢 **High** (≥0.7): Reliable
    - 🟡 **Low** (0.5-0.7): Use with caution
    - 🔴 **Very Low** (<0.5): Unreliable
    """)

# Trigger agent
if st.sidebar.button("Trigger Agent Cycle"):
    result = api_post("/agent/trigger")
    if result:
        st.sidebar.success(f"Cycle queued: {result.get('task_id', 'N/A')}")


# ===========================================================================
# Pages
# ===========================================================================


def confidence_badge(conf: float) -> str:
    """Return colored badge for confidence level."""
    if conf >= 0.7:
        return f"🟢 {conf:.2f}"
    elif conf >= 0.5:
        return f"🟡 {conf:.2f}"
    return f"🔴 {conf:.2f}"


# ---------------------------------------------------------------------------
# Status page
# ---------------------------------------------------------------------------
if page == "Status":
    st.title("Agent Status")

    status = api_get("/status")
    if status:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cycle", status["cycle_number"])
        col2.metric("Papers", status["papers_ingested"])
        col3.metric("Hypotheses", status["hypotheses_generated"])
        col4.metric("Simulations", status["simulations_run"])

        st.markdown(f"**Phase:** {status['phase']} | **Status:** {status['status']}")

        if status.get("prediction_accuracy") is not None:
            st.metric("Prediction Accuracy", f"{status['prediction_accuracy']:.1%}")

        if status.get("last_cycle_at"):
            st.caption(f"Last cycle: {status['last_cycle_at']}")

    # Clinical requirements
    st.subheader("Clinical Requirements Reference")
    reqs = api_get("/clinical-requirements")
    if reqs:
        cols = st.columns(3)
        cols[0].metric("MEG Target", f"{reqs['meg_sensitivity_ft']} fT/√Hz")
        cols[1].metric("MCG Target", f"{reqs['mcg_sensitivity_ft']} fT/√Hz")
        cols[2].metric("Biomarker Target", f"{reqs['biomarcador_sensitivity_pt']} pT/√Hz")


# ---------------------------------------------------------------------------
# Papers page
# ---------------------------------------------------------------------------
elif page == "Papers":
    st.title("Ingested Papers")

    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    papers = api_get("/papers", {"min_confidence": min_conf, "limit": 100})

    if papers:
        for p in papers:
            conf = p.get("confianza_fuente", p.get("confianza", 0))
            badge = confidence_badge(conf)
            with st.expander(f"{badge} {p['titulo'][:100]}"):
                st.markdown(f"**Type:** {p['tipo']} | **Date:** {p['fecha'][:10]}")
                st.markdown(f"**Authors:** {p['autores'][:200]}")
                if p.get("doi"):
                    st.markdown(f"**DOI:** {p['doi']}")
                if p.get("arxiv_id"):
                    st.markdown(f"**arXiv:** {p['arxiv_id']}")
                if p.get("parametros"):
                    st.json(p["parametros"])
                if p.get("calidad"):
                    q = p["calidad"]
                    st.progress(q["relevancia_biosensado"], text=f"Biosensing relevance: {q['relevancia_biosensado']:.2f}")
    else:
        st.info("No papers ingested yet. Trigger an agent cycle.")


# ---------------------------------------------------------------------------
# Hypotheses page
# ---------------------------------------------------------------------------
elif page == "Hypotheses":
    st.title("Generated Hypotheses")

    tipo_filter = st.selectbox("Type filter", ["All", "INCREMENTAL", "COMBINATORIA", "RADICAL"])
    params = {"limit": 50}
    if tipo_filter != "All":
        params["tipo"] = tipo_filter

    hyps = api_get("/hypotheses", params)
    if hyps:
        for h in hyps:
            conf = h.get("confianza", 0)
            badge = confidence_badge(conf)
            with st.expander(f"{badge} [{h['tipo']}] {h['enunciado'][:120]}"):
                st.markdown(f"**Action:** {h['accion']}")
                st.markdown(f"**Expected result:** {h['resultado_esperado']}")
                st.markdown(f"**Mechanism:** {h['mecanismo']}")
                cols = st.columns(4)
                cols[0].metric("Confidence", f"{conf:.2f}")
                cols[1].metric("Impact", f"{h['impacto']:.2f}")
                cols[2].metric("Test Cost", f"{h['costo_testeo']:.2f}")
                cols[3].metric("Rank", f"{h['rank_score']:.3f}")
                st.markdown(f"**Evidence:** {h['evidence_class']} | **Status:** {h['status']}")
    else:
        st.info("No hypotheses generated yet.")


# ---------------------------------------------------------------------------
# Simulations page
# ---------------------------------------------------------------------------
elif page == "Simulations":
    st.title("Simulations & Analysis")

    tab1, tab2, tab3 = st.tabs(["Run Simulation", "Parameter Sweep", "Pareto Frontier"])

    with tab1:
        st.subheader("Run Theoretical Simulation")
        col1, col2 = st.columns(2)
        with col1:
            material = st.selectbox("Material", ["NV_DIAMOND", "TRAPPED_ION", "SUPERCONDUCTOR"])
            n_qubits = st.slider("N spins", 5, 100, 15)
            temp_k = st.slider("Temperature (K)", 0.01, 400.0, 300.0)
        with col2:
            power_w = st.slider("Laser power (W)", 0.001, 1.0, 0.05)
            field_t = st.number_input("External field (T)", value=0.0, format="%.2e")

        if st.button("Run Simulation"):
            result = api_post("/simulations/run", json_body={
                "material_type": material,
                "temperatura_k": temp_k,
                "potencia_laser_w": power_w,
                "campo_externo_t": field_t,
                "n_qubits": n_qubits,
            })
            if result:
                st.markdown(f"**Confidence:** {confidence_badge(result['confidence'])}")
                if result.get("is_low_confidence"):
                    st.warning("LOW CONFIDENCE — results should be verified experimentally")

                sens = result["sensitivity"]
                cols = st.columns(3)
                cols[0].metric("0.5-10 Hz", f"{sens['0.5-10Hz']:.1f} pT/√Hz")
                cols[1].metric("10-50 Hz", f"{sens['10-50Hz']:.1f} pT/√Hz")
                cols[2].metric("50-100 Hz", f"{sens['50-100Hz']:.1f} pT/√Hz")

                st.json(result["clinical_comparison"])

    with tab2:
        st.subheader("1D Parameter Sweep")
        sweep_mat = st.selectbox("Material", ["NV_DIAMOND", "TRAPPED_ION", "SUPERCONDUCTOR"], key="sweep_mat")
        sweep_param = st.selectbox("Sweep parameter", ["temperatura_k", "potencia_laser_w", "n_spins", "frecuencia_hz"])
        col1, col2 = st.columns(2)
        min_v = col1.number_input("Min value", value=1.0)
        max_v = col2.number_input("Max value", value=400.0)

        if st.button("Run Sweep"):
            result = api_post("/simulations/sweep", params={
                "material_type": sweep_mat,
                "sweep_param": sweep_param,
                "min_val": min_v,
                "max_val": max_v,
                "n_points": 80,
            })
            if result:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result["param_values"],
                    y=result["sensitivity_10_50"],
                    name="10-50 Hz",
                    mode="lines",
                ))
                fig.add_trace(go.Scatter(
                    x=result["param_values"],
                    y=result["sensitivity_05_10"],
                    name="0.5-10 Hz",
                    mode="lines",
                    line=dict(dash="dash"),
                ))
                fig.update_layout(
                    title=f"Sensitivity vs {sweep_param} ({sweep_mat})",
                    xaxis_title=sweep_param,
                    yaxis_title="Sensitivity (pT/√Hz)",
                    yaxis_type="log",
                )
                st.plotly_chart(fig, use_container_width=True)

                if result.get("optimal"):
                    st.success(f"Optimal: {sweep_param}={result['optimal']['param_value']:.2f} → {result['optimal']['best_sensitivity_10_50']:.2f} pT/√Hz")

                if result.get("bifurcation_points"):
                    st.warning(f"Bifurcation points detected at: {[b['param_value'] for b in result['bifurcation_points']]}")

    with tab3:
        st.subheader("Pareto Frontier")
        n_samples = st.slider("Sample size", 100, 2000, 500)
        if st.button("Compute Pareto"):
            result = api_post("/simulations/pareto", params={"n_samples": n_samples})
            if result and result.get("pareto_optimal"):
                pts = result["pareto_optimal"]
                fig = px.scatter(
                    pts,
                    x="sensitivity_pT",
                    y="cost_usd",
                    color="material",
                    size="n_spins",
                    hover_data=["temperatura_k", "frecuencia_hz"],
                    log_x=True,
                    log_y=True,
                    title="Pareto Frontier: Sensitivity vs Cost",
                )
                fig.update_layout(
                    xaxis_title="Sensitivity (pT/√Hz)",
                    yaxis_title="Estimated Cost (USD)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Pareto-optimal configurations", result["n_pareto"])


# ---------------------------------------------------------------------------
# Protocols page
# ---------------------------------------------------------------------------
elif page == "Protocols":
    st.title("Experimental Protocols")

    protocols = api_get("/protocols", {"limit": 20})
    if protocols:
        for p in protocols:
            badge = confidence_badge(p.get("confianza", 0))
            with st.expander(f"{badge} {p['titulo']}"):
                st.markdown(f"**Objective:** {p['objetivo']}")
                st.markdown(f"**Material:** {p['material']} ({p['material_type']})")
                st.markdown(f"**Sensor config:** {p['sensor_config']}")
                st.markdown(f"**Driving:** {p['driving_config']}")
                st.markdown(f"**Detection:** {p['detection_method']}")

                if p.get("sensibilidad_predicha_pt"):
                    st.metric("Predicted sensitivity", f"{p['sensibilidad_predicha_pt']:.1f} pT/√Hz")
                if p.get("costo_estimado_usd"):
                    st.metric("Estimated cost", f"${p['costo_estimado_usd']:,.0f}")

                st.subheader("Steps")
                for i, step in enumerate(p.get("pasos", []), 1):
                    st.markdown(f"{i}. {step}")

                st.subheader("Success Metrics")
                st.json(p.get("metricas_exito", {}))

                if p.get("seguridad"):
                    st.subheader("Safety")
                    for s in p["seguridad"]:
                        st.warning(s)
    else:
        st.info("No protocols generated yet. Run agent cycles to generate experimental protocols.")


# ---------------------------------------------------------------------------
# Integrity page
# ---------------------------------------------------------------------------
elif page == "Integrity":
    st.title("Data Integrity Verification")

    checks = api_get("/integrity", {"limit": 20})
    if checks:
        for c in checks:
            cert = c["certificacion"]
            icon = {"INTEGRO": "✅", "CON_ADVERTENCIA": "⚠️", "NO_CONFIABLE": "❌"}.get(cert, "❓")
            with st.expander(f"{icon} {cert} — {c['created_at'][:19]}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("DTC Fidelity", f"{c['fidelidad_dtc']:.3f}", delta=c['coherencia_estado'])
                col2.metric("Correlation", f"{c['correlacion_media']:.3f}", delta=c['correlacion_estado'])
                col3.metric("KL Divergence", f"{c['divergencia_kl']:.3f}", delta=c['ruido_estado'])

                if c.get("fuentes_interferencia"):
                    st.warning(f"Interference sources: {', '.join(c['fuentes_interferencia'])}")

                st.caption(f"Hash: {c['hash_datos'][:32]}... | Protocol v{c['version_protocolo']}")
    else:
        st.info("No integrity checks performed yet.")


# ---------------------------------------------------------------------------
# Explorer page
# ---------------------------------------------------------------------------
elif page == "Explorer":
    st.title("DTC Sensor Explorer")

    st.subheader("Quick Sensitivity Calculator")
    col1, col2, col3 = st.columns(3)

    with col1:
        mat = st.selectbox("Material system", ["NV_DIAMOND", "TRAPPED_ION", "SUPERCONDUCTOR"], key="exp_mat")
    with col2:
        n_sp = st.number_input("Number of spins", 1, 1000, 15)
    with col3:
        temp = st.number_input("Temperature (K)", 0.01, 500.0, 300.0)

    col4, col5 = st.columns(2)
    with col4:
        power = st.number_input("Drive power (W)", 0.001, 10.0, 0.05)
    with col5:
        freq = st.number_input("Drive frequency (Hz)", 0.1, 1000.0, 10.0)

    if st.button("Calculate"):
        from src.ml.pinn_physics import compare_to_clinical, theoretical_sensitivity

        sens = theoretical_sensitivity(mat, n_sp, temp, power, freq)
        clinical = compare_to_clinical(sens)

        col1, col2, col3 = st.columns(3)
        col1.metric("0.5-10 Hz", f"{sens['0.5-10Hz']:.2f} pT/√Hz")
        col2.metric("10-50 Hz", f"{sens['10-50Hz']:.2f} pT/√Hz")
        col3.metric("50-100 Hz", f"{sens['50-100Hz']:.2f} pT/√Hz")

        st.markdown(f"**Order parameter:** {sens['estimated_order_param']:.4f}")
        st.markdown(f"**T2 effective:** {sens['T2_eff_us']:.2f} μs")

        st.subheader("Clinical Gap Analysis")
        for target, data in clinical.items():
            met = "✅" if data["meets_target"] else f"❌ ({data['gap_factor']:.0f}x gap)"
            st.markdown(f"**{target}:** {met} — Current: {data['current_pT']:.2f} pT, Target: {data['target_pT']:.4f} pT")
