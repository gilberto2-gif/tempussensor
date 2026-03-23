"""API routes: /papers, /hypotheses, /simulations, /protocols, /status, /integrity, /agent."""

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.confidence import EvidenceType, is_low_confidence, propagate_confidence
from src.api.schemas import (
    AgentStatus, ClinicalRequirements, HypothesisResponse,
    IntegrityCheckResponse, PaperResponse, ProtocolResponse,
    SimulationCreate, SimulationResponse,
)
from src.db import get_session
from src.ml.counterfactual import parameter_sweep_1d, pareto_frontier, robustness_analysis
from src.ml.pinn_physics import compare_to_clinical, theoretical_sensitivity
from src.models import (
    AgentCycle, Hypothesis, IntegrityCheck, Paper, Prediction, Protocol, Simulation,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

@router.get("/papers")
async def list_papers(
    limit: int = Query(50, le=200),
    offset: int = 0,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
):
    q = (
        select(Paper)
        .where(Paper.confianza_fuente >= min_confidence)
        .order_by(Paper.fecha.desc())
        .offset(offset).limit(limit)
    )
    result = await session.execute(q)
    papers = result.scalars().all()
    return [
        {
            "id": p.id, "doi": p.doi, "arxiv_id": p.arxiv_id,
            "titulo": p.titulo, "autores": p.autores,
            "fecha": p.fecha.isoformat() if p.fecha else None,
            "tipo": p.tipo, "parametros": p.parametros_json,
            "calidad": {
                "reproducibilidad": p.reproducibilidad,
                "novedad": p.novedad,
                "relevancia_biosensado": p.relevancia_biosensado,
            },
            "confianza": p.confianza_fuente,
            "confianza_baja": p.confianza_fuente < 0.7,
        }
        for p in papers
    ]


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Paper).where(Paper.id == paper_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Paper not found")
    return {
        "id": p.id, "doi": p.doi, "arxiv_id": p.arxiv_id,
        "titulo": p.titulo, "autores": p.autores, "abstract": p.abstract,
        "fecha": p.fecha.isoformat() if p.fecha else None,
        "tipo": p.tipo, "parametros": p.parametros_json,
        "confianza": p.confianza_fuente,
        "confianza_baja": p.confianza_fuente < 0.7,
    }


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

@router.get("/hypotheses")
async def list_hypotheses(
    limit: int = Query(50, le=200),
    tipo: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    q = select(Hypothesis).order_by(Hypothesis.rank_score.desc()).limit(limit)
    if tipo:
        q = q.where(Hypothesis.tipo == tipo)
    result = await session.execute(q)
    hyps = result.scalars().all()
    return [
        {
            "id": h.id, "enunciado": h.enunciado, "accion": h.accion,
            "resultado_esperado": h.resultado_esperado, "mecanismo": h.mecanismo,
            "tipo": h.tipo, "status": h.status,
            "confianza": h.confianza, "confianza_baja": h.confianza < 0.7,
            "impacto": h.impacto, "costo_testeo": h.costo_testeo,
            "rank_score": h.rank_score, "evidence_class": h.evidence_class,
            "created_at": h.created_at.isoformat() if h.created_at else None,
        }
        for h in hyps
    ]


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

@router.get("/simulations")
async def list_simulations(
    limit: int = Query(50, le=200),
    pareto_only: bool = False,
    session: AsyncSession = Depends(get_session),
):
    q = select(Simulation).order_by(Simulation.created_at.desc()).limit(limit)
    if pareto_only:
        q = q.where(Simulation.pareto_optimal == True)  # noqa
    result = await session.execute(q)
    sims = result.scalars().all()
    return [
        {
            "id": s.id, "material_type": s.material_type,
            "temperatura_k": s.temperatura_k, "potencia_laser_w": s.potencia_laser_w,
            "sensibilidad_05_10hz": s.sensibilidad_05_10hz,
            "sensibilidad_10_50hz": s.sensibilidad_10_50hz,
            "sensibilidad_50_100hz": s.sensibilidad_50_100hz,
            "confianza": s.confianza, "confianza_baja": s.confianza < 0.7,
            "pareto_optimal": s.pareto_optimal, "evidence_class": s.evidence_class,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sims
    ]


@router.post("/simulations/run")
async def run_simulation(params: SimulationCreate):
    sens = theoretical_sensitivity(
        material_type=params.material_type.value,
        n_spins=params.n_qubits,
        temperature_k=params.temperatura_k,
        drive_power_w=params.potencia_laser_w,
        drive_freq_hz=10.0,
    )
    clinical = compare_to_clinical(sens)
    confidence = propagate_confidence([0.52], EvidenceType.RESULTADO_SIMULACION)
    return {
        "sensitivity": sens,
        "clinical_comparison": {k: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv) for kk, vv in v.items()} for k, v in clinical.items()},
        "confidence": confidence,
        "is_low_confidence": is_low_confidence(confidence),
    }


@router.post("/simulations/sweep")
async def run_sweep(
    material_type: str = "NV_DIAMOND",
    sweep_param: str = "temperatura_k",
    n_points: int = Query(50, le=200),
    min_val: float = 1.0,
    max_val: float = 400.0,
):
    import numpy as np
    sweep = parameter_sweep_1d(
        material_type=material_type, sweep_param=sweep_param,
        sweep_range=np.linspace(min_val, max_val, n_points),
        fixed_params={},
    )
    return sweep


@router.post("/simulations/pareto")
async def run_pareto(n_samples: int = Query(500, le=2000)):
    result = pareto_frontier(n_samples=n_samples)
    return {"n_pareto": result["n_pareto"], "pareto_optimal": result["pareto_optimal"][:20]}


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@router.get("/protocols")
async def list_protocols(limit: int = Query(20, le=100), session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Protocol).order_by(Protocol.created_at.desc()).limit(limit))
    protos = result.scalars().all()
    return [
        {
            "id": p.id, "titulo": p.titulo, "objetivo": p.objetivo,
            "material": p.material, "material_type": p.material_type,
            "sensor_config": p.sensor_config, "driving_config": p.driving_config,
            "detection_method": p.detection_method,
            "pasos": p.pasos, "metricas_exito": p.metricas_exito, "seguridad": p.seguridad,
            "sensibilidad_predicha_pt": p.sensibilidad_predicha_pt,
            "temperatura_k": p.temperatura_k, "costo_estimado_usd": p.costo_estimado_usd,
            "confianza": p.confianza, "confianza_baja": p.confianza < 0.7,
            "evidence_class": p.evidence_class,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in protos
    ]


# ---------------------------------------------------------------------------
# Agent status + trigger
# ---------------------------------------------------------------------------

@router.get("/status")
async def agent_status(session: AsyncSession = Depends(get_session)):
    latest = await session.execute(select(AgentCycle).order_by(AgentCycle.created_at.desc()).limit(1))
    cycle = latest.scalar_one_or_none()
    papers_count = (await session.execute(select(func.count(Paper.id)))).scalar() or 0
    hyp_count = (await session.execute(select(func.count(Hypothesis.id)))).scalar() or 0
    sim_count = (await session.execute(select(func.count(Simulation.id)))).scalar() or 0
    proto_count = (await session.execute(select(func.count(Protocol.id)))).scalar() or 0

    return {
        "cycle_number": cycle.cycle_number if cycle else 0,
        "phase": cycle.phase if cycle else "IDLE",
        "status": cycle.status if cycle else "IDLE",
        "papers_ingested": papers_count,
        "hypotheses_generated": hyp_count,
        "simulations_run": sim_count,
        "protocols_generated": proto_count,
        "last_cycle_at": cycle.created_at.isoformat() if cycle and cycle.created_at else None,
    }


@router.post("/agent/trigger")
async def trigger_agent(background_tasks: BackgroundTasks):
    from src.agent.core import run_agent_cycle
    background_tasks.add_task(run_agent_cycle)
    return {"status": "queued", "message": "Agent cycle triggered in background"}


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

@router.get("/integrity")
async def list_integrity(limit: int = Query(20, le=100), session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(IntegrityCheck).order_by(IntegrityCheck.created_at.desc()).limit(limit))
    checks = result.scalars().all()
    return [
        {
            "id": c.id, "fidelidad_dtc": c.fidelidad_dtc,
            "coherencia_estado": c.coherencia_estado,
            "correlacion_media": c.correlacion_media, "correlacion_estado": c.correlacion_estado,
            "divergencia_kl": c.divergencia_kl, "ruido_estado": c.ruido_estado,
            "fuentes_interferencia": c.fuentes_interferencia,
            "certificacion": c.certificacion, "hash_datos": c.hash_datos,
            "version_protocolo": c.version_protocolo,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in checks
    ]


@router.get("/clinical-requirements")
async def clinical_requirements():
    return ClinicalRequirements().model_dump()


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

@router.get("/knowledge-graph")
async def knowledge_graph(session: AsyncSession = Depends(get_session)):
    from src.models import KnowledgeNode, KnowledgeEdge
    nodes = (await session.execute(select(KnowledgeNode))).scalars().all()
    edges = (await session.execute(select(KnowledgeEdge))).scalars().all()
    return {
        "nodes": [{"id": n.id, "type": n.node_type, "name": n.name, "properties": n.properties} for n in nodes],
        "edges": [{"source": e.source_id, "target": e.target_id, "relation": e.relation} for e in edges],
    }
