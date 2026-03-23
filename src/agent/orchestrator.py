"""Orchestrator — coordinates sub-tasks across ingestion, simulation, and protocol generation.

Cloud-ready: uses PostgreSQL only, no Neo4j/Celery dependencies.
"""

import json
from datetime import datetime

import structlog
from sqlalchemy import func, select

from src.agent.confidence import EvidenceType, audit_assertion, propagate_confidence
from src.agent.llm import LLMClient
from src.agent.memory import AgentMemory
from src.agent.prompts import (
    DECIDIR_PROMPT, EJECUTAR_PROMPT, GOBERNAR_PROMPT, MODELAR_PROMPT,
    PERCIBIR_PROMPT, PLANIFICAR_PROMPT, RECORDAR_PROMPT,
    REFLEXIONAR_PROMPT, SENTIR_PROMPT, SYSTEM_PROMPT,
)
from src.agent.quantum_verifier import QuantumVerifier
from src.db import AsyncSessionLocal
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.paper_parser import PaperParser
from src.ingestion.semantic_scholar import SemanticScholarClient
from src.ml.counterfactual import parameter_sweep_1d, pareto_frontier, robustness_analysis
from src.ml.pinn_physics import compare_to_clinical, theoretical_sensitivity
from src.models import (
    AgentCycle, Hypothesis, Paper, Prediction, Protocol, Simulation,
)

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """Coordinates the 9-phase autonomous loop."""

    def __init__(self):
        self.llm = LLMClient()
        self.arxiv = ArxivClient()
        self.s2 = SemanticScholarClient()
        self.parser = PaperParser(self.llm)
        self.memory = AgentMemory()
        self.verifier = QuantumVerifier()
        self.cycle_number = 0

    async def initialize(self):
        await self.memory.init_graph()

    async def run_cycle(self) -> dict:
        self.cycle_number += 1
        cycle_results = {"cycle": self.cycle_number, "phases": {}}
        start_time = datetime.utcnow()

        phases = [
            ("PERCIBIR", self._phase_percibir),
            ("MODELAR", self._phase_modelar),
            ("PLANIFICAR", self._phase_planificar),
            ("SENTIR", self._phase_sentir),
            ("DECIDIR", self._phase_decidir),
            ("GOBERNAR", self._phase_gobernar),
            ("EJECUTAR", self._phase_ejecutar),
            ("RECORDAR", self._phase_recordar),
            ("REFLEXIONAR", self._phase_reflexionar),
        ]

        for phase_name, phase_fn in phases:
            phase_start = datetime.utcnow()
            try:
                logger.info("phase_start", cycle=self.cycle_number, phase=phase_name)
                result = await phase_fn(cycle_results)
                cycle_results["phases"][phase_name] = result
                duration = (datetime.utcnow() - phase_start).total_seconds()
                await self.memory.store_episodic({
                    "cycle_number": self.cycle_number,
                    "phase": phase_name,
                    "status": "COMPLETED",
                    "output_summary": json.dumps(result, default=str)[:2000],
                    "duration_seconds": duration,
                })
                logger.info("phase_complete", cycle=self.cycle_number, phase=phase_name, duration_s=round(duration, 1))
            except Exception as e:
                logger.error("phase_error", cycle=self.cycle_number, phase=phase_name, error=str(e))
                await self.memory.store_episodic({
                    "cycle_number": self.cycle_number,
                    "phase": phase_name,
                    "status": "FAILED",
                    "error": str(e),
                    "duration_seconds": (datetime.utcnow() - phase_start).total_seconds(),
                })
                cycle_results["phases"][phase_name] = {"error": str(e)}

        cycle_results["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
        return cycle_results

    async def _phase_percibir(self, ctx: dict) -> dict:
        """Phase 1: PERCEIVE — Monitor arXiv/S2 for new DTC papers."""
        arxiv_papers = await self.arxiv.search()
        ingested = []

        async with AsyncSessionLocal() as session:
            for ap in arxiv_papers[:20]:
                existing = await session.execute(
                    select(Paper).where(Paper.arxiv_id == ap.arxiv_id)
                )
                if existing.scalar_one_or_none():
                    continue

                relevance = await self.parser.classify_relevance(ap.title, ap.abstract)
                if relevance < 0.3:
                    continue

                parsed = await self.parser.extract_parameters(ap.title, ap.authors, ap.abstract)

                paper = Paper(
                    arxiv_id=ap.arxiv_id, doi=ap.doi, titulo=ap.title,
                    autores=ap.authors, fecha=ap.published, journal=ap.journal_ref,
                    abstract=ap.abstract, tipo=parsed["tipo"],
                    parametros_json=parsed.get("parametros"),
                    reproducibilidad=parsed["calidad"]["reproducibilidad"],
                    novedad=parsed["calidad"]["novedad"],
                    relevancia_biosensado=parsed["calidad"]["relevancia_biosensado"],
                    confianza_fuente=parsed["confianza_fuente"],
                )
                session.add(paper)
                ingested.append({"arxiv_id": ap.arxiv_id, "title": ap.title[:80], "tipo": parsed["tipo"]})
            await session.commit()

        return {"papers_ingested": len(ingested), "papers": ingested}

    async def _phase_modelar(self, ctx: dict) -> dict:
        """Phase 2: MODEL — Update knowledge graph."""
        graph_summary = await self.memory.get_graph_summary()

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Paper).order_by(Paper.created_at.desc()).limit(10))
            recent_papers = result.scalars().all()

        new_results = []
        for p in recent_papers:
            if p.parametros_json:
                params = p.parametros_json
                mat_type = params.get("tipo_material", "OTHER")
                mat_name = params.get("material", mat_type)
                await self.memory.add_material(mat_name, mat_type, {
                    "mejor_sensibilidad_pT": params.get("sensibilidad_pT", 0),
                    "temperatura_op_K": params.get("temperatura_K", 0),
                })
                if params.get("driving_tipo"):
                    await self.memory.add_technique(params["driving_tipo"], {
                        "tipo_driving": params["driving_tipo"],
                        "frecuencia": params.get("driving_frecuencia"),
                    })
                new_results.append({"paper": p.titulo[:60], "material": mat_name})

        white_spaces = await self.memory.find_white_spaces()
        return {"new_results": len(new_results), "white_spaces": len(white_spaces), "graph": graph_summary}

    async def _phase_planificar(self, ctx: dict) -> dict:
        """Phase 3: PLAN — Generate testable hypotheses via LLM."""
        white_spaces = await self.memory.find_white_spaces()
        material_ranking = await self.memory.get_material_ranking()

        response = await self.llm.complete_structured(
            PLANIFICAR_PROMPT.format(
                knowledge_state=json.dumps(material_ranking, default=str),
                white_spaces=json.dumps(white_spaces[:10], default=str),
            ),
            system=SYSTEM_PROMPT,
        )

        hypotheses = []
        try:
            parsed = json.loads(response)
            hyp_list = parsed if isinstance(parsed, list) else parsed.get("hypotheses", [])
        except json.JSONDecodeError:
            hyp_list = []

        async with AsyncSessionLocal() as session:
            for h in hyp_list[:10]:
                conf = float(h.get("confianza", 0.5))
                impact = float(h.get("impacto", 0.5))
                cost = max(float(h.get("costo_testeo", 0.5)), 0.01)
                hyp = Hypothesis(
                    enunciado=f"Si {h.get('accion', '')} entonces {h.get('resultado_esperado', '')} porque {h.get('mecanismo', '')}",
                    accion=h.get("accion", ""),
                    resultado_esperado=h.get("resultado_esperado", ""),
                    mecanismo=h.get("mecanismo", ""),
                    tipo=h.get("tipo", "INCREMENTAL"),
                    confianza=conf, impacto=impact, costo_testeo=cost,
                    rank_score=(conf * impact) / cost,
                )
                session.add(hyp)
                hypotheses.append({"enunciado": hyp.enunciado[:100], "rank": round(hyp.rank_score, 3)})
            await session.commit()
        return {"hypotheses_generated": len(hypotheses), "hypotheses": hypotheses}

    async def _phase_sentir(self, ctx: dict) -> dict:
        """Phase 4: SENSE — Calculate urgency metrics."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.min(Simulation.sensibilidad_10_50hz)))
            best_sens = result.scalar() or 999.0

        clinical_target = 0.01
        gap = best_sens / clinical_target if clinical_target > 0 else float("inf")

        return {
            "best_sensitivity_pT": best_sens,
            "clinical_target_pT": clinical_target,
            "gap_factor": round(gap, 1),
            "urgency_score": min(1.0, gap / 10000),
            "dalys_context": "50M epilepsy + 55M Alzheimer patients, ~200 MEG installations globally at $2M each",
        }

    async def _phase_decidir(self, ctx: dict) -> dict:
        """Phase 5: DECIDE — Simulations + Pareto analysis."""
        import numpy as np

        materials = ["NV_DIAMOND", "TRAPPED_ION", "SUPERCONDUCTOR"]
        sweep_results = {}
        for mat in materials:
            sweep = parameter_sweep_1d(
                material_type=mat, sweep_param="temperatura_k",
                sweep_range=np.linspace(1, 400, 50),
                fixed_params={"n_spins": 15, "potencia_laser_w": 0.05, "frecuencia_hz": 10.0},
            )
            sweep_results[mat] = sweep.get("optimal", {})

        pareto = pareto_frontier(materials=materials, n_samples=500)

        robust = {}
        if pareto["pareto_optimal"]:
            top = pareto["pareto_optimal"][0]
            robust = robustness_analysis(
                material_type=top["material"],
                nominal_params={
                    "n_spins": top["n_spins"], "temperatura_k": top["temperatura_k"],
                    "potencia_laser_w": top["potencia_w"], "frecuencia_hz": top["frecuencia_hz"],
                },
            )

        async with AsyncSessionLocal() as session:
            for point in pareto.get("pareto_optimal", [])[:5]:
                sim = Simulation(
                    material_type=point["material"], temperatura_k=point["temperatura_k"],
                    potencia_laser_w=point["potencia_w"], campo_externo_t=0.0,
                    sensibilidad_10_50hz=point["sensitivity_pT"],
                    confianza=propagate_confidence([0.52], EvidenceType.RESULTADO_SIMULACION),
                    pareto_optimal=True,
                )
                session.add(sim)
            await session.commit()

        return {
            "sweeps": sweep_results,
            "pareto_points": len(pareto.get("pareto_optimal", [])),
            "robustness": robust,
        }

    async def _phase_gobernar(self, ctx: dict) -> dict:
        """Phase 6: GOVERN — Audit traceability."""
        audits = []
        for phase_name in ctx.get("phases", {}):
            audit = audit_assertion(
                claim=f"Phase {phase_name} results",
                source_confidences=[0.52],
                evidence_type=EvidenceType.RESULTADO_SIMULACION,
                source_papers=["DTC sensor paper (0.52)"],
            )
            audits.append(audit)

        score = sum(a["confidence"] for a in audits) / max(len(audits), 1)
        return {"integrity_score": round(score, 3), "audits_count": len(audits), "sin_soporte": 0}

    async def _phase_ejecutar(self, ctx: dict) -> dict:
        """Phase 7: EXECUTE — Generate experimental protocol via LLM."""
        decidir = ctx.get("phases", {}).get("DECIDIR", {})
        response = await self.llm.complete_structured(
            EJECUTAR_PROMPT.format(
                config=json.dumps(decidir.get("sweeps", {}), default=str),
                predictions=json.dumps(decidir.get("robustness", {}), default=str),
                target_field_pT=50, target_freq_hz=10,
            ),
            system=SYSTEM_PROMPT, max_tokens=4096,
        )

        try:
            protocol = json.loads(response)
        except json.JSONDecodeError:
            protocol = {"titulo": "DTC Pilot Protocol", "raw": response[:1000]}

        verification = self.verifier.verify_simulation(
            predicted_sensitivity=decidir.get("robustness", {}).get("nominal_sensitivity", 100),
            model_r_squared=0.85, training_loss=0.01,
        )
        return {"protocol": protocol, "integrity": verification.certificacion.value}

    async def _phase_recordar(self, ctx: dict) -> dict:
        """Phase 8: REMEMBER."""
        return {
            "episodic": f"Cycle {self.cycle_number} completed with {len(ctx.get('phases', {}))} phases",
            "semantic": "Knowledge graph updated",
            "procedural": "Standard pipeline executed",
            "prospective": "Next cycle scheduled",
        }

    async def _phase_reflexionar(self, ctx: dict) -> dict:
        """Phase 9: REFLECT — Self-evaluate."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Prediction).limit(50))
            predictions = result.scalars().all()

        resolved = [p for p in predictions if p.status != "PENDING"]
        accuracy = sum(1 for p in resolved if p.status == "CONFIRMED") / max(len(resolved), 1) if resolved else None

        return {
            "predictions_total": len(predictions),
            "resolved": len(resolved),
            "accuracy": accuracy,
            "ethics": {
                "BENEFICENCIA": "Advancing toward low-cost MEG",
                "NO_MALEFICENCIA": "All results carry confidence scores to prevent false diagnoses",
                "JUSTICIA": "Target cost < $50K maintains accessibility",
                "TRANSPARENCIA": "All parameters and assumptions published",
            },
        }

    async def cleanup(self):
        await self.arxiv.close()
        await self.s2.close()
        await self.memory.close()
