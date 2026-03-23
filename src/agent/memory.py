"""Agent memory system — 4 types stored in PostgreSQL.

Replaces Neo4j with KnowledgeNode/KnowledgeEdge tables for cloud deployment.

Memory types:
1. Episodic: each cycle with full context and lesson learned
2. Semantic: knowledge graph via KnowledgeNode/KnowledgeEdge
3. Procedural: how-to knowledge
4. Prospective: agenda with temporal/event/result triggers
"""

import structlog
from sqlalchemy import select, and_

from src.db import AsyncSessionLocal
from src.models import AgentCycle, KnowledgeNode, KnowledgeEdge

logger = structlog.get_logger(__name__)


class AgentMemory:
    """Multi-modal memory using PostgreSQL knowledge graph tables."""

    # -----------------------------------------------------------------------
    # Knowledge graph operations (replaces Neo4j)
    # -----------------------------------------------------------------------

    async def add_material(self, name: str, tipo: str, props: dict) -> int:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(KnowledgeNode).where(
                    and_(KnowledgeNode.node_type == "MATERIAL", KnowledgeNode.name == name)
                )
            )
            node = existing.scalar_one_or_none()
            if node:
                node.properties = {**(node.properties or {}), **props, "tipo": tipo}
            else:
                node = KnowledgeNode(node_type="MATERIAL", name=name, properties={**props, "tipo": tipo})
                session.add(node)
            await session.commit()
            await session.refresh(node)
            return node.id

    async def add_technique(self, name: str, props: dict) -> int:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(KnowledgeNode).where(
                    and_(KnowledgeNode.node_type == "TECHNIQUE", KnowledgeNode.name == name)
                )
            )
            node = existing.scalar_one_or_none()
            if node:
                node.properties = {**(node.properties or {}), **props}
            else:
                node = KnowledgeNode(node_type="TECHNIQUE", name=name, properties=props)
                session.add(node)
            await session.commit()
            await session.refresh(node)
            return node.id

    async def add_application(self, name: str, props: dict) -> int:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(KnowledgeNode).where(
                    and_(KnowledgeNode.node_type == "APPLICATION", KnowledgeNode.name == name)
                )
            )
            node = existing.scalar_one_or_none()
            if node:
                node.properties = {**(node.properties or {}), **props}
            else:
                node = KnowledgeNode(node_type="APPLICATION", name=name, properties=props)
                session.add(node)
            await session.commit()
            await session.refresh(node)
            return node.id

    async def add_edge(self, source_id: int, target_id: int, relation: str, props: dict | None = None):
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(KnowledgeEdge).where(and_(
                    KnowledgeEdge.source_id == source_id,
                    KnowledgeEdge.target_id == target_id,
                    KnowledgeEdge.relation == relation,
                ))
            )
            edge = existing.scalar_one_or_none()
            if not edge:
                edge = KnowledgeEdge(source_id=source_id, target_id=target_id, relation=relation, properties=props)
                session.add(edge)
                await session.commit()

    async def find_white_spaces(self) -> list[dict]:
        """Find material×technique combinations with no LOGRA edge."""
        async with AsyncSessionLocal() as session:
            materials = (await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.node_type == "MATERIAL")
            )).scalars().all()
            techniques = (await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.node_type == "TECHNIQUE")
            )).scalars().all()
            edges = (await session.execute(
                select(KnowledgeEdge).where(KnowledgeEdge.relation == "LOGRA")
            )).scalars().all()

            connected = {(e.source_id, e.target_id) for e in edges}
            white = []
            for m in materials:
                for t in techniques:
                    if (m.id, t.id) not in connected and (t.id, m.id) not in connected:
                        white.append({
                            "material": m.name,
                            "tipo": (m.properties or {}).get("tipo", ""),
                            "technique": t.name,
                            "driving": (t.properties or {}).get("tipo_driving", ""),
                        })
            return white[:50]

    async def get_material_ranking(self) -> list[dict]:
        async with AsyncSessionLocal() as session:
            mats = (await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.node_type == "MATERIAL")
            )).scalars().all()
            return [
                {
                    "material": m.name,
                    "tipo": (m.properties or {}).get("tipo", ""),
                    "best_sens": (m.properties or {}).get("mejor_sensibilidad_pT"),
                    "temp_k": (m.properties or {}).get("temperatura_op_K"),
                }
                for m in mats
            ]

    async def get_graph_summary(self) -> dict:
        async with AsyncSessionLocal() as session:
            nodes = (await session.execute(select(KnowledgeNode))).scalars().all()
            edges = (await session.execute(select(KnowledgeEdge))).scalars().all()

            node_counts = {}
            for n in nodes:
                node_counts[n.node_type] = node_counts.get(n.node_type, 0) + 1
            edge_counts = {}
            for e in edges:
                edge_counts[e.relation] = edge_counts.get(e.relation, 0) + 1

            return {"nodes": node_counts, "relationships": edge_counts}

    # -----------------------------------------------------------------------
    # Episodic memory
    # -----------------------------------------------------------------------

    async def store_episodic(self, cycle_data: dict) -> None:
        async with AsyncSessionLocal() as session:
            session.add(AgentCycle(**cycle_data))
            await session.commit()

    # -----------------------------------------------------------------------
    # Prospective memory
    # -----------------------------------------------------------------------

    async def add_prospective(self, trigger_type: str, trigger_value: str, action: str, priority: int = 5):
        async with AsyncSessionLocal() as session:
            node = KnowledgeNode(
                node_type="PROSPECTIVE",
                name=f"{trigger_type}:{trigger_value}",
                properties={
                    "trigger_type": trigger_type,
                    "trigger_value": trigger_value,
                    "action": action,
                    "priority": priority,
                    "status": "PENDING",
                },
            )
            session.add(node)
            await session.commit()

    async def get_pending_prospective(self) -> list[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(KnowledgeNode).where(
                    and_(
                        KnowledgeNode.node_type == "PROSPECTIVE",
                    )
                )
            )
            nodes = result.scalars().all()
            return [
                n.properties for n in nodes
                if (n.properties or {}).get("status") == "PENDING"
            ]

    async def init_graph(self):
        """Seed clinical applications."""
        for app_name, props in [
            ("MEG", {"sensitivity_fT": 10, "freq_lo": 1, "freq_hi": 100, "description": "Magnetoencephalography"}),
            ("MCG", {"sensitivity_fT": 50, "freq_lo": 0.1, "freq_hi": 50, "description": "Magnetocardiography"}),
            ("BIOMARCADORES", {"sensitivity_pT": 1, "freq_lo": 0.01, "freq_hi": 10, "description": "Magnetic biomarkers"}),
        ]:
            await self.add_application(app_name, props)
        logger.info("knowledge_graph_initialized")

    async def close(self):
        pass  # No external connections to close
