"""TempusSensor Agent Core — 9-phase neural loop via FastAPI BackgroundTasks.

Cloud-ready: no Celery/Redis dependency. Uses async background tasks.
"""

import asyncio
import json
from datetime import datetime

import structlog

from src.db import init_tables

logger = structlog.get_logger(__name__)

# Global state for the running agent
_agent_running = False
_last_cycle_result = None


async def run_agent_cycle() -> dict:
    """Execute one full 9-phase autonomous cycle."""
    global _agent_running, _last_cycle_result

    if _agent_running:
        return {"error": "Cycle already running"}

    _agent_running = True
    try:
        from src.agent.orchestrator import AgentOrchestrator

        await init_tables()
        orch = AgentOrchestrator()
        await orch.initialize()
        result = await orch.run_cycle()
        _last_cycle_result = result

        logger.info(
            "agent_cycle_complete",
            cycle=result.get("cycle"),
            duration=result.get("duration_seconds"),
        )
        return result
    except Exception as e:
        logger.error("agent_cycle_failed", error=str(e))
        return {"error": str(e)}
    finally:
        _agent_running = False


def get_agent_status() -> dict:
    return {
        "running": _agent_running,
        "last_result": _last_cycle_result,
    }
