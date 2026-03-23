"""TempusSensor global configuration via environment variables."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str = ""

    # PostgreSQL (Render provides DATABASE_URL)
    database_url: str = "sqlite+aiosqlite:///tempussensor.db"
    database_url_sync: str = "sqlite:///tempussensor.db"

    # External APIs
    semantic_scholar_api_key: str = ""

    # Agent
    agent_cycle_interval_minutes: int = 60
    paper_search_min_year: int = 2023
    confidence_low_threshold: float = 0.7

    # Frontend URL (Firebase Hosting)
    frontend_url: str = "http://localhost:3000"

    # Clinical reference thresholds (hardcoded defaults)
    meg_sensitivity_ft: float = 10.0
    mcg_sensitivity_ft: float = 50.0
    biomarcador_sensitivity_pt: float = 1.0
    target_temp_k: float = 250.0
    target_volume_l: float = 1.0
    target_weight_kg: float = 5.0
    target_cost_usd: float = 50_000.0

    model_config = {"env_file": ".env", "extra": "ignore"}

    @staticmethod
    def fix_render_url(url: str) -> str:
        """Render gives postgres:// but SQLAlchemy needs postgresql://."""
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url


def _build_settings() -> Settings:
    """Build settings, handling Render's DATABASE_URL format."""
    s = Settings()

    # Render provides DATABASE_URL as postgres://...
    render_db = os.environ.get("DATABASE_URL", "")
    if render_db and "postgres" in render_db:
        sync_url = Settings.fix_render_url(render_db)
        async_url = sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        s.database_url = async_url
        s.database_url_sync = sync_url

    return s


settings = _build_settings()
