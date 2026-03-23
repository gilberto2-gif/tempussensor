"""Semantic Scholar API client for enriching paper metadata and finding citations."""

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"

# Fields we request from S2
PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,venue,"
    "citationCount,referenceCount,influentialCitationCount,"
    "fieldsOfStudy,publicationDate"
)


class SemanticScholarClient:
    """Async client for Semantic Scholar API."""

    def __init__(self):
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        self.client = httpx.AsyncClient(
            base_url=S2_API,
            headers=headers,
            timeout=30.0,
        )
        self.min_year = settings.paper_search_min_year

    async def search_papers(
        self,
        query: str = "discrete time crystal sensor magnetic",
        limit: int = 100,
    ) -> list[dict]:
        """Search for papers, filtering post-min_year."""
        all_results = []
        offset = 0
        batch_size = min(limit, 100)

        while len(all_results) < limit:
            try:
                resp = await self.client.get(
                    "/paper/search",
                    params={
                        "query": query,
                        "offset": offset,
                        "limit": batch_size,
                        "fields": PAPER_FIELDS,
                        "year": f"{self.min_year}-",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                papers = data.get("data", [])
                if not papers:
                    break
                all_results.extend(papers)
                offset += len(papers)
                if data.get("next") is None:
                    break
            except httpx.HTTPStatusError as e:
                logger.error("s2_search_error", status=e.response.status_code, error=str(e))
                break
            except Exception as e:
                logger.error("s2_search_error", error=str(e))
                break

        logger.info("s2_search_complete", total=len(all_results))
        return all_results

    async def get_paper(self, paper_id: str) -> dict | None:
        """Get a single paper by S2 ID, DOI, or ArXiv ID."""
        try:
            resp = await self.client.get(
                f"/paper/{paper_id}",
                params={"fields": PAPER_FIELDS},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("s2_paper_not_found", paper_id=paper_id, status=e.response.status_code)
            return None

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers that cite the given paper."""
        try:
            resp = await self.client.get(
                f"/paper/{paper_id}/citations",
                params={"fields": PAPER_FIELDS, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return [c["citingPaper"] for c in data.get("data", []) if c.get("citingPaper")]
        except Exception as e:
            logger.error("s2_citations_error", paper_id=paper_id, error=str(e))
            return []

    async def get_references(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers referenced by the given paper."""
        try:
            resp = await self.client.get(
                f"/paper/{paper_id}/references",
                params={"fields": PAPER_FIELDS, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return [r["citedPaper"] for r in data.get("data", []) if r.get("citedPaper")]
        except Exception as e:
            logger.error("s2_references_error", paper_id=paper_id, error=str(e))
            return []

    async def multi_query_search(self, limit_per_query: int = 50) -> list[dict]:
        """Run multiple targeted queries for DTC sensor papers."""
        queries = [
            "discrete time crystal sensor",
            "discrete time crystal magnetic field",
            "time crystal magnetometry",
            "time crystal biosensor",
            "DTC NV center magnetometry",
            "time crystal trapped ion sensing",
            "Floquet time crystal sensor",
        ]
        all_papers = []
        seen_ids = set()

        for q in queries:
            papers = await self.search_papers(query=q, limit=limit_per_query)
            for p in papers:
                pid = p.get("paperId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)

        logger.info("s2_multi_search_complete", unique_papers=len(all_papers))
        return all_papers

    async def close(self):
        await self.client.aclose()
