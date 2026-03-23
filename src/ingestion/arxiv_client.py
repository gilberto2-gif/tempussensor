"""arXiv API client — searches for DTC + sensor + magnetic papers post-2023 ONLY."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# Search terms targeting discrete time crystal sensor papers
DEFAULT_QUERIES = [
    'all:"discrete time crystal" AND all:sensor',
    'all:"discrete time crystal" AND all:magnetic',
    'all:"time crystal" AND all:biosensor',
    'all:"time crystal" AND all:magnetometry',
    'all:DTC AND all:"magnetic field" AND all:sensing',
    'all:"time crystal" AND all:"NV center"',
    'all:"time crystal" AND all:"trapped ion" AND all:sensor',
]


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    published: datetime
    updated: datetime
    categories: list[str]
    doi: str | None = None
    journal_ref: str | None = None


class ArxivClient:
    """Async client for arXiv API with post-2023 filtering."""

    def __init__(self):
        self.min_year = settings.paper_search_min_year
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str | None = None,
        max_results: int = 50,
    ) -> list[ArxivPaper]:
        """Search arXiv. Filters out papers before min_year."""
        all_papers: list[ArxivPaper] = []

        queries = [query] if query else DEFAULT_QUERIES

        for q in queries:
            try:
                papers = await self._fetch_query(q, max_results=max_results)
                all_papers.extend(papers)
            except Exception as e:
                logger.error("arxiv_search_error", query=q, error=str(e))

        # Deduplicate by arxiv_id
        seen = set()
        unique = []
        for p in all_papers:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                unique.append(p)

        # Filter post-min_year ONLY
        filtered = [p for p in unique if p.published.year >= self.min_year]
        logger.info(
            "arxiv_search_complete",
            total_found=len(all_papers),
            unique=len(unique),
            post_filter=len(filtered),
        )
        return filtered

    async def _fetch_query(self, query: str, max_results: int) -> list[ArxivPaper]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = await self.client.get(ARXIV_API, params=params)
        resp.raise_for_status()
        return self._parse_response(resp.text)

    def _parse_response(self, xml_text: str) -> list[ArxivPaper]:
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            arxiv_id_url = entry.findtext("atom:id", "", ns)
            arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else arxiv_id_url

            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", "", ns)
                if name:
                    authors.append(name)

            published_str = entry.findtext("atom:published", "", ns)
            updated_str = entry.findtext("atom:updated", "", ns)
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))

            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term")
                if term:
                    categories.append(term)

            doi_el = entry.find("arxiv:doi", ns)
            doi = doi_el.text if doi_el is not None else None

            journal_el = entry.find("arxiv:journal_ref", ns)
            journal_ref = journal_el.text if journal_el is not None else None

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=", ".join(authors),
                    abstract=abstract,
                    published=published,
                    updated=updated,
                    categories=categories,
                    doi=doi,
                    journal_ref=journal_ref,
                )
            )

        return papers

    async def close(self):
        await self.client.aclose()
