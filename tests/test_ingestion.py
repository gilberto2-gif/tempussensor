"""Tests for paper ingestion pipeline."""

import pytest
from unittest.mock import AsyncMock, patch

from src.ingestion.arxiv_client import ArxivClient, ArxivPaper
from src.ingestion.semantic_scholar import SemanticScholarClient
from src.ingestion.paper_parser import PaperParser


class TestArxivClient:
    def test_init(self):
        client = ArxivClient()
        assert client.min_year >= 2023

    def test_parse_response_empty(self):
        client = ArxivClient()
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        papers = client._parse_response(xml)
        assert papers == []

    def test_parse_response_with_entry(self):
        client = ArxivClient()
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <title>DTC Sensor Test Paper</title>
            <summary>Abstract about discrete time crystals.</summary>
            <published>2024-01-15T00:00:00Z</published>
            <updated>2024-01-15T00:00:00Z</updated>
            <author><name>Test Author</name></author>
            <category term="quant-ph"/>
          </entry>
        </feed>"""
        papers = client._parse_response(xml)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.12345v1"
        assert "DTC Sensor" in papers[0].title
        assert papers[0].published.year == 2024

    @pytest.mark.asyncio
    async def test_search_filters_old_papers(self):
        client = ArxivClient()
        client.min_year = 2024

        old_paper = ArxivPaper(
            arxiv_id="2022.00001",
            title="Old paper",
            authors="Author",
            abstract="Old abstract",
            published=__import__("datetime").datetime(2022, 1, 1),
            updated=__import__("datetime").datetime(2022, 1, 1),
            categories=["quant-ph"],
        )

        new_paper = ArxivPaper(
            arxiv_id="2024.00001",
            title="New paper",
            authors="Author",
            abstract="New abstract",
            published=__import__("datetime").datetime(2024, 6, 1),
            updated=__import__("datetime").datetime(2024, 6, 1),
            categories=["quant-ph"],
        )

        with patch.object(client, "_fetch_query", return_value=[old_paper, new_paper]):
            results = await client.search(query="test")

        assert len(results) == 1
        assert results[0].arxiv_id == "2024.00001"
        await client.close()


class TestPaperParser:
    def test_default_extraction(self):
        parser = PaperParser()
        default = parser._default_extraction()
        assert default["tipo"] == "TEORICO"
        assert default["confianza_fuente"] == 0.3
        assert "material" in default["parametros"]

    @pytest.mark.asyncio
    async def test_classify_relevance_returns_float(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="0.85")
        parser = PaperParser(llm_client=mock_llm)

        relevance = await parser.classify_relevance(
            "DTC magnetic sensor", "A paper about DTC sensing."
        )
        assert 0.0 <= relevance <= 1.0
