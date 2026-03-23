"""Paper parameter extraction via Claude API.

Uses LLM to extract structured sensor parameters from paper abstracts/text,
classify paper type, and evaluate quality metrics.
"""

import json

import structlog

from src.agent.llm import LLMClient
from src.api.schemas import MaterialType, PaperType

logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """You are a quantum sensor physics expert. Extract structured parameters from this scientific paper about discrete time crystals (DTC) used as sensors.

Paper title: {title}
Authors: {authors}
Abstract: {abstract}

Extract ALL of the following parameters. Use null if not mentioned or not applicable.

Return ONLY valid JSON with this exact structure:
{{
  "tipo": "EXPERIMENTAL|TEORICO|REVIEW|SIMULACION",
  "parametros": {{
    "material": "description of the material system",
    "tipo_material": "NV_DIAMOND|TRAPPED_ION|SUPERCONDUCTOR|OTHER",
    "temperatura_K": null or float,
    "sensibilidad_pT": null or float (in pT/√Hz),
    "frecuencia_Hz": null or float (operating/target frequency),
    "SNR": null or float,
    "T2_us": null or float (coherence time in microseconds),
    "n_qubits": null or int,
    "geometria_red": null or string,
    "driving_tipo": null or "LASER|MICROWAVE|RF",
    "driving_frecuencia": null or float (Hz),
    "driving_potencia": null or float (W),
    "condiciones_ambientales": null or string
  }},
  "calidad": {{
    "reproducibilidad": float 0-1 (how reproducible are the results),
    "novedad": float 0-1 (how novel is the approach),
    "relevancia_biosensado": float 0-1 (relevance to biomagnetic sensing)
  }},
  "confianza_fuente": float 0-1 (overall confidence in the paper's claims)
}}

Be conservative with confidence scores. Mark experimental results from reputable labs higher.
For DTC sensor papers specifically, evaluate:
- Does it demonstrate actual magnetic field sensing? (high relevance)
- Is it a theoretical proposal? (medium relevance)
- Is it DTC physics without sensing application? (low relevance)
"""


class PaperParser:
    """Extracts structured parameters from papers using Claude API."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def extract_parameters(
        self, title: str, authors: str, abstract: str
    ) -> dict:
        """Extract structured parameters from a paper using Claude."""
        prompt = EXTRACTION_PROMPT.format(
            title=title,
            authors=authors,
            abstract=abstract,
        )

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system="You are a quantum physics parameter extraction system. Return ONLY valid JSON.",
                max_tokens=1024,
                temperature=0.1,
            )

            # Parse JSON from response
            text = response.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            parsed = json.loads(text)

            # Validate tipo
            tipo_raw = parsed.get("tipo", "TEORICO")
            if tipo_raw not in [t.value for t in PaperType]:
                tipo_raw = "TEORICO"
            parsed["tipo"] = tipo_raw

            # Validate tipo_material
            params = parsed.get("parametros", {})
            mat_type = params.get("tipo_material", "OTHER")
            if mat_type not in [m.value for m in MaterialType]:
                params["tipo_material"] = "OTHER"
            parsed["parametros"] = params

            # Clamp confidence values
            calidad = parsed.get("calidad", {})
            for key in ["reproducibilidad", "novedad", "relevancia_biosensado"]:
                val = calidad.get(key, 0.0)
                calidad[key] = max(0.0, min(1.0, float(val) if val else 0.0))
            parsed["calidad"] = calidad

            conf = parsed.get("confianza_fuente", 0.5)
            parsed["confianza_fuente"] = max(0.0, min(1.0, float(conf)))

            logger.info(
                "paper_parsed",
                title=title[:60],
                tipo=parsed["tipo"],
                confianza=parsed["confianza_fuente"],
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.error("paper_parse_json_error", title=title[:60], error=str(e))
            return self._default_extraction()
        except Exception as e:
            logger.error("paper_parse_error", title=title[:60], error=str(e))
            return self._default_extraction()

    def _default_extraction(self) -> dict:
        """Return a safe default when extraction fails."""
        return {
            "tipo": "TEORICO",
            "parametros": {
                "material": None,
                "tipo_material": "OTHER",
                "temperatura_K": None,
                "sensibilidad_pT": None,
                "frecuencia_Hz": None,
                "SNR": None,
                "T2_us": None,
                "n_qubits": None,
                "geometria_red": None,
                "driving_tipo": None,
                "driving_frecuencia": None,
                "driving_potencia": None,
                "condiciones_ambientales": None,
            },
            "calidad": {
                "reproducibilidad": 0.0,
                "novedad": 0.0,
                "relevancia_biosensado": 0.0,
            },
            "confianza_fuente": 0.3,
        }

    async def classify_relevance(self, title: str, abstract: str) -> float:
        """Quick relevance check before full extraction (saves API costs)."""
        prompt = f"""Rate 0.0 to 1.0 how relevant this paper is to using discrete time crystals as biomagnetic sensors.

Title: {title}
Abstract: {abstract[:500]}

Return ONLY a single float number between 0.0 and 1.0."""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system="Return only a float number.",
                max_tokens=10,
                temperature=0.0,
            )
            return max(0.0, min(1.0, float(response.strip())))
        except (ValueError, Exception):
            return 0.5
