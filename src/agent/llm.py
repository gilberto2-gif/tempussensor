"""Claude API client for TempusSensor agent.

Uses the anthropic SDK for all LLM interactions.
"""

import anthropic
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """Async Claude API client."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a completion request to Claude API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system

            response = await self.client.messages.create(**kwargs)
            text = response.content[0].text
            logger.debug("llm_complete", tokens_in=response.usage.input_tokens, tokens_out=response.usage.output_tokens)
            return text
        except anthropic.APIError as e:
            logger.error("llm_api_error", error=str(e))
            raise

    async def complete_structured(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Completion optimized for structured JSON output."""
        return await self.complete(
            prompt=prompt,
            system=system + "\n\nReturn ONLY valid JSON. No markdown, no explanation.",
            max_tokens=max_tokens,
            temperature=0.1,
        )

    async def analyze(
        self,
        context: str,
        question: str,
        system: str = "You are a quantum physics research assistant.",
        max_tokens: int = 2048,
    ) -> str:
        """Send an analysis request with context."""
        prompt = f"""Context:\n{context}\n\nQuestion:\n{question}"""
        return await self.complete(prompt=prompt, system=system, max_tokens=max_tokens)
