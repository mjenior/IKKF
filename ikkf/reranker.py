"""
Model-based skill reranking for escalated selections.

Provides LLM-based skill selection when embedding confidence is insufficient.
Uses litellm for provider-agnostic model access.
"""

import logging
from dataclasses import dataclass

import litellm

from ikkf.models import Skill

__all__ = ["ModelReranker", "RerankerResult", "create_reranker"]

logger = logging.getLogger(__name__)


@dataclass
class RerankerResult:
    """Result from model reranking operation."""

    skill_ids: list[str]  # Skill IDs ordered by relevance
    success: bool
    error: str | None = None


class ModelReranker:
    """
    Reranks skill candidates using an LLM.

    Formats skill descriptions for the model and parses its selection.
    Designed for fast, focused reranking with minimal token usage.

    Uses litellm for provider-agnostic model access. Model names use
    the litellm format: "provider/model" (e.g., "anthropic/claude-haiku-4-5-20251001").
    """

    PROMPT = '''Select relevant skills for this user message. Return skill IDs only.

Message: {message}

Skills:
{skills}

Return one skill ID per line, most relevant first. Return NONE if no skills apply.'''

    def __init__(self, model_name: str):
        """
        Initialize the reranker.

        Args:
            model_name: Model name in litellm format (e.g., "anthropic/claude-haiku-4-5-20251001")
        """
        self._model_name = model_name
        self._available = True
        self._init_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the reranker is ready to use."""
        return self._available

    def rerank(
        self,
        candidates: list[Skill],
        message: str,
    ) -> RerankerResult:
        """
        Rerank candidate skills for the given message.

        Args:
            candidates: Skills to choose from (should have descriptions)
            message: User message to match against

        Returns:
            RerankerResult with ordered skill IDs
        """
        if not self.available:
            return RerankerResult(
                skill_ids=[],
                success=False,
                error=self._init_error or "Reranker not initialized",
            )

        if not candidates:
            return RerankerResult(skill_ids=[], success=True)

        # Format skills for prompt
        skill_lines = []
        valid_ids = set()
        for s in candidates:
            desc = s.description or s.name
            skill_lines.append(f"- {s.id}: {desc}")
            valid_ids.add(s.id)

        prompt = self.PROMPT.format(
            message=message,
            skills="\n".join(skill_lines),
        )

        try:
            response = self._invoke(prompt)
            skill_ids = self._parse_response(response, valid_ids)
            logger.debug(f"Reranker selected {len(skill_ids)} skills: {skill_ids}")
            return RerankerResult(skill_ids=skill_ids, success=True)
        except Exception as e:
            logger.error(f"Reranker invocation failed: {e}")
            return RerankerResult(
                skill_ids=[],
                success=False,
                error=str(e),
            )

    def _invoke(self, prompt: str) -> str:
        """Invoke the model and return response text."""
        response = litellm.completion(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model response was empty")
        return str(content)

    def _parse_response(self, response: str, valid_ids: set[str]) -> list[str]:
        """
        Extract valid skill IDs from model response.

        Args:
            response: Raw model response text
            valid_ids: Set of valid skill IDs from candidates

        Returns:
            List of skill IDs in order of relevance
        """
        result = []
        for line in response.strip().split("\n"):
            # Clean up the line (remove bullet points, whitespace)
            cleaned = line.strip().lstrip("-•*").strip()

            # Handle "NONE" response
            if cleaned.upper() == "NONE":
                return []

            # Only include valid IDs, avoid duplicates
            if cleaned in valid_ids and cleaned not in result:
                result.append(cleaned)

        return result


def create_reranker(model_name: str) -> ModelReranker:
    """
    Factory function to create a reranker instance.

    Args:
        model_name: Model name in litellm format (e.g., "anthropic/claude-haiku-4-5-20251001")

    Returns:
        Configured ModelReranker instance
    """
    return ModelReranker(model_name)
