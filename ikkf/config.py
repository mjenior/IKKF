"""
Configuration for the skills system.

Provides a dataclass-based configuration that controls skill loading
and semantic matching thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SkillsConfig"]


@dataclass
class SkillsConfig:
    """
    Configuration for the skills system.

    Attributes:
        custom_skills_dirs: List of paths to directories containing custom
            skill definitions. Skills from later directories override
            earlier ones with the same ID. Loaded after core skills.
        skill_base_threshold: Base threshold for semantic matching (0.0-1.0).
            Lower values match more broadly. Default is 0.68.
        skill_importance_low_weight: Weight multiplier for low importance skills.
            Threshold = base_threshold / weight. Default is 0.9.
        skill_importance_medium_weight: Weight multiplier for medium importance skills.
            Default is 1.0 (no adjustment).
        skill_importance_high_weight: Weight multiplier for high importance skills.
            Default is 1.1 (slightly lower threshold).
    """

    # Custom skills directories (loaded after core, can override)
    custom_skills_dirs: list[Path] = field(default_factory=list)

    # Semantic matching thresholds
    skill_base_threshold: float = 0.68
    skill_importance_low_weight: float = 0.9
    skill_importance_medium_weight: float = 1.0
    skill_importance_high_weight: float = 1.1

    # Model escalation settings (opt-in, disabled by default)
    enable_model_escalation: bool = True
    escalation_model: str = "anthropic/claude-haiku-4-5-20251001"
    escalation_threshold: float = 0.75  # Unified confidence threshold
    escalation_max_candidates: int = 10  # Max skills to send to model

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        # Convert string paths to Path objects
        normalized_dirs: list[Path] = []
        for dir_path in self.custom_skills_dirs:
            if isinstance(dir_path, str):
                normalized_dirs.append(Path(dir_path))
            else:
                normalized_dirs.append(dir_path)
        self.custom_skills_dirs = normalized_dirs

        # Validate threshold range
        if not 0.0 <= self.skill_base_threshold <= 1.0:
            raise ValueError(
                f"skill_base_threshold must be between 0.0 and 1.0, got {self.skill_base_threshold}"
            )

        # Validate weights are positive
        for weight_name in [
            "skill_importance_low_weight",
            "skill_importance_medium_weight",
            "skill_importance_high_weight",
        ]:
            weight_value = getattr(self, weight_name)
            if weight_value <= 0:
                raise ValueError(f"{weight_name} must be positive, got {weight_value}")

        # Validate escalation threshold
        if not 0.0 <= self.escalation_threshold <= 1.0:
            raise ValueError(
                f"escalation_threshold must be between 0.0 and 1.0, "
                f"got {self.escalation_threshold}"
            )

        # Validate escalation_max_candidates is positive
        if self.escalation_max_candidates <= 0:
            raise ValueError(
                f"escalation_max_candidates must be positive, "
                f"got {self.escalation_max_candidates}"
            )

    def parse_escalation_model(self) -> str | None:
        """
        Return the escalation model name for use with litellm.

        litellm uses "provider/model" format natively, so this method
        simply returns the model string as-is.

        Returns:
            The model name string if escalation is enabled, None otherwise.

        Examples:
            "anthropic/gemini/gemini-2.5-flash" -> "gemini/gemini-2.5-flash"
            "anthropic/claude-haiku-4-5-20251001" -> "anthropic/claude-haiku-4-5-20251001"
            "openai/gpt-4o-mini" -> "openai/gpt-4o-mini"
        """
        if not self.enable_model_escalation:
            return None

        return self.escalation_model
