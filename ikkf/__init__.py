"""
Skills system for dynamic prompt composition.

ikkfs your LLM agent mid-campaign by providing modular, context-aware loading
of specialized guidance ("skills") based on conversation triggers. All matching
skills are included in the system prompt to dynamically reshape your agent's
capabilities on the fly.

Basic Usage:
    from ikkf import Skills

    # Create with defaults (includes core skills)
    skills = Skills()

    # Select relevant skills for a message
    selected = skills.select("What tools do you have?")

    # Access skill content
    for skill in selected:
        print(skill.content)

Custom Configuration:
    from pathlib import Path
    from ikkf import Skills, SkillsConfig

    config = SkillsConfig(
        custom_skills_dirs=[Path("./my_skills")],
    )
    skills = Skills(config)
"""

from pathlib import Path

from ikkf.config import SkillsConfig
from ikkf.models import (
    CATEGORY_DEFAULT_IMPORTANCE,
    SelectionResult,
    Skill,
    SkillCategory,
    SkillMatch,
    SkillTriggerType,
    TriggerConfig,
)
from ikkf.registry import SkillRegistry, get_core_skills_path
from ikkf.selector import SkillSelector


class Skills:
    """
    Main entry point for the skills system.

    Provides a unified interface for loading and selecting skills based on
    conversation context. Supports core skills bundled with the package and
    custom agent-specific skills from user-defined directories.

    Attributes:
        config: The configuration used to initialize this instance
        registry: The underlying skill registry (for advanced use)
        selector: The underlying skill selector (for advanced use)
    """

    def __init__(self, config: SkillsConfig | None = None):
        """
        Initialize the skills system.

        Args:
            config: Configuration for skills loading and selection.
                If None, uses default configuration (includes core skills,
                no semantic matching).
        """
        self.config = config or SkillsConfig()
        self._registry = self._build_registry()
        self._selector = self._build_selector()

    def _build_registry(self) -> SkillRegistry:
        """Build the skill registry based on configuration."""
        skills_dirs: list[Path] = []

        # Add custom directories (can override core skills)
        skills_dirs.extend(self.config.custom_skills_dirs)

        return SkillRegistry(skills_dirs=skills_dirs)

    def _build_selector(self) -> SkillSelector:
        """Build the skill selector with the current configuration."""
        return SkillSelector(
            registry=self._registry,
            config=self.config,
        )

    @property
    def registry(self) -> SkillRegistry:
        """Access the underlying skill registry."""
        return self._registry

    @property
    def selector(self) -> SkillSelector:
        """Access the underlying skill selector."""
        return self._selector

    def select(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> list[Skill]:
        """
        Select relevant skills based on current context.

        Args:
            user_message: The latest user message to analyze
            conversation_history: Full conversation for context (optional).
                Each entry should be a dict with at least a "content" key.
            pending_tool_calls: Tool names that might be called (optional).
                Used for tool_usage triggers.
            turn_count: Number of user turns so far. Used for turn_based triggers.

        Returns:
            List of selected skills, ordered by relevance (highest first).
            Skills include their content, metadata, and match information.
        """
        return self._selector.select_skills(
            user_message=user_message,
            conversation_history=conversation_history,
            pending_tool_calls=pending_tool_calls,
            turn_count=turn_count,
        )

    def select_with_metadata(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> SelectionResult:
        """
        Select skills with escalation metadata.

        Returns a SelectionResult that can be iterated like a list,
        but also contains information about whether model escalation
        was used.

        Args:
            user_message: The latest user message to analyze
            conversation_history: Full conversation for context (optional)
            pending_tool_calls: Tool names that might be called (optional)
            turn_count: Number of user turns so far

        Returns:
            SelectionResult with skills and escalation metadata

        Example:
            result = skills.select_with_metadata("query")

            # Iterate like a list
            for skill in result:
                print(skill.name)

            # Access metadata
            if result.escalated:
                print(f"Escalated: {result.escalation_reason}")
        """
        return self._selector.select_skills_with_metadata(
            user_message=user_message,
            conversation_history=conversation_history,
            pending_tool_calls=pending_tool_calls,
            turn_count=turn_count,
        )

    def get_skill(self, skill_id: str) -> Skill | None:
        """
        Get a specific skill by its ID.

        Args:
            skill_id: The unique identifier of the skill

        Returns:
            The skill if found, None otherwise
        """
        return self._registry.get_skill(skill_id)

    def list_skills(self) -> list[Skill]:
        """
        List all registered skills.

        Returns:
            List of all skills loaded from core and custom directories
        """
        return self._registry.get_all_skills()

    def list_skills_by_category(self, category: SkillCategory) -> list[Skill]:
        """
        List all skills in a specific category.

        Args:
            category: The category to filter by

        Returns:
            List of skills in the specified category
        """
        return self._registry.get_skills_by_category(category)

    def add_skills_directory(self, path: Path) -> int:
        """
        Add a custom skills directory at runtime.

        Skills from the new directory can override existing skills with the
        same ID. This is useful for adding agent-specific skills after
        initialization.

        Automatically rebuilds selector indexes to include the new skills
        in semantic matching.

        Args:
            path: Path to directory containing skill files (.yaml and .md pairs)

        Returns:
            Number of skills loaded from the directory
        """
        count = self._registry.add_skills_directory(path)
        if count > 0:
            self._selector = self._build_selector()
        return count

    def reload(self) -> None:
        """
        Reload all skills from disk.

        This is useful when skill files have been modified externally.
        Thread-safe: other operations will see either the old or new
        complete state, never a partial state.
        """
        self._registry.reload()
        self._selector = self._build_selector()


__all__ = [
    # Main entry point
    "Skills",
    # Configuration
    "SkillsConfig",
    # Models
    "Skill",
    "SkillCategory",
    "SkillMatch",
    "SkillTriggerType",
    "TriggerConfig",
    "SelectionResult",
    "CATEGORY_DEFAULT_IMPORTANCE",
    # Registry and Selector (for advanced use)
    "SkillRegistry",
    "SkillSelector",
    "get_core_skills_path",
]
