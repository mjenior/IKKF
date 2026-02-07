"""
Skill system models for dynamic prompt composition.

Skills are modular chunks of specialized guidance that get loaded dynamically
based on conversation context. Use these to build your agent's ikkf loadout.
"""

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "SkillCategory",
    "SkillTriggerType",
    "TriggerConfig",
    "Skill",
    "SkillMatch",
    "SelectionResult",
    "CATEGORY_DEFAULT_IMPORTANCE",
]


class SkillCategory(Enum):
    """Logical categories for grouping skills."""

    TECHNICAL = "technical"  # Technical depth and architecture skills
    CONVERSATION_MANAGEMENT = "conversation_management"  # Flow and structure skills
    KNOWLEDGE = "knowledge"  # Knowledge boundaries and evidence skills
    TOOL_GUIDANCE = "tool_guidance"  # Tool usage and selection skills
    CUSTOM = "custom"  # Default category for uncategorized skills


# Default importance values by category (higher = more important for selection)
# Hierarchy balances tool usage, sales context, and conversational quality:
#   - TOOL_GUIDANCE (85): Critical for correct tool usage but not dominant
#   - SALES (80): High priority for sales-focused agent conversations
#   - TECHNICAL (70): Important for technical depth when needed
#   - KNOWLEDGE (65): Evidence and boundary awareness
#   - CONVERSATION_MANAGEMENT (55): Foundational for good interactions
#   - CUSTOM (30): Catch-all for uncategorized skills
CATEGORY_DEFAULT_IMPORTANCE: dict[SkillCategory, int] = {
    SkillCategory.TOOL_GUIDANCE: 85,
    SkillCategory.TECHNICAL: 70,
    SkillCategory.KNOWLEDGE: 65,
    SkillCategory.CONVERSATION_MANAGEMENT: 55,
    SkillCategory.CUSTOM: 30,
}


class SkillTriggerType(Enum):
    """Types of triggers that can activate a skill."""

    KEYWORD = "keyword"  # Simple term matching in user message
    SEMANTIC = "semantic"  # Embedding similarity to reference phrases
    TOOL_USAGE = "tool_usage"  # Triggered by specific tool patterns in message
    EXPLICIT = "explicit"  # Always included in context

    # Semantic importance levels (for SEMANTIC trigger_type):
    #
    # Specify "importance" in trigger_config using one of: low, medium, high, critical.
    # Threshold is calculated as: base_threshold / weight (from config.py).
    # Higher importance = lower threshold = more likely to activate.
    #
    #   LOW:      Strict matching - only activates on strong signals (~0.75 threshold).
    #   MEDIUM:   Default. Good precision/recall balance (~0.68 threshold).
    #   HIGH:     Broad matching - activates more easily (~0.62 threshold).
    #   CRITICAL: Always included (bypasses threshold calculation entirely).
    #
    # If importance is omitted or invalid, defaults to MEDIUM.
    TURN_BASED = "turn_based"  # Based on conversation turn count


@dataclass
class TriggerConfig:
    """Configuration for a single trigger."""

    type: SkillTriggerType
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        # Convert string type to enum if needed
        if isinstance(self.type, str):
            self.type = SkillTriggerType(self.type)


@dataclass
class Skill:
    """
    A modular chunk of specialized guidance.

    Attributes:
        id: Unique identifier (e.g., "instrument_documentation")
        name: Human-readable name for logging/debugging
        content: The skill prompt content (markdown)
        trigger_type: Primary trigger type for activation
        trigger_config: Type-specific configuration for the primary trigger
        additional_triggers: Optional list of secondary triggers. Each skill can
            have multiple ways to be activated. The highest-scoring trigger wins.
        priority: Higher values = loaded first when multiple skills match (0-100)
        dependencies: Other skill IDs that should load with this one
        incompatible_with: Skill IDs that cannot be loaded alongside this one
        category: Logical category for grouping (defaults to CUSTOM)
        subcategory: Optional free-form subcategory for finer grouping
        importance: Importance level for selection (0-100). Higher values
            are selected first when multiple skills match.
            If not set, defaults based on category (TOOL_GUIDANCE=90, TECHNICAL=70, etc.)
    """

    id: str
    name: str
    content: str
    trigger_type: SkillTriggerType
    trigger_config: dict = field(default_factory=dict)
    additional_triggers: list[TriggerConfig] = field(default_factory=list)
    priority: int = 50
    dependencies: list[str] = field(default_factory=list)
    incompatible_with: list[str] = field(default_factory=list)
    category: SkillCategory = SkillCategory.CUSTOM
    subcategory: str | None = None
    importance: int | None = None  # None means use category default
    description: str | None = None  # For model-based selection reasoning

    def __post_init__(self):
        # Convert string trigger_type to enum if needed
        if isinstance(self.trigger_type, str):
            self.trigger_type = SkillTriggerType(self.trigger_type)

        # Convert string category to enum if needed
        if isinstance(self.category, str):
            try:
                self.category = SkillCategory(self.category)
            except ValueError:
                # Fall back to CUSTOM for unknown categories
                self.category = SkillCategory.CUSTOM

        # Set default importance based on category if not explicitly set
        if self.importance is None:
            self.importance = CATEGORY_DEFAULT_IMPORTANCE.get(self.category, 30)

        # Convert dict additional_triggers to TriggerConfig objects
        normalized_triggers = []
        for trigger in self.additional_triggers:
            if isinstance(trigger, dict):
                trigger_type = trigger.get("type", "explicit")
                trigger_config = trigger.get("config", {})
                normalized_triggers.append(TriggerConfig(type=SkillTriggerType(trigger_type), config=trigger_config))
            elif isinstance(trigger, TriggerConfig):
                normalized_triggers.append(trigger)
        self.additional_triggers = normalized_triggers

    def get_all_triggers(self) -> list[TriggerConfig]:
        """Return all triggers (primary + additional) as TriggerConfig objects."""
        primary = TriggerConfig(type=self.trigger_type, config=self.trigger_config)
        return [primary] + self.additional_triggers


@dataclass
class SkillMatch:
    """Result of matching a skill to context."""

    skill: Skill
    score: float  # 0.0 to 1.0
    trigger_reason: str  # Human-readable explanation

    def __lt__(self, other: "SkillMatch") -> bool:
        """Compare by score, then priority (ascending order).

        Use sorted(matches, reverse=True) to get highest scores first.
        """
        if self.score != other.score:
            return self.score < other.score
        return self.skill.priority < other.skill.priority


@dataclass
class SelectionResult:
    """
    Result of skill selection with optional escalation metadata.

    This dataclass can be iterated like a list for convenience,
    making it backward compatible with existing code patterns.

    Attributes:
        skills: List of selected skills, ordered by relevance
        escalated: Whether model-based escalation was used
        escalation_reason: Why escalation occurred (if applicable)

    Example:
        result = selector.select_skills_with_metadata("query")

        # Iterate like a list
        for skill in result:
            print(skill.name)

        # Access metadata
        if result.escalated:
            print(f"Escalated: {result.escalation_reason}")
    """
    skills: list[Skill]
    escalated: bool = False
    escalation_reason: str | None = None

    def __iter__(self):
        """Allow iteration over skills for convenience."""
        return iter(self.skills)

    def __len__(self):
        """Return number of selected skills."""
        return len(self.skills)

    def __getitem__(self, index):
        """Allow indexing into skills."""
        return self.skills[index]
