"""Tests for skill models."""


from ikkf import (
    CATEGORY_DEFAULT_IMPORTANCE,
    Skill,
    SkillCategory,
    SkillMatch,
    SkillTriggerType,
    TriggerConfig,
)


class TestSkillCategory:
    """Test SkillCategory enum."""

    def test_all_categories_exist(self) -> None:
        """All expected categories should exist."""
        assert SkillCategory.TECHNICAL is not None
        assert SkillCategory.CONVERSATION_MANAGEMENT is not None
        assert SkillCategory.KNOWLEDGE is not None
        assert SkillCategory.TOOL_GUIDANCE is not None
        assert SkillCategory.CUSTOM is not None

    def test_category_values(self) -> None:
        """Category values should be lowercase strings."""
        assert SkillCategory.CUSTOM.value == "custom"


class TestSkillTriggerType:
    """Test SkillTriggerType enum."""

    def test_all_trigger_types_exist(self) -> None:
        """All expected trigger types should exist."""
        assert SkillTriggerType.KEYWORD is not None
        assert SkillTriggerType.SEMANTIC is not None
        assert SkillTriggerType.TOOL_USAGE is not None
        assert SkillTriggerType.EXPLICIT is not None
        assert SkillTriggerType.TURN_BASED is not None

    def test_trigger_type_values(self) -> None:
        """Trigger type values should match expected strings."""
        assert SkillTriggerType.KEYWORD.value == "keyword"
        assert SkillTriggerType.SEMANTIC.value == "semantic"
        assert SkillTriggerType.EXPLICIT.value == "explicit"


class TestTriggerConfig:
    """Test TriggerConfig dataclass."""

    def test_creation_with_enum(self) -> None:
        """TriggerConfig can be created with enum type."""
        config = TriggerConfig(type=SkillTriggerType.KEYWORD, config={"keywords": ["test"]})
        assert config.type == SkillTriggerType.KEYWORD
        assert config.config == {"keywords": ["test"]}

    def test_creation_with_string(self) -> None:
        """TriggerConfig should convert string type to enum."""
        config = TriggerConfig(type="keyword", config={"keywords": ["test"]})
        assert config.type == SkillTriggerType.KEYWORD

    def test_default_config(self) -> None:
        """TriggerConfig should have empty dict as default config."""
        config = TriggerConfig(type=SkillTriggerType.EXPLICIT)
        assert config.config == {}


class TestSkill:
    """Test Skill dataclass."""

    def test_minimal_skill_creation(self) -> None:
        """Skill can be created with minimal required fields."""
        skill = Skill(
            id="test_skill",
            name="Test Skill",
            content="Test content",
            trigger_type=SkillTriggerType.EXPLICIT,
        )
        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"
        assert skill.content == "Test content"

    def test_default_values(self) -> None:
        """Skill should have sensible defaults."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
        )
        assert skill.priority == 50
        assert skill.dependencies == []
        assert skill.incompatible_with == []
        assert skill.category == SkillCategory.CUSTOM
        assert skill.subcategory is None
        assert skill.additional_triggers == []

    def test_string_trigger_type_converted(self) -> None:
        """String trigger_type should be converted to enum."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type="keyword",
        )
        assert skill.trigger_type == SkillTriggerType.KEYWORD

    def test_string_category_converted(self) -> None:
        """String category should be converted to enum."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            category="technical",
        )
        assert skill.category == SkillCategory.TECHNICAL

    def test_invalid_category_falls_back_to_custom(self) -> None:
        """Invalid category string should fall back to CUSTOM."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            category="invalid_category",
        )
        assert skill.category == SkillCategory.CUSTOM

    def test_importance_defaults_from_category(self) -> None:
        """Importance should default based on category."""
        skill_tool = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            category=SkillCategory.TOOL_GUIDANCE,
        )
        assert skill_tool.importance == CATEGORY_DEFAULT_IMPORTANCE[SkillCategory.TOOL_GUIDANCE]

        skill_sales = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            category=SkillCategory.SALES,
        )
        assert skill_sales.importance == CATEGORY_DEFAULT_IMPORTANCE[SkillCategory.SALES]

    def test_explicit_importance_overrides_default(self) -> None:
        """Explicit importance should override category default."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            category=SkillCategory.CUSTOM,
            importance=99,
        )
        assert skill.importance == 99

    def test_additional_triggers_converted(self) -> None:
        """Dict additional_triggers should be converted to TriggerConfig."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.KEYWORD,
            trigger_config={"keywords": ["primary"]},
            additional_triggers=[
                {"type": "semantic", "config": {"reference_phrases": ["test phrase"]}},
            ],
        )
        assert len(skill.additional_triggers) == 1
        assert isinstance(skill.additional_triggers[0], TriggerConfig)
        assert skill.additional_triggers[0].type == SkillTriggerType.SEMANTIC

    def test_get_all_triggers(self) -> None:
        """get_all_triggers should return primary + additional triggers."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.KEYWORD,
            trigger_config={"keywords": ["primary"]},
            additional_triggers=[
                {"type": "semantic", "config": {"reference_phrases": ["test"]}},
            ],
        )
        all_triggers = skill.get_all_triggers()
        assert len(all_triggers) == 2
        assert all_triggers[0].type == SkillTriggerType.KEYWORD
        assert all_triggers[1].type == SkillTriggerType.SEMANTIC


class TestSkillMatch:
    """Test SkillMatch dataclass."""

    def test_skill_match_creation(self) -> None:
        """SkillMatch can be created with required fields."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
        )
        match = SkillMatch(skill=skill, score=0.8, trigger_reason="test reason")
        assert match.skill == skill
        assert match.score == 0.8
        assert match.trigger_reason == "test reason"

    def test_skill_match_comparison_by_score(self) -> None:
        """SkillMatches should compare by score first."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
        )
        match_low = SkillMatch(skill=skill, score=0.5, trigger_reason="low")
        match_high = SkillMatch(skill=skill, score=0.9, trigger_reason="high")

        # Higher score should be "greater"
        assert match_low < match_high
        assert not match_high < match_low

    def test_skill_match_comparison_by_priority(self) -> None:
        """Equal scores should compare by priority."""
        skill_low_priority = Skill(
            id="low",
            name="Low",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            priority=10,
        )
        skill_high_priority = Skill(
            id="high",
            name="High",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
            priority=90,
        )
        match_low = SkillMatch(skill=skill_low_priority, score=0.5, trigger_reason="")
        match_high = SkillMatch(skill=skill_high_priority, score=0.5, trigger_reason="")

        # Higher priority should be "greater" when scores are equal
        assert match_low < match_high

    def test_skill_match_sorting(self) -> None:
        """SkillMatches should sort correctly."""
        skill = Skill(
            id="test",
            name="Test",
            content="Content",
            trigger_type=SkillTriggerType.EXPLICIT,
        )
        matches = [
            SkillMatch(skill=skill, score=0.5, trigger_reason=""),
            SkillMatch(skill=skill, score=0.9, trigger_reason=""),
            SkillMatch(skill=skill, score=0.7, trigger_reason=""),
        ]

        sorted_matches = sorted(matches, reverse=True)
        assert sorted_matches[0].score == 0.9
        assert sorted_matches[1].score == 0.7
        assert sorted_matches[2].score == 0.5


class TestCategoryDefaultImportance:
    """Test CATEGORY_DEFAULT_IMPORTANCE mapping."""

    def test_all_categories_have_default_importance(self) -> None:
        """All categories should have default importance defined."""
        for category in SkillCategory:
            assert category in CATEGORY_DEFAULT_IMPORTANCE

    def test_importance_values_in_range(self) -> None:
        """All importance values should be in reasonable range."""
        for _category, importance in CATEGORY_DEFAULT_IMPORTANCE.items():
            assert 0 <= importance <= 100

    def test_tool_guidance_highest_priority(self) -> None:
        """TOOL_GUIDANCE should have highest default importance."""
        max_importance = max(CATEGORY_DEFAULT_IMPORTANCE.values())
        assert CATEGORY_DEFAULT_IMPORTANCE[SkillCategory.TOOL_GUIDANCE] == max_importance
