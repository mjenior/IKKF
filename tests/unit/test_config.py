"""Tests for the SkillsConfig dataclass."""

from pathlib import Path

import pytest

from ikkf import SkillsConfig


class TestSkillsConfigDefaults:
    """Test default configuration values."""

    def test_default_custom_skills_dirs_empty(self) -> None:
        """Custom skills directories should be empty by default."""
        config = SkillsConfig()
        assert config.custom_skills_dirs == []

    def test_default_thresholds(self) -> None:
        """Check default threshold values."""
        config = SkillsConfig()
        assert config.skill_base_threshold == 0.68
        assert config.skill_importance_low_weight == 0.9
        assert config.skill_importance_medium_weight == 1.0
        assert config.skill_importance_high_weight == 1.1

    def test_default_escalation_model(self) -> None:
        """Default escalation model should be Anthropic Haiku 4.5."""
        config = SkillsConfig()
        assert config.escalation_model == "anthropic/claude-haiku-4-5-20251001"


class TestSkillsConfigValidation:
    """Test configuration validation."""

    def test_threshold_must_be_in_range(self) -> None:
        """Threshold must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="skill_base_threshold must be between"):
            SkillsConfig(skill_base_threshold=1.5)

        with pytest.raises(ValueError, match="skill_base_threshold must be between"):
            SkillsConfig(skill_base_threshold=-0.1)

    def test_threshold_at_boundaries(self) -> None:
        """Threshold at 0.0 and 1.0 should be valid."""
        config_low = SkillsConfig(skill_base_threshold=0.0)
        assert config_low.skill_base_threshold == 0.0

        config_high = SkillsConfig(skill_base_threshold=1.0)
        assert config_high.skill_base_threshold == 1.0

    def test_weights_must_be_positive(self) -> None:
        """Importance weights must be positive."""
        with pytest.raises(ValueError, match="skill_importance_low_weight must be positive"):
            SkillsConfig(skill_importance_low_weight=0)

        with pytest.raises(ValueError, match="skill_importance_medium_weight must be positive"):
            SkillsConfig(skill_importance_medium_weight=-1.0)

        with pytest.raises(ValueError, match="skill_importance_high_weight must be positive"):
            SkillsConfig(skill_importance_high_weight=0)

    def test_string_paths_converted_to_path_objects(self) -> None:
        """String paths should be converted to Path objects."""
        config = SkillsConfig(custom_skills_dirs=["./my_skills", "/absolute/path"])
        assert all(isinstance(p, Path) for p in config.custom_skills_dirs)
        assert config.custom_skills_dirs[0] == Path("./my_skills")
        assert config.custom_skills_dirs[1] == Path("/absolute/path")

    def test_path_objects_preserved(self) -> None:
        """Path objects should be preserved as-is."""
        paths = [Path("./my_skills"), Path("/absolute/path")]
        config = SkillsConfig(custom_skills_dirs=paths)
        assert config.custom_skills_dirs == paths


class TestSkillsConfigParseEscalationModel:
    """Test the parse_escalation_model method."""

    def test_returns_model_string_when_enabled(self) -> None:
        """parse_escalation_model returns the model string directly."""
        config = SkillsConfig()
        result = config.parse_escalation_model()
        assert result == "anthropic/claude-haiku-4-5-20251001"

    def test_returns_none_when_disabled(self) -> None:
        """parse_escalation_model returns None when escalation is disabled."""
        config = SkillsConfig(enable_model_escalation=False)
        assert config.parse_escalation_model() is None

    def test_returns_custom_model(self) -> None:
        """parse_escalation_model returns custom model string as-is."""
        config = SkillsConfig(escalation_model="openai/gpt-4o-mini")
        result = config.parse_escalation_model()
        assert result == "openai/gpt-4o-mini"
