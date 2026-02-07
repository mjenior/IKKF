"""Tests for the SkillRegistry class."""

import tempfile
from pathlib import Path

from ikkf import SkillCategory, SkillRegistry, get_core_skills_path


class TestSkillRegistryInitialization:
    """Test SkillRegistry initialization."""

    def test_empty_initialization(self) -> None:
        """Registry can be initialized with no directories."""
        registry = SkillRegistry()
        assert len(registry.get_all_skills()) == 0

    def test_single_directory(self) -> None:
        """Registry can be initialized with a single directory."""
        registry = SkillRegistry(skills_dir=get_core_skills_path())
        assert len(registry.get_all_skills()) > 0

    def test_multiple_directories(self) -> None:
        """Registry can be initialized with multiple directories."""
        registry = SkillRegistry(skills_dirs=[get_core_skills_path()])
        assert len(registry.get_all_skills()) > 0

    def test_skills_dirs_takes_precedence(self) -> None:
        """skills_dirs parameter should take precedence over skills_dir."""
        # Create empty temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir)

            # skills_dirs is empty, skills_dir points to core
            # skills_dirs should take precedence
            registry = SkillRegistry(
                skills_dir=get_core_skills_path(),
                skills_dirs=[empty_dir],
            )
            # Should have loaded from empty dir, not core
            assert len(registry.get_all_skills()) == 0


class TestSkillRegistryLoading:
    """Test skill loading from directories."""

    def test_loads_yaml_files(self) -> None:
        """Should load skills from YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            yaml_content = """
id: test_skill
name: Test Skill
category: custom
trigger_type: explicit
skill: |
  ### Test Skill

  This is test content for the skill.
"""
            (skills_dir / "test_skill.yaml").write_text(yaml_content)

            registry = SkillRegistry(skills_dir=skills_dir)
            assert len(registry.get_all_skills()) == 1

    def test_skips_yaml_without_skill_field(self) -> None:
        """Should skip YAML files without skill field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            yaml_content = """
id: orphan_skill
name: Orphan Skill
"""
            (skills_dir / "orphan_skill.yaml").write_text(yaml_content)
            # No skill field in YAML

            registry = SkillRegistry(skills_dir=skills_dir)
            assert len(registry.get_all_skills()) == 0

    def test_later_directories_override_earlier(self) -> None:
        """Skills from later directories should override earlier ones."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            dir1 = Path(tmpdir1)
            dir2 = Path(tmpdir2)

            # First directory skill
            (dir1 / "shared_skill.yaml").write_text("""
id: shared_skill
name: Original Skill
category: custom
trigger_type: explicit
priority: 10
skill: |
  ### Original Skill

  This is the original skill content.
""")

            # Second directory overrides
            (dir2 / "shared_skill.yaml").write_text("""
id: shared_skill
name: Override Skill
category: custom
trigger_type: explicit
priority: 99
skill: |
  ### Override Skill

  This is the override skill content.
""")

            registry = SkillRegistry(skills_dirs=[dir1, dir2])
            skill = registry.get_skill("shared_skill")

            assert skill is not None
            assert skill.name == "Override Skill"
            assert skill.priority == 99


class TestSkillRegistryAccess:
    """Test skill access methods."""

    def test_get_skill_returns_skill(self) -> None:
        """get_skill should return the requested skill."""
        registry = SkillRegistry(skills_dir=get_core_skills_path())
        skill = registry.get_skill("knowledge_boundaries")
        assert skill is not None
        assert skill.id == "knowledge_boundaries"

    def test_get_skill_returns_none_for_missing(self) -> None:
        """get_skill should return None for non-existent skills."""
        registry = SkillRegistry(skills_dir=get_core_skills_path())
        skill = registry.get_skill("nonexistent_skill")
        assert skill is None

    def test_get_all_skills_returns_list(self) -> None:
        """get_all_skills should return a list of all skills."""
        registry = SkillRegistry(skills_dir=get_core_skills_path())
        skills = registry.get_all_skills()
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_get_skills_by_category(self) -> None:
        """get_skills_by_category should return skills in that category."""
        registry = SkillRegistry(skills_dir=get_core_skills_path())
        skills = registry.get_skills_by_category(SkillCategory.KNOWLEDGE)
        assert isinstance(skills, list)
        for skill in skills:
            assert skill.category == SkillCategory.KNOWLEDGE


class TestSkillRegistryReload:
    """Test skill reload functionality."""

    def test_reload_reloads_from_disk(self) -> None:
        """reload should reload skills from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            # Create initial skill
            (skills_dir / "test_skill.yaml").write_text("""
id: test_skill
name: Initial Name
category: custom
trigger_type: explicit
skill: |
  ### Initial Name

  This is the initial content.
""")

            registry = SkillRegistry(skills_dir=skills_dir)
            skill = registry.get_skill("test_skill")
            assert skill.name == "Initial Name"

            # Modify the skill on disk
            (skills_dir / "test_skill.yaml").write_text("""
id: test_skill
name: Updated Name
category: custom
trigger_type: explicit
skill: |
  ### Updated Name

  This is the updated content.
""")

            # Reload
            registry.reload()

            skill = registry.get_skill("test_skill")
            assert skill.name == "Updated Name"


class TestSkillRegistryAddDirectory:
    """Test add_skills_directory functionality."""

    def test_add_skills_directory(self) -> None:
        """add_skills_directory should load skills from new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            (skills_dir / "new_skill.yaml").write_text("""
id: new_skill
name: New Skill
category: custom
trigger_type: explicit
skill: |
  ### New Skill

  This is a new skill.
""")

            registry = SkillRegistry()
            assert registry.get_skill("new_skill") is None

            count = registry.add_skills_directory(skills_dir)
            assert count == 1
            assert registry.get_skill("new_skill") is not None

    def test_add_same_directory_twice(self) -> None:
        """Adding the same directory twice should be a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)

            (skills_dir / "skill.yaml").write_text("""
id: skill
name: Skill
category: custom
trigger_type: explicit
skill: |
  ### Skill

  This is a skill.
""")

            registry = SkillRegistry()
            count1 = registry.add_skills_directory(skills_dir)
            count2 = registry.add_skills_directory(skills_dir)

            assert count1 == 1
            assert count2 == 0  # No new skills loaded
