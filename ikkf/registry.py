"""
Skill registry for loading and managing skill definitions.

Skills are stored as YAML files with embedded content.
Supports loading from multiple directories for core + agent-specific skills.

Thread-safe: All public methods are safe to call from multiple threads.
The reload() method uses atomic dictionary swapping to prevent inconsistent
states during concurrent access.
"""

import importlib.resources
import logging
import threading
from pathlib import Path

import yaml

from ikkf.models import Skill, SkillCategory, SkillTriggerType

__all__ = ["SkillRegistry", "get_core_skills_path"]

logger = logging.getLogger(__name__)


def get_core_skills_path() -> Path:
    """
    Get the path to the bundled core skills directory.

    Uses importlib.resources to locate the core/ directory within the
    installed package, ensuring it works regardless of installation method
    (pip install, editable install, etc.).

    Returns:
        Path to the core skills directory
    """
    package_files = importlib.resources.files("skills")
    return Path(str(package_files / "core"))


class SkillRegistry:
    """
    Registry for loading and accessing skill definitions.

    Skills are loaded from one or more directories containing:
    - {skill_id}.yaml: Skill manifest with metadata, trigger config, and content

    When multiple directories are provided, skills from later directories
    override skills with the same ID from earlier directories. This allows
    agent-specific skills to extend or override core skills.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        skills_dirs: list[Path] | None = None,
    ):
        """
        Initialize the registry and load all skills from the directories.

        Args:
            skills_dir: Single path to directory containing skill files (legacy API)
            skills_dirs: List of paths to skill directories. Skills from later
                directories override skills with the same ID from earlier ones.
                This allows core skills to be extended or overridden by
                agent-specific skills.

        Note:
            Either skills_dir or skills_dirs should be provided, not both.
            If skills_dir is provided, it's treated as a single-item list.
            If both are provided, skills_dirs takes precedence.
        """
        if skills_dirs is not None:
            self.skills_dirs = skills_dirs
        elif skills_dir is not None:
            self.skills_dirs = [skills_dir]
        else:
            self.skills_dirs = []

        # Thread safety: lock protects access to self._skills
        self._lock = threading.RLock()
        self._skills: dict[str, Skill] = {}

        # Load skills from all directories
        for skills_dir in self.skills_dirs:
            self._load_skills_from_dir(skills_dir, self._skills)

    def _parse_skill(self, manifest_path: Path, skills_dir: Path) -> Skill | None:
        """
        Parse a single skill from its manifest file.

        Args:
            manifest_path: Path to the YAML manifest file
            skills_dir: Directory containing the skill files

        Returns:
            The parsed Skill object, or None if parsing failed
        """
        manifest = yaml.safe_load(manifest_path.read_text())

        # Handle empty or invalid YAML files
        if not manifest or not isinstance(manifest, dict):
            logger.warning(f"Empty or invalid manifest: {manifest_path}")
            return None

        skill_id = manifest.get("id", manifest_path.stem)

        # Load content from skill field in YAML
        content = manifest.get("skill", "")
        if not content:
            logger.warning(f"Skill content (skill field) not found in: {manifest_path}")
            return None

        # Parse trigger type
        trigger_type_str = manifest.get("trigger_type", "explicit")
        try:
            trigger_type = SkillTriggerType(trigger_type_str)
        except ValueError:
            logger.warning(f"Unknown trigger type '{trigger_type_str}' for skill {skill_id}")
            trigger_type = SkillTriggerType.EXPLICIT

        # Parse category with fallback to CUSTOM
        category_str = manifest.get("category", "custom")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            logger.warning(f"Unknown category '{category_str}' for skill {skill_id}, using CUSTOM")
            category = SkillCategory.CUSTOM

        skill = Skill(
            id=skill_id,
            name=manifest.get("name", skill_id),
            content=content,
            trigger_type=trigger_type,
            trigger_config=manifest.get("trigger_config", {}),
            additional_triggers=manifest.get("additional_triggers", []),
            priority=manifest.get("priority", 50),
            dependencies=manifest.get("dependencies", []),
            incompatible_with=manifest.get("incompatible_with", []),
            category=category,
            subcategory=manifest.get("subcategory"),
            importance=manifest.get("importance"),
            description=manifest.get("description"),
        )

        logger.debug(
            f"Parsed skill: {skill_id} (trigger: {trigger_type.value}, "
            f"category: {category.value}, importance: {skill.importance})"
        )
        return skill

    def get_skill(self, skill_id: str) -> Skill | None:
        """
        Get a skill by ID.

        Thread-safe: uses internal locking.
        """
        with self._lock:
            return self._skills.get(skill_id)

    def get_all_skills(self) -> list[Skill]:
        """
        Get all registered skills.

        Thread-safe: returns a snapshot of the current skills list.
        """
        with self._lock:
            return list(self._skills.values())

    def get_skills_by_category(self, category: SkillCategory) -> list[Skill]:
        """
        Get all skills in a specific category.

        Thread-safe: uses internal locking.

        Args:
            category: The category to filter by

        Returns:
            List of skills in the specified category
        """
        with self._lock:
            return [s for s in self._skills.values() if s.category == category]

    def reload(self) -> None:
        """
        Reload all skills from disk.

        Thread-safe: loads into a temporary dict first, then atomically
        swaps. Other threads will either see the old complete state or
        the new complete state, never a partial state.
        """
        # Load into temporary dict (outside the lock to avoid blocking reads)
        new_skills: dict[str, Skill] = {}
        for skills_dir in self.skills_dirs:
            self._load_skills_from_dir(skills_dir, new_skills)

        # Atomic swap under lock
        with self._lock:
            self._skills = new_skills

        logger.info(f"Reloaded {len(new_skills)} skills")

    def add_skills_directory(self, skills_dir: Path) -> int:
        """
        Add a skills directory and load its skills.

        Skills from the new directory can override existing skills with the
        same ID. This is useful for adding agent-specific skills at runtime.

        Thread-safe: uses internal locking.

        Args:
            skills_dir: Path to directory containing skill files

        Returns:
            Number of *new* skills added (not counting overrides of existing skills)
        """
        with self._lock:
            if skills_dir in self.skills_dirs:
                logger.debug(f"Skills directory already registered: {skills_dir}")
                return 0

            self.skills_dirs.append(skills_dir)
            initial_count = len(self._skills)
            self._load_skills_from_dir(skills_dir, self._skills)
            loaded_count = len(self._skills) - initial_count

        logger.info(f"Added skills directory {skills_dir}: {loaded_count} new skills")
        return loaded_count

    def _load_skills_from_dir(
        self,
        skills_dir: Path,
        target_dict: dict[str, Skill],
    ) -> None:
        """Load all skill definitions from a directory into the target dict.

        Args:
            skills_dir: Directory containing skill files
            target_dict: Dictionary to load skills into
        """
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return

        loaded_count = 0
        for manifest_path in skills_dir.glob("*.yaml"):
            try:
                skill = self._parse_skill(manifest_path, skills_dir)
                if skill:
                    if skill.id in target_dict:
                        logger.debug(f"Overriding skill: {skill.id} from {skills_dir}")
                    target_dict[skill.id] = skill
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load skill from {manifest_path}: {e}")

        if loaded_count > 0:
            logger.debug(f"Loaded {loaded_count} skills from {skills_dir}")
