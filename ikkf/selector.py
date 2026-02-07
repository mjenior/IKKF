"""
Context-aware skill selection for dynamic prompt composition.

The selector evaluates each skill's trigger against the current context
and returns the most relevant skills.

Uses semantic-router for semantic matching with support for multiple
concurrent matches.
"""

import logging
import math
import re

from semantic_router import Route
from semantic_router.encoders import DenseEncoder
from semantic_router.routers import SemanticRouter

from ikkf.config import SkillsConfig
from ikkf.models import SelectionResult, Skill, SkillMatch, SkillTriggerType, TriggerConfig
from ikkf.registry import SkillRegistry
from ikkf.reranker import ModelReranker, create_reranker

__all__ = ["SkillSelector"]

logger = logging.getLogger(__name__)


class _LocalSemanticMatcher:
    """
    Simple local semantic matcher used as a fallback.
    Used primarily during testing to avoid API calls to Google.

    Pre-normalizes all vectors during initialization so similarity
    computation is a simple dot product.
    """

    def __init__(self, encoder, routes: list[Route]):
        self._encoder = encoder
        self._routes = routes
        self._route_embeddings: dict[str, list[list[float]]] = {}
        self._prepare_embeddings()

    def _normalize(self, vec: list[float]) -> list[float]:
        """Normalize a vector to unit length."""
        if not vec:
            return vec
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    def _prepare_embeddings(self) -> None:
        """Encode and pre-normalize all route utterance vectors."""
        for route in self._routes:
            utterances = route.utterances or []
            if not utterances:
                continue
            raw_embeddings = self._encoder(utterances)
            self._route_embeddings[route.name] = [self._normalize(vec) for vec in raw_embeddings]

    def match(self, message: str) -> dict[str, float]:
        if not message:
            return {}

        query_embeddings = self._encoder([message])
        if not query_embeddings:
            return {}

        query_vector = self._normalize(query_embeddings[0])
        matches: dict[str, float] = {}

        for route in self._routes:
            route_vectors = self._route_embeddings.get(route.name, [])
            if not route_vectors:
                continue

            best_score = max(self._dot_product(query_vector, vec) for vec in route_vectors)
            if best_score >= route.score_threshold:
                matches[route.name] = best_score

        return matches

    def _dot_product(self, vector_a: list[float], vector_b: list[float]) -> float:
        """Compute dot product of two pre-normalized vectors (equivalent to cosine similarity)."""
        if not vector_a or not vector_b:
            return 0.0
        return sum(a * b for a, b in zip(vector_a, vector_b, strict=False))


class SkillSelector:
    """
    Selects relevant skills based on conversation context.

    Supports multiple trigger types:
    - KEYWORD: Pattern matching against user message
    - SEMANTIC: Embedding similarity via semantic-router (supports multiple matches)
    - TOOL_USAGE: Detection of tool-related patterns
    - TURN_BASED: Based on conversation turn count
    - EXPLICIT: Always included

    Features:
    - Multi-match semantic routing: Can match multiple semantic skills per query
    - Delegated routing: SemanticRouter handles embedding + matching
    """

    def __init__(
        self,
        registry: SkillRegistry,
        config: "SkillsConfig",
    ):
        """
        Initialize the skill selector.

        Args:
            registry: The skill registry to select from
            config: Configuration object containing threshold settings
        """
        self.registry = registry
        self._config = config
        # Maps trigger_key -> patterns (trigger_key = "skill_id" or "skill_id:idx" for additional triggers)
        self._keyword_patterns: dict[str, list[re.Pattern[str]]] = {}
        # Set of trigger_keys that have semantic triggers (for validation in _score_semantic)
        self._semantic_trigger_keys: set[str] = set()
        # Semantic router for embedding-based matching
        self._semantic_router: SemanticRouter | None = None
        self._semantic_matcher: _LocalSemanticMatcher | None = None
        self._build_indexes()

        # Initialize model reranker if escalation enabled
        self._reranker: ModelReranker | None = None
        model_name = config.parse_escalation_model()
        if model_name:
            self._reranker = create_reranker(model_name)
            if not self._reranker.available:
                logger.warning(
                    "Model escalation enabled but reranker unavailable. "
                    "Escalation will be skipped."
                )
                self._reranker = None

    def _build_indexes(self) -> None:
        """Pre-compute keyword patterns and build semantic router."""
        has_semantic_triggers = False

        for skill in self.registry.get_all_skills():
            # Process all triggers (primary + additional)
            all_triggers = skill.get_all_triggers()

            for idx, trigger in enumerate(all_triggers):
                # Use skill.id for primary (idx=0), skill.id:idx for additional
                trigger_key = skill.id if idx == 0 else f"{skill.id}:{idx}"

                if trigger.type == SkillTriggerType.KEYWORD:
                    keywords = trigger.config.get("keywords", [])
                    if keywords:
                        self._keyword_patterns[trigger_key] = [
                            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords
                        ]
                        logger.debug(f"Registered {len(keywords)} keywords for trigger {trigger_key}")

                elif trigger.type == SkillTriggerType.SEMANTIC:
                    phrases = trigger.config.get("reference_phrases", [])
                    if phrases:
                        has_semantic_triggers = True
                        self._semantic_trigger_keys.add(trigger_key)

        # Build semantic router if we have semantic triggers
        if has_semantic_triggers:
            self._build_semantic_router()


    def _build_semantic_router(self) -> None:
        """Build the SemanticRouter with DenseEncoder."""
        routes = self._build_semantic_routes()

        if not routes:
            logger.debug("No semantic routes to build")
            return

        encoder = None
        try:
            encoder = DenseEncoder(name="ikkf-dense-encoder")

            total_utterances = sum(len(route.utterances) for route in routes)
            self._semantic_router = SemanticRouter(
                encoder=encoder,
                routes=routes,
                auto_sync="local",
                top_k=total_utterances,
            )

            logger.info(f"Built semantic router with {len(routes)} routes, {total_utterances} utterances")

        except Exception as e:
            logger.error(f"Failed to build semantic router: {e}")
            self._semantic_router = None
            if encoder is not None:
                self._semantic_matcher = self._build_local_semantic_matcher(encoder, routes)

    def _resolve_threshold(self, trigger_config: dict) -> float:
        """
        Resolve trigger config to a numeric threshold.

        Formula: final_threshold = base_threshold / importance_weight
        Higher importance weight = lower threshold = more likely to match.

        Priority:
        1. Explicit "threshold" value (deprecated, for backwards compatibility)
        2. "critical" importance returns 0.0 (always included)
        3. "importance" string mapped via config weights (low/medium/high)
        4. Default to base_threshold / medium_weight

        Args:
            trigger_config: The trigger configuration dict from the skill YAML

        Returns:
            The resolved numeric threshold (0.0-1.0)
        """
        # Backwards compatibility: explicit threshold takes precedence
        if "threshold" in trigger_config:
            return trigger_config["threshold"]

        # Get importance level, default to "medium"
        importance = trigger_config.get("importance", "medium")

        # Critical importance: always included (threshold of 0.0 matches any score)
        if importance == "critical":
            return 0.0

        # Map importance to weight from config
        importance_weights = {
            "low": self._config.skill_importance_low_weight,
            "medium": self._config.skill_importance_medium_weight,
            "high": self._config.skill_importance_high_weight,
        }

        weight = importance_weights.get(importance, self._config.skill_importance_medium_weight)
        return self._config.skill_base_threshold / weight

    def _build_semantic_routes(self) -> list[Route]:
        """
        Build Route objects from ikkf with semantic triggers.

        Returns:
            List of routes with per-route thresholds applied
        """
        routes = []

        for skill in self.registry.get_all_skills():
            for idx, trigger in enumerate(skill.get_all_triggers()):
                if trigger.type == SkillTriggerType.SEMANTIC:
                    trigger_key = skill.id if idx == 0 else f"{skill.id}:{idx}"
                    phrases = trigger.config.get("reference_phrases", [])
                    threshold = self._resolve_threshold(trigger.config)

                    if phrases:
                        route = Route(
                            name=trigger_key,
                            utterances=phrases,
                            score_threshold=threshold,
                        )
                        routes.append(route)
                        logger.debug(f"Created route {trigger_key} with {len(phrases)} utterances")

        return routes

    def _build_local_semantic_matcher(self, encoder, routes: list[Route]) -> _LocalSemanticMatcher | None:
        try:
            matcher = _LocalSemanticMatcher(encoder, routes)
            logger.warning("Semantic router unavailable. Using local semantic matcher.")
            return matcher
        except Exception as e:
            logger.error(f"Failed to build local semantic matcher: {e}")
            return None

    def _should_escalate(
        self,
        matches: list[SkillMatch],
        message: str,
    ) -> tuple[bool, str | None]:
        """
        Determine if selection should escalate to model-based reranking.

        Escalation triggers:
        1. No embedding matches found
        2. Top match score below confidence threshold
        3. Multiple close matches (ambiguous - within 15% of top)
        4. Very short message (< 4 words - embeddings less reliable)

        Args:
            matches: Embedding-based matches, sorted by score
            message: User message

        Returns:
            Tuple of (should_escalate, reason_string)
        """
        if self._reranker is None:
            return False, None

        threshold = self._config.escalation_threshold

        # No embedding matches
        if not matches:
            return True, "no_matches"

        top_score = matches[0].score

        # Low confidence on best match
        if top_score < threshold:
            return True, f"low_confidence:{top_score:.2f}"

        # Ambiguous: multiple close matches (within 15% of top)
        if len(matches) >= 2:
            second_score = matches[1].score
            if second_score >= top_score * 0.85 and top_score < 0.90:
                return True, f"ambiguous:{top_score:.2f}/{second_score:.2f}"

        # Very short message (< 4 words) AND not high confidence
        # Only escalate if embeddings are unreliable (short query + not confident)
        if len(message.split()) < 4 and top_score < 0.85:
            return True, f"short_message:{top_score:.2f}"

        return False, None

    def _get_rerank_candidates(self, matches: list[SkillMatch]) -> list[Skill]:
        """
        Get candidate skills for model reranking.

        If we have embedding matches, use top N.
        If no matches, fall back to skills with descriptions.

        Args:
            matches: Embedding-based matches

        Returns:
            List of candidate skills for reranking
        """
        max_candidates = self._config.escalation_max_candidates

        if matches:
            return [m.skill for m in matches[:max_candidates]]

        # No embedding matches - use skills with descriptions
        all_skills = self.registry.get_all_skills()
        with_descriptions = [s for s in all_skills if s.description]
        candidates = with_descriptions[:max_candidates]

        if not candidates:
            logger.debug("No reranking candidates available (no skills with descriptions)")

        return candidates

    def _apply_selection_constraints(self, matches: list[SkillMatch]) -> list[Skill]:
        """
        Apply dependency/incompatibility constraints to matches.

        This is refactored logic from select_skills to enable reuse
        for both embedding and model-based selection paths.

        Args:
            matches: Scored skill matches

        Returns:
            Final list of skills with constraints applied
        """
        selected: list[Skill] = []
        selected_ids: set[str] = set()

        for match in matches:
            skill = match.skill
            if skill.id in selected_ids:
                continue
            if self._is_incompatible(skill, selected_ids):
                continue

            selected.append(skill)
            selected_ids.add(skill.id)
            self._resolve_dependencies(
                skill=skill,
                selected=selected,
                selected_ids=selected_ids,
                resolution_chain=[skill.id],
            )

        return selected

    def _log_selection(self, selected: list[Skill], escalated: bool) -> None:
        """
        Log selection results with metadata.

        Args:
            selected: Final selected skills
            escalated: Whether model escalation was used
        """
        if selected:
            category_summary = self._get_category_summary(selected)
            skill_details = ", ".join([f"{s.id}({s.category.value})" for s in selected])
            prefix = "[ESCALATED] " if escalated else ""
            logger.info(
                f"{prefix}Skills selected: {len(selected)} skills, "
                f"categories: {category_summary} [{skill_details}]"
            )
        else:
            logger.debug("No skills matched for this context")

    def select_skills_with_metadata(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> SelectionResult:
        """
        Select skills with escalation metadata.

        This is the new primary selection method that returns full metadata
        about whether model escalation was used. The result can be iterated
        like a list for backward compatibility.

        Args:
            user_message: The latest user message
            conversation_history: Full conversation for context (optional)
            pending_tool_calls: Tool names that might be called (optional)
            turn_count: Number of user turns so far

        Returns:
            SelectionResult with skills and escalation metadata
        """
        conversation_history = conversation_history or []
        pending_tool_calls = pending_tool_calls or []

        # Phase 1: Embedding-based matching (existing logic)
        semantic_matches = self._get_all_semantic_matches(user_message)

        matches: list[SkillMatch] = []
        for skill in self.registry.get_all_skills():
            match = self._score_skill(
                skill,
                user_message,
                conversation_history,
                pending_tool_calls,
                turn_count,
                semantic_matches,
            )
            if match is not None and match.score > 0:
                matches.append(match)

        matches.sort(reverse=True)

        # Phase 2: Check for escalation
        should_escalate, reason = self._should_escalate(matches, user_message)
        escalated = False

        if should_escalate and self._reranker is not None:
            # Get candidates for reranking
            candidates = self._get_rerank_candidates(matches)

            if candidates:
                result = self._reranker.rerank(candidates, user_message)

                if result.success:
                    escalated = True
                    # Rebuild matches from reranker output
                    matches = []
                    for rank, skill_id in enumerate(result.skill_ids):
                        skill = self.registry.get_skill(skill_id)
                        if skill:
                            # Score decreases by rank (1.0, 0.95, 0.90, ...)
                            score = 1.0 - (rank * 0.05)
                            matches.append(SkillMatch(
                                skill=skill,
                                score=max(score, 0.5),
                                trigger_reason="model_selected",
                            ))
                        else:
                            logger.warning(f"Model selected unknown skill ID: {skill_id}")

                    logger.info(
                        f"Escalated ({reason}): "
                        f"{len(matches)} skills selected by model"
                    )
                else:
                    logger.warning(
                        f"Escalation triggered ({reason}) but reranking failed: "
                        f"{result.error}"
                    )

        # Phase 3: Apply constraints (dependencies, incompatibilities)
        selected = self._apply_selection_constraints(matches)

        # Phase 4: Logging
        self._log_selection(selected, escalated)

        return SelectionResult(
            skills=selected,
            escalated=escalated,
            escalation_reason=reason if escalated else None,
        )

    def select_skills(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> list[Skill]:
        """
        Select relevant skills based on current context.

        This is the backward-compatible API that returns just the list of skills.
        For escalation metadata, use select_skills_with_metadata() instead.

        Args:
            user_message: The latest user message
            conversation_history: Full conversation for context (optional)
            pending_tool_calls: Tool names that might be called (optional)
            turn_count: Number of user turns so far

        Returns:
            List of selected skills, ordered by relevance
        """
        result = self.select_skills_with_metadata(
            user_message=user_message,
            conversation_history=conversation_history,
            pending_tool_calls=pending_tool_calls,
            turn_count=turn_count,
        )
        return result.skills

    def _get_category_summary(self, skills: list[Skill]) -> str:
        """Get a summary string of categories represented in selected skills."""
        category_counts: dict[str, int] = {}
        for skill in skills:
            cat_name = skill.category.value
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        return ", ".join(f"{cat}:{count}" for cat, count in sorted(category_counts.items()))

    def _resolve_dependencies(
        self,
        skill: Skill,
        selected: list[Skill],
        selected_ids: set[str],
        resolution_chain: list[str],
    ) -> None:
        """
        Recursively resolve skill dependencies with cycle detection.

        Args:
            skill: The skill whose dependencies to resolve
            selected: List of selected skills (modified in-place)
            selected_ids: Set of selected skill IDs (modified in-place)
            resolution_chain: List of skill IDs in current resolution path (for cycle detection)
        """
        # Handle None dependencies (defensive)
        dependencies = skill.dependencies or []

        for dep_id in dependencies:
            # Skip if already selected
            if dep_id in selected_ids:
                continue

            # Cycle detection: if this dependency is already in our resolution chain
            if dep_id in resolution_chain:
                chain_str = " -> ".join(resolution_chain + [dep_id])
                logger.warning(f"Dependency cycle detected: {chain_str}")
                continue

            dep_skill = self.registry.get_skill(dep_id)
            if not dep_skill:
                logger.debug(f"Dependency {dep_id} not found in registry")
                continue

            # Check incompatibility in both directions
            if self._is_incompatible(dep_skill, selected_ids):
                logger.debug(f"Skipping dependency {dep_id}: incompatible with selected skills")
                continue

            # Add the dependency
            selected.append(dep_skill)
            selected_ids.add(dep_id)

            # Recursively resolve this dependency's dependencies (transitive resolution)
            self._resolve_dependencies(
                skill=dep_skill,
                selected=selected,
                selected_ids=selected_ids,
                resolution_chain=resolution_chain + [dep_id],
            )

    def _is_incompatible(self, skill: Skill, selected_ids: set[str]) -> bool:
        """
        Check if a skill is incompatible with any already-selected skills.

        Checks both directions:
        1. If the skill declares incompatibility with any selected skill
        2. If any selected skill declares incompatibility with this skill

        Args:
            skill: The skill to check
            selected_ids: Set of already-selected skill IDs

        Returns:
            True if the skill is incompatible with any selected skill
        """
        # Check if this skill is incompatible with any selected skill
        if any(sid in selected_ids for sid in (skill.incompatible_with or [])):
            return True

        # Check if any selected skill is incompatible with this skill
        for sid in selected_ids:
            selected_skill = self.registry.get_skill(sid)
            if selected_skill and skill.id in (selected_skill.incompatible_with or []):
                return True

        return False

    def _get_all_semantic_matches(self, message: str) -> dict[str, float]:
        """
        Get all semantic matches above their thresholds.

        Returns a dict mapping trigger_key -> similarity_score for all matching routes.
        This supports multiple semantic skills matching a single user message.

        Uses the SemanticRouter to perform routing. Per-route thresholds are
        configured on the routes themselves.
        """
        if (self._semantic_router is None and self._semantic_matcher is None) or not message:
            return {}

        try:
            if self._semantic_router is None:
                return self._semantic_matcher.match(message) if self._semantic_matcher else {}

            routed = self._semantic_router(message, limit=None)

            matches: dict[str, float] = {}
            if routed is None:
                return {}
            if not isinstance(routed, list):
                routed = [routed]

            for entry in routed:
                route_name = None
                score = None

                if isinstance(entry, tuple) and len(entry) >= 2:
                    route_obj, score = entry[0], entry[1]
                    route_name = getattr(route_obj, "name", None) if route_obj is not None else None
                else:
                    route_name = getattr(entry, "name", None)
                    score = getattr(entry, "score", None)
                    if route_name is None and hasattr(entry, "route"):
                        route_name = getattr(entry.route, "name", None)
                    if score is None:
                        score = getattr(entry, "similarity", None)

                if not route_name or score is None:
                    continue
                matches[route_name] = float(score)

            if matches:
                logger.debug(f"Semantic matches: {list(matches.keys())}")

            return matches

        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            if self._semantic_matcher is not None:
                return self._semantic_matcher.match(message)
            return {}

    def _score_skill(
        self,
        skill: Skill,
        message: str,
        history: list[dict],
        tools: list[str],
        turn: int,
        semantic_matches: dict[str, float] | None = None,
    ) -> SkillMatch | None:
        """
        Score a skill's relevance across all its triggers.

        Evaluates the primary trigger and all additional triggers,
        returning the best (highest-scoring) match.

        Args:
            skill: The skill to score
            message: User message
            history: Conversation history
            tools: Pending tool calls
            turn: Current turn count
            semantic_matches: Dict of trigger_key -> score for all semantic matches
        """
        all_triggers = skill.get_all_triggers()
        matches: list[SkillMatch] = []
        semantic_matches = semantic_matches or {}

        for idx, trigger in enumerate(all_triggers):
            trigger_key = skill.id if idx == 0 else f"{skill.id}:{idx}"
            match = self._score_single_trigger(
                skill=skill,
                trigger=trigger,
                trigger_key=trigger_key,
                message=message,
                history=history,
                tools=tools,
                turn=turn,
                semantic_matches=semantic_matches,
            )
            if match is not None and match.score > 0:
                matches.append(match)

        if not matches:
            return None

        # Return the best match (highest score)
        best_match = max(matches, key=lambda m: m.score)

        # If multiple triggers matched, combine reasons for debugging
        if len(matches) > 1:
            all_reasons = [m.trigger_reason for m in matches]
            combined_reason = f"{best_match.trigger_reason} (+{len(matches) - 1} other triggers)"
            logger.debug(f"Skill {skill.id} matched {len(matches)} triggers: {all_reasons}")
            return SkillMatch(
                skill=skill,
                score=best_match.score,
                trigger_reason=combined_reason,
            )

        return best_match

    def _score_single_trigger(
        self,
        skill: Skill,
        trigger: TriggerConfig,
        trigger_key: str,
        message: str,
        history: list[dict],
        tools: list[str],
        turn: int,
        semantic_matches: dict[str, float] | None = None,
    ) -> SkillMatch | None:
        """Score a single trigger configuration."""
        semantic_matches = semantic_matches or {}

        if trigger.type == SkillTriggerType.EXPLICIT:
            return SkillMatch(skill=skill, score=1.0, trigger_reason="always included")

        if trigger.type == SkillTriggerType.KEYWORD:
            return self._score_keyword(skill, trigger_key, trigger.config, message, history)

        if trigger.type == SkillTriggerType.SEMANTIC:
            return self._score_semantic(skill, trigger_key, semantic_matches)

        if trigger.type == SkillTriggerType.TOOL_USAGE:
            return self._score_tool_usage(skill, trigger.config, message, tools)

        if trigger.type == SkillTriggerType.TURN_BASED:
            return self._score_turn_based(skill, trigger.config, turn)

        return None

    def _score_keyword(
        self,
        skill: Skill,
        trigger_key: str,
        trigger_config: dict,
        message: str,
        history: list[dict],
    ) -> SkillMatch | None:
        """Score based on keyword pattern matching.

        Supports configurable history decay via trigger_config:
            - history_decay: float (0.0-1.0) - multiplier for history matches.
              Default is 0.5. Set to 0.0 to disable history matching, or higher
              values (e.g., 0.8) for skills that benefit from persistent context.
            - history_decay_by_position: list[float] - per-position decay multipliers
              for messages [-1], [-2], [-3]. If provided, overrides history_decay.
              Example: [0.8, 0.5, 0.2] means most recent = 0.8, second = 0.5, oldest = 0.2.
        """
        patterns = self._keyword_patterns.get(trigger_key, [])
        if not patterns:
            return None

        # Check current message
        matches_in_message = sum(1 for p in patterns if p.search(message or ""))

        # Get history decay configuration
        # history_decay_by_position takes precedence over history_decay if both are set
        history_decay_by_position = trigger_config.get("history_decay_by_position")
        history_decay = trigger_config.get("history_decay", 0.5)

        # Determine if history should be processed
        # Process history if: position decay is configured, OR flat decay is non-zero
        should_process_history = (
            bool(history_decay_by_position and len(history_decay_by_position) > 0) or history_decay != 0.0
        )

        # Calculate history matches with decay
        matches_in_history = 0.0
        if history and should_process_history:
            recent_history = history[-3:]  # Last 3 messages

            if history_decay_by_position and len(history_decay_by_position) > 0:
                # Position-aware decay: apply different multipliers per position
                # history_decay_by_position[0] = most recent, [1] = second, [2] = oldest
                for i, msg in enumerate(reversed(recent_history)):
                    content = msg.get("content") or ""
                    position_decay = (
                        history_decay_by_position[i]
                        if i < len(history_decay_by_position)
                        else history_decay_by_position[-1]
                    )
                    position_matches = sum(1 for p in patterns if p.search(content))
                    matches_in_history += position_matches * position_decay
            else:
                # Flat decay for all history matches
                recent_text = " ".join(m.get("content") or "" for m in recent_history)
                raw_history_matches = sum(1 for p in patterns if p.search(recent_text))
                matches_in_history = raw_history_matches * history_decay

        total_matches = matches_in_message + matches_in_history

        if total_matches == 0:
            return None

        # Normalize score (diminishing returns after 3 matches)
        score = min(total_matches / 3, 1.0)

        # Build recent_text for matched_keywords reporting
        recent_text = " ".join(m.get("content") or "" for m in history[-3:]) if history else ""
        matched_keywords = [p.pattern for p in patterns if p.search(message or "") or p.search(recent_text)]

        return SkillMatch(
            skill=skill,
            score=score,
            trigger_reason=f"keyword matches: {matched_keywords[:3]}",
        )

    def _score_semantic(
        self,
        skill: Skill,
        trigger_key: str,
        semantic_matches: dict[str, float],
    ) -> SkillMatch | None:
        """
        Score based on embedding similarity using multi-match semantic results.

        Args:
            skill: The skill being scored
            trigger_key: The trigger key to check for matches
            semantic_matches: Dict mapping trigger_key -> score for all matching routes
        """
        if trigger_key not in self._semantic_trigger_keys:
            return None

        # Check if this trigger matched
        if trigger_key in semantic_matches:
            score = semantic_matches[trigger_key]
            return SkillMatch(
                skill=skill,
                score=score,
                trigger_reason=f"semantic match (score: {score:.2f})",
            )

        return None

    def _score_tool_usage(
        self,
        skill: Skill,
        trigger_config: dict,
        message: str,
        pending_tools: list[str],
    ) -> SkillMatch | None:
        """Score based on tool usage patterns."""
        trigger_tools = trigger_config.get("tools", [])
        trigger_patterns = trigger_config.get("patterns", [])

        score = 0.0
        reasons = []

        # Check pending tool calls
        matching_tools = [t for t in pending_tools if t in trigger_tools]
        if matching_tools:
            score = max(score, 0.9)
            reasons.append(f"tools: {matching_tools}")

        # Check message patterns that suggest tool usage
        tool_indicators = [
            (r"\bstatus\b", ["get_device_properties", "list_submodules"]),
            (r"\bshow me\b.*\bprotocol", ["get_protocol_info", "get_protocols"]),
            (r"\bwhat.*running\b", ["list_protocol_runs", "get_batch"]),
            (r"\bevents?\b", ["get_events"]),
            (r"\bfiles?\b", ["get_files"]),
            (r"\binstruments?\b", ["list_submodules", "get_submodule_by_id"]),
        ]

        for pattern, tools in tool_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                if any(t in trigger_tools for t in tools):
                    score = max(score, 0.7)
                    reasons.append(f"pattern: {pattern}")

        # Also check for explicit tool patterns in config
        for pattern in trigger_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                score = max(score, 0.8)
                reasons.append(f"config pattern: {pattern}")

        if score > 0:
            return SkillMatch(
                skill=skill,
                score=score,
                trigger_reason="; ".join(reasons),
            )

        return None

    def _score_turn_based(
        self,
        skill: Skill,
        trigger_config: dict,
        turn: int,
    ) -> SkillMatch | None:
        """Score based on conversation turn count."""
        min_turn = trigger_config.get("min_turn", 0)
        max_turn = trigger_config.get("max_turn", float("inf"))

        if min_turn <= turn <= max_turn:
            # Handle infinite or very large ranges (open-ended skills)
            if max_turn == float("inf") or (max_turn - min_turn) > 100:
                # For open-ended ranges, score based on distance from min_turn
                # Skills are most relevant near their activation point
                distance = turn - min_turn
                score = max(0.6, 1.0 - (distance * 0.05))  # Decay but floor at 0.6
            elif max_turn == min_turn:
                # Exact turn match
                score = 1.0
            else:
                # Finite range: peak in middle of range
                range_size = max_turn - min_turn + 1
                position = turn - min_turn
                normalized_position = position / (range_size - 1)
                score = 1.0 - abs(normalized_position - 0.5) * 0.4

            return SkillMatch(
                skill=skill,
                score=score,
                trigger_reason=f"turn {turn} in range [{min_turn}, {max_turn}]",
            )

        return None
