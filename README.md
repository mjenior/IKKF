# I Know Kung Fu (IKKF)

**"I know Kung Fu." — Your LLM agent, probably.** A dynamic prompt composition system that lets your AI instantly load specialized skills mid-conversation, just like Neo downloading abilities in The Matrix.

## Overview

You know that scene? Neo sits in a chair, they plug in the neural interface, and suddenly he knows Kung Fu. That's IKKF—except for your language models, and without the whole "machines enslaving humanity" subplot.

Instead of bloating your agent's context window with every possible instruction from day one, `ikkf` dynamically injects the exact skills your agent needs, precisely when it needs them. Think of it as a specialized neural download for conversational scenarios. Skills load based on:

- **Keywords**: Spotting the right terminology in user messages
- **Semantic vibes**: Matching by meaning, not just words
- **Tool usage**: Activating when your agent needs specific capabilities
- **Conversation progression**: Leveling up as the chat gets deeper
- **Explicit inclusion**: Core skills that are always ready to go

## Getting Started

```bash
uv pip install ikkf
```

If you're setting up for development (which means you're about to become the One):

```bash
uv sync --all-groups
```

### API Keys

For model escalation (LLM-based skill selection), set your provider's API key:

```bash
# For Anthropic (default)
export ANTHROPIC_API_KEY="your-key-here"

# For Google Gemini
export GEMINI_API_KEY="your-key-here"

# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For other providers, see litellm docs
```

Model escalation is enabled by default. To disable it, see the [Model Escalation](#model-escalation) section.

## Quick Start

### The Basics

```python
from ikkf import Skills

# Load your skill set (uses bundled core skills out of the box)
skills = Skills()

# Ask what the user needs
selected = skills.select("What tools do you have?")

# Deploy the right skills for the situation
for skill in selected:
    print(f"Skill: {skill.name}")
    print(skill.content)
```

### Going Custom

```python
from pathlib import Path
from ikkf import Skills, SkillsConfig

# Teach IKKF about your own skill library
config = SkillsConfig(
    custom_skills_dirs=[Path("./my_agent/skills")],
)

skills = Skills(config)
```

### Stripping It Down

Want just your custom skills and nothing else? Easy:

```python
from ikkf import Skills, SkillsConfig

config = SkillsConfig(
    custom_skills_dirs=[Path("./my_skills")],
)
skills = Skills(config)
```

## Building Your Own Skills

Ready to expand your agent's abilities? Each skill is a pair of files: YAML for the configuration, Markdown for the actual content. No neural interface required.

### `my_skill.yaml` — The Blueprint

```yaml
id: my_skill
name: My Custom Skill
category: custom  # technical, conversation_management, knowledge, tool_guidance, custom
description: Custom skill that does a thing
trigger_type: keyword  # keyword, semantic, tool_usage, turn_based, explicit
trigger_config:
  keywords:
    - "specific term"
    - "another term"
priority: 50  # 0-100, higher = loads first
dependencies: []  # Companion skills that load together
incompatible_with: []  # Skills that don't mix well
```

### `my_skill.md` — The Download

```markdown
### My Custom Skill

When the user asks about specific terms:

1. First, do this
2. Then, do that
3. Finally, wrap up

Keep responses focused on the key concepts.
```

## Trigger Types

Your skills need to know *when* to activate. Here's how to set that up:

### Keyword Triggers

Match exact terms. Crude but effective—like recognizing "the One" by his vibes:

```yaml
trigger_type: keyword
trigger_config:
  keywords:
    - "pricing"
    - "cost"
    - "quote"
  history_decay: 0.5  # How much to weight older messages
```

### Semantic Triggers

Go by *meaning*, not just keywords. When the user's saying the same thing but in different words:

```yaml
trigger_type: semantic
trigger_config:
  importance: medium  # low, medium, high, critical
  reference_phrases:
    - "What's your pricing?"
    - "How much does it cost?"
```

### Tool Usage Triggers

Automatically activate when your agent might need specific tools. Perfect for tool-aware agents:

```yaml
trigger_type: tool_usage
trigger_config:
  tools:
    - get_device_properties
    - list_submodules
  patterns:
    - "\\bstatus\\b"
    - "\\bshow me\\b"
```

### Turn-Based Triggers

Load skills based on how far into the conversation you are. Got sophisticated moves? Save them for later in the fight:

```yaml
trigger_type: turn_based
trigger_config:
  min_turn: 3
  max_turn: 10
```

### Explicit Triggers

Always loaded. These are your *essential* moves—the ones you need locked and loaded from the start:

```yaml
trigger_type: explicit
```

## Model Escalation

When embedding-based matching isn't confident enough, IKKF can escalate to an LLM for smarter skill selection. Think of it as calling in expert backup when the situation gets ambiguous.

### How It Works

Escalation automatically triggers when:
- **No embedding matches** are found
- **Low confidence** on the best match (below threshold)
- **Ambiguous results** with multiple close matches
- **Very short messages** where embeddings are less reliable

When triggered, IKKF sends the candidate skills to an LLM for intelligent reranking based on the user's message.

### Configuration

IKKF uses [litellm](https://github.com/BerriAI/litellm) for provider-agnostic model access. Model names use the format `provider/model`:

```python
from ikkf import Skills, SkillsConfig

# Anthropic Haiku-4.5 (fast and cost-effective)
config = SkillsConfig(
    enable_model_escalation=True,  # Enabled by default
    escalation_model="anthropic/claude-haiku-4-5-20251001,
    escalation_threshold=0.75,  # Confidence threshold for triggering
    escalation_max_candidates=10,  # Max skills to send to model
)

skills = Skills(config)
```

### Using Different Models

litellm supports 100+ providers. Just use the `provider/model` format:

```python
# Google Gemini 2.0 Flash (experimental)
config = SkillsConfig(
    escalation_model="gemini/gemini-2.0-flash-exp"
)

# Anthropic Claude Sonnet
config = SkillsConfig(
    escalation_model="anthropic/claude-3-5-sonnet-20241022"
)

# OpenAI GPT-4o Mini
config = SkillsConfig(
    escalation_model="openai/gpt-4o-mini"
)

# Disable escalation entirely
config = SkillsConfig(
    enable_model_escalation=False
)
```

**Note:** Make sure you have the appropriate API keys set in your environment (e.g., `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). litellm handles provider authentication automatically.

### Checking Escalation Status

Want to know if model escalation was used? Use `select_with_metadata()`:

```python
from ikkf import Skills

skills = Skills()

# Get selection result with metadata
result = skills.select_with_metadata("What tools do you have?")

# Check if escalation happened
if result.escalated:
    print(f"Model escalation triggered: {result.escalation_reason}")
    print(f"Selected {len(result.skills)} skills via LLM")
else:
    print(f"Selected {len(result.skills)} skills via embeddings")

# Iterate over skills (result acts like a list)
for skill in result:
    print(skill.name)
```

The `SelectionResult` object is iterable (works like a list) but also includes:
- `escalated: bool` — Whether model escalation was triggered
- `escalation_reason: str | None` — Why escalation happened (e.g., "low_confidence:0.62")
- `skills: list[Skill]` — The selected skills

## API Reference

The main controls for managing your skill arsenal:

### Skills Class

Your central skill hub. Need a specific ability? This is where you ask for it:

```python
class Skills:
    def __init__(self, config: SkillsConfig | None = None): ...

    def select(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> list[Skill]: ...

    def select_with_metadata(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        pending_tool_calls: list[str] | None = None,
        turn_count: int = 0,
    ) -> SelectionResult: ...

    def get_skill(self, skill_id: str) -> Skill | None: ...
    def list_skills(self) -> list[Skill]: ...
    def list_skills_by_category(self, category: SkillCategory) -> list[Skill]: ...
    def add_skills_directory(self, path: Path) -> int: ...
    def reload(self) -> None: ...
```

### SkillsConfig

The dials and switches that control how aggressively IKKF swaps out skills. Turn these knobs to fine-tune your agent's behavior:

```python
@dataclass
class SkillsConfig:
    # Skill loading
    custom_skills_dirs: list[Path] = field(default_factory=list)

    # Semantic matching thresholds
    skill_base_threshold: float = 0.68  # Base similarity threshold (0.0-1.0)
    skill_importance_low_weight: float = 0.9  # Multiplier for low importance
    skill_importance_medium_weight: float = 1.0  # Multiplier for medium importance
    skill_importance_high_weight: float = 1.1  # Multiplier for high importance

    # Model escalation (LLM-based reranking)
    enable_model_escalation: bool = True  # Enable/disable escalation
    escalation_model: str = "anthropic/claude-haiku-4-5-20251001"  # litellm model
    escalation_threshold: float = 0.75  # Confidence threshold for triggering
    escalation_max_candidates: int = 10  # Max skills to send to model
```

## Contributing & Development

This project uses [uv](https://docs.astral.sh/uv/) for dependencies and [Task](https://taskfile.dev/) for automation. No red pills necessary—just good tooling.

### Setting Up Your Dojo

```bash
uv sync --all-groups
```

### Running the Tests

```bash
task test
# Want coverage stats too?
task test:cov
```

### Lint & Format

Keep the code clean and consistent:

```bash
task lint      # Static analysis (ruff + mypy)
task format    # Auto-format with ruff
```

### All Checks at Once

```bash
task check     # Runs lint + test together
```

### Building for Distribution

```bash
task build
```

## License

MIT
