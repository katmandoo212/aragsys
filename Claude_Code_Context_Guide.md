# Claude Code CLI: Context Information & File Structure Guide

## Best Ways to Include Context from Another LLM

### Option 1: Use a File (Recommended for Substantial Context)
**Best for:** Large context, structured information, or information you want to persist

1. **Create a markdown file** in your project directory or `.claude/` folder
2. **Reference it in CLAUDE.md** so Claude knows to read it
3. **Or mention it directly** in your prompt: "Read the context in `context.md` and help me..."

Example structure:
```
my-project/
├── .claude/
│   ├── CLAUDE.md           # Reference the context file here
│   └── llm-context.md      # Your context from another LLM
```

In your CLAUDE.md:
```markdown
# Project Context

For additional context about [specific topic], see @.claude/llm-context.md
```

### Option 2: Pipe Directly into Claude Code
**Best for:** One-time use, log files, CSV data, or quick context injection

```bash
# Pipe file contents
cat context.txt | claude "Using this context, help me..."

# Pipe command output
echo "Context from another LLM: ..." | claude "Given this information..."
```

### Option 3: Use the @ Symbol in Prompts
**Best for:** Referencing specific files during a conversation

```bash
claude

# Then in the conversation:
> @context.md Review this context and help me implement the RAG system
```

### Option 4: Create a Slash Command (For Repeated Use)
**Best for:** Frequently used context that you want as a shortcut

Create `.claude/commands/load-context.md`:
```markdown
---
description: Load context from another LLM session
---

Read the context file at `.claude/llm-context.md` and use it to inform your responses about the project architecture and decisions made.
```

Then use: `/load-context` in any Claude session

### Recommendation
**For your RAG project context:** I recommend creating a file in `.claude/` (e.g., `.claude/architecture-notes.md` or `.claude/llm-context.md`) and referencing it in your CLAUDE.md file. This ensures the context is:
- Persistent across sessions
- Version controlled (if you commit it)
- Easy to update as decisions evolve
- Automatically available to Claude when working on the project

---

## Claude Code Markdown Files: Purpose & Structure

Here's a comprehensive table of the established .md files Claude Code uses:

| File Name | Location | Purpose | When Used | Scope | Auto-loaded? |
|-----------|----------|---------|-----------|-------|--------------|
| **CLAUDE.md** | Project root or `.claude/` | Main project-specific instructions, commands, conventions, architecture | Every session start | Project | ✅ Yes |
| **CLAUDE.local.md** | Project root or `.claude/` | Personal project preferences (not committed to git) | Every session start | Project (personal) | ✅ Yes |
| **AGENTS.md** | Project root or `.claude/` | Alternative to CLAUDE.md (used by other AI tools like Cursor, Zed) | Session start (if using compatible tools) | Project | ✅ Yes |
| **SKILL.md** | `.claude/skills/{skill-name}/` | Defines a reusable skill with instructions, scripts, and resources | When Claude detects relevance or user invokes `/skill-name` | Project or User | ⚠️ Metadata only (full content on-demand) |
| **{command}.md** | `.claude/commands/` | Custom slash command definitions (legacy, now merged into skills) | When user types `/command` | Project | ❌ No (invoked only) |
| **{rule}.md** | `.claude/rules/` | Conditional or unconditional rules that extend CLAUDE.md | Session start (unconditional) or when working with matching files (conditional) | Project | ✅ Yes (if conditions met) |

### Additional Files in User's Home Directory:

| File Name | Location | Purpose | When Used | Scope |
|-----------|----------|---------|-----------|-------|
| **CLAUDE.md** | `~/.claude/` | Global preferences across all projects | Every session start | User (all projects) |
| **SKILL.md** | `~/.claude/skills/{skill-name}/` | Personal skills available in all projects | When relevant or invoked | User (all projects) |
| **{command}.md** | `~/.claude/commands/` | Personal slash commands available everywhere | When user types `/command` | User (all projects) |

---

## Detailed File Descriptions

### 1. CLAUDE.md (Primary Configuration)
**Purpose:** The "constitution" of your project - persistent memory for Claude Code

**Contains:**
- Project overview and architecture
- Tech stack and dependencies
- Common terminal commands (build, test, lint, deploy)
- Code style and conventions
- Workflows (how to implement features, create PRs)
- File organization patterns
- Project-specific warnings and gotchas
- References to other documentation

**Priority:** Loaded first in the hierarchy, treated as immutable system rules

**Best Practices:**
- Keep concise (avoid redundancy)
- Focus on what Claude struggles with
- Document commands with exact arguments
- Use markdown headings for organization
- Reference external files with `@path/to/file.md`
- Iterate based on real mistakes

**Example Structure:**
```markdown
# Project Overview
This is a Scientific Agentic RAG Framework using PocketFlow for orchestration.

# Tech Stack
- Python 3.13+
- PocketFlow for workflow orchestration
- ChromaDB for vector storage
- Neo4j for graph relationships

# Common Commands
- **Run pipeline:** `python main.py --config config/pipelines.yaml`
- **Run tests:** `pytest tests/`
- **Evaluate:** `python -m evaluation.evaluate --dataset benchmarks/scientific_qa.json`

# Architecture
- RAG techniques are in `techniques/` as standalone .py files
- Each technique implements the BaseTechnique protocol (duck typing)
- Registry pattern loads techniques dynamically from `config/techniques.yaml`

# Coding Standards
- Use strict type hints (Python 3.13+)
- Follow SOLID principles (SRP and DI are priorities)
- No try-except in utility functions (let PocketFlow retry handle errors)

# Workflows
## Adding a New RAG Technique
1. Create `techniques/your_technique.py` implementing BaseTechnique
2. Add entry to `config/techniques.yaml`
3. Write unit tests in `tests/test_techniques/`
4. Update pipeline configs if needed
```

---

### 2. CLAUDE.local.md (Personal Overrides)
**Purpose:** Your personal preferences that shouldn't be shared with the team

**Contains:**
- Personal coding style preferences
- Custom tool configurations
- IDE-specific settings
- Experimental workflows
- Shortcuts you prefer

**Automatically added to `.gitignore`**

**Example:**
```markdown
# Personal Preferences

- Always use `ruff` for linting, not `flake8`
- When creating tests, include both positive and negative cases
- Prefer functional programming style over OOP when possible
```

---

### 3. SKILL.md (Reusable Capabilities)
**Purpose:** Modular, reusable expertise packages that Claude loads when relevant

**Structure:**
```markdown
---
name: technique-creator
description: Creates new RAG techniques following project patterns. Use when user asks to add/create a new retrieval technique.
allowed-tools: Read, Write, Bash
context: fork
agent: general-purpose
---

# RAG Technique Creator

## Overview
This skill helps create new RAG techniques that follow our project's design patterns.

## When to Use
- User wants to add a new retrieval technique
- Implementing a novel RAG approach
- Extending the technique registry

## Process
1. Read existing techniques in `techniques/` for patterns
2. Create new technique file implementing BaseTechnique protocol
3. Add YAML entry to `config/techniques.yaml`
4. Create corresponding unit tests
5. Update documentation

## Template
[See @templates/technique_template.py for the base structure]
```

**Key Features:**
- **Progressive disclosure:** Metadata loaded at startup, full content on-demand
- **Cross-platform:** Works in Claude Code, Claude.ai, Claude Desktop
- **Can include scripts:** Python scripts in `scripts/`, templates in `templates/`
- **Can run in subagents:** `context: fork` creates isolated execution

---

### 4. Rules Files (.claude/rules/)
**Purpose:** Modular, conditional rules that extend CLAUDE.md

**Can be:**
- **Unconditional:** Apply to all files
- **Conditional:** Only apply when working with specific file patterns

**Example: `.claude/rules/testing.md`**
```markdown
---
paths:
  - "tests/**/*.py"
---

# Testing Standards

When working with test files:
- Use pytest fixtures for setup/teardown
- Follow AAA pattern (Arrange, Act, Assert)
- Name tests descriptively: `test_should_<expected_behavior>_when_<condition>()`
- Mock external dependencies
```

---

### 5. Slash Commands (.claude/commands/)
**Purpose:** Saved prompts as shortcuts (now merged into skills)

**Example: `.claude/commands/create-technique.md`**
```markdown
---
description: Create a new RAG technique
argument-hint: [technique-name]
---

Create a new RAG technique named "$ARGUMENTS" following our project standards:
1. Implement BaseTechnique protocol in `techniques/$ARGUMENTS.py`
2. Add entry to `config/techniques.yaml`
3. Create tests in `tests/test_techniques/test_$ARGUMENTS.py`
4. Follow SOLID principles (especially SRP and DI)
```

**Usage:** `/create-technique hyde-advanced`

---

## Memory Hierarchy (Load Order)

Claude Code loads files in this priority order:

1. **Managed/Organization level** (if configured via MDM/Group Policy)
2. **User global:** `~/.claude/CLAUDE.md`
3. **Project ancestors:** Any CLAUDE.md files from current dir up to root
4. **Project level:** `{project}/.claude/CLAUDE.md`
5. **Project local:** `{project}/.claude/CLAUDE.local.md`
6. **Project rules:** `{project}/.claude/rules/*.md` (conditional or unconditional)
7. **Skill metadata:** All SKILL.md frontmatter (full content loaded on-demand)

**Files higher in the hierarchy take precedence**

---

## Special Syntax & Features

### Importing Files
Reference other files from CLAUDE.md:

```markdown
See @README.md for project overview
See @package.json for available npm commands

# Git Workflow
@docs/git-instructions.md
```

### Running Commands Before Loading
In SKILL.md, use `!` syntax to run commands and inject output:

```markdown
---
name: pr-summary
description: Summarize pull request changes
---

## PR Context
- Diff: !`git diff HEAD~1`
- Changed files: !`git diff --name-only HEAD~1`
- Branch: !`git branch --show-current`

Summarize these changes...
```

### Using Arguments
In commands/skills:

```markdown
Fix issue #$ARGUMENTS following our coding standards
```

### Conditional Rules
In `.claude/rules/{rule}.md`:

```yaml
---
paths:
  - "src/api/**/*.ts"
  - "src/services/**/*.ts"
---
```

---

## Best Practices for Your RAG Project

Based on your Scientific Agentic RAG project, here's my recommendation:

### Create These Files:

1. **`.claude/CLAUDE.md`** (main configuration)
```markdown
# Scientific Agentic RAG Framework

## Overview
Multi-hop reasoning system using PocketFlow for orchestration.

## Architecture
- **Registry Pattern:** RAG techniques loaded from `config/techniques.yaml`
- **SOLID Principles:** SRP and DI prioritized
- **Duck Typing:** All techniques follow BaseTechnique protocol
- **PocketFlow:** Node-based workflows with prep/exec/post pattern

## Common Commands
- Run pipeline: `python main.py --config config/pipelines.yaml --query "your query"`
- Run tests: `pytest tests/`
- Evaluate: `python -m evaluation.evaluate --dataset benchmarks/scientific_qa.json`

## Key Design Patterns
- Registry: Dynamic technique loading from YAML
- Strategy: Swappable retrieval strategies
- Chain of Responsibility: Retrieve → Rerank → Compress pipeline
- Decorator: Wrapping retrievers with enhancements

## Coding Standards
- Python 3.13+ type hints
- No try-except in utility functions (PocketFlow retry handles)
- Each RAG technique: single responsibility
- Dependency injection via YAML config

## Project Structure
- `techniques/`: Standalone RAG technique implementations
- `config/`: YAML configuration files
- `nodes/`: PocketFlow node definitions
- `flows/`: PocketFlow flow compositions
- `registry/`: Technique registry implementation

## Context from Other LLM
For detailed architecture decisions and trade-offs, see @.claude/architecture-notes.md
```

2. **`.claude/architecture-notes.md`** (your LLM context)
```markdown
# Architecture Decisions & Context

[Paste your context from the other LLM here]

## Key Decisions
...

## Trade-offs Considered
...

## Implementation Notes
...
```

3. **`.claude/skills/technique-creator/SKILL.md`** (for creating techniques)
```markdown
---
name: technique-creator
description: Creates new RAG techniques following project patterns
---

When creating a new RAG technique:
1. Read @techniques/base.py for the protocol
2. Create standalone .py file in `techniques/`
3. Implement BaseTechnique protocol (duck typing)
4. Add YAML entry to `config/techniques.yaml`
5. Create unit tests in `tests/test_techniques/`
6. Follow SRP - single responsibility only
```

4. **`.claude/commands/add-technique.md`** (quick command)
```markdown
---
description: Add a new RAG technique
argument-hint: [technique-name]
---

Create RAG technique "$ARGUMENTS" following our registry pattern and SOLID principles.
```

---

## Quick Reference: When to Use Each File

| Scenario | Use This File | Example |
|----------|---------------|---------|
| Project setup instructions | `CLAUDE.md` | Architecture, tech stack, commands |
| Personal IDE preferences | `CLAUDE.local.md` | "Always use ruff instead of flake8" |
| Context from another LLM | Create `.claude/llm-context.md`, reference in CLAUDE.md | Architecture decisions, trade-offs |
| Reusable workflow | `SKILL.md` | Creating techniques, running evaluations |
| Quick command shortcut | `.claude/commands/{name}.md` | `/add-technique`, `/run-eval` |
| API-specific rules | `.claude/rules/api-standards.md` with `paths:` | Validation, error handling |
| Testing conventions | `.claude/rules/testing.md` with `paths:` | Pytest patterns, fixtures |

---

## Checking What's Loaded

To see what memory files Claude has loaded in your current session:

```bash
claude

# Then in the conversation:
> /memory
```

This shows:
- All loaded CLAUDE.md files
- All loaded rules
- Available skills
- Active configuration

---

## Summary

**For your RAG project context from another LLM:**
1. ✅ Create `.claude/architecture-notes.md` with the context
2. ✅ Reference it in `.claude/CLAUDE.md` using `@.claude/architecture-notes.md`
3. ✅ This makes it persistent, version-controlled, and automatically available

**No slash commands needed** - Claude automatically loads all `.claude/CLAUDE.md` and referenced files at session start!

The file hierarchy ensures your context is always available without manual intervention.
