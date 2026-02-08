---
name: ui-guidelines
description: Use before implementing ANY UI components - enforces design system consistency including color palettes, tokens, layouts, and interactions across Python, Dash, Flask, React, JavaScript, Tailwind, HTML, CSS
---

<EXTREMELY-IMPORTANT>
This skill MUST be invoked before creating, modifying, or reviewing any UI components, layouts, or visual elements. This is not optional.
</EXTREMELY-IMPORTANT>

# Purpose

This skill ensures all UI work follows the project's design system standards as documented in `design-documentation.md`. It enforces consistency across the entire frontend stack (React/JavaScript) and Python-templated UIs (Dash, Flask, HTML templates).

## When to Use This Skill

Invoke this skill when:
- Creating new UI components (React components, Dash components, Flask templates)
- Modifying existing UI layouts or styling
- Adding new visual patterns or interactions
- Reviewing UI code for design compliance
- Setting up Tailwind configuration or design tokens

## The Workflow

### Step 1: Read Design Documentation

Before any UI work, read `design-documentation.md` to understand:
- Color palette (primary, secondary, semantic colors)
- Design tokens (spacing, typography, shadows, borders)
- Layout patterns (grid systems, breakpoints, container styles)
- Component patterns (buttons, cards, forms, navigation)
- Interaction standards (hover states, focus states, transitions)

### Step 2: Check Current Design State

Before implementing or modifying UI:
1. Examine existing components to find patterns to follow
2. Identify similar components that already exist
3. Check if there's a reusable component pattern available

### Step 3: Apply Guidelines

When creating UI components, ensure:

**Color Palette**
- Use named color variables/tokens, never hardcoded hex values
- Follow semantic color usage (primary for CTAs, destructive for delete actions)
- Maintain proper contrast ratios for accessibility

**Design Tokens**
- Use spacing scale from design system (e.g., space-1, space-4, space-8)
- Follow typography hierarchy (h1-h6, body, caption, etc.)
- Apply consistent border radii, shadows, and transitions

**Layouts**
- Use responsive breakpoints defined in design system
- Follow grid/flex patterns specified in documentation
- Maintain consistent padding and margins

**Interactions**
- Apply standard hover, active, and focus states
- Use consistent transition durations and easing functions
- Follow interaction patterns (e.g., loading states, disabled states)

### Step 4: Validate Against Guidelines

After implementation, verify:
- [ ] All colors use tokens/variables, not hardcoded values
- [ ] Spacing follows the defined scale
- [ ] Typography uses proper semantic hierarchy
- [ ] Components match existing patterns or extend them consistently
- [ ] Responsive behavior matches design system breakpoints
- [ ] Interactive states (hover, focus, active) are implemented
- [ ] Accessibility standards (contrast, keyboard navigation) are met

## Technology-Specific Guidance

### React/JavaScript Components
- Use Tailwind utility classes following design system patterns
- Extract common patterns into reusable components
- Use Tailwind's config for design tokens where appropriate

### Dash (Python)
- Apply design tokens through Dash component properties
- Use consistent spacing, colors, and fonts via component style props
- Map design system patterns to Dash component conventions

### Flask Templates (HTML/CSS)
- Use Tailwind classes consistently with React components
- Reference the same design tokens and patterns
- Ensure template structure matches component patterns

### CSS/Tailwind Configuration
- Define custom colors in Tailwind config from design palette
- Set up spacing scale, border radius, shadows as design tokens
- Configure fonts and typography from design system

## Common Violations to Avoid

- ❌ Hardcoded hex colors (e.g., `#3B82F6`) → Use tokens (e.g., `color-primary`)
- ❌ Arbitrary spacing values (e.g., `mr-[17px]`) → Use scale (e.g., `mr-4`)
- ❌ Inconsistent border radii across components
- ❌ Missing hover/focus states on interactive elements
- ❌ Creating new patterns when existing ones exist
- ❌ Inconsistent typography sizes and weights

## Documentation Requirements

If the design system in `design-documentation.md` is incomplete:
1. Identify what's missing before implementing
2. Ask user to clarify or document the missing guidelines
3. Do not create arbitrary design decisions without documentation

## Skill Completion

After following this skill, you should:
1. Have read and understood the current design system
2. Have a clear pattern to follow for the UI work
3. Be ready to implement with design system compliance
4. Know what to validate after implementation