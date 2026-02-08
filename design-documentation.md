# Design System Documentation

This document defines the design system for the project. All UI components, layouts, and visual elements must follow these guidelines. The `ui-guidelines` skill enforces compliance with this system.

---

## Color Palette

### Primary Colors

| Name | Value | Usage |
|------|-------|-------|
| primary-50 | `#______` | Light backgrounds, hover states |
| primary-100 | `#______` | Subtle backgrounds |
| primary-500 | `#______` | Primary actions, links, accents |
| primary-600 | `#______` | Primary hover states |
| primary-900 | `#______` | Dark primary text, heavy backgrounds |

### Secondary Colors

| Name | Value | Usage |
|------|-------|-------|
| secondary-50 | `#______` | Light secondary backgrounds |
| secondary-500 | `#______` | Secondary actions, subtle accents |
| secondary-900 | `#______` | Dark secondary backgrounds |

### Semantic Colors

| Name | Value | Usage |
|------|-------|-------|
| success | `#______` | Success states, confirmations |
| warning | `#______` | Warnings, pending states |
| error | `#______` | Errors, destructive actions |
| info | `#______` | Informational messages |

### Neutral Colors

| Name | Value | Usage |
|------|-------|-------|
| gray-50 | `#______` | Page backgrounds |
| gray-100 | `#______` | Card backgrounds |
| gray-200 | `#______` | Borders, dividers |
| gray-400 | `#______` | Disabled text, placeholders |
| gray-600 | `#______` | Secondary text, labels |
| gray-900 | `#______` | Primary text, headings |

---

## Design Tokens

### Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| space-0 | `0` | No spacing |
| space-1 | `0.25rem` (4px) | Tight spacing, icon padding |
| space-2 | `0.5rem` (8px) | Small spacing, compact components |
| space-3 | `0.75rem` (12px) | Default element spacing |
| space-4 | `1rem` (16px) | Standard padding, section spacing |
| space-6 | `1.5rem` (24px) | Card padding, grouped items |
| space-8 | `2rem` (32px) | Section separation, large gaps |
| space-12 | `3rem` (48px) | Major sections, page margins |
| space-16 | `4rem` (64px) | Container margins, hero sections |

### Typography

| Token | Value | Usage |
|-------|-------|-------|
| font-sans | `[font-family]` | Body text, UI elements |
| font-mono | `[font-family]` | Code, technical content |

#### Font Sizes

| Token | Value | Usage |
|-------|-------|-------|
| text-xs | `0.75rem` (12px) | Captions, labels |
| text-sm | `0.875rem` (14px) | Small text, secondary labels |
| text-base | `1rem` (16px) | Body text, default |
| text-lg | `1.125rem` (18px) | Large body text |
| text-xl | `1.25rem` (20px) | Subheadings, cards |
| text-2xl | `1.5rem` (24px) | Section headings |
| text-3xl | `1.875rem` (30px) | Page headings |
| text-4xl | `2.25rem` (36px) | Hero headings |

#### Font Weights

| Token | Value | Usage |
|-------|-------|-------|
| font-normal | `400` | Body text |
| font-medium | `500` | Emphasis, labels |
| font-semibold | `600` | Headings, important text |
| font-bold | `700` | Strong emphasis |

### Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| rounded-none | `0` | No rounding |
| rounded-sm | `0.125rem` (2px) | Small elements, tags |
| rounded-md | `0.375rem` (6px) | Default, cards, buttons |
| rounded-lg | `0.5rem` (8px) | Large cards, modals |
| rounded-xl | `0.75rem` (12px) | Hero elements |
| rounded-full | `9999px` | Pills, badges, avatars |

### Shadows

| Token | Value | Usage |
|-------|-------|-------|
| shadow-sm | Small elevation | Tooltips, dropdowns |
| shadow-md | Medium elevation | Cards, buttons |
| shadow-lg | Large elevation | Modals, drawers |
| shadow-xl | Extra elevation | Popovers, overlays |

### Transitions

| Token | Value | Usage |
|-------|-------|-------|
| transition-fast | `150ms ease-out` | Instant feedback |
| transition-base | `200ms ease-out` | Default animations |
| transition-slow | `300ms ease-out` | Modal enter/exit |

---

## Layout Patterns

### Breakpoints

| Breakpoint | Min Width | Max Width | Usage |
|------------|-----------|-----------|-------|
| xs | `0px` | `639px` | Mobile devices |
| sm | `640px` | `767px` | Large phones, small tablets |
| md | `768px` | `1023px` | Tablets, small laptops |
| lg | `1024px` | `1279px` | Desktops |
| xl | `1280px` | `1535px` | Large desktops |
| 2xl | `1536px`+ | Extra large screens |

### Container Widths

| Token | Value | Usage |
|-------|-------|-------|
| container-sm | `640px` | Narrow content |
| container-md | `768px` | Standard content |
| container-lg | `1024px` | Wide content |
| container-xl | `1280px` | Maximum content |

### Grid Systems

| Token | Columns | Gap | Usage |
|-------|---------|-----|-------|
| grid-2 | 2 columns | `space-4` | Simple 2-column layouts |
| grid-3 | 3 columns | `space-4` | Card grids, feature lists |
| grid-4 | 4 columns | `space-4` | Dashboard widgets |
| grid-6 | 6 columns | `space-4` | Complex dashboards |

---

## Component Patterns

### Buttons

**Primary Button**
- Background: `primary-500`
- Text: `gray-50` or white
- Padding: `space-2` horizontal, `space-3` vertical
- Border radius: `rounded-md`
- Hover: `primary-600` with `transition-base`
- Focus: Ring `primary-500` with `offset-2`

**Secondary Button**
- Background: `gray-100`
- Text: `gray-900`
- Padding: `space-2` horizontal, `space-3` vertical
- Border radius: `rounded-md`
- Hover: `gray-200` with `transition-base`

**Destructive Button**
- Background: `error`
- Text: `gray-50` or white
- Padding: `space-2` horizontal, `space-3` vertical
- Border radius: `rounded-md`
- Hover: Darker error shade with `transition-base`

### Cards

**Standard Card**
- Background: `gray-50`
- Padding: `space-6`
- Border radius: `rounded-lg`
- Shadow: `shadow-md`
- Border: `1px solid gray-200`

### Forms

**Input Fields**
- Background: `gray-50`
- Border: `1px solid gray-200`
- Border radius: `rounded-md`
- Padding: `space-2` horizontal, `space-3` vertical
- Focus: Border `primary-500`, ring `primary-500` with `offset-2`

**Labels**
- Text: `gray-600`
- Font weight: `font-medium`
- Font size: `text-sm`
- Margin bottom: `space-1`

### Navigation

**Header**
- Background: `gray-900`
- Padding: `space-4` horizontal
- Height: `64px`
- Text: `gray-50`

**Sidebar**
- Background: `gray-900`
- Width: `250px`
- Text: `gray-50`

---

## Interaction Standards

### Hover States

- Buttons: Darker shade or opacity change
- Links: Underline appears
- Cards: `shadow-lg` elevation increase
- Interactive rows: Background `gray-100`

### Focus States

- All interactive elements must have visible focus indicator
- Focus ring: `2px solid primary-500` with `offset-2`
- Never remove outline without providing alternative

### Active States

- Buttons: Slightly darker than hover, brief scale transform
- Interactive elements: Immediate visual feedback

### Disabled States

- Buttons: `opacity-50`, no hover effects, `cursor-not-allowed`
- Inputs: `opacity-50`, `cursor-not-allowed`, no focus ring

### Loading States

- Buttons: Spinner icon, text disabled
- Cards: Skeleton loader with `gray-200` background
- Lists: Loading spinner centered

---

## Tailwind Configuration Reference

```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        // Add color palette from above
      },
      spacing: {
        // Add spacing scale from above
      },
      borderRadius: {
        // Add border radii from above
      },
      boxShadow: {
        // Add shadows from above
      },
      transitionDuration: {
        'fast': '150ms',
        'base': '200ms',
        'slow': '300ms',
      },
    },
  },
  screens: {
    // Add breakpoints from above
  },
}
```

---

## Usage Examples

### React Component Example

```tsx
import React from 'react';

export function PrimaryButton({ children, onClick, disabled }) {
  const baseClasses = 'px-2 py-3 rounded-md font-medium transition-base';
  const enabledClasses = 'bg-primary-500 text-gray-50 hover:bg-primary-600 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2';
  const disabledClasses = 'opacity-50 cursor-not-allowed hover:bg-primary-500';

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${disabled ? disabledClasses : enabledClasses}`}
    >
      {children}
    </button>
  );
}
```

### Dash Component Example

```python
import dash.html as html

def create_primary_button(label, id_name, disabled=False):
    return html.Button(
        label,
        id=id_name,
        disabled=disabled,
        style={
            'backgroundColor': '#______',  # primary-500 from palette
            'color': '#______',  # gray-50
            'padding': '8px 16px',  # space-2, space-4
            'borderRadius': '6px',  # rounded-md
            'fontWeight': 500,  # font-medium
            'transition': 'background-color 200ms ease-out',  # transition-base
            'opacity': 0.5 if disabled else 1,
            'cursor': 'not-allowed' if disabled else 'pointer',
        }
    )
```

### Flask Template Example

```html
<button
  class="px-2 py-3 rounded-md font-medium bg-primary-500 text-gray-50 hover:bg-primary-600 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-base"
  {% if disabled %}disabled{% endif %}>
  {{ label }}
</button>
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | [Date] | Initial design system definition |

---

## Notes

- All design decisions should be documented here before implementation
- When adding new components, document their patterns in the Component Patterns section
- Update version history when making significant changes to the design system