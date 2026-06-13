# Blog Writing Skill for 11zhouxuan.github.io

This document describes how to write blog posts for this Hexo + NexT theme blog. Any AI agent should follow these rules when creating or editing posts.

## Tech Stack

- **Static Site Generator**: Hexo
- **Theme**: NexT v8.27.0, Mist scheme (top navigation bar)
- **Markdown Renderer**: `hexo-renderer-markdown-it` (does NOT interfere with LaTeX)
- **Math Rendering**: MathJax (enabled per-post via `mathjax: true` in front matter)
- **Deployment**: GitHub Actions → push to `main` → auto-build → deploy to `gh-pages`
- **Custom CSS**: `source/_data/styles.styl`
- **Site URL**: https://11zhouxuan.github.io/

## Post File Location

All posts go in `source/_posts/`. Use kebab-case filenames, e.g., `flow-matching-three-steps.md`.

## Front Matter Template

```yaml
---
title: "Your Title Here"
date: 2025-06-12 10:00:00
tags:
  - tag1
  - tag2
categories:
  - category
mathjax: true
---
```

## Bilingual (EN/ZH) Structure

Use language toggle buttons and `<div>` blocks:

```html
<div class="lang-switch">
  <button id="btn-en" class="lang-btn active" onclick="switchLang('en')">English</button>
  <button id="btn-zh" class="lang-btn" onclick="switchLang('zh')">中文</button>
</div>

<!-- Chinese Version -->
<div class="lang-content lang-zh" style="display:none;">
... Chinese content ...
</div>

<!-- English Version -->
<div class="lang-content lang-en">
... English content ...
</div>
```

Add the `switchLang()` script at the end of the post. Reference implementation:

```html
<script>
function switchLang(lang) {
  document.querySelectorAll('.lang-content').forEach(function(el) {
    el.style.display = 'none';
  });
  document.querySelectorAll('.lang-btn').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelector('.lang-' + lang).style.display = 'block';
  document.getElementById('btn-' + lang).classList.add('active');

  // Switch title
  var titleEl = document.querySelector('.post-title');
  if (titleEl) {
    if (lang === 'zh') {
      titleEl.textContent = '中文标题';
    } else {
      titleEl.textContent = 'English Title';
    }
  }

  // Switch TOC: hide headings from the other language
  var tocLinks = document.querySelectorAll('.post-toc a');
  tocLinks.forEach(function(link) {
    var li = link.closest('li');
    if (!li) return;
    var isChinese = /[\u4e00-\u9fff]/.test(link.textContent);
    if (lang === 'zh') {
      li.style.display = isChinese ? '' : 'none';
    } else {
      li.style.display = isChinese ? 'none' : '';
    }
  });
}

document.addEventListener('DOMContentLoaded', function() {
  switchLang('en');
});
</script>
```

## Math Formula Rules

We use `hexo-renderer-markdown-it` which does NOT interfere with LaTeX syntax. Write standard LaTeX freely:

### What works directly (no escaping needed):
- `$X_0$`, `$p_1$`, `$v_t^\theta$` — underscores work normally
- `\;`, `\,`, `\quad` — all spacing commands work
- `\|`, `\lVert`, `\rVert` — all norm notations work
- `\\` in `aligned` environments — works normally
- `\big\|`, `\Big(` etc. — all delimiter sizing works

### Display math
Use `$$...$$` on its own line:
```
$$\frac{\mathrm{d}}{\mathrm{d}t}Z(t) = v_t(Z(t)), \qquad Z(0) = x_0. \tag{1}$$
```

### Inline math
Use `$...$` for inline math: `$X_0 \sim p_0$`

### Multi-line equations
`aligned` environments with `\\` work:
```
$$\begin{aligned}
a &= b + c \\
d &= e + f
\end{aligned}$$
```

### Equation numbering
Use `\tag{N}` at the end of display math for manual numbering.

## Footnotes

Use traditional bottom-of-article footnotes with anchor links (NOT sidenotes):

**In-text reference:**
```html
<a href="#note-1" style="text-decoration:none"><sup>[1]</sup></a>
```

**At the bottom of the article:**
```html
<a id="note-1"></a>**[1]** Your footnote text here.
```

The CSS removes underlines from these footnote links. Do NOT use markdown-style `[1]` references — they get interpreted as link references by the markdown parser.

## NexT Tag Plugins (Callout Boxes)

Use these for highlighted content blocks:

```
{% note info %}
**Title**: Content here...
{% endnote %}

{% note danger %}
**Warning**: Content here...
{% endnote %}

{% note success %}
**Result**: Content here...
{% endnote %}
```

Available types: `info` (blue), `danger` (red), `success` (green), `warning` (orange), `default` (grey).

## Styling Notes

- **Body font**: Source Serif 4 + Noto Serif SC (serif, academic style)
- **Headings**: Source Sans 3 + Noto Sans SC (sans-serif)
- **Code**: JetBrains Mono
- **Content width**: 780px centered
- **TOC**: Default hidden in sidebar; user clicks hamburger icon to open; `toc.number: false`

## Mobile Considerations

- All content forced to `max-width: 100%` on mobile
- Math formulas have `overflow-x: auto` for horizontal scrolling on small screens
- NEVER use `float: right` or large `padding-right` — these cause "grey right half" on mobile
- NEVER use sidenotes/margin notes — they overflow on mobile

## Deployment

Just push to `main` branch. GitHub Actions will:
1. Run `npx hexo generate`
2. Deploy `./public` to `gh-pages` branch
3. Site is live at https://11zhouxuan.github.io/

No manual build steps needed. Wait ~1-2 minutes after push for deployment to complete.

## Common Pitfalls Summary

| Problem | Cause | Fix |
|---------|-------|-----|
| Footnote has underline | Theme default link style | Already fixed in CSS |
| Mobile right-half grey | `float: right` or padding | Never use side-floats |
| `[1]` disappearing | Markdown interprets as link ref | Wrap in `<a><sup>[1]</sup></a>` |
| Equation tag not showing | Missing `\tag{N}` | Add `\tag{N}` at end of display math |
| Inline math on own line | CSS `display:block` on mjx-container | Fixed — only display math is block |
