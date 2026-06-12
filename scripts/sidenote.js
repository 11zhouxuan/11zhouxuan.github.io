/**
 * sidenote.js - Custom tag plugin for margin notes / sidenotes
 * 
 * Usage in Markdown (MUST be on same line as surrounding text):
 *   ...text{% sidenote 1 %}Content{% endsidenote %}more text...
 * 
 * Renders as:
 *   - In text: [1] as superscript
 *   - In right margin: the numbered note content
 */

'use strict';

hexo.extend.tag.register('sidenote', function(args, content) {
  // Do NOT render through markdown - let the page's own renderer handle it
  // Just wrap in the appropriate HTML structure
  const num = args[0] || '';
  // Trim whitespace from content
  const text = content.trim();
  
  if (num) {
    // Inline [N] superscript + margin note with full content
    return `<sup class="sidenote-ref">[${num}]</sup><span class="sidenote"><span class="sidenote-num">[${num}]</span> ${text}</span>`;
  } else {
    return `<span class="sidenote">${text}</span>`;
  }
}, { ends: true });

// Also register a simpler "marginnote" tag for inline margin notes without numbers
hexo.extend.tag.register('marginnote', function(args, content) {
  const text = content.trim();
  return `<span class="marginnote">${text}</span>`;
}, { ends: true });
