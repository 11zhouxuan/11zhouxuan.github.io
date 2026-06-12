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
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  const num = args[0] || '';
  
  if (num) {
    // Inline [N] superscript + margin note with full content
    return `<sup class="sidenote-ref">[${num}]</sup><span class="sidenote"><span class="sidenote-num">[${num}]</span> ${rendered}</span>`;
  } else {
    return `<span class="sidenote">${rendered}</span>`;
  }
}, { ends: true });

// Also register a simpler "marginnote" tag for inline margin notes without numbers
hexo.extend.tag.register('marginnote', function(args, content) {
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  return `<span class="marginnote">${rendered}</span>`;
}, { ends: true });
