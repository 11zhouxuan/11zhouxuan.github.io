/**
 * sidenote.js - Custom tag plugin for margin notes / sidenotes
 * 
 * Usage in Markdown (MUST be on same line as surrounding text):
 *   ...text{% sidenote 1 %}Content{% endsidenote %}more text...
 * 
 * Renders as:
 *   - In text: a small superscript number (like footnote markers)
 *   - In right margin: the numbered note content
 */

'use strict';

hexo.extend.tag.register('sidenote', function(args, content) {
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  const num = args[0] || '';
  
  if (num) {
    // Use <label> as inline element for the superscript number,
    // and <span class="sidenote"> for the margin content.
    // The label is inline so it stays attached to surrounding text.
    return `<label class="sidenote-toggle sidenote-number" for="sn-${num}">${num}</label><span class="sidenote"><span class="sidenote-number">${num}</span>${rendered}</span>`;
  } else {
    return `<span class="sidenote">${rendered}</span>`;
  }
}, { ends: true });

// Also register a simpler "marginnote" tag for inline margin notes without numbers
hexo.extend.tag.register('marginnote', function(args, content) {
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  return `<span class="marginnote">${rendered}</span>`;
}, { ends: true });
