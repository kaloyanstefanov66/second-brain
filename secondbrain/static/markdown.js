// Minimal, safe Markdown renderer: input is HTML-escaped first, so notes can never inject markup.
// Supports headings, lists (incl. tasks), quotes, code, tables, links, [[wikilinks]], #tags,
// PDF page markers, and optional citation chips [n].
(function () {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  function inline(text, opts) {
    const codes = [];
    let s = esc(text).replace(/`([^`\n]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
    s = s.replace(/\[\[([^\]|#\n]+)(?:#[^\]|\n]*)?(?:\|([^\]\n]+))?\]\]/g,
      (_, target, alias) => `<a class="wikilink" data-title="${target.trim()}">${(alias || target).trim()}</a>`);
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt) => `<span class="muted">[image: ${alt || "untitled"}]</span>`);
    s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (m, label, href) => {
      if (/^https?:\/\//i.test(href)) return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
      if (/^mailto:/i.test(href)) return `<a href="${href}">${label}</a>`;
      if (!/^[a-z][a-z0-9+.-]*:/i.test(href) && !href.startsWith("#")) {
        const name = decodeURIComponent(href.split("#")[0].split("/").pop().replace(/\.(md|markdown)$/i, ""));
        return `<a class="wikilink" data-title="${name}">${label}</a>`;
      }
      return label;
    });
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
    s = s.replace(/\*\*([^*\n]+)\*\*|__([^_\n]+)__/g, (_, a, b) => `<strong>${a || b}</strong>`);
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)|(^|[^_\w])_([^_\n]+)_(?!\w)/g,
      (_, p1, a, p3, b) => `${p1 ?? p3 ?? ""}<em>${a || b}</em>`);
    s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    s = s.replace(/(^|\s)#([A-Za-z][\w/-]*)/g, '$1<span class="hashtag">#$2</span>');
    if (opts.citations) {
      s = s.replace(/\[(\d{1,2})\]/g, (m, n) => opts.citations.has(+n)
        ? `<button class="cite" data-n="${n}">${n}</button>` : m);
    }
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
  }

  function render(src, opts = {}) {
    src = (src || "").replace(/\r\n?/g, "\n").replace(/^---\n[\s\S]*?\n---\n?/, "");
    const lines = src.split("\n");
    const out = [];
    let i = 0;
    const isBlockStart = (l) => /^(#{1,6})\s|^```|^>\s?|^\s*([-*+]|\d+[.)])\s+|^\s*(---|\*\*\*|___)\s*$|^<!-- page \d+ -->|^\|/.test(l);

    while (i < lines.length) {
      const line = lines[i];
      let m;
      if (!line.trim()) { i++; continue; }
      if ((m = line.match(/^<!-- page (\d+) -->\s*$/))) {
        out.push(`<div class="page-break" data-page="${m[1]}">Page ${m[1]}</div>`); i++; continue;
      }
      if (line.startsWith("```")) {
        const body = [];
        i++;
        while (i < lines.length && !lines[i].startsWith("```")) body.push(lines[i++]);
        i++;
        out.push(`<pre><code>${esc(body.join("\n"))}</code></pre>`);
        continue;
      }
      if ((m = line.match(/^(#{1,6})\s+(.+?)\s*#*$/))) {
        const lvl = Math.min(m[1].length + (opts.shiftHeadings || 0), 6);
        out.push(`<h${lvl}>${inline(m[2], opts)}</h${lvl}>`); i++; continue;
      }
      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
      if (line.startsWith(">")) {
        const body = [];
        while (i < lines.length && lines[i].startsWith(">")) body.push(lines[i++].replace(/^>\s?/, ""));
        out.push(`<blockquote>${render(body.join("\n"), opts)}</blockquote>`);
        continue;
      }
      if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim(), opts));
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
          rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      if ((m = line.match(/^\s*([-*+]|\d+[.)])\s+/))) {
        const ordered = /\d/.test(m[1]);
        const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let item = lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, "");
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) item += " " + lines[i++].trim();
          const task = item.match(/^\[([ xX])\]\s+(.*)$/);
          items.push(task
            ? `<li style="list-style:none;margin-left:-1.2em"><input type="checkbox" disabled ${task[1] !== " " ? "checked" : ""}> ${inline(task[2], opts)}</li>`
            : `<li>${inline(item, opts)}</li>`);
        }
        out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
        continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i]))) para.push(lines[i++]);
      out.push(`<p>${inline(para.join("\n"), opts).replace(/\n/g, opts.breaks ? "<br>" : " ")}</p>`);
    }
    return out.join("\n");
  }

  window.Markdown = { render, escape: esc };
})();
