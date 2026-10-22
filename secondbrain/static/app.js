// Second Brain OS front-end. Plain JS, no build step, no network access beyond this local server.
(function () {
  "use strict";
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const esc = window.Markdown.escape;
  const KIND_LABEL = { note: "Note", pdf: "PDF", bookmark: "Bookmark", web: "Web page", doc: "Document" };
  const KIND_ICON = { note: "i-note", pdf: "i-pdf", bookmark: "i-bookmark", web: "i-web", doc: "i-doc" };
  const icon = (id, style = "") => `<svg class="i" style="${style}"><use href="#${id}"/></svg>`;
  const kindBadge = (k) => `<span class="kind">${icon(KIND_ICON[k] || "i-note")}${esc(KIND_LABEL[k] || k)}</span>`;

  const state = { status: null, view: null, graph: null, graphData: null, history: [], asking: false,
                  docs: null, docKind: "", searchKind: "", readerStack: [], wasBusy: false };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opts });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch {}
      throw new Error(msg);
    }
    return res.json();
  }

  function toast(text, ms = 3500) {
    const t = document.createElement("div");
    t.className = "toast card";
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                  set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };

  // ================================================================ navigation

  function show(view) {
    if (!document.getElementById(`view-${view}`)) view = "ask";
    state.view = view;
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
    $$("[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
    if (location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
    if (view === "graph") loadGraph();
    if (view === "connections") loadConnections();
    if (view === "library") loadLibrary();
    if (view === "search") setTimeout(() => $("#search-input").focus(), 0);
    if (view === "ask") setTimeout(() => $("#ask-input").focus(), 0);
  }
  $$("[data-view]").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));

  document.addEventListener("keydown", (ev) => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
    if (ev.key === "/" && !typing) { ev.preventDefault(); show("search"); }
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") { ev.preventDefault(); show("search"); }
    if (ev.key === "Escape" && $("#reader").classList.contains("open") && !typing) closeReader();
  });

  // ================================================================ theme

  const themes = ["auto", "light", "dark"];
  function applyTheme(t) {
    if (t === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    $("#theme-toggle span").textContent = `Theme: ${t}`;
    if (state.graph) state.graph.readColors();
  }
  let theme = store.get("theme") || "auto";
  applyTheme(theme);
  $("#theme-toggle").addEventListener("click", () => {
    theme = themes[(themes.indexOf(theme) + 1) % themes.length];
    store.set("theme", theme);
    applyTheme(theme);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.graph && state.graph.readColors());

  // ================================================================ status

  async function pollStatus() {
    try {
      const s = await api("/api/status");
      state.status = s;
      renderStatus(s);
      if (state.wasBusy && !s.busy) onIndexFinished(s);
      state.wasBusy = s.busy;
    } catch (e) {
      $("#status-index").textContent = "Server unreachable";
      $("#dot-index").className = "dot warn";
    }
    setTimeout(pollStatus, state.wasBusy ? 1000 : 8000);
  }

  function renderStatus(s) {
    const p = s.progress || {};
    if (s.busy) {
      $("#dot-index").className = "dot busy";
      $("#status-index").textContent = p.stage === "indexing" && p.total ? `Indexing ${p.current + 1}/${p.total}` : `${cap(p.stage || "working")}…`;
    } else {
      $("#dot-index").className = s.documents ? "dot ok" : "dot warn";
      $("#status-index").textContent = s.documents ? `${s.documents.toLocaleString()} documents indexed` : "Nothing indexed yet";
    }
    const lexical = s.embedder.id.startsWith("hashing");
    $("#dot-embed").className = lexical ? "dot warn" : "dot ok";
    $("#status-embed").textContent = lexical ? "Lexical embeddings (add a model for semantic search)" : `Embeddings: ${s.embedder.description}`;
    $("#dot-llm").className = s.llm ? "dot ok" : "dot warn";
    $("#status-llm").textContent = s.llm ? `Answers: ${s.llm}` : "Extractive answers (no local LLM running)";
    $("#nav-docs").textContent = s.documents || "";
    $("#nav-conn").textContent = s.discovered || "";
    $("#ask-hint").textContent = s.llm
      ? `Answered locally by ${s.llm} · Enter to ask · Shift+Enter for a new line`
      : "No local LLM found: answers are quoted from your files. Start Ollama for written answers.";
    if (state.view === "library") renderProgress(s);
    if (!state.kindsRendered && s.by_kind) renderKindChips(s.by_kind);
  }
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function onIndexFinished(s) {
    state.graphData = null; state.docs = null; state.connections = null;
    if (s.last_report) {
      const r = s.last_report;
      toast(`Indexed: ${r.added} added, ${r.updated} updated, ${r.removed} removed · ${r.graph.discovered ?? 0} discovered connections`);
    }
    renderKindChips(s.by_kind);
    loadSuggestions(true);
    if (state.view === "graph") loadGraph();
    if (state.view === "connections") loadConnections();
    if (state.view === "library") loadLibrary();
  }

  // ================================================================ ask

  const askInput = $("#ask-input");
  askInput.addEventListener("input", () => { askInput.style.height = "auto"; askInput.style.height = Math.min(askInput.scrollHeight, 160) + "px"; });
  askInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $("#ask-form").requestSubmit(); }
  });
  $("#ask-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = askInput.value.trim();
    if (q && !state.asking) { askInput.value = ""; askInput.style.height = "auto"; ask(q); }
  });

  async function loadSuggestions(force) {
    if (state.suggested && !force) return;
    state.suggested = true;
    const box = $("#ask-suggestions");
    let qs = [];
    try {
      const g = state.graphData || (state.graphData = await api("/api/graph"));
      // one question per cluster, about its best-connected note (dates and bookmarks make poor topics)
      for (const c of g.clusters.filter((c) => c.size >= 2).slice(0, 4)) {
        const hub = g.nodes.filter((n) => n.cluster === c.id && n.kind !== "bookmark" && !/^\d{4}-\d{2}-\d{2}/.test(n.title))
          .sort((a, b) => b.degree - a.degree)[0];
        if (hub) qs.push(`What do my notes say about ${hub.title.replace(/^(Book|Meeting|Paper):\s*/i, "").replace(/\s*\(.*\)$/, "")}?`);
      }
      if (g.nodes.length) qs.push("Which ideas show up across very different notes?");
    } catch {}
    if (!qs.length) qs = ["What did I decide in my last project meeting?", "Summarize what I know about habits", "Which papers mention feedback loops?"];
    box.innerHTML = qs.map((q) => `<button class="chip">${esc(q)}</button>`).join("");
    $$(".chip", box).forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
  }

  async function ask(question) {
    state.asking = true;
    $("#ask-send").disabled = true;
    $("#ask-hero")?.remove();
    const inner = $("#thread-inner");
    const block = document.createElement("div");
    block.innerHTML = `<div class="msg-user">${esc(question)}</div>
      <div class="answer-meta"><span class="spinner"></span><span>Searching your files…</span></div>
      <div class="answer md"></div><div class="sources"></div>`;
    inner.appendChild(block);
    const meta = $(".answer-meta", block), body = $(".answer", block), srcBox = $(".sources", block);
    const scroll = () => { const t = $("#thread"); t.scrollTop = t.scrollHeight; };
    scroll();

    let text = "", sources = [], mode = "", pending = false;
    const nums = () => new Set(sources.map((s) => s.n));
    const paint = () => {
      pending = false;
      body.innerHTML = window.Markdown.render(text, { citations: nums(), shiftHeadings: 2 });
      wireCitations(body, sources);
      scroll();
    };
    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: state.history.slice(-4) }) });
      if (!res.ok) throw new Error(res.statusText);
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
          if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.type === "sources") {
            sources = ev.sources;
            renderSources(srcBox, sources);
            meta.innerHTML = `<span class="spinner"></span><span>Reading ${sources.length} passages…</span>`;
          } else if (ev.type === "meta") {
            mode = ev.mode;
            meta.innerHTML = mode === "llm" ? `<span class="spinner"></span><span class="badge">${esc(ev.model)}</span><span>writing…</span>`
              : mode === "extractive" ? `<span class="badge">Quoted from your files</span><span>No local LLM running: showing the most relevant sentences.</span>` : "";
          } else if (ev.type === "delta") {
            text += ev.text;
            if (!pending) { pending = true; requestAnimationFrame(paint); }
          } else if (ev.type === "error") {
            throw new Error(ev.message);
          }
        }
      }
      paint();
      if (mode === "llm") meta.innerHTML = `<span class="badge">${esc(meta.querySelector(".badge")?.textContent || "local model")}</span><span>answered on this machine</span>`;
      const cited = new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => +m[1]));
      $$(".source", srcBox).forEach((el) => el.classList.toggle("dim", cited.size > 0 && !cited.has(+el.dataset.n)));
      state.history.push({ role: "user", content: question }, { role: "assistant", content: text });
    } catch (e) {
      meta.innerHTML = "";
      body.innerHTML = `<div class="notice error">Something went wrong: ${esc(e.message)}</div>`;
    } finally {
      state.asking = false;
      $("#ask-send").disabled = false;
      askInput.focus();
    }
  }

  function renderSources(box, sources) {
    box.innerHTML = sources.map((s) => `
      <div class="source card" data-n="${s.n}">
        <span class="n">${s.n}</span>
        <div class="t">${esc(s.title)}</div>
        <div class="loc">${kindBadge(s.kind)} ${esc(locLabel(s))}</div>
      </div>`).join("");
    $$(".source", box).forEach((el) => {
      const s = sources.find((x) => x.n === +el.dataset.n);
      el.addEventListener("click", () => openDoc(s.doc_id, { chunk: s }));
      el.addEventListener("mouseenter", (ev) => showCitePop(s, el));
      el.addEventListener("mouseleave", hideCitePop);
    });
  }
  const locLabel = (s) => [s.heading, s.page ? `p. ${s.page}` : ""].filter(Boolean).join(" · ");

  // Plain text for previews: drop Markdown syntax, and bookmark URL / folder lines.
  function cleanText(text, kind) {
    let t = text || "";
    if (kind === "bookmark") t = t.split("\n").filter((l, i) => i > 0 && !/^(https?:\/\/|Folder:)/.test(l)).join(" ") || text.split("\n")[0];
    return t.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, a, b) => b || a).replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/(\*\*|__|`)/g, "").replace(/^#{1,6}\s+/gm, "").replace(/\s+/g, " ").trim();
  }

  function wireCitations(el, sources) {
    $$(".cite", el).forEach((b) => {
      const s = sources.find((x) => x.n === +b.dataset.n);
      if (!s) return;
      b.addEventListener("click", () => openDoc(s.doc_id, { chunk: s }));
      b.addEventListener("mouseenter", () => showCitePop(s, b));
      b.addEventListener("mouseleave", hideCitePop);
    });
  }

  function showCitePop(s, anchor) {
    const pop = $("#cite-pop");
    const plain = cleanText(s.text, s.kind);
    const text = plain.length > 420 ? plain.slice(0, 420) + "…" : plain;
    pop.innerHTML = `<div class="t">${esc(s.title)}</div><div class="muted" style="font-size:11.5px;margin-bottom:4px">${esc(locLabel(s))}</div><div>${esc(text)}</div>`;
    pop.classList.remove("hidden");
    const r = anchor.getBoundingClientRect(), pr = pop.getBoundingClientRect();
    let top = r.top - pr.height - 8;
    if (top < 8) top = r.bottom + 8;
    pop.style.top = `${top}px`;
    pop.style.left = `${Math.max(8, Math.min(r.left, innerWidth - pr.width - 8))}px`;
  }
  const hideCitePop = () => $("#cite-pop").classList.add("hidden");

  // ================================================================ search

  function renderKindChips(byKind) {
    state.kindsRendered = true;
    const kinds = Object.keys(byKind || {}).sort();
    const chips = (current) => [`<button class="chip ${current ? "" : "active"}" data-kind="">All</button>`,
      ...kinds.map((k) => `<button class="chip ${current === k ? "active" : ""}" data-kind="${k}">${esc(KIND_LABEL[k] || k)} <span class="muted">${byKind[k]}</span></button>`)].join("");
    $("#search-kinds").innerHTML = chips(state.searchKind);
    $$("#search-kinds .chip").forEach((c) => c.addEventListener("click", () => { state.searchKind = c.dataset.kind; renderKindChips(byKind); runSearch(); }));
    $("#doc-kinds").innerHTML = chips(state.docKind);
    $$("#doc-kinds .chip").forEach((c) => c.addEventListener("click", () => { state.docKind = c.dataset.kind; renderKindChips(byKind); renderDocs(); }));
  }

  function highlight(text, q) {
    const terms = q.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || [];
    let html = esc(text);
    if (!terms.length) return html;
    const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
    return html.replace(re, "<mark>$1</mark>");
  }

  let searchSeq = 0;
  const runSearch = debounce(async () => {
    const q = $("#search-input").value.trim();
    const box = $("#search-results");
    if (!q) {
      box.innerHTML = `<div class="empty">${icon("i-search", "width:32px;height:32px")}<h2>Search everything you know</h2><p>Finds passages by meaning, not just exact words. Try a question or a half-remembered idea.</p></div>`;
      return;
    }
    const seq = ++searchSeq;
    box.innerHTML = `<div class="muted"><span class="spinner"></span> Searching…</div>`;
    try {
      const params = new URLSearchParams({ q, k: 30 });
      if (state.searchKind) params.set("kind", state.searchKind);
      const { results } = await api(`/api/search?${params}`);
      if (seq !== searchSeq) return;
      if (!results.length) { box.innerHTML = `<div class="empty"><h2>No matches</h2><p>Try different words, or add more sources in the Library.</p></div>`; return; }
      const groups = new Map();
      for (const r of results) { if (!groups.has(r.doc_id)) groups.set(r.doc_id, []); groups.get(r.doc_id).push(r); }
      box.innerHTML = [...groups.values()].map((hits, gi) => {
        const h = hits[0];
        const why = [h.keyword ? "keyword" : "", h.similarity > 0 ? "meaning" : ""].filter(Boolean).join(" + ");
        return `<div class="result card" data-g="${gi}">
          <div class="result-head"><span class="result-title">${esc(h.title)}</span><span class="result-loc">${esc(locLabel(h))}</span>
            <span class="result-why">${kindBadge(h.kind)}<span>${esc(why)}</span></span></div>
          <div class="result-snippet">${highlight(cleanText(h.snippet, h.kind), q)}</div>
          ${hits.slice(1).map((x) => `<div class="result-more" data-c="${x.chunk_id}">${x.heading || x.page ? `<b>${esc(locLabel(x))}</b> · ` : ""}${highlight(cleanText(x.snippet, x.kind).slice(0, 180), q)}…</div>`).join("")}
        </div>`;
      }).join("");
      const groupList = [...groups.values()];
      $$(".result", box).forEach((el) => el.addEventListener("click", (ev) => {
        const hits = groupList[+el.dataset.g];
        const more = ev.target.closest(".result-more");
        const hit = more ? hits.find((x) => x.chunk_id === +more.dataset.c) : hits[0];
        openDoc(hit.doc_id, { chunk: hit });
      }));
    } catch (e) {
      box.innerHTML = `<div class="notice error">${esc(e.message)}</div>`;
    }
  }, 220);
  $("#search-input").addEventListener("input", runSearch);

  // ================================================================ graph

  async function loadGraph() {
    if (!state.graph) {
      state.graph = new window.ForceGraph($("#graph-canvas"), {
        onHover: (n, x, y) => {
          const tip = $("#graph-tip");
          if (!n) { tip.classList.add("hidden"); return; }
          const cl = state.graphData.clusters.find((c) => c.id === n.cluster);
          tip.innerHTML = `<div class="t">${esc(n.title)}</div><div class="muted">${esc(KIND_LABEL[n.kind] || n.kind)} · ${n.nbrs.size} connection${n.nbrs.size === 1 ? "" : "s"}${cl ? ` · ${esc(cl.label)}` : ""}</div>${n.tags.length ? `<div style="margin-top:4px">${n.tags.slice(0, 6).map((t) => `<span class="tag">#${esc(t)}</span>`).join("")}</div>` : ""}`;
          tip.classList.remove("hidden");
          const w = $("#view-graph").clientWidth;
          tip.style.left = `${Math.min(x + 14, w - 290)}px`;
          tip.style.top = `${y + 14}px`;
        },
        onClick: (n) => {
          state.graph.selected = n;
          if (n) openDoc(n.id, { fromGraph: true });
        },
      });
      const bind = (id, key) => $(id).addEventListener("change", (e) => { state.graph.filter[key] = e.target.checked; state.graph.reheat(0.3); });
      bind("#g-links", "link"); bind("#g-semantic", "semantic"); bind("#g-discover", "discovered"); bind("#g-labels", "labels");
      $("#g-zoom-in").addEventListener("click", () => state.graph.zoomAt(1.3));
      $("#g-zoom-out").addEventListener("click", () => state.graph.zoomAt(1 / 1.3));
      $("#g-fit").addEventListener("click", () => state.graph.fit());
      $("#graph-find").addEventListener("input", (e) => state.graph.setSearch(e.target.value));
      $("#graph-find").addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        const m = state.graph.setSearch(e.target.value);
        if (m.length) state.graph.focusNode(m[0].id);
      });
      const panel = $("#graph-panel"), toggle = $("#graph-panel-toggle");
      const setCollapsed = (c) => { panel.classList.toggle("collapsed", c); toggle.setAttribute("aria-expanded", String(!c)); };
      setCollapsed(innerWidth <= 820);
      toggle.addEventListener("click", () => setCollapsed(!panel.classList.contains("collapsed")));
      $("#shape-key").innerHTML = ["note", "pdf", "bookmark", "web"].map((k) => `<span>${shapeSvg(k)}${KIND_LABEL[k]}</span>`).join("");
    }
    if (state.graphLoadedFor === state.graphData && state.graphData) return;
    try {
      const data = state.graphData || (state.graphData = await api("/api/graph"));
      const sizes = Object.fromEntries(data.clusters.map((c) => [c.id, c.size]));
      state.graph.setData(data, sizes);
      state.graphLoadedFor = data;
      $("#graph-empty").classList.toggle("hidden", data.nodes.length > 0);
      $("#graph-panel").classList.toggle("hidden", !data.nodes.length);
      renderLegend(data);
    } catch (e) { toast(`Could not load graph: ${e.message}`); }
  }

  function shapeSvg(kind) {
    const s = { circle: '<circle cx="6" cy="6" r="4.5"/>', square: '<rect x="2" y="2" width="8" height="8"/>',
      diamond: '<path d="M6 .8 11.2 6 6 11.2.8 6z"/>', triangle: '<path d="M6 1.2 11 10H1z"/>' }[window.GRAPH_SHAPES[kind]];
    return `<svg width="12" height="12" viewBox="0 0 12 12" style="fill:var(--text-3)">${s}</svg>`;
  }

  function renderLegend(data) {
    const clusters = data.clusters.filter((c) => c.size >= 2);
    const colored = clusters.filter((c) => c.id < window.CLUSTER_VARS.length);
    const otherCount = data.nodes.length - colored.reduce((a, c) => a + c.size, 0);
    const g = state.graph;
    $("#graph-legend").innerHTML = colored.map((c) => `
      <div class="legend-item ${g.filter.hiddenClusters.has(c.id) ? "off" : ""}" data-c="${c.id}" title="Click to hide/show · double-click to show only this">
        <span class="sw" style="background:var(${window.CLUSTER_VARS[c.id]})"></span><span class="lbl">${esc(c.label)}</span><span class="ct">${c.size}</span></div>`).join("")
      + (otherCount > 0 ? `<div class="legend-item" style="cursor:default"><span class="sw" style="background:var(--series-other)"></span><span class="lbl">Other / unclustered</span><span class="ct">${otherCount}</span></div>` : "");
    $$("#graph-legend [data-c]").forEach((el) => {
      const id = +el.dataset.c;
      el.addEventListener("click", () => {
        g.filter.hiddenClusters.has(id) ? g.filter.hiddenClusters.delete(id) : g.filter.hiddenClusters.add(id);
        renderLegend(data); g.reheat(0.3);
      });
      el.addEventListener("dblclick", () => {
        const all = [...new Set(data.nodes.map((n) => n.cluster))];
        const onlyThis = g.filter.hiddenClusters.size === all.length - 1 && !g.filter.hiddenClusters.has(id);
        g.filter.hiddenClusters = onlyThis ? new Set() : new Set(all.filter((c) => c !== id));
        // nodes recolored to -1 (singletons) must be hidden too when isolating
        if (!onlyThis) g.filter.hiddenClusters.add(-1);
        renderLegend(data); g.reheat(0.3); setTimeout(() => g.fit(), 400);
      });
    });
  }

  // ================================================================ connections

  async function loadConnections() {
    const box = $("#conn-list");
    if (state.connections) return;
    box.innerHTML = `<div class="muted"><span class="spinner"></span> Loading…</div>`;
    try {
      const { connections } = await api("/api/connections");
      state.connections = connections;
      if (!connections.length) {
        box.innerHTML = `<div class="empty">${icon("i-spark", "width:32px;height:32px")}<h2>No hidden connections yet</h2><p>They appear once you have enough documents that talk about related things without linking to each other.</p></div>`;
        return;
      }
      const sims = connections.map((c) => c.similarity).sort((a, b) => a - b);
      const strength = (s) => (s >= sims[Math.floor(sims.length * 0.67)] ? 3 : s >= sims[Math.floor(sims.length * 0.33)] ? 2 : 1);
      box.innerHTML = connections.map((c, i) => {
        const [ea, eb] = c.evidence;
        const st = strength(c.similarity);
        return `<div class="conn card" data-i="${i}">
          <div class="conn-pair">
            <div class="conn-doc" data-side="a">${kindBadge(c.a.kind)}<div class="t">${esc(c.a.title)}</div><div class="q">${ea ? "“" + esc(cleanText(ea.text, c.a.kind)) + "”" : ""}</div></div>
            <div class="conn-mid" title="cosine similarity ${c.similarity.toFixed(3)}">${icon("i-spark")}<span class="pct">${["", "weak", "medium", "strong"][st]}</span></div>
            <div class="conn-doc" data-side="b">${kindBadge(c.b.kind)}<div class="t">${esc(c.b.title)}</div><div class="q">${eb ? "“" + esc(cleanText(eb.text, c.b.kind)) + "”" : ""}</div></div>
          </div>
          <div class="conn-foot">${c.terms.length ? `<span>Bridged by</span>${c.terms.map((t) => `<span class="term">${esc(t)}</span>`).join("")}` : "<span>Similar in meaning</span>"}
            <button class="btn small" data-act="graph">${icon("i-target", "width:14px;height:14px")}Show in graph</button></div>
        </div>`;
      }).join("");
      $$(".conn", box).forEach((el) => {
        const c = connections[+el.dataset.i];
        $$(".conn-doc", el).forEach((d) => d.addEventListener("click", () => {
          const side = d.dataset.side, ev = c.evidence[side === "a" ? 0 : 1];
          openDoc(c[side].id, { chunk: ev ? { chunk_id: ev.chunk_id, text: ev.text, heading: ev.heading, page: ev.page } : null });
        }));
        $("[data-act=graph]", el).addEventListener("click", async () => {
          show("graph");
          await loadGraph();
          state.graph.focusNode(c.a.id, 1.3);
          state.graph.searchMatches = new Set([state.graph.byId.get(c.a.id), state.graph.byId.get(c.b.id)].filter(Boolean));
        });
      });
    } catch (e) { box.innerHTML = `<div class="notice error">${esc(e.message)}</div>`; }
  }

  // ================================================================ library

  async function loadLibrary() {
    const s = state.status;
    if (s) renderLibraryStats(s);
    if (!state.docs) {
      try { state.docs = (await api("/api/documents")).documents; } catch (e) { state.docs = []; }
    }
    renderDocs();
  }

  function renderLibraryStats(s) {
    const tiles = [["Documents", s.documents], ["Passages", s.chunks], ["Words", s.words], ["Discovered links", s.discovered]];
    $("#lib-stats").innerHTML = tiles.map(([l, v]) => `<div class="stat card"><div class="v">${(v || 0).toLocaleString()}</div><div class="l">${l}</div></div>`).join("");
    $("#root-list").innerHTML = s.roots.length ? s.roots.map((r) => `<div class="root card">${icon("i-folder", "width:15px;height:15px;color:var(--text-3)")}<span title="${esc(r)}">${esc(r)}</span><button class="btn small" data-root="${esc(r)}">Remove</button></div>`).join("")
      : `<div class="notice">No sources yet. Add a folder of Markdown notes (an Obsidian vault works), a folder of PDFs, or a browser bookmarks export.</div>`;
    $$("#root-list [data-root]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Stop tracking ${b.dataset.root}?\n\nIts documents are removed from the index. Your files are not touched.`)) return;
      try {
        const r = await api("/api/sources/remove", { method: "POST", body: JSON.stringify({ path: b.dataset.root }) });
        toast(`Removed ${r.removed} documents`);
        onIndexFinished(await api("/api/status"));
      } catch (e) { toast(e.message); }
    }));
    renderProgress(s);
  }

  function renderProgress(s) {
    const p = s.progress || {};
    $("#index-progress").classList.toggle("hidden", !s.busy);
    if (s.busy) {
      const pct = p.total ? ((p.current + 1) / p.total) * 100 : 100;
      $("#progress-bar").style.width = `${pct}%`;
      $("#progress-label").textContent = p.file ? `${cap(p.stage)} · ${p.file}` : `${cap(p.stage || "working")}…`;
    }
    const r = s.last_report;
    $("#index-report").innerHTML = !s.busy && r && r.failed.length
      ? `<div class="notice error">${r.failed.length} file(s) could not be indexed:<br>${r.failed.slice(0, 5).map((f) => `${esc(f.path)}: ${esc(f.error)}`).join("<br>")}</div>` : "";
    if (p.stage === "error") $("#index-report").innerHTML = `<div class="notice error">Indexing failed: ${esc(p.error)}</div>`;
  }

  function renderDocs() {
    const docs = state.docs || [];
    const f = $("#doc-filter").value.trim().toLowerCase();
    const rows = docs.filter((d) => (!state.docKind || d.kind === state.docKind)
      && (!f || d.title.toLowerCase().includes(f) || d.tags.some((t) => t.includes(f.replace(/^#/, "")))));
    $("#doc-count").textContent = `${rows.length} of ${docs.length}`;
    $("#doc-rows").innerHTML = rows.slice(0, 500).map((d) => `<tr data-id="${d.id}"><td>${esc(d.title)}</td><td>${kindBadge(d.kind)}</td>
      <td>${d.tags.slice(0, 4).map((t) => `<span class="tag">#${esc(t)}</span>`).join("")}</td><td class="num">${(d.words || 0).toLocaleString()}</td></tr>`).join("")
      + (rows.length > 500 ? `<tr><td colspan="4" class="muted">…and ${rows.length - 500} more. Use the filter.</td></tr>` : "");
    $$("#doc-rows tr[data-id]").forEach((tr) => tr.addEventListener("click", () => openDoc(+tr.dataset.id)));
  }
  $("#doc-filter").addEventListener("input", debounce(renderDocs, 120));

  async function startJob(promise, label) {
    try {
      await promise;
      toast(label);
      state.wasBusy = true;
      setTimeout(pollStatus, 300);
    } catch (e) { toast(e.message, 5000); }
  }

  $("#add-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const path = $("#add-path").value.trim();
    if (!path) return;
    startJob(api("/api/sources", { method: "POST", body: JSON.stringify({ path }) }), `Indexing ${path}…`);
    $("#add-path").value = "";
  });
  $("#reindex-btn").addEventListener("click", () => startJob(api("/api/reindex", { method: "POST" }), "Re-scanning sources…"));

  const dz = $("#dropzone"), fileInput = $("#file-input");
  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); upload(e.dataTransfer.files); });
  fileInput.addEventListener("change", () => { upload(fileInput.files); fileInput.value = ""; });
  function upload(files) {
    if (!files || !files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    startJob(api("/api/upload", { method: "POST", body: fd }), `Uploading ${files.length} file(s) to your local inbox…`);
  }

  // ================================================================ reader

  const reader = $("#reader");
  function closeReader() { reader.classList.remove("open"); state.readerStack = []; if (state.graph) state.graph.selected = null; }
  $("#reader-close").addEventListener("click", closeReader);
  $("#reader-back").addEventListener("click", () => {
    state.readerStack.pop();
    const prev = state.readerStack.pop();
    if (prev) openDoc(prev.id, prev.opts);
  });

  async function openDoc(id, opts = {}) {
    state.readerStack.push({ id, opts });
    $("#reader-back").disabled = state.readerStack.length < 2;
    reader.classList.add("open");
    const body = $("#reader-body");
    body.innerHTML = `<div class="muted"><span class="spinner"></span> Opening…</div>`;
    let doc;
    try { doc = await api(`/api/documents/${id}`); } catch (e) { body.innerHTML = `<div class="notice error">${esc(e.message)}</div>`; return; }

    const chunk = opts.chunk;
    const open = $("#reader-open");
    if (doc.kind === "bookmark") { open.href = doc.url; open.classList.remove("hidden"); }
    else if (doc.has_file) { open.href = `/api/documents/${id}/file${doc.kind === "pdf" && chunk && chunk.page ? `#page=${chunk.page}` : ""}`; open.classList.remove("hidden"); }
    else open.classList.add("hidden");
    $("#reader-graph").onclick = async () => { show("graph"); await loadGraph(); state.graph.focusNode(id); };

    let content = doc.content || "";
    if (doc.kind === "bookmark") content = content.split("\n").filter((l, i) => i > 0 && !/^https?:\/\//.test(l)).join("\n\n");

    const rel = (title, items, render) => items.length ? `<div class="rel-section"><h4>${title}</h4>${items.map(render).join("")}</div>` : "";
    const relItem = (x, sub = "", badge = "") => `<div class="rel-item" data-id="${x.id}">${icon(KIND_ICON[x.kind] || "i-note", "width:15px;height:15px;color:var(--text-3)")}<span class="t">${esc(x.title)}</span>${badge}<span class="sub">${esc(sub)}</span></div>`;

    body.innerHTML = `
      ${kindBadge(doc.kind)}
      <div class="reader-title">${esc(doc.title)}</div>
      <div class="reader-path">${doc.kind === "bookmark" ? `<a href="${esc(doc.url)}" target="_blank" rel="noopener noreferrer">${esc(doc.url)}</a>` : esc(doc.path)}</div>
      ${doc.tags.length ? `<div>${doc.tags.map((t) => `<span class="tag">#${esc(t)}</span>`).join("")}</div>` : ""}
      ${chunk ? `<div class="cited"><div class="l">Cited passage${chunk.heading || chunk.page ? " · " + esc(locLabel(chunk)) : ""}</div>${esc((t => t.length > 600 ? t.slice(0, 600) + "…" : t)(cleanText(chunk.text, doc.kind)))}</div>` : ""}
      <div class="md" id="reader-md">${window.Markdown.render(content, { shiftHeadings: 1, breaks: doc.kind === "pdf" })}</div>
      ${rel("Linked from", doc.backlinks, (x) => relItem(x))}
      ${rel("Links to", doc.outlinks, (x) => relItem(x))}
      ${rel("Similar ideas", doc.related, (x) => relItem(x, (x.terms || []).slice(0, 3).join(", "), x.discovered ? '<span class="new-badge" title="Discovered: you never linked these">new</span>' : ""))}
    `;
    body.scrollTop = 0;
    $$(".rel-item", body).forEach((el) => el.addEventListener("click", () => {
      openDoc(+el.dataset.id);
      if (state.graph && state.view === "graph") state.graph.focusNode(+el.dataset.id, state.graph.view.k);
    }));
    $$("a.wikilink", body).forEach((a) => a.addEventListener("click", async () => {
      try { const r = await api(`/api/resolve?title=${encodeURIComponent(a.dataset.title)}`); openDoc(r.id); }
      catch { toast(`“${a.dataset.title}” isn't in your index yet`); }
    }));
    if (chunk) markPassage($("#reader-md"), chunk);
    if (state.graph && !opts.fromGraph) state.graph.selected = state.graph.byId.get(id) || null;
  }

  // Best-effort: find the cited chunk inside the rendered document and highlight + scroll to it.
  function markPassage(container, chunk) {
    const plain = (s) => s.replace(/\[\[([^\]|]+)(\|([^\]]+))?\]\]/g, (_, a, __, b) => b || a)
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/[*_`#>]+/g, "").replace(/^\s*([-+]|\d+[.)])\s+/gm, "").replace(/\s+/g, " ").trim();
    const text = plain(chunk.text);
    const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => s.length > 20);
    if (!sentences.length) return;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodes = []; let full = "";
    while (walker.nextNode()) { nodes.push({ node: walker.currentNode, start: full.length }); full += walker.currentNode.nodeValue; }
    const toRe = (s) => new RegExp(s.slice(0, 70).trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*"));
    const first = toRe(sentences[0]).exec(full);
    if (!first) return;
    let endIdx = first.index + first[0].length;
    const lastS = sentences[sentences.length - 1];
    const tail = new RegExp(lastS.slice(-50).trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*"));
    const last = tail.exec(full.slice(first.index));
    if (last) endIdx = first.index + last.index + last[0].length;
    const locate = (idx) => { for (let i = nodes.length - 1; i >= 0; i--) if (nodes[i].start <= idx) return [nodes[i].node, idx - nodes[i].start]; return [nodes[0].node, 0]; };
    const range = document.createRange();
    const [sn, so] = locate(first.index), [en, eo] = locate(endIdx);
    range.setStart(sn, Math.min(so, sn.nodeValue.length));
    range.setEnd(en, Math.min(eo, en.nodeValue.length));
    if (window.CSS && CSS.highlights && window.Highlight) {
      CSS.highlights.set("cited", new Highlight(range));
    } else if (sn === en) {
      const m = document.createElement("mark"); m.className = "hit"; range.surroundContents(m);
    }
    const el = sn.parentElement;
    setTimeout(() => el && el.scrollIntoView({ block: "center", behavior: "smooth" }), 60);
  }

  // ================================================================ boot

  show((location.hash || "#ask").slice(1));
  pollStatus();
  loadSuggestions();
  runSearch();
})();
