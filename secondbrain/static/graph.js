// Interactive force-directed graph on <canvas>. No dependencies.
// Barnes-Hut repulsion (O(n log n)) keeps it smooth for a few thousand notes.
(function () {
  const SHAPES = { note: "circle", pdf: "square", bookmark: "diamond", web: "triangle", doc: "square" };
  // Orange (series-2) is reserved for discovered connections, so clusters use the other seven slots.
  const CLUSTER_VARS = ["--series-1", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];

  class QuadTree {
    constructor(nodes) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const n of nodes) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
      const size = Math.max(x1 - x0, y1 - y0, 1) + 1;
      this.root = { x: x0, y: y0, size, mass: 0, cx: 0, cy: 0, node: null, kids: null };
      for (const n of nodes) this.insert(this.root, n, 0);
    }
    insert(q, n, depth) {
      q.cx = (q.cx * q.mass + n.x) / (q.mass + 1);
      q.cy = (q.cy * q.mass + n.y) / (q.mass + 1);
      q.mass++;
      if (!q.kids && !q.node) { q.node = n; return; }
      if (depth > 40) return; // coincident points
      if (!q.kids) {
        const h = q.size / 2;
        q.kids = [0, 1, 2, 3].map((i) => ({ x: q.x + (i & 1) * h, y: q.y + (i >> 1) * h, size: h, mass: 0, cx: 0, cy: 0, node: null, kids: null }));
        const old = q.node; q.node = null;
        this.insert(this.child(q, old), old, depth + 1);
      }
      this.insert(this.child(q, n), n, depth + 1);
    }
    child(q, n) {
      const h = q.size / 2;
      return q.kids[(n.x >= q.x + h ? 1 : 0) + (n.y >= q.y + h ? 2 : 0)];
    }
    force(n, strength, theta = 0.85) {
      const stack = [this.root];
      while (stack.length) {
        const q = stack.pop();
        if (!q.mass || q.node === n) continue;
        const dx = q.cx - n.x, dy = q.cy - n.y;
        let d2 = dx * dx + dy * dy;
        if (q.kids && (q.size * q.size) / d2 > theta * theta) { stack.push(...q.kids); continue; }
        if (d2 < 1) d2 = 1;
        const f = (strength * q.mass) / d2;
        n.vx -= dx * f; n.vy -= dy * f;
      }
    }
  }

  class ForceGraph {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.opts = opts;
      this.nodes = []; this.edges = []; this.byId = new Map();
      this.view = { x: 0, y: 0, k: 1 };
      this.alpha = 0;
      this.hover = null; this.selected = null;
      this.filter = { link: true, semantic: true, discovered: true, labels: true, hiddenClusters: new Set() };
      this.colors = {};
      this.readColors();
      this.bindEvents();
      new ResizeObserver(() => this.resize()).observe(canvas);
      const loop = () => { this.tick(); this.draw(); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }

    readColors() {
      const cs = getComputedStyle(document.documentElement);
      const v = (name) => cs.getPropertyValue(name).trim();
      this.colors = {
        series: CLUSTER_VARS.map(v), other: v("--series-other"),
        edge: v("--edge"), edgeLink: v("--edge-link"), discover: v("--discover"),
        text: v("--text"), text2: v("--text-2"), surface: v("--surface"), bg: v("--bg"), accent: v("--accent"),
      };
      this.dirty = true;
    }

    clusterColor(c) { return c >= 0 && c < CLUSTER_VARS.length ? this.colors.series[c] : this.colors.other; }

    setData({ nodes, edges }, clusterSizes = {}) {
      const old = this.byId;
      this.byId = new Map();
      this.nodes = nodes.map((n, i) => {
        const prev = old.get(n.id);
        const a = i * 2.39996, r = 12 * Math.sqrt(i + 1);
        const node = { ...n, x: prev ? prev.x : Math.cos(a) * r, y: prev ? prev.y : Math.sin(a) * r, vx: 0, vy: 0, nbrs: new Set() };
        // clusters of one have no identity worth a color
        if ((clusterSizes[n.cluster] || 0) < 2) node.cluster = -1;
        node.r = 4 + Math.min(Math.sqrt(n.degree) * 1.7, 10);
        this.byId.set(n.id, node);
        return node;
      });
      this.edges = edges.map((e) => ({ ...e, s: this.byId.get(e.source), t: this.byId.get(e.target) })).filter((e) => e.s && e.t);
      for (const e of this.edges) { e.s.nbrs.add(e.t); e.t.nbrs.add(e.s); }
      this.alpha = 1;
      this.fitPending = !old.size;
      this.settleFit = !old.size; // refit once the layout has settled, unless the user takes over
    }

    edgeVisible(e) {
      if (this.filter.hiddenClusters.has(e.s.cluster) || this.filter.hiddenClusters.has(e.t.cluster)) return false;
      if (e.kind === "link") return this.filter.link;
      if (e.discovered) return this.filter.discovered || this.filter.semantic;
      return this.filter.semantic;
    }
    nodeVisible(n) { return !this.filter.hiddenClusters.has(n.cluster); }

    reheat(a = 0.5) { this.alpha = Math.max(this.alpha, a); }

    tick() {
      if (this.alpha < 0.005 || !this.nodes.length) {
        if (this.fitPending && this.nodes.length) { this.fit(false); this.fitPending = false; }
        if (this.settleFit && this.nodes.length) { this.fit(); this.settleFit = false; }
        return;
      }
      const nodes = this.nodes.filter((n) => this.nodeVisible(n));
      const a = this.alpha;
      const qt = new QuadTree(nodes);
      for (const n of nodes) qt.force(n, 60 * a);
      for (const e of this.edges) {
        if (!this.edgeVisible(e)) continue;
        const dx = e.t.x - e.s.x, dy = e.t.y - e.s.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const target = e.kind === "link" ? 70 : 110 - 40 * Math.min(e.weight, 1);
        const k = (e.kind === "link" ? 0.06 : 0.035) * a;
        const f = ((d - target) / d) * k;
        const bs = e.s.nbrs.size, bt = e.t.nbrs.size, bias = bt / (bs + bt);
        e.t.vx -= dx * f * (1 - bias); e.t.vy -= dy * f * (1 - bias);
        e.s.vx += dx * f * bias; e.s.vy += dy * f * bias;
      }
      for (const n of nodes) {
        n.vx -= n.x * 0.004 * a; n.vy -= n.y * 0.004 * a; // gentle gravity keeps islands on screen
        if (n === this.dragNode) { n.vx = n.vy = 0; continue; }
        n.vx *= 0.6; n.vy *= 0.6;
        n.x += n.vx; n.y += n.vy;
      }
      this.alpha *= 0.975;
      if (this.fitPending && this.alpha < 0.3) { this.fit(false); this.fitPending = false; }
    }

    // ---------------------------------------------------------------- drawing

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = this.canvas.getBoundingClientRect();
      if (!width || !height) return;
      this.w = width; this.h = height; this.dpr = dpr;
      this.canvas.width = width * dpr; this.canvas.height = height * dpr;
    }

    shape(ctx, kind, x, y, r) {
      ctx.beginPath();
      switch (SHAPES[kind] || "circle") {
        case "square": ctx.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76); break;
        case "diamond": ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath(); break;
        case "triangle": ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.1, y + r * 0.8); ctx.lineTo(x - r * 1.1, y + r * 0.8); ctx.closePath(); break;
        default: ctx.arc(x, y, r, 0, Math.PI * 2);
      }
    }

    draw() {
      if (!this.w) this.resize();
      if (!this.w) return;
      const { ctx, view, colors } = this;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      ctx.translate(this.w / 2 + view.x, this.h / 2 + view.y);
      ctx.scale(view.k, view.k);

      const focus = this.hover || this.selected;
      const lit = focus ? new Set([focus, ...focus.nbrs]) : null;
      const searchHit = this.searchMatches;

      // edges: plain first, then discovered on top
      const ordered = [...this.edges].sort((a, b) => (a.discovered && this.filter.discovered) - (b.discovered && this.filter.discovered));
      for (const e of ordered) {
        if (!this.edgeVisible(e)) continue;
        const on = !lit || (lit.has(e.s) && lit.has(e.t) && (e.s === focus || e.t === focus));
        const disc = e.discovered && this.filter.discovered;
        ctx.globalAlpha = on ? 1 : 0.15;
        ctx.strokeStyle = disc ? colors.discover : e.kind === "link" ? colors.edgeLink : colors.edge;
        ctx.lineWidth = (disc ? 1.8 : e.kind === "link" ? 1.3 : 1) / Math.sqrt(view.k);
        ctx.setLineDash(e.kind === "semantic" && !disc ? [4 / view.k, 3 / view.k] : []);
        ctx.beginPath(); ctx.moveTo(e.s.x, e.s.y); ctx.lineTo(e.t.x, e.t.y); ctx.stroke();
      }
      ctx.setLineDash([]);

      for (const n of this.nodes) {
        if (!this.nodeVisible(n)) continue;
        const dim = (lit && !lit.has(n)) || (searchHit && !searchHit.has(n));
        ctx.globalAlpha = dim ? 0.18 : 1;
        this.shape(ctx, n.kind, n.x, n.y, n.r);
        ctx.fillStyle = this.clusterColor(n.cluster);
        ctx.fill();
        ctx.lineWidth = 2 / view.k;
        ctx.strokeStyle = colors.surface; // 2px surface ring separates overlapping marks
        ctx.stroke();
        if (n === this.selected || (searchHit && searchHit.has(n))) {
          ctx.lineWidth = 2.5 / view.k; ctx.strokeStyle = colors.text;
          this.shape(ctx, n.kind, n.x, n.y, n.r + 3.5 / view.k); ctx.stroke();
        }
      }

      // labels: hovered/selected neighbourhood, hubs, or everything when zoomed in
      ctx.globalAlpha = 1;
      const fontPx = 12 / view.k;
      ctx.font = `500 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      // Greedy declutter: important labels first, then hubs; skip any label that would overlap one already placed.
      const candidates = [];
      for (const n of this.nodes) {
        if (!this.nodeVisible(n)) continue;
        const important = lit ? lit.has(n) : (searchHit ? searchHit.has(n) : false);
        const show = important || (this.filter.labels && (view.k > 1.15 || n.degree >= 5 || this.nodes.length < 40));
        if (!show || (lit && !lit.has(n))) continue;
        candidates.push({ n, important, prio: (important ? 1e6 : 0) + (n === focus ? 1e7 : 0) + n.degree });
      }
      candidates.sort((a, b) => b.prio - a.prio);
      const placed = [];
      for (const { n, important } of candidates) {
        const label = n.title.length > 38 ? n.title.slice(0, 36) + "…" : n.title;
        const y = n.y + n.r + 3 / view.k;
        const w = ctx.measureText(label).width, box = { x0: n.x - w / 2, x1: n.x + w / 2, y0: y, y1: y + fontPx * 1.2 };
        if (placed.some((p) => box.x0 < p.x1 && box.x1 > p.x0 && box.y0 < p.y1 && box.y1 > p.y0)) continue;
        placed.push(box);
        ctx.lineWidth = 3 / view.k; ctx.strokeStyle = colors.bg; ctx.lineJoin = "round";
        ctx.strokeText(label, n.x, y);
        ctx.fillStyle = important ? colors.text : colors.text2;
        ctx.fillText(label, n.x, y);
      }
    }

    // ---------------------------------------------------------------- interaction

    toWorld(px, py) {
      return { x: (px - this.w / 2 - this.view.x) / this.view.k, y: (py - this.h / 2 - this.view.y) / this.view.k };
    }

    nodeAt(px, py) {
      const p = this.toWorld(px, py);
      let best = null, bestD = Infinity;
      for (const n of this.nodes) {
        if (!this.nodeVisible(n)) continue;
        const d = Math.hypot(n.x - p.x, n.y - p.y);
        const hit = n.r + 6 / this.view.k; // hit target bigger than the mark
        if (d < hit && d < bestD) { best = n; bestD = d; }
      }
      return best;
    }

    bindEvents() {
      const c = this.canvas;
      let down = null;
      c.addEventListener("pointerdown", (ev) => {
        c.setPointerCapture(ev.pointerId);
        const r = c.getBoundingClientRect();
        const node = this.nodeAt(ev.clientX - r.left, ev.clientY - r.top);
        down = { x: ev.clientX, y: ev.clientY, vx: this.view.x, vy: this.view.y, node, moved: false };
        this.settleFit = false;
        if (node) { this.dragNode = node; this.reheat(0.3); }
        c.classList.add("dragging");
      });
      c.addEventListener("pointermove", (ev) => {
        const r = c.getBoundingClientRect();
        const px = ev.clientX - r.left, py = ev.clientY - r.top;
        if (down) {
          if (Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 3) down.moved = true;
          if (down.node) {
            const p = this.toWorld(px, py);
            down.node.x = p.x; down.node.y = p.y; this.reheat(0.25);
          } else {
            this.view.x = down.vx + ev.clientX - down.x; this.view.y = down.vy + ev.clientY - down.y;
          }
          return;
        }
        const n = this.nodeAt(px, py);
        if (n !== this.hover) {
          this.hover = n;
          c.classList.toggle("pointer", !!n);
          this.opts.onHover && this.opts.onHover(n, px, py);
        } else if (n && this.opts.onHover) this.opts.onHover(n, px, py);
      });
      const up = () => {
        if (!down) return;
        if (!down.moved && this.opts.onClick) this.opts.onClick(down.node);
        this.dragNode = null; down = null;
        c.classList.remove("dragging");
      };
      c.addEventListener("pointerup", up);
      c.addEventListener("pointercancel", up);
      c.addEventListener("pointerleave", () => { if (!down && this.hover) { this.hover = null; this.opts.onHover && this.opts.onHover(null); } });
      c.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        this.settleFit = false;
        const r = c.getBoundingClientRect();
        this.zoomAt(Math.exp(-ev.deltaY * 0.0015), ev.clientX - r.left, ev.clientY - r.top);
      }, { passive: false });
    }

    zoomAt(factor, px = this.w / 2, py = this.h / 2) {
      this.settleFit = false;
      const k = Math.min(6, Math.max(0.08, this.view.k * factor));
      const p = this.toWorld(px, py);
      this.view.k = k;
      this.view.x = px - this.w / 2 - p.x * k;
      this.view.y = py - this.h / 2 - p.y * k;
    }

    animateTo(target, ms = 450) {
      const start = { ...this.view }, t0 = performance.now();
      const step = (t) => {
        const u = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - u, 3);
        for (const key of ["x", "y", "k"]) this.view[key] = start[key] + (target[key] - start[key]) * e;
        if (u < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    fit(animate = true) {
      const vis = this.nodes.filter((n) => this.nodeVisible(n));
      if (!vis.length || !this.w) return;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const n of vis) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
      const padX = Math.min(150, this.w * 0.15), padY = 60, leftPanel = this.w > 820 ? 280 : 0; // room for labels
      const k = Math.min(2, Math.max(0.1, Math.min((this.w - padX * 2 - leftPanel) / (x1 - x0 || 1), (this.h - padY * 2) / (y1 - y0 || 1))));
      const target = { k, x: -((x0 + x1) / 2) * k + leftPanel / 2, y: -((y0 + y1) / 2) * k };
      animate ? this.animateTo(target) : Object.assign(this.view, target);
    }

    focusNode(id, zoom = 1.6) {
      const n = this.byId.get(id);
      if (!n) return;
      this.selected = n;
      this.animateTo({ k: zoom, x: -n.x * zoom, y: -n.y * zoom });
    }

    setSearch(q) {
      q = (q || "").trim().toLowerCase();
      this.searchMatches = q ? new Set(this.nodes.filter((n) => n.title.toLowerCase().includes(q) || n.tags.some((t) => t.includes(q)))) : null;
      return this.searchMatches ? [...this.searchMatches] : [];
    }
  }

  window.ForceGraph = ForceGraph;
  window.GRAPH_SHAPES = SHAPES;
  window.CLUSTER_VARS = CLUSTER_VARS;
})();
