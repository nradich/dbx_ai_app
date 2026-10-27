'use strict';

/*
 * tailspot. — "Model performance" tab (view tabs, comparison summary, drill-down,
 * training runs, app submissions). Read-only views of /api/performance/* (see app/API.md).
 * Pure helpers first; DOM wiring only when a document exists. Data is only ever written
 * with textContent / attributes, never innerHTML.
 */
(function (root) {
  const CLASS_ORDER = ['alaska', 'american', 'delta', 'southwest', 'united'];
  const SCALE_ORDER = ['context_1.0', 'context_1.5', 'context_2.0', 'context_3.0', 'context_full'];
  const PALETTE = ['#285441', '#c9873e', '#4f7fb0', '#ad4a3c', '#8a6bb8', '#779b7e', '#3b8f8a'];
  // Longer than the server's own SQL wait (about 25 s) so its 503 arrives before we give up.
  const REQUEST_TIMEOUT_MS = 60000;
  const HISTORY_POLL_MS = 10000;
  const RUNS_POLL_MS = 30000;
  const CHART = { w: 520, h: 250, l: 48, r: 16, t: 14, b: 40 };
  const API = {
    summary: '/api/performance/summary',
    predictions: '/api/performance/predictions',
    runs: '/api/performance/training-runs',
    submissions: '/api/performance/submissions',
  };

  // ---------- Pure helpers ----------
  function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }
  function round2(x) { return Math.round(x * 100) / 100; }
  function pct(x, digits = 1) { return isNum(x) ? `${(x * 100).toFixed(digits)}%` : '—'; }
  function fixed(x, digits = 2) { return isNum(x) ? x.toFixed(digits) : '—'; }
  function dash(x) { return x === null || x === undefined || x === '' ? '—' : String(x); }
  function versionLabel(v) { return v === null || v === undefined || v === '' ? '—' : `v${v}`; }
  function titleCase(s) {
    if (!s) return '—';
    return String(s).split(/[_\s]+/).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  }

  /** "7/8 (87.5%)": correct images (rounded from accuracy × n) out of n. */
  function accuracyCell(acc, n) {
    if (!isNum(acc)) return '—';
    if (!isNum(n) || n <= 0) return pct(acc);
    return `${Math.round(acc * n)}/${n} (${pct(acc)})`;
  }

  function variantLabel(variant) {
    if (variant === 'tail_crop') return 'Tail crop';
    if (variant === 'whole_image' || variant === 'context_full') return 'Whole image';
    const m = /^context_(.+)$/.exec(variant || '');
    if (m) return `${m[1]}× box`;
    return titleCase(variant);
  }

  /** Context variants ordered by scale, whole image last; non-context variants dropped. */
  function scaleOrder(variants) {
    const key = (v) => (v === 'context_full' ? Infinity : parseFloat(v.slice(8)));
    return (variants || []).filter((v) => /^context_/.test(v) && !Number.isNaN(key(v)))
      .sort((a, b) => key(a) - key(b));
  }

  /** Class order: the five known airlines, then any extra labels seen in the rows (sorted). */
  function classesFor(rows, base = CLASS_ORDER) {
    const extra = new Set();
    (rows || []).forEach((r) => [r.actual, r.predicted].forEach((c) => {
      if (c && !base.includes(c)) extra.add(c);
    }));
    return base.concat([...extra].sort());
  }

  /** Confusion counts: counts[actualIndex][predictedIndex]. Rows without a label are skipped. */
  function confusion(rows, classes) {
    const cls = classes || classesFor(rows);
    const counts = cls.map(() => cls.map(() => 0));
    (rows || []).forEach((r) => {
      const a = cls.indexOf(r.actual), p = cls.indexOf(r.predicted);
      if (a >= 0 && p >= 0) counts[a][p] += 1;
    });
    return { classes: cls, counts };
  }

  /** Booleans marking the maximum finite value(s); ties are all marked. */
  function bestFlags(values) {
    const finite = values.filter(isNum);
    if (!finite.length) return values.map(() => false);
    const max = Math.max(...finite);
    return values.map((v) => isNum(v) && Math.abs(v - max) < 1e-9);
  }

  /** Index of long-format metric rows by version|set|variant|metric. */
  function indexMetrics(metrics) {
    const map = new Map();
    (metrics || []).forEach((m) => {
      map.set(`${m.model_version}|${m.eval_set}|${m.variant}|${m.metric}`, m);
    });
    return {
      get(version, set, variant, metric) { return map.get(`${version}|${set}|${variant}|${metric}`) || null; },
      value(version, set, variant, metric) {
        const m = map.get(`${version}|${set}|${variant}|${metric}`);
        return m && isNum(m.value) ? m.value : null;
      },
    };
  }

  function scaleLinear(domain, range) {
    const [d0, d1] = domain, [r0, r1] = range;
    return (v) => (d1 === d0 ? (r0 + r1) / 2 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0));
  }

  /** SVG path geometry for [x, y] points; non-finite points break the line. */
  function linePath(points, xDomain, yDomain, box = CHART) {
    const sx = scaleLinear(xDomain, [box.l, box.w - box.r]);
    const sy = scaleLinear(yDomain, [box.h - box.b, box.t]);
    const coords = [], parts = [];
    let pen = false;
    (points || []).forEach(([x, y]) => {
      if (!isNum(x) || !isNum(y)) { pen = false; return; }
      const c = { x: round2(sx(x)), y: round2(sy(y)), vx: x, vy: y };
      coords.push(c);
      parts.push(`${pen ? 'L' : 'M'}${c.x},${c.y}`);
      pen = true;
    });
    return { d: parts.join(' '), coords };
  }

  /**
   * About `max` evenly spaced integer ticks covering [lo, hi]; the last integer step (hi) is
   * always the final tick, replacing the previous tick when that would sit within 20% of a step.
   */
  function integerTicks(lo, hi, max = 8) {
    if (!isNum(lo) || !isNum(hi)) return [];
    const step = Math.max(1, Math.ceil((hi - lo) / Math.max(1, max - 1)));
    const ticks = [];
    for (let v = Math.ceil(lo); v <= hi; v += step) ticks.push(v);
    const last = Math.floor(hi);
    if (ticks.length && ticks[ticks.length - 1] !== last) {
      if (ticks.length > 1 && last - ticks[ticks.length - 1] <= 0.2 * step) ticks.pop();
      ticks.push(last);
    }
    return ticks;
  }

  function pad2(n) { return String(n).padStart(2, '0'); }
  function clockTime(date) { return `${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())}`; }
  function localDateTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function errorMessage(status, text) {
    try {
      const body = JSON.parse(text);
      if (body && typeof body.detail === 'string' && body.detail) return body.detail;
      if (body && Array.isArray(body.detail)) {
        const msgs = body.detail.map((d) => d && d.msg).filter(Boolean);
        if (msgs.length) return msgs.join('; ');
      }
    } catch (e) { /* not JSON */ }
    if (status === 503) return 'Performance data cannot be read right now. Try again shortly.';
    if (status === 404) return 'Not found.';
    return `Request failed (HTTP ${status}).`;
  }

  function submissionsAccuracyText(s) {
    const reviewed = s && isNum(s.reviewed) ? s.reviewed : 0;
    if (!reviewed) return 'No reviewed submissions yet';
    return `${isNum(s.reviewed_correct) ? s.reviewed_correct : 0}/${reviewed} reviewed correct`;
  }

  const RECENT_FILTERS = [['all', 'All'], ['correct', 'Correct'], ['wrong', 'Wrong'], ['unreviewed', 'Not reviewed']];

  /** Outcome of a recent submission: correct | wrong | unreviewed | failed. */
  function recentOutcome(r) {
    if (!r || r.status === 'failed' || !r.predicted_airline) return 'failed';
    if (r.correct === true) return 'correct';
    if (r.correct === false) return 'wrong';
    return 'unreviewed';
  }

  /** Rows matching a filter; failed submissions count as "not reviewed" (nothing to judge). */
  function filterRecent(rows, filter) {
    const list = Array.isArray(rows) ? rows : [];
    if (filter === 'correct' || filter === 'wrong') return list.filter((r) => recentOutcome(r) === filter);
    if (filter === 'unreviewed') return list.filter((r) => ['unreviewed', 'failed'].includes(recentOutcome(r)));
    return list;
  }

  /** Badge text for a recent submission. */
  function outcomeText(r) {
    const outcome = recentOutcome(r);
    if (outcome === 'correct') return '✓ Correct';
    if (outcome === 'wrong') {
      const actual = r.review_label === 'other' ? 'another airline' : titleCase(r.review_label);
      return `✗ Wrong — was ${actual}`;
    }
    if (outcome === 'failed') return 'No prediction';
    return 'Not reviewed';
  }

  /** "2/3 correct (67%)" for a by_model_version entry, or "No reviewed photos yet". */
  function versionAccuracyText(v) {
    const reviewed = v && isNum(v.reviewed) ? v.reviewed : 0;
    if (!reviewed) return 'No reviewed photos yet';
    const correct = isNum(v.correct) ? v.correct : 0;
    return `${correct}/${reviewed} correct (${Math.round((correct / reviewed) * 100)}%)`;
  }

  /** Same-origin photo URL; kind is thumb | crop | original. */
  function photoUrl(submissionId, kind) {
    return `/api/submissions/${encodeURIComponent(String(submissionId))}/photo?kind=${encodeURIComponent(kind)}`;
  }

  const helpers = {
    pct, fixed, accuracyCell, variantLabel, scaleOrder, classesFor, confusion, bestFlags,
    indexMetrics, linePath, integerTicks, clockTime, errorMessage, submissionsAccuracyText,
    versionLabel, titleCase, recentOutcome, filterRecent, outcomeText, versionAccuracyText, photoUrl,
    CLASS_ORDER, SCALE_ORDER, CHART, RECENT_FILTERS,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  root.TailspotPerformance = helpers;

  if (typeof document === 'undefined') return;

  // ---------- DOM utilities ----------
  const SVG_NS = 'http://www.w3.org/2000/svg';
  let uid = 0;

  function fill(node, attrs, children) {
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') node.textContent = String(v);
      else node.setAttribute(k, v === true ? '' : String(v));
    });
    (children || []).forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.append(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    });
    return node;
  }
  function el(tag, attrs, children) { return fill(document.createElement(tag), attrs, children); }
  function svgEl(tag, attrs, children) { return fill(document.createElementNS(SVG_NS, tag), attrs, children); }
  function $(id) { return document.getElementById(id); }

  function showLoading(node, text) {
    node.replaceChildren(el('p', { class: 'perf-loading', role: 'status', text }));
  }
  function showError(node, message, retry) {
    const button = el('button', { type: 'button', class: 'crop-button perf-retry', text: 'Retry' });
    button.addEventListener('click', retry);
    node.replaceChildren(el('div', { class: 'perf-error', role: 'alert' }, [el('p', { text: message }), button]));
  }
  function note(text) { return el('p', { class: 'perf-muted', text }); }
  function badge(text, kind) { return el('span', { class: `perf-badge${kind ? ` perf-badge-${kind}` : ''}`, text }); }

  /** Scrollable table. Cells: string | Node | {content, cls, th}. Rows: array or {cells, cls}. */
  function table(headers, rows, opts = {}) {
    const cell = (c, tag) => {
      const spec = c && typeof c === 'object' && !(c instanceof Node) ? c : { content: c };
      const node = el(spec.th ? 'th' : tag, { class: spec.cls, scope: spec.th ? 'row' : null });
      const content = spec.content;
      (Array.isArray(content) ? content : [content]).forEach((x) => fill(node, {}, [x === null || x === undefined ? '—' : x]));
      return node;
    };
    const thead = el('thead', {}, [el('tr', {}, headers.map((h) => fill(cell(h, 'th'), { scope: 'col' })))]);
    const tbody = el('tbody', {}, rows.map((r) => {
      const spec = Array.isArray(r) ? { cells: r } : r;
      return el('tr', { class: spec.cls }, spec.cells.map((c) => cell(c, 'td')));
    }));
    const t = el('table', { class: `perf-table ${opts.cls || ''}`.trim() }, [
      opts.caption ? el('caption', { class: opts.captionVisible ? 'perf-caption' : 'visually-hidden', text: opts.caption }) : null,
      thead, tbody,
    ]);
    // Plain overflow wrapper: not a focusable landmark (there are many tables per view).
    return el('div', { class: 'perf-table-wrap' }, [t]);
  }

  /** Responsive SVG line chart with title/desc, y gridlines and a legend. */
  function lineChart(o) {
    const id = `perf-chart-${++uid}`;
    const box = CHART;
    const sx = scaleLinear(o.xDomain, [box.l, box.w - box.r]);
    const sy = scaleLinear(o.yDomain, [box.h - box.b, box.t]);
    const svg = svgEl('svg', { viewBox: `0 0 ${box.w} ${box.h}`, class: 'perf-chart', role: 'img',
      'aria-labelledby': `${id}-t ${id}-d`, preserveAspectRatio: 'xMidYMid meet' }, [
      svgEl('title', { id: `${id}-t` }, [o.title]), svgEl('desc', { id: `${id}-d` }, [o.desc]),
    ]);
    o.yTicks.forEach((v) => {
      const y = round2(sy(v));
      svg.append(svgEl('line', { x1: box.l, x2: box.w - box.r, y1: y, y2: y, class: 'perf-grid' }),
        svgEl('text', { x: box.l - 8, y: y + 4, class: 'perf-axis', 'text-anchor': 'end' }, [o.yFmt(v)]));
    });
    o.xTicks.forEach((t) => svg.append(svgEl('text', { x: round2(sx(t.v)), y: box.h - box.b + 18,
      class: 'perf-axis', 'text-anchor': 'middle' }, [t.label])));
    if (o.xLabel) {
      svg.append(svgEl('text', { x: (box.l + box.w - box.r) / 2, y: box.h - 4, class: 'perf-axis perf-axis-label',
        'text-anchor': 'middle' }, [o.xLabel]));
    }
    o.series.forEach((s) => {
      const g = linePath(s.points, o.xDomain, o.yDomain, box);
      const group = svgEl('g', { class: 'perf-series' });
      if (g.coords.length > 1) {
        group.append(svgEl('path', { d: g.d, fill: 'none', stroke: s.color, 'stroke-width': 2.5,
          'stroke-linejoin': 'round', 'stroke-dasharray': s.dashed ? '6 4' : null }));
      }
      const r = g.coords.length > 25 ? 2.2 : 3.6;
      g.coords.forEach((c) => group.append(svgEl('circle', { cx: c.x, cy: c.y, r, fill: s.color },
        [svgEl('title', {}, [`${s.name} · ${o.pointFmt(c.vx, c.vy)}`])])));
      svg.append(group);
    });
    const legend = el('ul', { class: 'perf-legend' }, o.series.map((s) => el('li', {}, [
      el('span', { class: `perf-swatch${s.dashed ? ' is-dashed' : ''}`, style: `--swatch:${s.color}`, 'aria-hidden': 'true' }), s.name,
    ])));
    return el('figure', { class: 'perf-figure' }, [el('figcaption', { class: 'perf-figure-title', text: o.title }), svg, legend]);
  }

  /** Horizontal bar row: label, track with fill (0..1), value text. */
  function barRow(label, fraction, valueText) {
    const width = isNum(fraction) ? Math.max(0, Math.min(1, fraction)) * 100 : 0;
    return el('div', { class: 'perf-bar-row' }, [
      el('span', { class: 'perf-bar-label', text: label }),
      el('span', { class: 'perf-bar-track', 'aria-hidden': 'true' }, [el('span', { class: 'perf-bar-fill', style: `width:${width.toFixed(1)}%` })]),
      el('span', { class: 'perf-bar-value', text: valueText }),
    ]);
  }

  async function fetchJson(url) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
    try {
      let res, text;
      try {
        res = await fetch(url, { cache: 'no-store', credentials: 'same-origin',
          headers: { Accept: 'application/json' }, signal: ctrl.signal });
        text = await res.text();
      } catch (err) {
        throw new Error(err && err.name === 'AbortError' ? 'The request timed out. Try again.'
          : 'Could not reach the server. Check your connection and try again.');
      }
      if (!res.ok) throw new Error(errorMessage(res.status, text));
      try { return JSON.parse(text); } catch (e) { throw new Error('The server returned an unreadable response.'); }
    } finally { clearTimeout(timer); }
  }

  // ---------- State ----------
  const state = {
    loaded: false, summary: null, metrics: indexMetrics([]), versions: [],
    drillRows: null, drillToken: 0,
    runs: [], selectedRunId: null, historyToken: 0, historyTimer: null, runsTimer: null, lastHistory: null,
    submissions: null, recentFilter: 'all',
  };
  const colorFor = (version) => PALETTE[Math.max(0, state.versions.indexOf(version)) % PALETTE.length];
  const perfVisible = () => !$('panel-performance').hidden && !document.hidden;

  // ---------- Summary ----------
  async function loadSummary() {
    const slot = $('perf-summary-state');
    showLoading(slot, 'Loading the latest comparison…');
    try {
      const data = await fetchJson(API.summary);
      slot.replaceChildren();
      state.summary = data;
      // The submissions section may have rendered first; refresh its "serving" marker.
      if (state.submissions && $('perf-submissions').querySelector('.perf-version-cards')) renderSubmissions(state.submissions);
      if (!data || data.available === false) {
        $('perf-summary-content').hidden = true;
        $('perf-empty').hidden = false;
        $('perf-caveat-stress').hidden = true;
        return;
      }
      $('perf-empty').hidden = true;
      $('perf-summary-content').hidden = false;
      renderSummary(data);
    } catch (err) {
      $('perf-summary-content').hidden = true;
      $('perf-empty').hidden = true;
      showError(slot, err.message, loadSummary);
    }
  }

  function renderSummary(data) {
    const models = data.models || [];
    state.metrics = indexMetrics(data.metrics);
    state.versions = models.map((m) => m.model_version);
    (data.metrics || []).forEach((m) => { if (!state.versions.includes(m.model_version)) state.versions.push(m.model_version); });
    const serving = data.serving || {};
    const card = (label, value, sub) => el('div', { class: 'perf-card' }, [
      el('span', { class: 'perf-card-label', text: label }), el('strong', { class: 'perf-card-value', text: value }),
      sub ? el('span', { class: 'perf-card-sub', text: sub }) : null,
    ]);
    $('perf-cards').replaceChildren(
      card('Serving now', versionLabel(serving.model_version), serving.endpoint || 'Endpoint unknown'),
      card('Versions compared', String(state.versions.length), models.length === 1 ? 'model version' : 'model versions'),
      card('Evaluated at', localDateTime(data.evaluated_at), 'local time'),
      card('Evaluation', data.eval_run_id ? String(data.eval_run_id).slice(0, 8) : '—', 'run ID'),
    );
    $('perf-models').replaceChildren(models.length ? table(
      ['Version', 'Recipe', 'Dataset', 'Aliases when evaluated', 'Flags', 'Val accuracy'],
      models.map((m) => [
        { content: versionLabel(m.model_version), th: true }, dash(m.recipe), dash(m.dataset_version),
        { content: (m.aliases || []).length ? m.aliases.map((a) => badge(a, 'alias')) : '—' },
        { content: [m.serving ? badge('Serving', 'serving') : null, m.selected ? badge('Selected by validation', 'selected') : null]
          .filter(Boolean).concat(m.serving || m.selected ? [] : ['—']) },
        pct(m.val_accuracy),
      ]), { caption: 'Model versions in this comparison' }) : note('No model versions in this comparison.'));
    renderMatrix(data);
    renderContext(data);
    const stress = (data.eval_sets || []).find((s) => s.eval_set === 'full_plane_stress');
    $('perf-caveat-stress').hidden = !stress;
    $('perf-caveat-stress-text').textContent = stress ? stress.description || '' : '';
    populateDrill(data);
  }

  function renderMatrix(data) {
    const rows = [];
    (data.eval_sets || []).filter((s) => s.eval_set !== 'context_sensitivity').forEach((set) => {
      (set.variants || []).forEach((variant, i) => {
        const accs = state.versions.map((v) => state.metrics.value(v, set.eval_set, variant, 'accuracy'));
        const best = bestFlags(accs);
        const header = { th: true, content: [
          el('span', { class: 'perf-row-title', text: set.label || set.eval_set }),
          el('span', { class: 'perf-row-variant', text: variantLabel(variant) }),
          i === 0 && set.description ? el('span', { class: 'perf-row-desc', text: set.description }) : null,
        ] };
        rows.push([header].concat(state.versions.map((v, j) => {
          const acc = state.metrics.get(v, set.eval_set, variant, 'accuracy');
          if (!acc) return { content: '—', cls: 'perf-cell' };
          const f1 = state.metrics.value(v, set.eval_set, variant, 'macro_f1');
          const ece = state.metrics.value(v, set.eval_set, variant, 'ece');
          return { cls: `perf-cell${best[j] ? ' is-best' : ''}`, content: [
            el('span', { class: 'perf-cell-main' }, [accuracyCell(acc.value, acc.n),
              best[j] ? el('span', { class: 'perf-star', title: 'Best in this row' }, ['★', el('span', { class: 'visually-hidden', text: ' best' })]) : null]),
            el('span', { class: 'perf-cell-sub', text: `F1 ${fixed(f1)} · ECE ${fixed(ece)}` }),
          ] };
        })));
      });
    });
    $('perf-matrix').replaceChildren(rows.length
      ? table(['Evaluation set'].concat(state.versions.map(versionLabel)), rows, { cls: 'perf-matrix', caption: 'Accuracy by evaluation set and model version' })
      : note('No evaluation sets in this comparison.'));
  }

  function renderContext(data) {
    const host = $('perf-context');
    const set = (data.eval_sets || []).find((s) => s.eval_set === 'context_sensitivity');
    const scales = set ? scaleOrder(set.variants) : [];
    if (!scales.length) { host.replaceChildren(note('No context-sensitivity results in this comparison.')); return; }
    const short = (v) => (v === 'context_full' ? 'Whole' : variantLabel(v).replace(' box', ''));
    const chartFor = (metric, title, desc) => lineChart({
      title, desc, xDomain: [0, scales.length - 1], yDomain: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1],
      yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: 'Crop: tail box scale',
      xTicks: scales.map((s, i) => ({ v: i, label: short(s) })),
      pointFmt: (x, y) => `${variantLabel(scales[x])}: ${pct(y)}`,
      series: state.versions.map((v) => ({ name: versionLabel(v), color: colorFor(v),
        points: scales.map((s, i) => [i, state.metrics.value(v, 'context_sensitivity', s, metric)]) })),
    });
    const fallback = (metric, caption) => el('div', { class: 'visually-hidden' }, [table(
      ['Version'].concat(scales.map(variantLabel)),
      state.versions.map((v) => [{ content: versionLabel(v), th: true }].concat(scales.map((s) => {
        const m = state.metrics.get(v, 'context_sensitivity', s, metric);
        return m ? (metric === 'accuracy' ? accuracyCell(m.value, m.n) : pct(m.value)) : '—';
      }))), { caption })]);
    host.replaceChildren(el('div', { class: 'perf-chart-grid' }, [
      el('div', {}, [chartFor('accuracy', 'Accuracy vs crop context',
        'Accuracy of each model version as the crop grows from the tail box to the whole image.'),
      fallback('accuracy', 'Accuracy by crop context')]),
      el('div', {}, [chartFor('share_pred_southwest', 'Share predicted Southwest vs crop context',
        'Share of images each model version predicted as Southwest at each crop size.'),
      fallback('share_pred_southwest', 'Share predicted Southwest by crop context')]),
    ]));
  }

  // ---------- Drill-down ----------
  function populateDrill(data) {
    const setSel = $('perf-drill-set');
    const prev = setSel.value;
    const sets = data.eval_sets || [];
    setSel.replaceChildren(...sets.map((s) => el('option', { value: s.eval_set, text: s.label || s.eval_set })));
    if (sets.some((s) => s.eval_set === prev)) setSel.value = prev;
    populateVariants();
  }

  function currentSet() {
    return ((state.summary && state.summary.eval_sets) || []).find((s) => s.eval_set === $('perf-drill-set').value) || null;
  }

  function populateVariants() {
    const sel = $('perf-drill-variant');
    const prev = sel.value;
    const set = currentSet();
    const variants = set ? (set.eval_set === 'context_sensitivity' ? scaleOrder(set.variants) : set.variants || []) : [];
    sel.replaceChildren(...variants.map((v) => el('option', { value: v, text: variantLabel(v) })));
    if (variants.includes(prev)) sel.value = prev;
    sel.disabled = variants.length < 2;
    $('perf-drill-description').textContent = set ? set.description || '' : '';
    loadDrill();
  }

  async function loadDrill() {
    const body = $('perf-drill-body');
    const set = $('perf-drill-set').value, variant = $('perf-drill-variant').value;
    const token = ++state.drillToken;
    state.drillRows = null;
    if (!set || !variant) { body.replaceChildren(note('Choose an evaluation set.')); return; }
    showLoading(body, 'Loading predictions…');
    // Pin the drill-down to the comparison shown in the summary.
    const runId = state.summary && state.summary.eval_run_id;
    const query = `eval_set=${encodeURIComponent(set)}&variant=${encodeURIComponent(variant)}`
      + (runId ? `&eval_run_id=${encodeURIComponent(runId)}` : '');
    try {
      const data = await fetchJson(`${API.predictions}?${query}`);
      if (token !== state.drillToken) return;
      state.drillRows = (data && data.rows) || [];
      renderDrill();
    } catch (err) {
      if (token === state.drillToken) showError(body, err.message, loadDrill);
    }
  }

  function renderDrill() {
    const body = $('perf-drill-body');
    const rows = state.drillRows || [];
    if (!rows.length) { body.replaceChildren(note('No predictions were stored for this set and variant.')); return; }
    const set = $('perf-drill-set').value, variant = $('perf-drill-variant').value;
    const onlyMistakes = $('perf-drill-mistakes').checked;
    const classes = classesFor(rows);
    const versions = state.versions.filter((v) => rows.some((r) => r.model_version === v))
      .concat([...new Set(rows.map((r) => r.model_version))].filter((v) => !state.versions.includes(v)));
    body.replaceChildren(...versions.map((v) => {
      const vr = rows.filter((r) => r.model_version === v);
      const correct = vr.filter((r) => r.correct).length;
      const { counts } = confusion(vr, classes);
      const matrix = table([{ content: 'Actual \\ predicted' }].concat(classes.map(titleCase)),
        classes.map((c, i) => [{ content: titleCase(c), th: true }].concat(counts[i].map((n, j) => ({
          content: String(n), cls: `perf-cm${i === j ? ' is-diag' : ''}${n === 0 ? ' is-zero' : (i !== j ? ' is-off' : '')}`,
        })))), { cls: 'perf-confusion', caption: `Confusion matrix for ${versionLabel(v)}: rows are the actual airline, columns the predicted airline` });
      const f1 = el('div', { class: 'perf-bars' }, [el('h4', { text: 'F1 by airline' })].concat(classes.map((c) => {
        const val = state.metrics.value(v, set, variant, `f1_${c}`);
        return barRow(titleCase(c), val, fixed(val));
      })));
      const shown = onlyMistakes ? vr.filter((r) => !r.correct) : vr;
      const preds = shown.length ? table(['Item', 'Actual', 'Predicted', 'Confidence', 'p(actual)', 'Result'],
        shown.map((r) => ({ cls: r.correct ? null : 'is-wrong', cells: [
          { content: dash(r.item_id), th: true }, titleCase(r.actual), titleCase(r.predicted), pct(r.confidence), pct(r.true_prob),
          { content: el('span', { class: r.correct ? 'perf-ok' : 'perf-bad', 'aria-label': r.correct ? 'correct' : 'wrong', text: r.correct ? '✓' : '✗' }) },
        ] })), { cls: 'perf-preds', caption: `Predictions of ${versionLabel(v)}` })
        : note(onlyMistakes ? 'No mistakes for this version.' : 'No predictions.');
      return el('div', { class: 'perf-drill-version' }, [
        el('h3', {}, [`${versionLabel(v)} `, el('span', { class: 'perf-muted', text: `${correct}/${vr.length} correct` })]),
        el('div', { class: 'perf-drill-grid' }, [matrix, f1]), preds,
      ]);
    }));
  }

  // ---------- Training runs ----------
  // Two independent timers while the selected run is RUNNING: its history every 10 s and the
  // runs list every 30 s. Both stop when the tab or the document is hidden.
  function clearHistoryPoll() { clearTimeout(state.historyTimer); state.historyTimer = null; }
  function clearRunsPoll() { clearTimeout(state.runsTimer); state.runsTimer = null; }
  function clearPolls() { clearHistoryPoll(); clearRunsPoll(); }

  function scheduleHistoryPoll(running) {
    clearHistoryPoll();
    if (!running || !perfVisible()) return;
    state.historyTimer = setTimeout(() => { state.historyTimer = null; if (perfVisible()) loadHistory(true); }, HISTORY_POLL_MS);
  }

  function scheduleRunsPoll(running) {
    if (!running || !perfVisible()) { clearRunsPoll(); return; }
    if (state.runsTimer !== null) return; // already armed: do not push it back
    state.runsTimer = setTimeout(() => { state.runsTimer = null; if (perfVisible()) loadTraining(true); }, RUNS_POLL_MS);
  }

  function selectedRunning() { return !!selectedRun() && selectedRun().status === 'RUNNING'; }

  async function loadTraining(quiet) {
    const list = $('perf-training-list');
    if (quiet !== true) { showLoading(list, 'Loading training runs…'); $('perf-training-detail').replaceChildren(); }
    try {
      const data = await fetchJson(API.runs);
      if (!data || data.experiment_id === null || data.experiment_id === undefined) {
        clearPolls(); state.runs = [];
        list.replaceChildren(note('No training experiment is configured.'));
        $('perf-training-detail').replaceChildren();
        return;
      }
      state.runs = data.runs || [];
      if (!state.runs.length) {
        clearPolls();
        list.replaceChildren(note('No training runs yet.'));
        $('perf-training-detail').replaceChildren();
        return;
      }
      if (!state.runs.some((r) => r.run_id === state.selectedRunId)) state.selectedRunId = state.runs[0].run_id;
      renderRuns();
      scheduleRunsPoll(selectedRunning());
      if (quiet !== true) await loadHistory(false);
      else if (state.historyTimer === null) scheduleHistoryPoll(selectedRunning());
    } catch (err) {
      if (quiet === true) {
        scheduleRunsPoll(selectedRunning());
        if (state.historyTimer === null) scheduleHistoryPoll(selectedRunning());
      } else showError(list, err.message, () => loadTraining(false));
    }
  }

  function selectedRun() { return state.runs.find((r) => r.run_id === state.selectedRunId) || null; }

  function renderRuns() {
    $('perf-training-list').replaceChildren(table(
      ['Status', 'Run', 'Recipe', 'Dataset', 'Model', 'Best val acc', 'Test acc', 'Epochs', 'Started', ''],
      state.runs.map((r) => {
        const button = el('button', { type: 'button', class: 'perf-link-button', 'aria-pressed': String(r.run_id === state.selectedRunId),
          text: 'Show curves', 'aria-label': `Show curves for ${r.run_name || r.run_id}` });
        button.addEventListener('click', () => { state.selectedRunId = r.run_id; renderRuns(); loadHistory(false); });
        return { cls: r.run_id === state.selectedRunId ? 'is-selected' : null, cells: [
          { content: badge(titleCase(String(r.status || 'unknown').toLowerCase()), `status-${String(r.status || '').toLowerCase()}`) },
          { content: dash(r.run_name || r.run_id), th: true }, dash(r.recipe), dash(r.dataset_version),
          versionLabel(r.registered_model_version), pct(r.best_val_accuracy), pct(r.test_accuracy),
          dash(r.epochs_completed), localDateTime(r.start_time), { content: button },
        ] };
      }), { cls: 'perf-runs', caption: 'Training runs, newest first' }));
  }

  async function loadHistory(quiet) {
    const detail = $('perf-training-detail');
    const run = selectedRun();
    if (!run) return;
    const token = ++state.historyToken;
    if (quiet !== true) showLoading(detail, 'Loading training curves…');
    try {
      const data = await fetchJson(`${API.runs}/${encodeURIComponent(run.run_id)}/history`);
      if (token !== state.historyToken) return;
      const status = data.status || run.status;
      const last = state.lastHistory;
      const previous = last && last.runId === run.run_id ? last.status : run.status;
      state.lastHistory = { runId: run.run_id, status };
      renderHistory(run, data);
      scheduleHistoryPoll(status === 'RUNNING');
      if (status === 'RUNNING') scheduleRunsPoll(true); // an armed list timer is left to run once
      // The run just finished: refresh the list once so it stops showing RUNNING.
      if (previous === 'RUNNING' && status !== 'RUNNING') loadTraining(true);
    } catch (err) {
      if (token !== state.historyToken) return;
      if (quiet === true) {
        const last = state.lastHistory;
        scheduleHistoryPoll((last && last.runId === run.run_id ? last.status : run.status) === 'RUNNING');
      } else showError(detail, err.message, () => loadHistory(false));
    }
  }

  function renderHistory(run, data) {
    const m = (data && data.metrics) || {};
    const series = (keys) => keys.map((k, i) => ({ name: k.replace('_', ' '), color: PALETTE[i], dashed: i === 1,
      points: (m[k] || []).filter((p) => Array.isArray(p)) }));
    const steps = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'].flatMap((k) => (m[k] || []).map((p) => p[0])).filter(isNum);
    const heading = el('h3', { class: 'perf-detail-title' }, [`Curves for ${run.run_name || run.run_id} `,
      badge(titleCase(String(data.status || run.status || '').toLowerCase()), `status-${String(data.status || run.status || '').toLowerCase()}`)]);
    if (!steps.length) { $('perf-training-detail').replaceChildren(heading, note('No per-epoch metrics logged yet.')); return; }
    const lo = Math.min(...steps), hi = Math.max(...steps);
    const xTicks = integerTicks(lo, hi).map((v) => ({ v, label: String(v) }));
    const losses = ['train_loss', 'val_loss'].flatMap((k) => (m[k] || []).map((p) => p[1])).filter(isNum);
    const lossMax = losses.length ? Math.max(...losses) * 1.1 || 1 : 1;
    const common = { xDomain: [lo, hi], xTicks, xLabel: 'Epoch' };
    $('perf-training-detail').replaceChildren(heading, el('div', { class: 'perf-chart-grid' }, [
      lineChart(Object.assign({}, common, { title: 'Loss by epoch', desc: `Training and validation loss per epoch for ${run.run_name || run.run_id}.`,
        yDomain: [0, lossMax], yTicks: [0, 0.25, 0.5, 0.75, 1].map((f) => f * lossMax), yFmt: (v) => fixed(v),
        pointFmt: (x, y) => `epoch ${x}: ${fixed(y, 3)}`, series: series(['train_loss', 'val_loss']) })),
      lineChart(Object.assign({}, common, { title: 'Accuracy by epoch', desc: `Training and validation accuracy per epoch for ${run.run_name || run.run_id}.`,
        yDomain: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], yFmt: (v) => `${Math.round(v * 100)}%`,
        pointFmt: (x, y) => `epoch ${x}: ${pct(y)}`, series: series(['train_accuracy', 'val_accuracy']) })),
    ]));
  }

  // ---------- App submissions ----------
  async function loadSubmissions() {
    const host = $('perf-submissions');
    showLoading(host, 'Loading app submissions…');
    try { renderSubmissions(await fetchJson(API.submissions)); } catch (err) { showError(host, err.message, loadSubmissions); }
  }

  function renderSubmissions(s) {
    const tile = (label, value) => el('div', { class: 'perf-card' }, [el('span', { class: 'perf-card-label', text: label }),
      el('strong', { class: 'perf-card-value', text: value })]);
    const byAirline = s.by_predicted_airline || [];
    const maxCount = Math.max(1, ...byAirline.map((a) => a.count || 0));
    const buckets = s.confidence_buckets || [];
    const maxBucket = Math.max(1, ...buckets.map((b) => b.count || 0));
    const hist = el('div', { class: 'perf-hist', role: 'img', 'aria-label': `Confidence histogram: ${buckets.map((b) =>
      `${pct(b.lower, 0)}–${pct(b.upper, 0)}: ${b.count}`).join(', ')}` }, buckets.map((b) => el('div', { class: 'perf-hist-col' }, [
      el('span', { class: 'perf-hist-count', text: String(b.count) }),
      el('span', { class: 'perf-hist-bar', style: `height:${(((b.count || 0) / maxBucket) * 100).toFixed(1)}%` }),
      el('span', { class: 'perf-hist-label', text: `${Math.round(b.lower * 100)}–${Math.round(b.upper * 100)}%` }),
    ])));
    const recent = s.recent || [];
    const versions = (s.by_model_version || []).filter((v) => v.model_version !== null && v.model_version !== undefined);
    const recentHost = el('div', { id: 'perf-recent' });
    state.submissions = s;
    $('perf-submissions').replaceChildren(
      el('div', { class: 'perf-cards' }, [tile('Total', dash(s.total)), tile('Succeeded', dash(s.succeeded)), tile('Failed', dash(s.failed)),
        tile('Reviewed accuracy', submissionsAccuracyText(s))]),
      el('h4', { text: 'Live accuracy by model version' }),
      versions.length ? el('div', { class: 'perf-cards perf-version-cards' }, versions.map((v) => el('div', { class: 'perf-card' }, [
        el('span', { class: 'perf-card-label', text: `${versionLabel(v.model_version)}${v.model_version === servingVersion() ? ' · serving' : ''}` }),
        el('strong', { class: 'perf-card-value perf-card-value-small', text: versionAccuracyText(v) }),
        el('span', { class: 'perf-card-sub', text: `${v.submissions} submission${v.submissions === 1 ? '' : 's'} · ${v.reviewed} reviewed` }),
      ]))) : note('No submissions with a model version yet.'),
      note('Based on app photos whose airline was confirmed (Yes/No answers or a reviewer). These are small, self-selected samples, not a test set.'),
      el('div', { class: 'perf-sub-grid' }, [
        el('div', { class: 'perf-bars' }, [el('h4', { text: 'Predicted airline' })].concat(byAirline.length
          ? byAirline.map((a) => barRow(titleCase(a.airline), (a.count || 0) / maxCount, String(a.count)))
          : [note('No successful predictions yet.')])),
        el('div', {}, [el('h4', { text: 'Confidence (uncalibrated)' }), hist]),
      ]),
      el('h4', { id: 'perf-recent-title', text: `Recent submissions (latest ${recent.length})` }),
      recent.length ? recentHost : note('No submissions yet.'),
    );
    if (recent.length) renderRecent(recentHost, recent);
  }

  function servingVersion() {
    const serving = state.summary && state.summary.serving;
    return serving && serving.model_version ? String(serving.model_version) : null;
  }

  function renderRecent(host, recent) {
    const counts = Object.fromEntries(RECENT_FILTERS.map(([key]) => [key, filterRecent(recent, key).length]));
    const buttons = el('div', { class: 'perf-filter', role: 'group', 'aria-label': 'Filter recent submissions' },
      RECENT_FILTERS.map(([key, label]) => {
        const button = el('button', { type: 'button', class: 'perf-filter-button', 'aria-pressed': String(state.recentFilter === key),
          'data-filter': key, text: `${label} (${counts[key]})` });
        button.addEventListener('click', () => {
          state.recentFilter = key;
          renderRecent(host, recent);
          const again = host.querySelector(`[data-filter="${key}"]`);
          if (again) again.focus();
        });
        return button;
      }));
    const rows = filterRecent(recent, state.recentFilter);
    const list = rows.length
      ? el('ul', { class: 'perf-recent-list', 'aria-labelledby': 'perf-recent-title' }, rows.map(recentItem))
      : note('No submissions match this filter.');
    host.replaceChildren(buttons, el('div', { class: 'perf-recent-scroll', tabindex: '0', role: 'region',
      'aria-label': 'Recent submissions, scrollable' }, [list]));
  }

  function recentItem(r) {
    const outcome = recentOutcome(r);
    const airline = r.predicted_airline ? titleCase(r.predicted_airline) : 'No prediction';
    const thumb = r.has_photo
      ? el('a', { class: 'perf-thumb', href: photoUrl(r.submission_id, 'original'), target: '_blank', rel: 'noopener',
        title: 'Open the full photo in a new tab' }, [
        el('img', { src: photoUrl(r.submission_id, 'thumb'), alt: `Tail crop sent to the model (guess: ${airline})`,
          loading: 'lazy', decoding: 'async', width: '72', height: '72' }),
        el('span', { class: 'visually-hidden', text: ' (opens full photo in a new tab)' }),
      ])
      : el('span', { class: 'perf-thumb perf-thumb-missing', text: 'No photo' });
    const img = thumb.querySelector && thumb.querySelector('img');
    if (img) {
      img.addEventListener('error', () => {
        thumb.replaceChildren(el('span', { class: 'perf-thumb-missing', text: 'Photo unavailable' }));
      }, { once: true });
    }
    return el('li', { class: `perf-recent-item is-${outcome}` }, [
      thumb,
      el('div', { class: 'perf-recent-main' }, [
        el('div', { class: 'perf-recent-guess' }, [
          el('strong', { text: airline }),
          r.predicted_airline ? el('span', { class: 'perf-muted-inline', text: ` ${pct(r.confidence, 0)} confidence` }) : null,
        ]),
        el('div', { class: 'perf-recent-meta', text: `${localDateTime(r.submitted_at)} · Model ${versionLabel(r.model_version)}` }),
      ]),
      el('span', { class: `perf-outcome perf-outcome-${outcome}`, text: outcomeText(r) }),
    ]);
  }

  // ---------- Loading & tabs ----------
  async function loadAll() {
    const button = $('perf-refresh');
    button.disabled = true;
    state.loaded = true;
    clearPolls();
    await Promise.allSettled([loadSummary(), loadTraining(false), loadSubmissions()]);
    button.disabled = false;
    $('perf-updated').textContent = `Updated ${clockTime(new Date())}`;
  }

  function selectTab(name, opts = {}) {
    const tabs = [...document.querySelectorAll('.view-tab[role="tab"]')];
    if (!tabs.some((t) => t.dataset.tab === name)) name = 'identify';
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      t.classList.toggle('is-active', on);
      const panel = $(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
      if (on && opts.focus) t.focus();
    });
    if (opts.updateHash !== false && root.history && root.history.replaceState) {
      const base = location.pathname + location.search;
      root.history.replaceState(null, '', name === 'performance' ? `${base}#performance` : base);
    }
    if (name === 'performance') {
      if (!state.loaded) loadAll();
      else if (selectedRun() && selectedRun().status === 'RUNNING') loadTraining(true);
    } else clearPolls();
  }

  function initPerformance() {
    const tabs = [...document.querySelectorAll('.view-tab[role="tab"]')];
    if (!tabs.length || !$('panel-performance')) return;
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => selectTab(t.dataset.tab));
      t.addEventListener('keydown', (e) => {
        let next = null;
        if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
        else if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (e.key === 'Home') next = tabs[0];
        else if (e.key === 'End') next = tabs[tabs.length - 1];
        if (!next) return;
        e.preventDefault();
        selectTab(next.dataset.tab, { focus: true });
      });
    });
    $('perf-refresh').addEventListener('click', loadAll);
    $('perf-drill-set').addEventListener('change', populateVariants);
    $('perf-drill-variant').addEventListener('change', loadDrill);
    $('perf-drill-mistakes').addEventListener('change', () => { if (state.drillRows) renderDrill(); });
    root.addEventListener('hashchange', () => selectTab(location.hash === '#performance' ? 'performance' : 'identify', { updateHash: false }));
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) clearPolls();
      else if (perfVisible() && selectedRun() && selectedRun().status === 'RUNNING') loadTraining(true);
    });
    if (location.hash === '#performance') selectTab('performance', { updateHash: false });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initPerformance);
  else initPerformance();
})(typeof window !== 'undefined' ? window : globalThis);
