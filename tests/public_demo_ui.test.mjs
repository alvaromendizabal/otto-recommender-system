/** Execute the actual inline report script with its actual generated DOM/data. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const generate = String.raw`
import json,sys,tempfile
from pathlib import Path
from html.parser import HTMLParser
sys.path.insert(0,str(Path(sys.argv[1])/"src"))
from otto_recsys.public_demo import run_demo
class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root={"tag":"#document","attrs":{},"children":[]}
        self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        node={"tag":tag,"attrs":dict(attrs),"children":[]}
        self.stack[-1]["children"].append(node)
        if tag not in {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}:
            self.stack.append(node)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if self.stack[-1]["tag"]==tag:self.stack.pop()
    def handle_endtag(self,tag):
        for index in range(len(self.stack)-1,0,-1):
            if self.stack[index]["tag"]==tag:
                del self.stack[index:]
                return
    def handle_data(self,data):self.stack[-1]["children"].append(data)
with tempfile.TemporaryDirectory() as temporary:
    output=Path(temporary)/"demo"
    result=run_demo(output)
    html=(output/"report.html").read_text()
    parsed=Tree();parsed.feed(html)
    predictions=json.loads((output/"predictions.json").read_text())
    manifest=json.loads((output/"manifest.json").read_text())
    print(json.dumps(dict(result=result,predictions=predictions,manifest=manifest,html=html,dom=parsed.root),allow_nan=False))
`;
const generated = spawnSync(process.env.PYTHON || 'python3', ['-S', '-B', '-c', generate, root], {
  encoding: 'utf8', maxBuffer: 32 * 1024 * 1024, timeout: 30000,
});
assert.equal(generated.status, 0, generated.stderr);
const fixture = JSON.parse(generated.stdout);
const plain = value => JSON.parse(JSON.stringify(value));
const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

function browser({ mutate = null, downloadFailure = false } = {}) {
  const ids = new Map(), downloads = [], revoked = [], timers = [];
  let active = null;
  function matches(node, selector) {
    return selector.split(',').some(part => {
      const expression = part.trim();
      if (!expression) return false;
      const tag = expression.match(/^[a-z][a-z0-9-]*/i)?.[0];
      if (tag && node.tag !== tag.toLowerCase()) return false;
      for (const match of expression.matchAll(/#([a-z0-9_-]+)/gi)) if (node.id !== match[1]) return false;
      for (const match of expression.matchAll(/\.([a-z0-9_-]+)/gi)) if (!node.className.split(/\s+/).includes(match[1])) return false;
      for (const match of expression.matchAll(/\[([^\]=\s]+)(?:=["']?([^\]"']*)["']?)?\]/g)) {
        if (!node.hasAttribute(match[1])) return false;
        if (match[2] !== undefined && node.getAttribute(match[1]) !== match[2]) return false;
      }
      return true;
    });
  }
  class Element {
    constructor(tag, attrs = {}) {
      this.tag = tag; this.attrs = { ...attrs }; this.childNodes = []; this.parentElement = null;
      this.listeners = new Map(); this.dataset = {};
      this.hidden = Object.hasOwn(attrs, 'hidden'); this.disabled = Object.hasOwn(attrs, 'disabled'); this.open = Object.hasOwn(attrs, 'open');
      this.className = attrs.class || ''; this._value = attrs.value;
      this.style = { setProperty(name, value) { this[name] = String(value); } };
      this.classList = {
        contains: name => this.className.split(/\s+/).includes(name),
        add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(' '); },
        remove: (...names) => { this.className = this.className.split(/\s+/).filter(name => !names.includes(name)).join(' '); },
        toggle: (name, force) => { const present = this.classList.contains(name); const wanted = force === undefined ? !present : force; wanted ? this.classList.add(name) : this.classList.remove(name); return wanted; },
      };
      for (const [name, value] of Object.entries(attrs)) this.setAttribute(name, value ?? '');
    }
    setAttribute(name, value) {
      this.attrs[name] = String(value);
      if (name === 'id') { this.id = String(value); ids.set(this.id, this); }
      if (name === 'class') this.className = String(value);
      if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
    }
    getAttribute(name) { return Object.hasOwn(this.attrs, name) ? this.attrs[name] : null; }
    hasAttribute(name) { return Object.hasOwn(this.attrs, name); }
    removeAttribute(name) { delete this.attrs[name]; if (name === 'open') this.open = false; }
    get children() { return this.childNodes.filter(child => child instanceof Element); }
    get textContent() { return this.childNodes.map(child => typeof child === 'string' ? child : child.textContent).join(''); }
    set textContent(value) { this.childNodes = [String(value)]; }
    get text() { return this.textContent; }
    get options() { return this.querySelectorAll('option'); }
    get selectedIndex() { return this.options.findIndex(option => option.value === this.value); }
    get value() { return this._value ?? (this.tag === 'select' ? this.options.find(option => option.hasAttribute('selected'))?.value ?? this.options[0]?.value ?? '' : ''); }
    set value(value) { this._value = String(value); }
    append(...children) { for (const child of children) { this.childNodes.push(child); if (child instanceof Element) child.parentElement = this; } }
    appendChild(child) { this.append(child); return child; }
    replaceChildren(...children) { this.childNodes = []; this.append(...children); }
    remove() { if (this.parentElement) this.parentElement.childNodes = this.parentElement.childNodes.filter(child => child !== this); this.parentElement = null; this.removed = true; }
    descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
    querySelectorAll(selector) { return this.descendants().filter(node => matches(node, selector)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
    closest(selector) { let node = this; while (node) { if (matches(node, selector)) return node; node = node.parentElement; } return null; }
    addEventListener(name, listener) { if (!this.listeners.has(name)) this.listeners.set(name, []); this.listeners.get(name).push(listener); }
    fire(name, options = {}) {
      let stopped = false;
      const event = { target: this, currentTarget: this, preventDefault() {}, stopPropagation() { stopped = true; }, ...options };
      let node = this;
      while (node) { event.currentTarget = node; for (const listener of node.listeners.get(name) ?? []) listener(event); if (stopped) break; node = node.parentElement; }
    }
    click() { if (this.disabled) return; this.fire('click'); if (this.tag === 'a' && this.download) downloads.at(-1).link = this; }
    focus() { active = this; }
  }
  function build(raw) {
    if (typeof raw === 'string') return raw;
    const node = new Element(raw.tag, raw.attrs);
    node.append(...raw.children.map(build));
    return node;
  }
  const tree = build(fixture.dom);
  const document = {
    body: tree.querySelector('body'),
    get activeElement() { return active; },
    getElementById: id => {
      const found = ids.get(id);
      let parent = found;
      while (parent?.parentElement) parent = parent.parentElement;
      return parent === tree ? found : null;
    },
    querySelectorAll: selector => tree.querySelectorAll(selector),
    querySelector: selector => tree.querySelector(selector),
    createElement: tag => new Element(tag),
    addEventListener: (...args) => tree.addEventListener(...args),
  };
  const ui = {
    document, tree, downloads, revoked,
    el(id) { const node = document.getElementById(id); assert.ok(node, `Element #${id} exists`); return node; },
    panels: () => document.querySelectorAll('[data-demo-panel]'),
    visible: () => document.querySelectorAll('[data-demo-panel]').filter(panel => !panel.hidden),
    select(id, value) { const node = this.el(id); node.value = value; node.fire('change'); },
    click(id) { this.el(id).click(); },
    flushTimers() { for (const callback of timers.splice(0)) callback(); },
    predictionData: () => JSON.parse(document.getElementById('demo-prediction-data').textContent),
  };
  if (mutate) mutate(ui);
  const runtime = vm.createContext({ document, Blob, URL: {
    createObjectURL(blob) { if (downloadFailure) throw new Error('download fixture unavailable'); downloads.push({ blob }); return `blob:synthetic-${downloads.length}`; },
    revokeObjectURL(url) { revoked.push(url); },
  }, setTimeout(callback) { timers.push(callback); return timers.length; }, console });
  runtime.window = runtime;
  const scripts = tree.querySelectorAll('script').filter(node => node.getAttribute('type') !== 'application/json');
  assert.equal(scripts.length, 1, 'Run the single actual inline application script');
  vm.runInContext(scripts[0].textContent, runtime, { filename: 'generated-otto-report.js', timeout: 5000 });
  return ui;
}

test('real Python generation and seal remain separate from displayed evaluation targets', () => {
  assert.equal(fixture.result.synthetic, true);
  assert.equal(fixture.result.provenance.targets_used_in_fit_or_inference, false);
  assert.equal(fixture.result.provenance.predictions_sealed_before_evaluation, true);
  assert.equal(fixture.result.provenance.predictions_sha256, fixture.manifest.files['predictions.json'].sha256);
  const ui = browser(), embedded = ui.predictionData();
  assert.equal(embedded.evidence_type, 'SYNTHETIC_ONLY');
  assert.equal(embedded.predictions_sha256, fixture.result.provenance.predictions_sha256);
  assert.doesNotMatch(JSON.stringify(embedded), /"targets"|"evaluation"|"metrics"/);
  for (const example of embedded.examples) {
    const original = fixture.result.examples.find(row => row.session === example.session);
    const sealed = fixture.predictions.rows.find(row => row.session === example.session);
    assert.deepEqual(example.prefix, original.prefix);
    assert.deepEqual(example.objectives, sealed.objectives);
    assert.ok(example.prefix.every(event => event.ts < example.query_ts));
  }
});

test('actual bootstrap selects exactly the first click panel and enables ready controls', () => {
  const ui = browser();
  assert.equal(ui.panels().length, fixture.result.examples.length * 3);
  assert.equal(ui.visible().length, 1);
  assert.equal(ui.visible()[0].dataset.session, '0');
  assert.equal(ui.visible()[0].dataset.objective, 'clicks');
  for (const id of ['session-select', 'objective-select', 'reset-selection', 'export-selection-json', 'export-selection-csv']) assert.equal(ui.el(id).disabled, false);
  assert.equal(ui.el('selection-error').hidden, true);
  assert.match(ui.el('selection-status').textContent, /10,?000/);
});

test('all session and objective change handlers select the matching real prediction panel', () => {
  const ui = browser();
  for (let index = 0; index < fixture.result.examples.length; index++) for (const action of ['clicks', 'carts', 'orders']) {
    ui.select('session-select', index); ui.select('objective-select', action);
    assert.equal(ui.visible().length, 1);
    const panel = ui.visible()[0];
    assert.equal(panel.dataset.session, String(index)); assert.equal(panel.dataset.objective, action);
    const ranked = panel.querySelectorAll('[data-ranked-item]').map(row => Number(row.dataset.rankedItem));
    assert.deepEqual(ranked, fixture.result.examples[index].recommendations[action]);
  }
});

test('selection changes close future-evaluation disclosures and retain keyboard focus', () => {
  const ui = browser();
  ui.document.querySelectorAll('details.evaluation').forEach(node => { node.open = true; });
  const control = ui.el('objective-select'); control.focus();
  ui.select('objective-select', 'orders');
  assert.ok(ui.document.querySelectorAll('details.evaluation').every(node => !node.open));
  assert.equal(ui.document.activeElement, control);
});

test('prefix buttons highlight matching ranked items without changing order, scores or targets', () => {
  const ui = browser(), panel = ui.visible()[0], rows = panel.querySelectorAll('[data-ranked-item]');
  const before = rows.map(row => [row.dataset.rankedItem, row.textContent]);
  const events = panel.querySelectorAll('[data-event-item]');
  assert.equal(events.length, fixture.result.examples[0].prefix.length);
  events.forEach((button, index) => {
    assert.equal(Number(button.dataset.eventItem), fixture.result.examples[0].prefix[index].item);
    button.focus(); button.click();
    assert.equal(ui.document.activeElement, button);
    assert.equal(button.getAttribute('aria-pressed'), 'true');
    for (const row of rows) assert.equal(row.classList.contains('is-observed-match'), row.dataset.rankedItem === button.dataset.eventItem);
    assert.ok(events.filter(event => event !== button).every(event => event.getAttribute('aria-pressed') === 'false'));
    assert.deepEqual(rows.map(row => [row.dataset.rankedItem, row.textContent]), before);
    assert.ok(panel.querySelector('[data-event-status]').textContent.length > 0);
    assert.ok(panel.querySelectorAll('details.evaluation').every(node => !node.open));
    button.click();
    assert.equal(button.getAttribute('aria-pressed'), 'false');
    assert.ok(rows.every(row => !row.classList.contains('is-observed-match')));
  });
});

test('rendered score disclosures reconcile the three actual contributions without calling them probabilities', () => {
  const ui = browser(), weights = { cooccurrence: .6, prefix_recency: .3, popularity: .1 };
  for (const panel of ui.panels()) {
    const example = fixture.result.examples[Number(panel.dataset.session)], action = panel.dataset.objective;
    panel.querySelectorAll('[data-ranked-item]').forEach((row, index) => {
      const expected = example.explanations[action][index], detail = row.querySelector('details.score-detail');
      assert.equal(detail.open, false);
      const cells = detail.querySelectorAll('tbody')[0].children.map(tr => tr.children.map(td => td.textContent));
      assert.equal(cells.length, 3);
      for (const [label, value, weight, contribution] of cells) {
        const key = label.replaceAll(' ', '_');
        close(Number(value), expected.features[key], .0000051);
        assert.equal(Number(weight), weights[key]);
        close(Number(contribution), weights[key] * expected.features[key], .0000051);
      }
      close(cells.reduce((sum, item) => sum + Number(item[3]), 0), expected.score, .0000151);
      assert.match(detail.textContent, /before rounding/);
    });
  }
});

test('candidate coverage and achieved ranking use the same deduplicated denominator inside future evaluation only', () => {
  const ui = browser();
  for (const panel of ui.panels()) {
    const example = fixture.result.examples[Number(panel.dataset.session)], action = panel.dataset.objective;
    const targets = [...new Set(example.targets[action])], denominator = Math.min(20, targets.length);
    const retrieved = Math.min(20, targets.filter(item => example.candidates[action].includes(item)).length);
    const ranked = targets.filter(item => example.recommendations[action].includes(item)).length;
    const coverage = panel.querySelector('.coverage-comparison');
    assert.ok(coverage.closest('details.evaluation'));
    assert.equal(coverage.closest('details.evaluation').open, false);
    assert.deepEqual(coverage.querySelectorAll('strong').map(node => node.textContent), [`${retrieved} / ${denominator}`, `${ranked} / ${denominator}`]);
    const bars = coverage.querySelectorAll('i').map(node => Number(node.getAttribute('style').match(/width:([\d.]+)%/)[1]));
    close(bars[0], denominator ? 100 * retrieved / denominator : 0, 1e-8);
    close(bars[1], denominator ? 100 * ranked / denominator : 0, 1e-8);
    assert.match(coverage.textContent, new RegExp(`Retrieval shortfall: ${denominator - retrieved}`));
    assert.match(coverage.textContent, new RegExp(`Additional ranking shortfall: ${retrieved - ranked}`));
  }
});

test('selected JSON export exactly preserves sealed predictions and excludes future outcomes', async () => {
  const ui = browser(); ui.select('session-select', '2'); ui.select('objective-select', 'orders');
  ui.click('export-selection-json'); assert.equal(ui.downloads.length, 1);
  const downloaded = JSON.parse(await ui.downloads[0].blob.text());
  const original = fixture.result.examples[2], objective = fixture.predictions.rows.find(row => row.session === original.session).objectives.orders;
  assert.equal(downloaded.schema, 'otto-synthetic-selected-predictions-v1');
  assert.equal(downloaded.evidence_type, 'SYNTHETIC_ONLY');
  assert.equal(downloaded.session, original.session); assert.equal(downloaded.query_ts, original.query_ts); assert.equal(downloaded.objective, 'orders');
  assert.deepEqual(downloaded.prefix, original.prefix);
  for (const key of ['candidates', 'recommendations', 'explanations']) assert.deepEqual(downloaded[key], objective[key]);
  assert.equal(downloaded.predictions_sha256, fixture.result.provenance.predictions_sha256);
  assert.doesNotMatch(JSON.stringify(downloaded), /"targets"|"evaluation"|"metrics"/);
  assert.match(ui.downloads[0].link.download, /\.json$/); assert.equal(ui.downloads[0].link.removed, true);
  ui.flushTimers(); assert.equal(ui.revoked.length, 1);
});

test('selected CSV reconciles each score with the exact public feature components', async () => {
  const ui = browser(); ui.select('session-select', '1'); ui.select('objective-select', 'carts'); ui.click('export-selection-csv');
  const lines = (await ui.downloads[0].blob.text()).trim().split('\n');
  const header = lines.shift().split(','), rows = lines.map(line => Object.fromEntries(line.split(',').map((value, index) => [header[index], value])));
  const explanations = fixture.result.examples[1].explanations.carts;
  assert.equal(rows.length, explanations.length);
  rows.forEach((row, index) => {
    const expected = explanations[index];
    assert.equal(Number(row.rank), index + 1); assert.equal(Number(row.item), expected.item);
    close(Number(row.score), expected.score, 1e-10);
    close(Number(row.cooccurrence), expected.features.cooccurrence, 1e-10);
    close(Number(row.prefix_recency), expected.features.prefix_recency, 1e-10);
    close(Number(row.popularity), expected.features.popularity, 1e-10);
    close(Number(row.score), .6 * Number(row.cooccurrence) + .3 * Number(row.prefix_recency) + .1 * Number(row.popularity), 1e-10);
    assert.equal(Number(row.session), fixture.result.examples[1].session); assert.equal(row.objective, 'carts');
  });
  assert.match(lines[0], /synthetic/i);
});

test('invalid selection hides previous results and blocks stale exports until reset', () => {
  const ui = browser(); ui.select('session-select', '999');
  assert.equal(ui.visible().length, 0); assert.equal(ui.el('selection-error').hidden, false);
  for (const id of ['export-selection-json', 'export-selection-csv']) assert.equal(ui.el(id).disabled, true);
  ui.click('export-selection-json'); assert.equal(ui.downloads.length, 0);
  ui.click('reset-selection'); assert.equal(ui.visible().length, 1); assert.equal(ui.el('selection-error').hidden, true);
  assert.equal(ui.el('session-select').value, '0'); assert.equal(ui.el('objective-select').value, 'clicks');
});

test('missing or malformed embedded prediction data cannot enable export', () => {
  for (const mutate of [
    ui => { ui.el('demo-prediction-data').remove(); },
    ui => { ui.el('demo-prediction-data').textContent = '{invalid'; },
    ui => { const data = ui.predictionData(); data.evidence_type = 'REAL_DATA'; ui.el('demo-prediction-data').textContent = JSON.stringify(data); },
    ui => { const data = ui.predictionData(); data.examples = []; ui.el('demo-prediction-data').textContent = JSON.stringify(data); },
    ui => { const data = ui.predictionData(); data.examples[0].prefix[0].ts = data.examples[0].query_ts; ui.el('demo-prediction-data').textContent = JSON.stringify(data); },
    ui => { const data = ui.predictionData(); data.examples[0].objectives.clicks.recommendations[1] = data.examples[0].objectives.clicks.recommendations[0]; ui.el('demo-prediction-data').textContent = JSON.stringify(data); },
    ui => { const data = ui.predictionData(); data.examples[0].objectives.clicks.explanations[0].score = null; ui.el('demo-prediction-data').textContent = JSON.stringify(data); },
  ]) {
    const ui = browser({ mutate });
    assert.equal(ui.visible().length, 0); assert.equal(ui.el('selection-error').hidden, false);
    assert.equal(ui.el('export-selection-json').disabled, true); assert.equal(ui.el('export-selection-csv').disabled, true);
  }
});

test('download failures are visible without falsely reporting successful export', () => {
  const ui = browser({ downloadFailure: true }); ui.click('export-selection-json');
  assert.equal(ui.downloads.length, 0);
  assert.match(ui.el('export-status').textContent, /unavailable|failed|error/i);
});

test('static report preserves synthetic scope, self-contained runtime and closed evaluation panels', () => {
  const ui = browser();
  assert.ok(ui.document.querySelectorAll('details.evaluation').every(node => !node.open));
  assert.ok(ui.document.querySelectorAll('details.score-detail').length > 0);
  assert.match(fixture.html, /All item IDs, sessions and results on this page are synthetic/);
  assert.match(fixture.html, /retrieval ceiling/);
  assert.match(fixture.html, /not a calibrated probability/);
  const scripts = ui.tree.querySelectorAll('script'); assert.ok(scripts.every(node => !node.hasAttribute('src')));
  const actualScript = scripts.find(node => node.getAttribute('type') !== 'application/json').textContent;
  assert.doesNotMatch(actualScript, /\bfetch\s*\(|XMLHttpRequest|WebSocket|localStorage|\beval\s*\(/);
  const css = ui.tree.querySelector('style').textContent;
  assert.match(css, /prefers-reduced-motion/); assert.match(css, /focus-visible/); assert.match(css, /\[hidden\]/);
  for (const match of css.matchAll(/font(?:-size)?\s*:\s*(?:\d+\s+)?(\d+)px/g)) assert.ok(Number(match[1]) >= 12, `Readable text: ${match[0]}`);
});
