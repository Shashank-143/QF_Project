const INITIAL_CAPITAL = 10000;
const isLocal =
  !window.location.hostname ||
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const API_BASE = isLocal ? "http://localhost:8000" : window.location.origin;
const SUMMARY_PATH = `${API_BASE}/summary`;
const POSITIVE_PAIRS_PATH = `${API_BASE}/pairs/positive`;
const NEGATIVE_PAIRS_PATH = `${API_BASE}/pairs/negative`;
const BACKTEST_SUMMARY_PATH = `${API_BASE}/backtest_summary`;
const DEFAULT_MODEL = "SAC";
let selectedModel = DEFAULT_MODEL;

const fields = {
  final: "stat-final",
  delta: "stat-delta",
  initial: "stat-initial",
  pnl: "stat-pnl",
  sharpe: "stat-sharpe",
  alpha: "stat-alpha",
  dd: "stat-dd",
  ntrades: "stat-ntrades",
  positiveList: "positive-list",
  negativeList: "negative-list",
  summaryBody: "summary-tbody",
  chart: "chart",
  detailPairName: "t-pairname",
  detailPairTag: "t-pairtag",
  detailModel: "t-model",
  detailQty: "t-qty",
  detailSharesA: "t-shares-a",
  detailSharesB: "t-shares-b",
  detailCost: "t-cost",
  detailCapitalAfter: "t-capital-after",
  detailEntry: "t-entry-price",
  detailExit: "t-exit-price",
  detailFinal: "t-finaleq",
  detailPnl: "t-pnl",
};

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function parseNumber(value) {
  if (value == null || value === "") return NaN;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function fmtINR(amount) {
  const parsed = parseNumber(amount);
  if (Number.isNaN(parsed)) return "—";
  const sign = parsed < 0 ? "-" : "";
  return sign + new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(Math.abs(parsed));
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function fmtPricePair(priceA, priceB) {
  const parsedA = parseNumber(priceA);
  const parsedB = parseNumber(priceB);
  if (!Number.isFinite(parsedA) && !Number.isFinite(parsedB)) return "—";
  if (!Number.isFinite(parsedB)) return fmtINR(parsedA);
  if (!Number.isFinite(parsedA)) return fmtINR(parsedB);
  return `${fmtINR(parsedA)} / ${fmtINR(parsedB)}`;
}

function clearChildren(container) {
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
}

function createPairItem(pair) {
  const li = document.createElement("li");
  li.className = "pair-item";
  li.addEventListener("click", () => selectPair(pair));

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.color = pair.pair_type === "positive" ? "var(--positive)" : "var(--negative)";

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = pair.pair;

  const corr = document.createElement("span");
  corr.className = "corr";
  corr.textContent = pair.pair_type === "positive" ? "Positive" : "Negative";

  const strengthValue = Number(pair.gnn_strength);
  const strength = document.createElement("span");
  strength.className = `pnl ${strengthValue >= 0 ? "pos" : "neg"}`;
  strength.textContent = Number.isFinite(strengthValue) ? strengthValue.toFixed(3) : "—";

  li.appendChild(swatch);
  li.appendChild(name);
  li.appendChild(corr);
  li.appendChild(strength);

  return li;
}

function renderPairLists(positivePairs, negativePairs) {
  const positive = document.getElementById(fields.positiveList);
  const negative = document.getElementById(fields.negativeList);
  if (!positive || !negative) return;

  clearChildren(positive);
  clearChildren(negative);

  positivePairs.forEach((item) => positive.appendChild(createPairItem(item)));
  negativePairs.forEach((item) => negative.appendChild(createPairItem(item)));

  const positiveCount = document.querySelector(".pairs-block.positive .count");
  const negativeCount = document.querySelector(".pairs-block.negative .count");
  if (positiveCount) positiveCount.textContent = positivePairs.length;
  if (negativeCount) negativeCount.textContent = negativePairs.length;
}

function renderSummaryTable(summary) {
  const body = document.getElementById(fields.summaryBody);
  if (!body) return;

  clearChildren(body);
  summary.forEach((pair) => {
    const tradeCount = pair.trade_count ?? "—";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${pair.pair}</td>
      <td>${pair.pair_type}</td>
      <td>${fmtINR(pair.pnl)}</td>
      <td>${tradeCount}</td>
      <td>${typeof pair.sharpe === "number" ? pair.sharpe.toFixed(2) : "—"}</td>
      <td>${typeof pair.max_drawdown === "number" ? fmtPct(-pair.max_drawdown * 100) : "—"}</td>
    `;
    body.appendChild(row);
  });
}

function renderStats(summary, stats = null) {
  const totalPairs = summary.length;
  const totalCapital = stats?.initial_capital ?? INITIAL_CAPITAL * totalPairs;
  const totalPnl = stats?.total_pnl ?? summary.reduce((sum, item) => sum + Number(item.pnl || 0), 0);
  const finalEquity = stats?.final_equity ?? totalCapital + totalPnl;
  const delta = stats?.return_pct ?? (totalCapital ? (totalPnl / totalCapital) * 100 : 0);
  const trades = summary.reduce((sum, item) => sum + Number(item.trade_count || 0), 0);

  setText(fields.final, fmtINR(finalEquity));
  setText(fields.delta, fmtPct(delta));
  setText(fields.initial, fmtINR(totalCapital));
  setText(fields.pnl, fmtINR(totalPnl));
  setText(fields.sharpe, stats?.sharpe_ratio != null ? stats.sharpe_ratio.toFixed(2) : "—");
  setText(fields.alpha, stats?.alpha != null ? stats.alpha.toFixed(3) : "—");
  setText(fields.dd, stats?.max_drawdown != null ? fmtPct(-stats.max_drawdown * 100) : "—");
  setText(fields.ntrades, `${trades}`);
}

function renderChart(summary) {
  const chart = document.getElementById(fields.chart);
  if (!chart) return;

  const width = 760;
  const height = 180;
  const padding = 40;
  const data = summary.slice(0, 12);
  const values = data.map((item) => Number(item.pnl || 0));
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const range = max - min || 1;
  const barWidth = Math.max(24, Math.floor((width - padding * 2) / data.length) - 10);

  chart.innerHTML = "";
  chart.setAttribute("viewBox", `0 0 ${width} ${height + 40}`);
  chart.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", padding);
  axis.setAttribute("x2", width - padding);
  axis.setAttribute("y1", height);
  axis.setAttribute("y2", height);
  axis.setAttribute("stroke", "#394150");
  axis.setAttribute("stroke-width", "1");
  chart.appendChild(axis);

  data.forEach((item, index) => {
    const value = Number(item.pnl || 0);
    const barHeight = Math.round(((value - min) / range) * height);
    const x = padding + index * (barWidth + 10);
    const y = height - barHeight;

    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", x);
    bar.setAttribute("y", y);
    bar.setAttribute("width", barWidth);
    bar.setAttribute("height", Math.max(2, barHeight));
    bar.setAttribute("fill", item.pnl >= 0 ? "#4ade80" : "#f87171");
    chart.appendChild(bar);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x + barWidth / 2);
    label.setAttribute("y", height + 16);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", "#9aa5b4");
    label.textContent = item.pair.split("-")[0];
    chart.appendChild(label);
  });
}

function selectPair(pair) {
  setText(fields.detailPairName, pair.pair);
  setText(fields.detailPairTag, pair.pair_type.toUpperCase());
  setText(fields.detailModel, selectedModel);
  setText(fields.detailQty, pair.trade_count || "—");
  setText(fields.detailSharesA, "—");
  setText(fields.detailSharesB, "—");
  setText(fields.detailEntry, "—");
  setText(fields.detailExit, "—");
  setText(fields.detailCost, "—");
  setText(fields.detailCapitalAfter, "—");
  setText(fields.detailFinal, fmtINR(pair.equity));
  setText(fields.detailPnl, fmtINR(pair.pnl));

  loadPairDetails(pair.pair);
}

async function loadPairDetails(pairName) {
  try {
    const response = await fetch(`${API_BASE}/pair/${encodeURIComponent(pairName)}`);
    if (!response.ok) throw new Error("Unable to load pair details");
    const detail = await response.json();

    if (!detail || detail.error) {
      return;
    }

    const sharesA = parseNumber(detail.shares_a);
    const sharesB = parseNumber(detail.shares_b);
    setText(fields.detailSharesA, Number.isFinite(sharesA) && sharesA !== 0 ? sharesA.toFixed(2) : "—");
    setText(fields.detailSharesB, Number.isFinite(sharesB) && sharesB !== 0 ? sharesB.toFixed(2) : "—");
    setText(fields.detailCost, fmtINR(detail.total_cost));
    setText(fields.detailCapitalAfter, fmtINR(detail.capital_after));
    setText(fields.detailEntry, fmtPricePair(detail.entry_price_a, detail.entry_price_b));
    setText(fields.detailExit, fmtPricePair(detail.exit_price_a, detail.exit_price_b));
    setText(fields.detailFinal, fmtINR(detail.final_equity));
    setText(fields.detailPnl, fmtINR(detail.total_pnl));
    setText(fields.detailQty, detail.trade_count ?? "—");
  } catch (error) {
    console.error(error);
  }
}

function setActiveModel(model) {
  selectedModel = model;
  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.model === model);
  });
  setText(fields.detailModel, model);
  updateTogglePill();
}

function updateTogglePill() {
  const toggle = document.querySelector(".model-toggle");
  const pill = toggle?.querySelector(".toggle-pill");
  const active = toggle?.querySelector(".toggle-btn.active");
  if (!toggle || !pill || !active) return;

  const toggleRect = toggle.getBoundingClientRect();
  const activeRect = active.getBoundingClientRect();
  const offset = activeRect.left - toggleRect.left;

  pill.style.width = `${activeRect.width}px`;
  pill.style.transform = `translateX(${offset}px)`;
}

async function loadDashboard() {
  try {
    const [summaryResponse, positiveResponse, negativeResponse, backtestResponse] = await Promise.all([
      fetch(SUMMARY_PATH),
      fetch(POSITIVE_PAIRS_PATH),
      fetch(NEGATIVE_PAIRS_PATH),
      fetch(BACKTEST_SUMMARY_PATH),
    ]);

    if (!summaryResponse.ok) throw new Error("Unable to load backend summary");
    if (!positiveResponse.ok) throw new Error("Unable to load positive pairs");
    if (!negativeResponse.ok) throw new Error("Unable to load negative pairs");
    if (!backtestResponse.ok) throw new Error("Unable to load backtest summary");

    const payload = await summaryResponse.json();
    const positivePayload = await positiveResponse.json();
    const negativePayload = await negativeResponse.json();
    const backtestPayload = await backtestResponse.json();

    const summary = Array.isArray(payload.summary) ? payload.summary : [];
    const stats = payload.stats || null;
    const positivePairs = Array.isArray(positivePayload.pairs) ? positivePayload.pairs : [];
    const negativePairs = Array.isArray(negativePayload.pairs) ? negativePayload.pairs : [];
    const backtestSummary = Array.isArray(backtestPayload.summary) ? backtestPayload.summary : [];

    renderStats(summary, stats);
    renderPairLists(positivePairs, negativePairs);
    renderSummaryTable(backtestSummary);
    renderChart(summary);

    if (summary.length > 0) {
      selectPair(summary[0]);
    }
  } catch (error) {
    console.error(error);
    setText(fields.final, "Error");
    setText(fields.delta, "Error");
    setText(fields.initial, "Error");
    setText(fields.pnl, "Error");
  }
}

window.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
  setActiveModel(selectedModel);
  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => setActiveModel(btn.dataset.model));
  });

  window.addEventListener("resize", updateTogglePill);
});
