"""Live browser dashboard for unitree_rl_lab safety value records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_CSV = Path(
    "log/safety_value/safety_value_records.csv"
)

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safety Value Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1318;
      --panel: #171d24;
      --panel-2: #1f2630;
      --line: #46b6a6;
      --line-soft: rgba(70, 182, 166, 0.18);
      --text: #edf2f7;
      --muted: #a5b1bf;
      --border: #303946;
      --accent: #e6b450;
      --signal: #df6b6b;
      --danger: #df6b6b;
      --ok: #78c679;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }

    header {
      padding: 18px 24px 14px;
      border-bottom: 1px solid var(--border);
      background: #11171d;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }

    .source {
      color: var(--muted);
      font-size: 12px;
      max-width: 54vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1.2fr)) repeat(4, minmax(110px, 1fr));
      gap: 1px;
      background: var(--border);
      border-bottom: 1px solid var(--border);
    }

    .metric {
      min-height: 88px;
      padding: 14px 18px;
      background: var(--panel);
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 8px;
    }

    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }

    .value {
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }

    .metric.small .value {
      font-size: 19px;
      font-weight: 650;
    }

    .main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      min-height: 0;
    }

    .chart-area {
      min-width: 0;
      padding: 20px 24px 24px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }

    button,
    input {
      height: 34px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      font: inherit;
      font-size: 13px;
    }

    button {
      padding: 0 12px;
      cursor: pointer;
    }

    button.active {
      border-color: var(--line);
      color: var(--line);
    }

    .field {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .legend {
      display: inline-flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }

    .swatch {
      width: 18px;
      height: 3px;
      border-radius: 999px;
      background: var(--line);
    }

    .swatch.signal { background: var(--signal); }

    input {
      width: 92px;
      padding: 0 10px;
      font-variant-numeric: tabular-nums;
    }

    .plot {
      position: relative;
      min-height: 420px;
      border: 1px solid var(--border);
      background: #121820;
      border-radius: 8px;
      overflow: hidden;
    }

    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      pointer-events: none;
    }

    aside {
      border-left: 1px solid var(--border);
      background: #11171d;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .side-block {
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
    }

    .side-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 7px 0;
      font-size: 13px;
      color: var(--muted);
    }

    .side-row strong {
      color: var(--text);
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--muted);
    }

    .dot.live { background: var(--ok); }
    .dot.warn { background: var(--accent); }
    .dot.error { background: var(--danger); }

    @media (max-width: 900px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .source {
        max-width: 100%;
      }

      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .main {
        grid-template-columns: 1fr;
      }

      aside {
        border-left: 0;
        border-top: 1px solid var(--border);
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Safety Value Monitor</h1>
      <div class="source" id="csvPath"></div>
    </header>

    <section class="metrics">
      <div class="metric">
        <div class="label">Current Value</div>
        <div class="value" id="currentValue">--</div>
      </div>
      <div class="metric">
        <div class="label">Current Signal</div>
        <div class="value" id="currentSignal">--</div>
      </div>
      <div class="metric small">
        <div class="label">Step</div>
        <div class="value" id="currentStep">--</div>
      </div>
      <div class="metric small">
        <div class="label">Min</div>
        <div class="value" id="minValue">--</div>
      </div>
      <div class="metric small">
        <div class="label">Max</div>
        <div class="value" id="maxValue">--</div>
      </div>
      <div class="metric small">
        <div class="label">Samples</div>
        <div class="value" id="sampleCount">--</div>
      </div>
    </section>

    <main class="main">
      <section class="chart-area">
        <div class="toolbar">
          <button id="pauseButton" type="button">Pause</button>
          <button id="autoScaleButton" class="active" type="button">Auto Scale</button>
          <label class="field">Window <input id="windowInput" type="number" min="10" max="5000" step="10"></label>
          <label class="field">Threshold <input id="thresholdInput" type="number" step="0.01" placeholder="none"></label>
          <span class="legend">
            <span class="legend-item"><span class="swatch"></span>Value</span>
            <span class="legend-item"><span class="swatch signal"></span>Signal</span>
          </span>
          <span class="status"><span id="statusDot" class="dot"></span><span id="statusText">Starting</span></span>
        </div>
        <div class="plot">
          <canvas id="chart"></canvas>
          <div class="empty" id="emptyText">Waiting for safety value data</div>
        </div>
      </section>

      <aside>
        <div class="side-block">
          <div class="side-row"><span>Obs key</span><strong id="obsKey">--</strong></div>
          <div class="side-row"><span>Time</span><strong id="simTime">--</strong></div>
          <div class="side-row"><span>CSV size</span><strong id="csvSize">--</strong></div>
          <div class="side-row"><span>Updated</span><strong id="updatedAt">--</strong></div>
          <div class="side-row"><span>Record age</span><strong id="recordAge">--</strong></div>
          <div class="side-row"><span>Stream</span><strong id="streamState">--</strong></div>
        </div>
        <div class="side-block">
          <div class="side-row"><span>Visible range</span><strong id="rangeText">--</strong></div>
          <div class="side-row"><span>Poll</span><strong id="pollText">--</strong></div>
          <div class="side-row"><span>Threshold state</span><strong id="thresholdState">--</strong></div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const SERVER_CONFIG = __SERVER_CONFIG__;
    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");
    const fmtValue = new Intl.NumberFormat(undefined, { maximumFractionDigits: 6 });
    const fmtCompact = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
    const state = {
      rows: [],
      paused: false,
      autoScale: true,
      window: SERVER_CONFIG.window,
      threshold: SERVER_CONFIG.threshold,
      staleSeconds: SERVER_CONFIG.staleSeconds,
      stale: false,
      recordAgeS: null,
      lastError: "",
      timer: null
    };

    const el = (id) => document.getElementById(id);
    el("csvPath").textContent = SERVER_CONFIG.csv;
    el("windowInput").value = state.window;
    if (state.threshold !== null && state.threshold !== undefined) {
      el("thresholdInput").value = state.threshold;
    }
    el("pollText").textContent = `${SERVER_CONFIG.pollMs} ms`;

    function setStatus(kind, text) {
      const dot = el("statusDot");
      dot.className = `dot ${kind}`;
      el("statusText").textContent = text;
    }

    function formatBytes(bytes) {
      if (!Number.isFinite(bytes)) return "--";
      const units = ["B", "KB", "MB", "GB"];
      let value = bytes;
      let idx = 0;
      while (value >= 1024 && idx < units.length - 1) {
        value /= 1024;
        idx += 1;
      }
      return `${fmtCompact.format(value)} ${units[idx]}`;
    }

    function formatDuration(seconds) {
      if (!Number.isFinite(seconds)) return "--";
      if (seconds < 60) return `${fmtCompact.format(seconds)} s`;
      const minutes = Math.floor(seconds / 60);
      const rem = seconds - minutes * 60;
      return `${minutes}m ${fmtCompact.format(rem)}s`;
    }

    function setText(id, value) {
      el(id).textContent = value;
    }

    function updateMetrics(payload) {
      const latest = payload.latest;
      const stats = payload.stats || {};
      if (!latest) {
        setText("currentValue", "--");
        setText("currentSignal", "--");
        setText("currentStep", "--");
        setText("minValue", "--");
        setText("maxValue", "--");
        setText("sampleCount", "0");
        setText("obsKey", "--");
        setText("simTime", "--");
        setText("csvSize", formatBytes(payload.size_bytes));
        setText("updatedAt", new Date().toLocaleTimeString());
        setText("recordAge", formatDuration(payload.record_age_s));
        setText("streamState", payload.stale ? "stale" : "waiting");
        setText("rangeText", "--");
        setText("thresholdState", "--");
        return;
      }

      setText("currentValue", fmtValue.format(latest.safety_value));
      setText(
        "currentSignal",
        Number.isFinite(latest.safety_signal) ? fmtValue.format(latest.safety_signal) : "--"
      );
      setText("currentStep", String(latest.step));
      setText("minValue", fmtValue.format(stats.min));
      setText("maxValue", fmtValue.format(stats.max));
      setText("sampleCount", String(payload.count));
      setText("obsKey", latest.obs_key || "--");
      setText("simTime", `${fmtCompact.format(latest.time_s)} s`);
      setText("csvSize", formatBytes(payload.size_bytes));
      setText("updatedAt", new Date().toLocaleTimeString());
      setText("recordAge", formatDuration(payload.record_age_s));
      setText("streamState", payload.stale ? "holding last" : "recording");
      setText("rangeText", `${fmtValue.format(stats.min)} .. ${fmtValue.format(stats.max)}`);

      if (state.threshold === null || state.threshold === undefined || Number.isNaN(state.threshold)) {
        setText("thresholdState", "--");
      } else {
        const referenceValue = Number.isFinite(latest.safety_signal) ? latest.safety_signal : latest.safety_value;
        const side = referenceValue >= state.threshold ? "above" : "below";
        setText("thresholdState", `${side} ${fmtValue.format(state.threshold)}`);
      }
    }

    function resizeCanvas() {
      const rect = chart.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(360, Math.floor(rect.width * ratio));
      const height = Math.max(320, Math.floor(rect.height * ratio));
      if (chart.width !== width || chart.height !== height) {
        chart.width = width;
        chart.height = height;
      }
      drawChart();
    }

    function drawChart() {
      const rows = state.rows;
      const width = chart.width;
      const height = chart.height;
      ctx.clearRect(0, 0, width, height);
      if (!rows.length) return;

      const ratio = window.devicePixelRatio || 1;
      const padL = 56 * ratio;
      const padR = 20 * ratio;
      const padT = 24 * ratio;
      const padB = 42 * ratio;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const values = rows.flatMap((row) => {
        const out = [row.safety_value];
        if (Number.isFinite(row.safety_signal)) out.push(row.safety_signal);
        return out;
      });
      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (Number.isFinite(state.threshold)) {
        minV = Math.min(minV, state.threshold);
        maxV = Math.max(maxV, state.threshold);
      }
      if (state.autoScale) {
        const pad = Math.max(0.0001, (maxV - minV) * 0.12);
        minV -= pad;
        maxV += pad;
      } else {
        minV = -1;
        maxV = 1;
      }
      if (Math.abs(maxV - minV) < 1e-8) {
        minV -= 1;
        maxV += 1;
      }

      const latestReal = rows[rows.length - 1];
      const firstReal = rows[0];
      const liveMinT = Number.isFinite(firstReal.time_s) ? firstReal.time_s : 0;
      const liveMaxT = Number.isFinite(latestReal.time_s) ? latestReal.time_s : liveMinT;
      const visibleDuration = Math.max(1.0, liveMaxT - liveMinT);
      let maxT = liveMaxT;
      if (state.stale && Number.isFinite(state.recordAgeS)) {
        maxT += Math.max(0, state.recordAgeS);
      }
      let minT = maxT - visibleDuration;

      const xAt = (point) => {
        if (!Number.isFinite(point.time_s)) return padL + plotW;
        return padL + ((point.time_s - minT) / (maxT - minT)) * plotW;
      };
      const yAt = (value) => padT + (1 - (value - minV) / (maxV - minV)) * plotH;

      ctx.save();
      ctx.font = `${12 * ratio}px ui-sans-serif, system-ui, sans-serif`;
      ctx.lineWidth = 1 * ratio;
      ctx.strokeStyle = "#2c3542";
      ctx.fillStyle = "#a5b1bf";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i += 1) {
        const y = padT + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
        const label = maxV - ((maxV - minV) * i) / 4;
        ctx.fillText(fmtCompact.format(label), 12 * ratio, y);
      }

      ctx.strokeStyle = "#303946";
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.stroke();

      if (Number.isFinite(state.threshold)) {
        const y = yAt(state.threshold);
        ctx.strokeStyle = "#e6b450";
        ctx.setLineDash([6 * ratio, 5 * ratio]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#e6b450";
        ctx.fillText(`threshold ${fmtCompact.format(state.threshold)}`, padL + 8 * ratio, y - 14 * ratio);
      }

      function drawSeries(key, color, widthScale, fill) {
        const series = rows
          .map((row) => ({ time_s: row.time_s, value: row[key], held: row.held }))
          .filter((point) => Number.isFinite(point.value));
        if (!series.length) return;

        if (fill && series.length > 1) {
          const grad = ctx.createLinearGradient(0, padT, 0, height - padB);
          grad.addColorStop(0, "rgba(70, 182, 166, 0.22)");
          grad.addColorStop(1, "rgba(70, 182, 166, 0.02)");
          ctx.beginPath();
          series.forEach((point, idx) => {
            const x = xAt(point);
            const y = yAt(point.value);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.lineTo(xAt(series[series.length - 1]), height - padB);
          ctx.lineTo(xAt(series[0]), height - padB);
          ctx.closePath();
          ctx.fillStyle = grad;
          ctx.fill();
        }

        ctx.beginPath();
        series.forEach((point, idx) => {
          const x = xAt(point);
          const y = yAt(point.value);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = widthScale * ratio;
        ctx.stroke();
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();

      drawSeries("safety_value", "#46b6a6", 2.2, true);
      drawSeries("safety_signal", "#df6b6b", 1.9, false);

      if (state.stale && Number.isFinite(state.recordAgeS) && rows.length) {
        const holdEnd = {
          time_s: latestReal.time_s + Math.max(0, state.recordAgeS)
        };
        const x0 = xAt(latestReal);
        const x1 = xAt(holdEnd);
        if (x1 > x0) {
          ctx.lineWidth = 2.2 * ratio;
          ctx.strokeStyle = "#46b6a6";
          ctx.beginPath();
          ctx.moveTo(x0, yAt(latestReal.safety_value));
          ctx.lineTo(x1, yAt(latestReal.safety_value));
          ctx.stroke();

          if (Number.isFinite(latestReal.safety_signal)) {
            ctx.lineWidth = 1.9 * ratio;
            ctx.strokeStyle = "#df6b6b";
            ctx.beginPath();
            ctx.moveTo(x0, yAt(latestReal.safety_signal));
            ctx.lineTo(x1, yAt(latestReal.safety_signal));
            ctx.stroke();
          }
        }
      }

      const latest = rows[rows.length - 1];
      const markerPoint = state.stale && Number.isFinite(state.recordAgeS)
        ? { time_s: latest.time_s + Math.max(0, state.recordAgeS) }
        : latest;
      const x = xAt(markerPoint);
      const y = yAt(latest.safety_value);
      ctx.fillStyle = "#edf2f7";
      ctx.beginPath();
      ctx.arc(x, y, 4.5 * ratio, 0, Math.PI * 2);
      ctx.fill();
      if (Number.isFinite(latest.safety_signal)) {
        const ys = yAt(latest.safety_signal);
        ctx.fillStyle = "#df6b6b";
        ctx.beginPath();
        ctx.arc(x, ys, 4.0 * ratio, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      ctx.fillStyle = "#a5b1bf";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(`${fmtCompact.format(minT)} s`, padL, height - 14 * ratio);
      ctx.textAlign = "right";
      ctx.fillText(`${fmtCompact.format(maxT)} s`, width - padR, height - 14 * ratio);
      ctx.restore();
    }

    async function refresh() {
      if (state.paused) return;
      try {
        const response = await fetch(`/api/latest?window=${encodeURIComponent(state.window)}&t=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        state.rows = payload.rows || [];
        state.stale = Boolean(payload.stale);
        state.recordAgeS = Number.isFinite(payload.record_age_s) ? payload.record_age_s : null;
        updateMetrics(payload);
        el("emptyText").style.display = state.rows.length ? "none" : "grid";
        if (payload.stale) {
          setStatus("warn", `Holding ${formatDuration(payload.record_age_s)}`);
        } else {
          setStatus(state.rows.length ? "live" : "warn", state.rows.length ? "Recording" : "Waiting");
        }
        drawChart();
      } catch (err) {
        state.lastError = err.message || String(err);
        setStatus("error", state.lastError);
      }
    }

    el("pauseButton").addEventListener("click", () => {
      state.paused = !state.paused;
      el("pauseButton").textContent = state.paused ? "Resume" : "Pause";
      el("pauseButton").classList.toggle("active", state.paused);
      setStatus(state.paused ? "warn" : "live", state.paused ? "Paused" : "Live");
    });

    el("autoScaleButton").addEventListener("click", () => {
      state.autoScale = !state.autoScale;
      el("autoScaleButton").classList.toggle("active", state.autoScale);
      drawChart();
    });

    el("windowInput").addEventListener("change", (event) => {
      const value = Number(event.target.value);
      if (Number.isFinite(value)) state.window = Math.max(10, Math.min(5000, Math.floor(value)));
      event.target.value = state.window;
      refresh();
    });

    el("thresholdInput").addEventListener("change", (event) => {
      const value = Number(event.target.value);
      state.threshold = event.target.value === "" || !Number.isFinite(value) ? null : value;
      updateMetrics({ latest: state.rows[state.rows.length - 1], stats: statsFromRows(state.rows), count: state.rows.length });
      drawChart();
    });

    function statsFromRows(rows) {
      if (!rows.length) return {};
      const values = rows.flatMap((row) => {
        const out = [row.safety_value];
        if (Number.isFinite(row.safety_signal)) out.push(row.safety_signal);
        return out;
      });
      return { min: Math.min(...values), max: Math.max(...values) };
    }

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    refresh();
    state.timer = setInterval(refresh, SERVER_CONFIG.pollMs);
  </script>
</body>
</html>
"""


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_recent_rows(csv_path: Path, limit: int, tail_bytes: int, stale_seconds: float) -> dict:
    now = time.time()
    if not csv_path.exists():
        return {
            "ok": False,
            "error": f"CSV does not exist: {csv_path}",
            "csv": str(csv_path),
            "rows": [],
            "count": 0,
            "latest": None,
            "stats": {},
            "size_bytes": 0,
            "mtime": None,
            "record_age_s": None,
            "stale": True,
            "stale_seconds": stale_seconds,
        }

    stat = csv_path.stat()
    record_age_s = max(0.0, now - stat.st_mtime)
    stale = record_age_s > stale_seconds
    if stat.st_size == 0:
        return {
            "ok": True,
            "csv": str(csv_path),
            "rows": [],
            "count": 0,
            "latest": None,
            "stats": {},
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "record_age_s": record_age_s,
            "stale": stale,
            "stale_seconds": stale_seconds,
        }

    with csv_path.open("rb") as f:
        header_line = f.readline()
        if not header_line:
            header = ""
            data_start = 0
        else:
            header = header_line.decode("utf-8", errors="replace").strip()
            data_start = f.tell()

        if stat.st_size <= data_start:
            tail_text = ""
        elif stat.st_size - data_start <= tail_bytes:
            f.seek(data_start)
            tail_text = f.read().decode("utf-8", errors="replace")
        else:
            f.seek(max(data_start, stat.st_size - tail_bytes))
            f.readline()
            tail_text = f.read().decode("utf-8", errors="replace")

    if not header:
        return {
            "ok": False,
            "error": f"CSV header is empty: {csv_path}",
            "csv": str(csv_path),
            "rows": [],
            "count": 0,
            "latest": None,
            "stats": {},
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "record_age_s": record_age_s,
            "stale": stale,
            "stale_seconds": stale_seconds,
        }

    lines = tail_text.splitlines()
    if not tail_text.endswith("\n") and lines:
        lines = lines[:-1]

    rows: list[dict] = []
    reader = csv.DictReader([header, *lines])
    for raw in reader:
        value = parse_float(raw.get("safety_value"))
        signal = parse_float(raw.get("safety_signal"))
        time_s = parse_float(raw.get("time_s"))
        step = parse_int(raw.get("step"))
        if value is None or time_s is None or step is None:
            continue
        rows.append(
            {
                "time_s": time_s,
                "step": step,
                "safety_value": value,
                "safety_signal": signal,
                "obs_key": raw.get("obs_key", ""),
            }
        )

    rows = rows[-limit:]
    values = [row["safety_value"] for row in rows]
    visible_values = [
        value
        for row in rows
        for value in (row["safety_value"], row.get("safety_signal"))
        if value is not None
    ]
    signal_values = [row["safety_signal"] for row in rows if row.get("safety_signal") is not None]
    stats = {}
    if visible_values:
        stats = {
            "min": min(visible_values),
            "max": max(visible_values),
            "mean": sum(values) / len(values),
        }
        if signal_values:
            stats["signal_min"] = min(signal_values)
            stats["signal_max"] = max(signal_values)
            stats["signal_mean"] = sum(signal_values) / len(signal_values)

    return {
        "ok": True,
        "csv": str(csv_path),
        "rows": rows,
        "count": len(rows),
        "latest": rows[-1] if rows else None,
        "stats": stats,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "record_age_s": record_age_s,
        "stale": stale,
        "stale_seconds": stale_seconds,
    }


def make_handler(csv_path: Path, default_window: int, tail_mb: int, stale_seconds: float, server_config: dict):
    class SafetyValueDashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_html(self) -> None:
            html = INDEX_HTML.replace("__SERVER_CONFIG__", json.dumps(server_config))
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_html()
                return
            if parsed.path == "/api/latest":
                query = parse_qs(parsed.query)
                requested_window = parse_int(query.get("window", [None])[0])
                window = requested_window or default_window
                window = max(10, min(5000, window))
                bytes_per_row = 8192
                tail_bytes = max(1024 * 1024, min(tail_mb * 1024 * 1024, window * bytes_per_row))
                payload = read_recent_rows(csv_path, window, tail_bytes, stale_seconds)
                status = HTTPStatus.OK if payload.get("ok", False) else HTTPStatus.NOT_FOUND
                self.send_json(payload, status=status)
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    return SafetyValueDashboardHandler


def bind_server(host: str, port: int, handler, retries: int) -> tuple[ThreadingHTTPServer, int]:
    if port == 0:
        server = ThreadingHTTPServer((host, 0), handler)
        return server, int(server.server_address[1])

    last_error: OSError | None = None
    for candidate in range(port, port + retries + 1):
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
            return server, candidate
        except OSError as exc:
            last_error = exc
    raise OSError(f"Could not bind to {host}:{port}-{port + retries}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live dashboard for safety_value_records.csv.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to safety_value_records.csv.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--port-retries", type=int, default=20)
    parser.add_argument("--window", type=int, default=600, help="Default number of recent samples to display.")
    parser.add_argument("--poll-ms", type=int, default=500, help="Browser polling interval in milliseconds.")
    parser.add_argument("--tail-mb", type=int, default=64, help="Maximum CSV tail read per API request.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional reference line value.")
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=2.0,
        help="Mark the stream stale when safety_value_records.csv has not changed for this long.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    window = max(10, min(5000, int(args.window)))
    poll_ms = max(100, int(args.poll_ms))
    stale_seconds = max(0.1, float(args.stale_seconds))
    server_config = {
        "csv": str(csv_path),
        "window": window,
        "pollMs": poll_ms,
        "threshold": args.threshold,
        "staleSeconds": stale_seconds,
    }
    handler = make_handler(csv_path, window, max(1, int(args.tail_mb)), stale_seconds, server_config)
    server, bound_port = bind_server(args.host, args.port, handler, max(0, int(args.port_retries)))
    url = f"http://{args.host}:{bound_port}"
    print(f"[OK] Safety value dashboard: {url}")
    print(f"[INFO] Reading CSV: {csv_path}")
    print("[INFO] Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
