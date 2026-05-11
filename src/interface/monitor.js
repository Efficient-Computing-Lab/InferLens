document.addEventListener("DOMContentLoaded", () => {
  let frames = [];
  let texts = [];
  let currentFrame = 0;
  let liveMode = true;
  let accumulatedText = "";
  let streamAbortController = null;
  let chatHistory = [];
  let prefillInfo = null;
  let historyEnabled = true;
  let pclEnabled = false;

  const els = {
    status: document.getElementById("status"),
    message: document.getElementById("message"),
    run: document.getElementById("run"),
    clear: document.getElementById("clear"),
    clearConversation: document.getElementById("clearConversation"),
    live: document.getElementById("live"),
    model: document.getElementById("model"),
    promptTokens: document.getElementById("promptTokens"),
    generatedTokens: document.getElementById("generatedTokens"),
    kvLen: document.getElementById("kvLen"),
    historySlider: document.getElementById("historySlider"),
    framePos: document.getElementById("framePos"),
    frameMax: document.getElementById("frameMax"),
    output: document.getElementById("output"),
    tokenId: document.getElementById("tokenId"),
    tokenStr: document.getElementById("tokenStr"),
    step: document.getElementById("step"),
    logitsBody: document.getElementById("logitsBody"),
    resourceChart: document.getElementById("resourceChart"),
    resourceLegend: document.getElementById("resourceLegend"),
    resourceDetails: document.getElementById("resourceDetails"),
    layerNormChart: document.getElementById("layerNormChart"),
    layerNormLegend: document.getElementById("layerNormLegend"),
    layers: document.getElementById("layers"),
    conversationHistory: document.getElementById("conversationHistory"),
    prefillTimeline: document.getElementById("prefillTimeline"),
    prefillLegend: document.getElementById("prefillLegend"),
    prefillGrid: document.getElementById("prefillGrid"),
    layerTimingChart: document.getElementById("layerTimingChart"),
    pclStatus: document.getElementById("pclStatus"),
    pclGrid: document.getElementById("pclGrid"),
    pclActivation: document.getElementById("pclActivation"),
    pclWrap: document.getElementById("pclWrap"),
    historyToggle: document.getElementById("historyToggle"),
    pclToggle: document.getElementById("pclToggle")
  };

  function setStatus(text) {
    els.status.textContent = text;
    if(text === "Running"){
      els.run.disabled = true;
    }else{
      els.run.disabled = false;
    }
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function fmtNumber(value, digits = 3) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "—";
  }

  function fmtInt(value) {
    const n = Number(value);
    return Number.isFinite(n) ? String(Math.round(n)) : "—";
  }

  function fmtMs(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(2)} ms` : "—";
  }

  function fmtMB(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(1)} MB` : "—";
  }

  function fmtPct(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(1)}%` : "—";
  }

  function scaleWidth(value, factor = 8, max = 100) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0) return 0;
    return Math.min(max, Math.max(0, n * factor));
  }

  function getLayerMetricRanges(frame) {
    const layers = frame?.layers || {};
    const keys = Object.keys(layers);

    const metricKeys = [
      "in_norm",
      "attn_o_proj",
      "post_attn_norm",
      "mlp_in_proj",
      "mlp_out_proj"
    ];

    const ranges = {};

    metricKeys.forEach((metricKey) => {
      const values = keys
        .map((layerKey) => {
          const layer = layers[layerKey] || {};
          const v = Number(layer?.[metricKey]?.l2);
          return Number.isFinite(v) ? v : null;
        })
        .filter((v) => v != null);

      if (!values.length) {
        ranges[metricKey] = { min: 0, max: 0 };
        return;
      }

      ranges[metricKey] = {
        min: Math.min(...values),
        max: Math.max(...values)
      };
    });

    return ranges;
  }

  function scaleMetricToRange(value, min, max) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return 0;

    if (max <= min) {
      return 50;
    }

    const pct = ((n - min) / (max - min)) * 100;
    return Math.max(0, Math.min(100, pct));
  }

  function exitTokenBlock(layer) {
    const tokId = layer?.exit_token_id;
    const tokStr = layer?.exit_token_str;
    const tokText = layer?.exit_token_text;

    if (tokId == null && !tokStr && !tokText) {
      return null;
    }

    const wrap = document.createElement("div");
    wrap.className = "exit-token";
    wrap.innerHTML = `
      <div class="label">Early-exit token</div>
      <div class="exit-token-chip">
        <span class="exit-token-main">${escapeHtml(tokText || tokStr || "—")}</span>
        <span class="exit-token-meta">id=${escapeHtml(tokId ?? "—")} · ${escapeHtml(tokStr || "—")}</span>
      </div>
    `;
    return wrap;
  }

  function metricBar(label, value, extraText = "", range = null) {
    const width = range
      ? scaleMetricToRange(value, range.min, range.max)
      : scaleWidth(value, 8, 100);

    const wrap = document.createElement("div");
    wrap.className = "metric";
    wrap.innerHTML = `
      <div class="label">${escapeHtml(label)}: ${escapeHtml(fmtNumber(value))}</div>
      <div class="bar"><span style="width:${width}%;"></span></div>
      ${
        extraText
          ? `<div class="tiny">${escapeHtml(extraText)}</div>`
          : ""
      }
    `;
    return wrap;
  }

  function clearLogits() {
    els.logitsBody.innerHTML = "";
  }

  function clearLayers() {
    els.layers.innerHTML = "";
  }

  function clearResourceChart() {
    els.resourceChart.innerHTML = "";
    els.resourceLegend.innerHTML = "";
    els.resourceDetails.innerHTML = "";
  }

  function clearLayerNormChart() {
    els.layerNormChart.innerHTML = "";
    els.layerNormLegend.innerHTML = "";
  }

  function clearLayerTiming() {
    els.layerTimingChart.innerHTML = `<div class="sub">No per-layer timing yet.</div>`;
  }

  function clearPclStats() {
    els.pclStatus.textContent = "—";
    els.pclStatus.className = "pcl-decision";
    els.pclGrid.innerHTML = `<div class="sub">No PCL data yet.</div>`;
    els.pclActivation.innerHTML = "";
  }

  function renderPclStats(pcl) {
    if (!pcl) { clearPclStats(); return; }

    if (pcl.error) {
      els.pclStatus.textContent = `Error: ${pcl.error}`;
      els.pclStatus.className = "pcl-decision pcl-error";
    } else if (pcl.trained) {
      els.pclStatus.textContent = "Trained";
      els.pclStatus.className = "pcl-decision pcl-trained";
    } else if (pcl.skipped) {
      const reason = pcl.skip_reason ? ` — ${pcl.skip_reason}` : "";
      els.pclStatus.textContent = `Skipped${reason}`;
      els.pclStatus.className = "pcl-decision pcl-skipped";
    } else {
      els.pclStatus.textContent = "—";
      els.pclStatus.className = "pcl-decision";
    }

    els.pclGrid.innerHTML = "";
    const items = [
      ["LM loss",        pcl.lm_loss       != null ? fmtNumber(pcl.lm_loss)       : "—"],
      ["Entropy",        pcl.entropy        != null ? fmtNumber(pcl.entropy)        : "—"],
      ["Loss EMA",       pcl.loss_ema       != null ? fmtNumber(pcl.loss_ema)       : "—"],
      ["Loss EMA peak",  pcl.loss_ema_peak  != null ? fmtNumber(pcl.loss_ema_peak)  : "—"],
      ["Drift detected", pcl.drift_detected ? "Yes" : "No"],
      ["Experts total",  pcl.experts_total  ?? "—"],
      ["Experts spawned",pcl.experts_spawned ?? "—"],
      ["Expert updated", pcl.expert_updated_idx != null ? String(pcl.expert_updated_idx) : "—"],
      ["Total turns",    pcl.total_turns    ?? "—"],
      ["Total trained",  pcl.total_trained  ?? "—"],
      ["Total skipped",  pcl.total_skipped  ?? "—"],
      ["Replay buffer",  pcl.replay_buffer_size ?? "—"],
      ["Observe time",   pcl.observe_ms     != null ? fmtMs(pcl.observe_ms) : "—"],
    ];
    items.forEach(([k, v]) => {
      const box = document.createElement("div");
      box.className = "prefill-box";
      box.innerHTML = `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div>`;
      els.pclGrid.appendChild(box);
    });

    els.pclActivation.innerHTML = "";
    const profile = pcl.activation_profile || {};
    const entries = Object.entries(profile);
    if (entries.length) {
      const header = document.createElement("div");
      header.style.cssText = "font-size:12px;color:var(--muted);margin-bottom:8px;margin-top:12px;";
      header.textContent = "Activation profile";
      els.pclActivation.appendChild(header);

      entries.forEach(([name, value]) => {
        const pct = Math.max(0, Math.min(100, Number(value) * 100));
        const row = document.createElement("div");
        row.className = "layer-timing-row";
        row.innerHTML = `
          <div class="layer-timing-label">${escapeHtml(name)}</div>
          <div class="layer-timing-bar"><span style="width:${pct.toFixed(1)}%;"></span></div>
          <div class="layer-timing-value">${escapeHtml(fmtNumber(Number(value), 4))}</div>
        `;
        els.pclActivation.appendChild(row);
      });
    }
  }

  function clearPrefill() {
    prefillInfo = null;
    els.prefillTimeline.innerHTML = "";
    els.prefillLegend.innerHTML = "";
    els.prefillGrid.innerHTML = `<div class="sub">No prefill data yet.</div>`;
    clearLayerTiming();
  }

  function renderConversationHistory() {
    els.conversationHistory.innerHTML = "";

    if (!chatHistory.length) {
      els.conversationHistory.innerHTML = `<div class="sub">No conversation history yet.</div>`;
      return;
    }

    chatHistory.forEach((msg) => {
      const entry = document.createElement("div");
      entry.className = "history-entry";
      entry.innerHTML = `
        <div class="history-role">${escapeHtml(msg.role || "unknown")}</div>
        <div>${escapeHtml(msg.content || "")}</div>
      `;
      els.conversationHistory.appendChild(entry);
    });

    els.conversationHistory.scrollTop = els.conversationHistory.scrollHeight;
  }

  function resetView() {
    frames = [];
    texts = [];
    currentFrame = 0;
    accumulatedText = "";
    liveMode = true;

    els.live.textContent = "Pause live";
    els.historySlider.min = "0";
    els.historySlider.max = "0";
    els.historySlider.value = "0";
    els.framePos.textContent = "0";
    els.frameMax.textContent = "0";
    els.output.textContent = "";
    els.tokenId.textContent = "—";
    els.tokenStr.textContent = "—";
    els.step.textContent = "—";
    els.model.textContent = "—";
    els.promptTokens.textContent = "0";
    els.generatedTokens.textContent = "0";
    els.kvLen.textContent = "0";
    clearLogits();
    clearPrefill();
    clearResourceChart();
    clearLayerNormChart();
    clearLayers();
    clearPclStats();
    renderConversationHistory();
  }

  function clearConversationState() {
    chatHistory = [];
    renderConversationHistory();
  }

  function renderLogits(frame) {
    clearLogits();
    const logits = Array.isArray(frame.logit_top5) ? frame.logit_top5 : [];

    logits.forEach((item, idx) => {
      let tokenId = "—";
      let tokenStr = "—";
      let logit = "—";

      if (Array.isArray(item)) {
        tokenId = item[0];
        logit = item[1];
      } else if (item && typeof item === "object") {
        tokenId = item.id ?? item.token_id ?? "—";
        tokenStr = item.token ?? item.token_str ?? "—";
        logit = item.logit ?? item.value ?? "—";
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${escapeHtml(tokenStr)}</td>
        <td>${escapeHtml(tokenId)}</td>
        <td>${escapeHtml(fmtNumber(logit))}</td>
      `;
      els.logitsBody.appendChild(tr);
    });
  }

  function getFrameResourcePercentages(frame) {
    const rm = frame?.resource_metrics || {};

    const cpu = Number(rm.cpu_percent);
    const ram = Number(rm.ram_percent);
    const gpuUtil = Number(rm.gpu_util_percent);

    const gpuUsed = Number(rm.gpu_mem_used_mb);
    const gpuTotal = Number(rm.gpu_mem_total_mb);
    const gpuMemPct = (Number.isFinite(gpuUsed) && Number.isFinite(gpuTotal) && gpuTotal > 0)
      ? (gpuUsed / gpuTotal) * 100
      : null;

    const torchAlloc = Number(rm.torch_cuda_mem_allocated_mb);
    const torchReserved = Number(rm.torch_cuda_mem_reserved_mb);

    const torchAllocPct = (Number.isFinite(torchAlloc) && Number.isFinite(gpuTotal) && gpuTotal > 0)
      ? (torchAlloc / gpuTotal) * 100
      : null;

    const torchReservedPct = (Number.isFinite(torchReserved) && Number.isFinite(gpuTotal) && gpuTotal > 0)
      ? (torchReserved / gpuTotal) * 100
      : null;

    return {
      cpu_percent: Number.isFinite(cpu) ? cpu : null,
      ram_percent: Number.isFinite(ram) ? ram : null,
      gpu_util_percent: Number.isFinite(gpuUtil) ? gpuUtil : null,
      gpu_mem_percent: gpuMemPct,
      torch_alloc_percent: torchAllocPct,
      torch_reserved_percent: torchReservedPct
    };
  }

  function buildSeries() {
    return [
      { key: "cpu_percent", label: "CPU %", color: "#59d0ff" },
      { key: "ram_percent", label: "RAM %", color: "#9b7bff" },
      { key: "gpu_util_percent", label: "GPU util %", color: "#5df2a4" },
      { key: "gpu_mem_percent", label: "GPU mem %", color: "#ffb454" },
      { key: "torch_alloc_percent", label: "Torch alloc %", color: "#ff7c7c" },
      { key: "torch_reserved_percent", label: "Torch reserved %", color: "#c3d16b" }
    ];
  }

  function buildLayerNormSeries() {
    return [
      { key: "in_norm", label: "Input norm L2", color: "#59d0ff" },
      { key: "attn_o_proj", label: "Attention out L2", color: "#9b7bff" },
      { key: "post_attn_norm", label: "Post-attn norm L2", color: "#5df2a4" },
      { key: "mlp_in_proj", label: "MLP in L2", color: "#ffb454" },
      { key: "mlp_out_proj", label: "MLP out L2", color: "#ff7c7c" }
    ];
  }

  function renderResourceLegend(series) {
    els.resourceLegend.innerHTML = "";
    series.forEach((s) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `
        <span class="legend-swatch" style="background:${s.color};"></span>
        <span>${escapeHtml(s.label)}</span>
      `;
      els.resourceLegend.appendChild(item);
    });
  }

  function renderLayerNormLegend(series) {
    els.layerNormLegend.innerHTML = "";
    series.forEach((s) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `
        <span class="legend-swatch" style="background:${s.color};"></span>
        <span>${escapeHtml(s.label)}</span>
      `;
      els.layerNormLegend.appendChild(item);
    });
  }

  function createSvgEl(tag, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) {
      el.setAttribute(k, String(v));
    }
    return el;
  }

  function renderResourceDetails(frame) {
    els.resourceDetails.innerHTML = "";
    const rm = frame?.resource_metrics || {};
    const pct = getFrameResourcePercentages(frame);

    const items = [
      ["CPU %", pct.cpu_percent != null ? fmtNumber(pct.cpu_percent, 2) : "—"],
      ["RAM %", pct.ram_percent != null ? fmtNumber(pct.ram_percent, 2) : "—"],
      ["GPU util %", pct.gpu_util_percent != null ? fmtNumber(pct.gpu_util_percent, 2) : "—"],
      ["GPU mem %", pct.gpu_mem_percent != null ? fmtNumber(pct.gpu_mem_percent, 2) : "—"],
      ["Torch alloc %", pct.torch_alloc_percent != null ? fmtNumber(pct.torch_alloc_percent, 2) : "—"],
      ["Torch reserved %", pct.torch_reserved_percent != null ? fmtNumber(pct.torch_reserved_percent, 2) : "—"],
      ["GPU", rm.gpu_name ? String(rm.gpu_name) : "—"],
      ["Timestamp", rm.timestamp != null ? fmtInt(rm.timestamp) : "—"]
    ];

    items.forEach(([k, v]) => {
      const box = document.createElement("div");
      box.className = "resource-box";
      box.innerHTML = `
        <div class="k">${escapeHtml(k)}</div>
        <div class="v">${escapeHtml(v)}</div>
      `;
      els.resourceDetails.appendChild(box);
    });
  }

  function renderResourceChart(selectedIndex) {
    clearResourceChart();

    const svg = els.resourceChart;
    const width = 900;
    const height = 280;
    const margin = { top: 18, right: 20, bottom: 34, left: 44 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    const series = buildSeries();
    renderResourceLegend(series);

    if (!frames.length) {
      const text = createSvgEl("text", {
        x: width / 2,
        y: height / 2,
        "text-anchor": "middle",
        fill: "#9aa6d1",
        "font-size": "14"
      });
      text.textContent = "No resource metrics yet";
      svg.appendChild(text);
      return;
    }

    const xForIndex = (i) => {
      if (frames.length <= 1) return margin.left + plotW / 2;
      return margin.left + (i / (frames.length - 1)) * plotW;
    };

    const yForPercent = (pct) => {
      const clamped = Math.max(0, Math.min(100, pct));
      return margin.top + plotH - (clamped / 100) * plotH;
    };

    for (let p = 0; p <= 100; p += 25) {
      const y = yForPercent(p);
      svg.appendChild(createSvgEl("line", {
        x1: margin.left,
        y1: y,
        x2: margin.left + plotW,
        y2: y,
        stroke: "#28325f",
        "stroke-width": 1
      }));

      const label = createSvgEl("text", {
        x: margin.left - 8,
        y: y + 4,
        "text-anchor": "end",
        fill: "#9aa6d1",
        "font-size": "11"
      });
      label.textContent = `${p}`;
      svg.appendChild(label);
    }

    svg.appendChild(createSvgEl("line", {
      x1: margin.left,
      y1: margin.top,
      x2: margin.left,
      y2: margin.top + plotH,
      stroke: "#6f7cad",
      "stroke-width": 1.2
    }));

    svg.appendChild(createSvgEl("line", {
      x1: margin.left,
      y1: margin.top + plotH,
      x2: margin.left + plotW,
      y2: margin.top + plotH,
      stroke: "#6f7cad",
      "stroke-width": 1.2
    }));

    const xTickCount = Math.min(frames.length, 8);
    for (let i = 0; i < xTickCount; i++) {
      const idx = xTickCount === 1
        ? 0
        : Math.round((i / (xTickCount - 1)) * (frames.length - 1));
      const x = xForIndex(idx);

      svg.appendChild(createSvgEl("line", {
        x1: x,
        y1: margin.top + plotH,
        x2: x,
        y2: margin.top + plotH + 5,
        stroke: "#6f7cad",
        "stroke-width": 1
      }));

      const label = createSvgEl("text", {
        x,
        y: margin.top + plotH + 18,
        "text-anchor": "middle",
        fill: "#9aa6d1",
        "font-size": "11"
      });
      label.textContent = String(idx);
      svg.appendChild(label);
    }

    const yAxisLabel = createSvgEl("text", {
      x: 14,
      y: margin.top + plotH / 2,
      fill: "#9aa6d1",
      "font-size": "11",
      transform: `rotate(-90 14 ${margin.top + plotH / 2})`,
      "text-anchor": "middle"
    });
    yAxisLabel.textContent = "Resource %";
    svg.appendChild(yAxisLabel);

    const xAxisLabel = createSvgEl("text", {
      x: margin.left + plotW / 2,
      y: height - 6,
      fill: "#9aa6d1",
      "font-size": "11",
      "text-anchor": "middle"
    });
    xAxisLabel.textContent = "Frame";
    svg.appendChild(xAxisLabel);

    series.forEach((s) => {
      let d = "";
      let hasPoint = false;

      frames.forEach((frame, idx) => {
        const vals = getFrameResourcePercentages(frame);
        const v = vals[s.key];
        if (v == null || !Number.isFinite(v)) return;

        const x = xForIndex(idx);
        const y = yForPercent(v);
        d += hasPoint ? ` L ${x} ${y}` : `M ${x} ${y}`;
        hasPoint = true;
      });

      if (!hasPoint) return;

      svg.appendChild(createSvgEl("path", {
        d,
        fill: "none",
        stroke: s.color,
        "stroke-width": 2.2,
        "stroke-linejoin": "round",
        "stroke-linecap": "round"
      }));

      const selectedVals = getFrameResourcePercentages(frames[selectedIndex]);
      const selectedValue = selectedVals[s.key];
      if (selectedValue != null && Number.isFinite(selectedValue)) {
        svg.appendChild(createSvgEl("circle", {
          cx: xForIndex(selectedIndex),
          cy: yForPercent(selectedValue),
          r: 3.6,
          fill: s.color,
          stroke: "#0a1022",
          "stroke-width": 1.5
        }));
      }
    });

    if (selectedIndex >= 0 && selectedIndex < frames.length) {
      const x = xForIndex(selectedIndex);
      svg.appendChild(createSvgEl("line", {
        x1: x,
        y1: margin.top,
        x2: x,
        y2: margin.top + plotH,
        stroke: "#ffffff",
        "stroke-width": 1.2,
        "stroke-dasharray": "5 4",
        opacity: 0.85
      }));

      const marker = createSvgEl("text", {
        x: Math.min(width - 24, Math.max(24, x)),
        y: margin.top - 4,
        "text-anchor": "middle",
        fill: "#e8ecff",
        "font-size": "11"
      });
      marker.textContent = `frame ${selectedIndex}`;
      svg.appendChild(marker);
    }

    renderResourceDetails(frames[selectedIndex]);
  }

  function getLayerNormSeriesData(frame) {
    const layers = frame?.layers || {};
    const layerKeys = Object.keys(layers)
      .map((k) => Number(k))
      .filter((n) => Number.isFinite(n))
      .sort((a, b) => a - b);

    const seriesDefs = buildLayerNormSeries();

    return {
      layerKeys,
      series: seriesDefs.map((def) => ({
        ...def,
        values: layerKeys.map((layerIdx) => {
          const layer = layers[String(layerIdx)] || layers[layerIdx] || {};
          const value = Number(layer?.[def.key]?.l2);
          return Number.isFinite(value) ? value : null;
        })
      }))
    };
  }

  function renderLayerNormChart(frame) {
    clearLayerNormChart();

    const svg = els.layerNormChart;
    const width = 900;
    const height = 280;
    const margin = { top: 18, right: 20, bottom: 40, left: 56 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    const { layerKeys, series } = getLayerNormSeriesData(frame);
    renderLayerNormLegend(series);

    if (!layerKeys.length) {
      const text = createSvgEl("text", {
        x: width / 2,
        y: height / 2,
        "text-anchor": "middle",
        fill: "#9aa6d1",
        "font-size": "14"
      });
      text.textContent = "No layer norm data yet";
      svg.appendChild(text);
      return;
    }

    const allValues = series.flatMap(s => s.values).filter(v => Number.isFinite(v));
    if (!allValues.length) {
      const text = createSvgEl("text", {
        x: width / 2,
        y: height / 2,
        "text-anchor": "middle",
        fill: "#9aa6d1",
        "font-size": "14"
      });
      text.textContent = "No L2 norm values in this frame";
      svg.appendChild(text);
      return;
    }

    const maxVal = Math.max(...allValues, 1e-9);
    const paddedMax = maxVal * 1.1;

    const xForLayerIndex = (i) => {
      if (layerKeys.length <= 1) return margin.left + plotW / 2;
      return margin.left + (i / (layerKeys.length - 1)) * plotW;
    };

    const yForValue = (v) => {
      const clamped = Math.max(0, Math.min(paddedMax, v));
      return margin.top + plotH - (clamped / paddedMax) * plotH;
    };

    for (let i = 0; i <= 4; i++) {
      const v = (i / 4) * paddedMax;
      const y = yForValue(v);

      svg.appendChild(createSvgEl("line", {
        x1: margin.left,
        y1: y,
        x2: margin.left + plotW,
        y2: y,
        stroke: "#28325f",
        "stroke-width": 1
      }));

      const label = createSvgEl("text", {
        x: margin.left - 8,
        y: y + 4,
        "text-anchor": "end",
        fill: "#9aa6d1",
        "font-size": "11"
      });
      label.textContent = fmtNumber(v, 2);
      svg.appendChild(label);
    }

    svg.appendChild(createSvgEl("line", {
      x1: margin.left,
      y1: margin.top,
      x2: margin.left,
      y2: margin.top + plotH,
      stroke: "#6f7cad",
      "stroke-width": 1.2
    }));

    svg.appendChild(createSvgEl("line", {
      x1: margin.left,
      y1: margin.top + plotH,
      x2: margin.left + plotW,
      y2: margin.top + plotH,
      stroke: "#6f7cad",
      "stroke-width": 1.2
    }));

    layerKeys.forEach((layer, i) => {
      const x = xForLayerIndex(i);

      svg.appendChild(createSvgEl("line", {
        x1: x,
        y1: margin.top + plotH,
        x2: x,
        y2: margin.top + plotH + 5,
        stroke: "#6f7cad",
        "stroke-width": 1
      }));

      const label = createSvgEl("text", {
        x,
        y: margin.top + plotH + 18,
        "text-anchor": "middle",
        fill: "#9aa6d1",
        "font-size": "11"
      });
      label.textContent = String(layer);
      svg.appendChild(label);
    });

    const yAxisLabel = createSvgEl("text", {
      x: 18,
      y: margin.top + plotH / 2,
      fill: "#9aa6d1",
      "font-size": "11",
      transform: `rotate(-90 18 ${margin.top + plotH / 2})`,
      "text-anchor": "middle"
    });
    yAxisLabel.textContent = "L2 norm";
    svg.appendChild(yAxisLabel);

    const xAxisLabel = createSvgEl("text", {
      x: margin.left + plotW / 2,
      y: height - 8,
      fill: "#9aa6d1",
      "font-size": "11",
      "text-anchor": "middle"
    });
    xAxisLabel.textContent = "Layer";
    svg.appendChild(xAxisLabel);

    series.forEach((s) => {
      let d = "";
      let hasPoint = false;

      s.values.forEach((value, i) => {
        if (!Number.isFinite(value)) return;
        const x = xForLayerIndex(i);
        const y = yForValue(value);
        d += hasPoint ? ` L ${x} ${y}` : `M ${x} ${y}`;
        hasPoint = true;
      });

      if (!hasPoint) return;

      svg.appendChild(createSvgEl("path", {
        d,
        fill: "none",
        stroke: s.color,
        "stroke-width": 2.2,
        "stroke-linejoin": "round",
        "stroke-linecap": "round"
      }));

      s.values.forEach((value, i) => {
        if (!Number.isFinite(value)) return;
        svg.appendChild(createSvgEl("circle", {
          cx: xForLayerIndex(i),
          cy: yForValue(value),
          r: 3.2,
          fill: s.color,
          stroke: "#0a1022",
          "stroke-width": 1.2
        }));
      });
    });
  }

  function renderLayers(frame) {
    clearLayers();
    const layers = frame.layers || {};
    const keys = Object.keys(layers).sort((a, b) => Number(a) - Number(b));
    const ranges = getLayerMetricRanges(frame);

    keys.forEach((layerKey) => {
      const layer = layers[layerKey] || {};
      const card = document.createElement("div");
      card.className = "layer";

      const title = document.createElement("h3");
      title.textContent = `Layer ${layerKey}`;
      card.appendChild(title);

      const exitTokenEl = exitTokenBlock(layer);
      if (exitTokenEl) {
        card.appendChild(exitTokenEl);
      }

      if (layer.in_norm?.l2 != null) {
        card.appendChild(
          metricBar(
            "Input norm L2",
            layer.in_norm.l2,
            `mean=${fmtNumber(layer.in_norm.mean)} std=${fmtNumber(layer.in_norm.std)}`,
            ranges.in_norm
          )
        );
      }

      if (layer.attn_o_proj?.l2 != null) {
        card.appendChild(
          metricBar(
            "Attention out L2",
            layer.attn_o_proj.l2,
            `mean=${fmtNumber(layer.attn_o_proj.mean)} std=${fmtNumber(layer.attn_o_proj.std)}`,
            ranges.attn_o_proj
          )
        );
      }

      if (layer.post_attn_norm?.l2 != null) {
        card.appendChild(
          metricBar(
            "Post-attn norm L2",
            layer.post_attn_norm.l2,
            `mean=${fmtNumber(layer.post_attn_norm.mean)} std=${fmtNumber(layer.post_attn_norm.std)}`,
            ranges.post_attn_norm
          )
        );
      }

      if (layer.mlp_in_proj?.l2 != null) {
        const topk = Array.isArray(layer.mlp_in_proj.topk_idx)
          ? layer.mlp_in_proj.topk_idx.slice(0, 8).join(", ")
          : "";

        card.appendChild(
          metricBar(
            "MLP gate L2",
            layer.mlp_in_proj.l2,
            topk ? `top neurons: ${topk}` : "",
            ranges.mlp_in_proj
          )
        );
      }

      if (layer.mlp_out_proj?.l2 != null) {
        const topk = Array.isArray(layer.mlp_out_proj.topk_idx)
          ? layer.mlp_out_proj.topk_idx.slice(0, 8).join(", ")
          : "";

        card.appendChild(
          metricBar(
            "MLP up L2",
            layer.mlp_out_proj.l2,
            topk ? `top neurons: ${topk}` : "",
            ranges.mlp_out_proj
          )
        );
      }

      els.layers.appendChild(card);
    });
  }

  function renderLayerTiming(prefill) {
    els.layerTimingChart.innerHTML = "";

    const layers = prefill?.layers || {};
    const keys = Object.keys(layers).sort((a, b) => Number(a) - Number(b));

    if (!keys.length) {
      clearLayerTiming();
      return;
    }

    const rows = keys
      .map((k) => {
        const layer = layers[k] || {};
        const timeMs =
          Number(layer.prefill_time_ms) ||
          Number(layer.prefill_ms) ||
          Number(layer.layer_time_ms);
        return { layer: k, time_ms: timeMs };
      })
      .filter((r) => Number.isFinite(r.time_ms));

    if (!rows.length) {
      clearLayerTiming();
      return;
    }

    const maxMs = Math.max(...rows.map(r => r.time_ms), 1);

    rows.forEach((row) => {
      const widthPct = Math.max(0, Math.min(100, (row.time_ms / maxMs) * 100));

      const el = document.createElement("div");
      el.className = "layer-timing-row";
      el.innerHTML = `
        <div class="layer-timing-label">L${escapeHtml(row.layer)}</div>
        <div class="layer-timing-bar"><span style="width:${widthPct}%;"></span></div>
        <div class="layer-timing-value">${escapeHtml(fmtMs(row.time_ms))}</div>
      `;
      els.layerTimingChart.appendChild(el);
    });
  }

  function renderFrame(index) {
    if (!frames.length) return;
    if (index < 0) index = 0;
    if (index >= frames.length) index = frames.length - 1;

    currentFrame = index;
    const frame = frames[index];

    els.framePos.textContent = String(index);
    els.frameMax.textContent = String(Math.max(frames.length - 1, 0));
    els.historySlider.value = String(index);

    els.output.textContent = texts[index] || "";
    els.generatedTokens.textContent = String(index + 1);
    els.kvLen.textContent = frame.kv_seq_len ?? "—";
    els.tokenId.textContent = frame.next_token_id ?? "—";
    els.tokenStr.textContent = frame.next_token_str ?? "—";
    els.step.textContent = frame.step ?? index;

    if (frame.model) {
      els.model.textContent = frame.model;
    }

    renderLogits(frame);
    renderResourceChart(index);
    renderLayerNormChart(frame);
    renderLayers(frame);
  }

  function parsePromptTokenEstimate(prompt, history) {
    const parts = [];

    if (Array.isArray(history)) {
      for (const msg of history) {
        if (msg && typeof msg.content === "string") {
          parts.push(msg.content);
        }
      }
    }

    if (typeof prompt === "string") {
      parts.push(prompt);
    }

    const combined = parts.join(" ").trim();
    if (!combined) return 0;
    return combined.split(/\s+/).length;
  }

  function renderPrefill(prefill) {
    prefillInfo = prefill || null;
    els.prefillTimeline.innerHTML = "";
    els.prefillLegend.innerHTML = "";
    els.prefillGrid.innerHTML = "";

    if (!prefill) {
      els.prefillGrid.innerHTML = `<div class="sub">No prefill data yet.</div>`;
      clearLayerTiming();
      return;
    }

    const tokenize = Number(prefill.tokenize_ms) || 0;
    const transfer = Number(prefill.transfer_ms) || 0;
    const prefillMs = Number(prefill.prefill_ms) || 0;
    const sample = Number(prefill.first_token_sample_ms) || 0;
    const total = tokenize + transfer + prefillMs + sample;

    const segments = [
      { label: "Tokenize", value: tokenize, color: "#59d0ff" },
      { label: "Transfer", value: transfer, color: "#9b7bff" },
      { label: "Prefill forward", value: prefillMs, color: "#5df2a4" },
      { label: "First-token sample", value: sample, color: "#ffb454" }
    ];

    if (total > 0) {
      segments.forEach((seg) => {
        const pct = Math.max(0, (seg.value / total) * 100);

        const bar = document.createElement("div");
        bar.className = "timeline-seg";
        bar.style.width = `${pct}%`;
        bar.style.background = seg.color;
        els.prefillTimeline.appendChild(bar);

        const legend = document.createElement("div");
        legend.className = "legend-item";
        legend.innerHTML = `
          <span class="legend-swatch" style="background:${seg.color};"></span>
          <span>${escapeHtml(seg.label)}: ${escapeHtml(fmtMs(seg.value))}</span>
        `;
        els.prefillLegend.appendChild(legend);
      });
    }

    const mbt = prefill.memory_before_transfer || {};
    const mb = prefill.memory_before_prefill || {};
    const ma = prefill.memory_after_prefill || {};

    const totalMb = Number(ma.gpu_total_mb ?? mb.gpu_total_mb ?? mbt.gpu_total_mb);
    const reservedAfter = Number(ma.torch_reserved_mb);
    const reservedPctAfter = Number.isFinite(totalMb) && totalMb > 0 && Number.isFinite(reservedAfter)
      ? (reservedAfter / totalMb) * 100
      : null;

    const items = [
      ["Prompt tokens", prefill.prompt_tokens ?? "—"],
      ["KV len after prefill", prefill.kv_seq_len_after_prefill ?? "—"],
      ["Tokenize", fmtMs(prefill.tokenize_ms)],
      ["Transfer", fmtMs(prefill.transfer_ms)],
      ["Prefill forward", fmtMs(prefill.prefill_ms)],
      ["Sample first token", fmtMs(prefill.first_token_sample_ms)],
      ["Total first-token latency", fmtMs(prefill.first_token_latency_ms)],
      ["Torch alloc before", fmtMB(mb.torch_allocated_mb)],
      ["Torch reserved before", fmtMB(mb.torch_reserved_mb)],
      ["Torch alloc after", fmtMB(ma.torch_allocated_mb)],
      ["Torch reserved after", fmtMB(ma.torch_reserved_mb)],
      ["Reserved after / total", reservedPctAfter != null ? fmtPct(reservedPctAfter) : "—"],
      ["Peak allocated", fmtMB(ma.torch_peak_allocated_mb)],
      ["Peak reserved", fmtMB(ma.torch_peak_reserved_mb)],
      ["GPU total", fmtMB(totalMb)]
    ];

    items.forEach(([k, v]) => {
      const box = document.createElement("div");
      box.className = "prefill-box";
      box.innerHTML = `
        <div class="k">${escapeHtml(k)}</div>
        <div class="v">${escapeHtml(v)}</div>
      `;
      els.prefillGrid.appendChild(box);
    });

    renderLayerTiming(prefill);
  }

  async function startStream(prompt) {
    const trimmedPrompt = prompt.trim();
    resetView();
    els.promptTokens.textContent = String(parsePromptTokenEstimate(trimmedPrompt, chatHistory));

    if (!trimmedPrompt) {
      setStatus("Enter a prompt");
      return;
    }

    setStatus("Running");

    if (streamAbortController) {
      streamAbortController.abort();
    }
    streamAbortController = new AbortController();

    const outgoingHistory = historyEnabled ? [...chatHistory] : [];

    try {
      const formData = new FormData();
      formData.append("message", prompt);
      formData.append("history", JSON.stringify(outgoingHistory));
      formData.append("enable_pcl", pclEnabled ? "true" : "false");
      formData.append("max_tokens", "256");
      formData.append("temperature", "0.7");
      formData.append("top_p", "0.9");

      const response = await fetch("/api/chat_monitor_stream", {
        method: "POST",
        body: formData,
        signal: streamAbortController.signal
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }

      if (!response.body) {
        throw new Error("Readable stream not available in this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;

          let data;
          try {
            data = JSON.parse(line);
          } catch (err) {
            console.warn("Skipping malformed JSON line:", line, err);
            continue;
          }

          if (data.model) {
            els.model.textContent = data.model;
          }

          if (data.type === "prefill" && data.prefill) {
            renderPrefill(data.prefill);

            if (data.prefill.prompt_tokens != null) {
              els.promptTokens.textContent = String(data.prefill.prompt_tokens);
            }

            if (data.prefill.kv_seq_len_after_prefill != null) {
              els.kvLen.textContent = String(data.prefill.kv_seq_len_after_prefill);
            }
          }

          if (typeof data.response === "string" && data.response.length > 0) {
            accumulatedText += data.response;
          }

          if (data.trace) {
            const traceFrame = {
              ...data.trace,
              model: data.model || null
            };

            frames.push(traceFrame);
            texts.push(accumulatedText);

            els.historySlider.max = String(Math.max(frames.length - 1, 0));
            els.frameMax.textContent = String(Math.max(frames.length - 1, 0));

            if (liveMode) {
              renderFrame(frames.length - 1);
            } else {
              renderResourceChart(currentFrame);
              renderLayerNormChart(frames[currentFrame]);
            }
          }

          if (data.error) {
            setStatus("Error");
            els.output.innerHTML = `<span class="error">${escapeHtml(data.error)}</span>`;
          }

          if (data.done) {
            setStatus("Done");
            if (data.pcl) {
              renderPclStats(data.pcl);
            }
          }
        }
      }

      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer.trim());
          if (data.model) {
            els.model.textContent = data.model;
          }
          if (data.error) {
            setStatus("Error");
            els.output.innerHTML = `<span class="error">${escapeHtml(data.error)}</span>`;
          }
          if (data.done) {
            setStatus("Done");
          }
        } catch (_) {}
      }

      if (frames.length === 0 && accumulatedText.length === 0) {
        setStatus("No trace returned");
      }

      if (historyEnabled) {
        if (trimmedPrompt) {
          chatHistory.push({ role: "user", content: prompt });
        }
        if (accumulatedText) {
          chatHistory.push({ role: "assistant", content: accumulatedText });
        }
      }
      renderConversationHistory();

    } catch (err) {
      if (err.name === "AbortError") {
        setStatus("Stopped");
        return;
      }
      setStatus("Error");
      els.output.innerHTML = `<span class="error">${escapeHtml(err.message || String(err))}</span>`;
      console.error(err);
    }
  }

  els.historySlider.addEventListener("input", () => {
    if (!frames.length) return;
    liveMode = false;
    els.live.textContent = "Resume live";
    renderFrame(Number(els.historySlider.value));
  });

  els.live.addEventListener("click", () => {
    liveMode = !liveMode;
    els.live.textContent = liveMode ? "Pause live" : "Resume live";

    if (liveMode && frames.length) {
      renderFrame(frames.length - 1);
    } else if (frames.length) {
      renderFrame(currentFrame);
    }
  });

  els.run.addEventListener("click", () => {
    if(!els.run.disabled){
      startStream(els.message.value);
    }
  });

  els.clear.addEventListener("click", () => {
    if (streamAbortController) {
      streamAbortController.abort();
    }
    resetView();
    setStatus("Idle");
  });

  els.clearConversation.addEventListener("click", () => {
    if (streamAbortController) {
      streamAbortController.abort();
    }
    clearConversationState();
    resetView();
    setStatus("Idle");
  });

  els.message.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
      startStream(els.message.value);
    }
  });

  els.historyToggle.addEventListener("change", () => {
    historyEnabled = els.historyToggle.checked;
  });

  els.pclToggle.addEventListener("change", () => {
    pclEnabled = els.pclToggle.checked;
    els.pclWrap.style.display = pclEnabled ? "" : "none";
  });

  els.pclWrap.style.display = "none";

  renderConversationHistory();
  resetView();
  setStatus("Idle");
});