"use strict";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("csv-file");
const dropZone = document.getElementById("drop-zone");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const clearFileButton = document.getElementById("clear-file");
const submitButton = document.getElementById("submit-button");
const progressPanel = document.querySelector(".progress-panel");
const statusTitle = document.getElementById("status-title");
const resultSection = document.getElementById("result");

const phaseLabels = {
  waiting: "等待上传",
  reading: "正在读取文件",
  inference: "模型正在推理",
  complete: "检测完成",
  error: "检测失败",
};

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatProbability(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(6) : "—";
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
}

function setPhase(phase, message) {
  progressPanel.classList.toggle("error", phase === "error");
  progressPanel.classList.toggle("complete", phase === "complete");
  statusTitle.textContent = message || phaseLabels[phase];
  const orderedPhases = ["waiting", "reading", "inference", "complete"];
  const currentIndex = orderedPhases.indexOf(phase);
  document.querySelectorAll(".progress-steps li").forEach((step, index) => {
    step.classList.toggle("active", phase !== "error" && index === currentIndex);
    step.classList.toggle("done", phase === "complete" ? index < 3 : currentIndex > index);
  });
}

function setSelectedFile(file) {
  if (!file) {
    fileInfo.hidden = true;
    fileName.textContent = "";
    fileSize.textContent = "";
    submitButton.disabled = true;
    setPhase("waiting");
    return;
  }
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  fileInfo.hidden = false;
  submitButton.disabled = false;
  setPhase("waiting", "文件已就绪");
}

function addTextCell(row, value, className) {
  const cell = document.createElement("td");
  if (className) {
    const badge = document.createElement("span");
    badge.className = className;
    badge.textContent = String(value);
    cell.appendChild(badge);
  } else {
    cell.textContent = String(value);
  }
  row.appendChild(cell);
}

function renderRiskDistribution(counts) {
  const container = document.getElementById("risk-distribution");
  container.replaceChildren();
  const levels = [
    ["Low", "低风险", "low"],
    ["Medium", "中风险", "medium"],
    ["High", "高风险", "high"],
  ];
  const total = levels.reduce((sum, [key]) => sum + Number(counts[key] || 0), 0);
  levels.forEach(([key, label, cssClass]) => {
    const count = Number(counts[key] || 0);
    const row = document.createElement("div");
    row.className = `risk-row ${cssClass}`;
    const labelNode = document.createElement("span");
    labelNode.className = "risk-label";
    labelNode.textContent = label;
    const track = document.createElement("div");
    track.className = "risk-track";
    const fill = document.createElement("div");
    fill.className = "risk-fill";
    fill.style.width = `${total ? (count / total) * 100 : 0}%`;
    track.appendChild(fill);
    const countNode = document.createElement("strong");
    countNode.className = "risk-count";
    countNode.textContent = String(count);
    row.append(labelNode, track, countNode);
    container.appendChild(row);
  });
}

function renderResults(data) {
  document.getElementById("metric-flows").textContent = String(data.total_flows);
  document.getElementById("metric-windows").textContent = String(data.evaluated_windows);
  document.getElementById("metric-attacks").textContent = String(data.attack_count);
  document.getElementById("metric-ratio").textContent = formatPercent(data.attack_ratio);
  document.getElementById("metric-max").textContent = formatProbability(data.maximum_attack_probability);
  document.getElementById("metric-mean").textContent = formatProbability(data.mean_attack_probability);
  document.getElementById("model-version").textContent = data.model_version || "—";
  document.getElementById("decision-threshold").textContent = formatProbability(data.decision_threshold);
  document.getElementById("result-time").textContent = `完成于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
  renderRiskDistribution(data.risk_level_counts || {});

  const tableBody = document.getElementById("predictions");
  tableBody.replaceChildren();
  (data.sample_predictions || []).forEach((item, index) => {
    const row = document.createElement("tr");
    const riskClass = String(item.risk_level || "").toLowerCase();
    addTextCell(row, index + 1);
    addTextCell(row, item.target_row_number);
    addTextCell(row, formatProbability(item.attack_probability));
    addTextCell(row, item.detected_attack ? "是" : "否", `attack-pill ${item.detected_attack ? "yes" : "no"}`);
    addTextCell(row, item.risk_level, `risk-pill ${riskClass}`);
    tableBody.appendChild(row);
  });
  resultSection.hidden = false;
  setPhase("complete");
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));
clearFileButton.addEventListener("click", () => {
  fileInput.value = "";
  resultSection.hidden = true;
  setSelectedFile(null);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setPhase("error", "请先选择 CSV 文件");
    return;
  }
  resultSection.hidden = true;
  submitButton.disabled = true;
  try {
    setPhase("reading");
    await file.arrayBuffer();
    setPhase("inference");
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "检测请求失败，请检查文件后重试。");
    renderResults(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "检测请求失败，请重试。";
    setPhase("error", message);
  } finally {
    submitButton.disabled = false;
  }
});
