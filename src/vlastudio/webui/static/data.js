"use strict";
const $ = id => document.getElementById(id);
Object.assign(chinese, {
  "Data management":"数据管理", "Datasets":"数据集", "Refresh":"刷新", "Clear filters":"清空条件", "Apply":"应用",
  "Choose a dataset and an episode.":"请选择数据集和片段。", "Annotations":"标注",
  "Export annotations":"导出标注", "Save annotations":"保存标注", "Annotation versions":"标注版本",
  "Add separator":"添加分隔符", "Delete separator":"删除分隔符",
  "Clear selection":"清除选区", "Drag the red arrow to preview. Click a separator to select it, or drag it to change its frame. Click a segment to loop it.":"拖动红色箭头预览。点击分隔符可选中，拖动可调整位置。点击区间可循环播放。",
  "Separator":"分隔符", "Episode endpoint":"片段端点", "Current frame":"当前帧", "Frame":"帧", "Loop segment":"循环子片段", "Show options":"展开选项", "Add separator here":"在此添加分隔符", "Latest revision":"最新修订", "Recorded signals":"已录制信号", "Delete dataset":"删除数据集", "Whole episode":"整个片段",
  "Subtask segment":"子任务片段", "Language description":"语言描述", "Reviewed":"已人工复核",
  "unknown":"未知", "success":"成功", "failure":"失败", "Unsaved edits · kept in this browser":"修改已暂存在本浏览器，尚未保存",
  "Annotations saved":"标注已保存", "Cancel":"取消", "Confirm":"确认", "No matching datasets":"没有匹配的数据集",
  "Move this dataset and all its episodes to trash?":"将数据集及全部片段移到回收目录？",
  "Move this recording to trash?":"将此片段移到回收目录？",
  "Source recording changed. Previous annotations remain in history.":"源片段已更新，旧标注仍保留在历史中。",
  "Temporary edits restored.":"已恢复浏览器暂存的修改。", "Legacy dataset":"旧数据集", "Not recorded":"未记录",
  "Replace current edits with this annotation version?":"用此历史标注替换当前编辑？",
  "Annotations changed in another window. Reload before saving.":"其他窗口已更新标注，请重新加载后再保存。",
});
const date = value => value ? new Date(value).toLocaleString(language === "zh" ? "zh-CN" : "en") : tr("Not recorded");
let datasets = [], selectedDataset, episodeRows = [], selected, metadata, documentData, history = [];
let frame = 0, loopId = null, dirty = false, playing = false, selectionToken = 0, playbackToken = 0;
let scrub = null;
// A movable separator marks the start of its right-hand segment.
let selectedSeparator = null;
let pendingFrame, currentRevision = null, saving = false;
const cameraViews = new Map();
function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}
async function api(path, body, signal) {
  const response = await fetch("/api/" + path, {
    method: body ? "POST" : "GET", headers: {"Content-Type":"application/json"},
    body: body ? JSON.stringify(body) : undefined, cache:"no-store", signal,
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || response.status);
  return data;
}
function message(text, error = false) {
  $("message").textContent = tr(text);
  $("message").classList.toggle("error", error);
}
function applyLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  $("language").value = language;
  document.querySelectorAll("[data-i18n]").forEach(element => element.textContent = tr(element.dataset.i18n));
  for (const id of ["filterRobot","filterTask","filterSuffix"]) $(id).placeholder = language === "zh" ? "输入前缀筛选…" : "Type a prefix…";
  renderDatasets(); renderEpisodes();
  document.querySelectorAll(".filter-toggle").forEach(button => button.title = button.ariaLabel = tr("Show options"));
  if (documentData) renderAnnotations();
  if (selectedDataset) datasetInfo();
  renderThemeToggle();
}
function renderThemeToggle() {
  const dark = document.documentElement.dataset.theme === "dark";
  const button = $("themeToggle");
  button.title = button.ariaLabel = tr(dark ? "Light mode" : "Dark mode");
  button.setAttribute("aria-pressed", String(dark));
  button.innerHTML = dark
    ? '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>'
    : '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M20.5 14A9 9 0 0 1 10 3.5 9 9 0 1 0 20.5 14Z"/></svg>';
}
function applyTheme() {
  window.VLATheme.apply();
  renderThemeToggle();
}
$("themeToggle").onclick = () => {
  localStorage.setItem("vla-theme", document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  applyTheme();
};
$("language").onchange = () => {
  const defaultSelected = $("filterSuffix").value === tr("Default");
  language = $("language").value;
  if (defaultSelected) $("filterSuffix").value = tr("Default");
  closeFilters(); localStorage.setItem("vla-language", language); applyLanguage();
};
function confirmAction(title, text) {
  $("confirmTitle").textContent = tr(title);
  $("confirmMessage").textContent = tr(text);
  $("confirm").showModal();
  return new Promise(resolve => $("confirm").addEventListener("close", () => resolve($("confirm").returnValue === "ok"), {once:true}));
}
const filters = [
  {input:"filterRobot", list:"robots", key:"robot"},
  {input:"filterTask", list:"tasks", key:"task"},
  {input:"filterSuffix", list:"suffixes", key:"suffix"},
];
function value(item, key) { return key === "suffix" ? item.suffix || tr("Default") : item[key] || ""; }
let appliedPrefixes = ["", "", ""];
let appliedDefaultLabel = tr("Default");
function matches(item) {
  return filters.every(({key}, index) => {
    const label = key === "suffix" && !item.suffix ? appliedDefaultLabel : value(item,key);
    return label.toLocaleLowerCase().startsWith(appliedPrefixes[index]);
  });
}
function applyFilters() {
  appliedPrefixes = filters.map(({input}) => $(input).value.trim().toLocaleLowerCase());
  appliedDefaultLabel = tr("Default");
  closeFilters();
  if (selectedDataset && !matches(selectedDataset)) {
    clearEpisode(); selectedDataset = null; episodeRows = [];
    $("datasetTitle").textContent = $("datasetInfo").textContent = "";
    $("deleteDataset").hidden = true; renderEpisodes();
  }
  renderDatasets();
}
$("datasetFilters").onsubmit = event => { event.preventDefault(); applyFilters(); };
let openFilter = null;
function filterOptions(filter, all = false) {
  const prefix = all ? "" : $(filter.input).value.trim().toLocaleLowerCase();
  const previousFilters = filters.slice(0, filters.indexOf(filter));
  const available = datasets.filter(item => previousFilters.every(previous =>
    value(item, previous.key).toLocaleLowerCase().startsWith($(previous.input).value.trim().toLocaleLowerCase())));
  return [...new Set(available.map(item => value(item, filter.key)))]
    .filter(option => option.toLocaleLowerCase().startsWith(prefix)).sort();
}
function closeFilters() {
  for (const filter of filters) {
    $(filter.list).hidden = true;
    $(filter.input).setAttribute("aria-expanded", "false");
    $(filter.input).removeAttribute("aria-activedescendant");
  }
  openFilter = null;
}
function openOptions(filter, all = false) {
  closeFilters(); openFilter = filter;
  const list = $(filter.list);
  list.replaceChildren(...filterOptions(filter, all).map((option, index) => {
    const row = node("button", option);
    row.type = "button"; row.id = filter.list + "-option-" + index;
    row.setAttribute("role", "option"); row.setAttribute("aria-selected", "false");
    row.onmousedown = event => event.preventDefault();
    row.onclick = () => chooseFilter(filter, option);
    return row;
  }));
  if (!list.children.length) list.append(node("small", tr("No matching datasets")));
  list.hidden = false;
  $(filter.input).setAttribute("aria-expanded", "true");
}
function chooseFilter(filter, option) {
  $(filter.input).value = option; closeFilters();
  $(filter.input).focus();
}
function renderDatasets() {
  const shown = datasets.filter(item => matches(item));
  $("datasetList").replaceChildren(...shown.map(item => {
    const button = node("button");
    button.append(node("strong", item.legacy ? item.dataset : [item.robot, item.task, value(item,"suffix")].filter(Boolean).join(" · ")));
    button.append(node("small", tr("Updated") + " " + date(item.modified_at)));
    button.setAttribute("aria-pressed", String(item.dataset === selectedDataset?.dataset));
    button.onclick = () => chooseDataset(item).catch(error => message(error.message, true));
    return button;
  }));
  if (!shown.length) $("datasetList").append(node("small", tr("No matching datasets")));
}
for (const filter of filters) {
  const input = $(filter.input);
  input.oninput = () => openOptions(filter);
  input.onclick = () => openOptions(filter, true);
  input.onkeydown = event => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter") {
      event.preventDefault();
      const option = openFilter === filter && $(filter.list).querySelector('[aria-selected="true"]');
      if (option) chooseFilter(filter, option.textContent);
      $("datasetFilters").requestSubmit();
      return;
    }
    if (event.key === "Escape") { closeFilters(); return; }
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    if (openFilter !== filter) openOptions(filter, true);
    const rows = [...$(filter.list).querySelectorAll('[role="option"]')];
    if (!rows.length) return;
    let index = rows.findIndex(row => row.getAttribute("aria-selected") === "true");
    index = index < 0 ? (event.key === "ArrowDown" ? 0 : rows.length - 1) :
      (index + (event.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length;
    rows.forEach((row,i) => row.setAttribute("aria-selected", String(i === index)));
    input.setAttribute("aria-activedescendant", rows[index].id);
    rows[index].scrollIntoView({block:"nearest"});
  };
  document.querySelector('[data-filter="' + filter.input + '"]').onclick = () => {
    if (openFilter === filter) closeFilters();
    else { input.focus(); openOptions(filter, true); }
  };
}
document.addEventListener("pointerdown",event => { if (!event.target.closest(".data-combobox")) closeFilters(); });
$("clearFilters").onclick = () => { filters.forEach(filter => $(filter.input).value = ""); applyFilters(); };
async function refresh() {
  datasets = await api("data/catalog");
  if (selectedDataset) selectedDataset = datasets.find(item => item.dataset === selectedDataset.dataset);
  renderDatasets();
  if (selectedDataset) datasetInfo();
  renderThemeToggle();
}
$("refresh").onclick = () => refresh().catch(error => message(error.message, true));
function datasetInfo() {
  $("datasetTitle").textContent = selectedDataset.dataset;
  $("datasetInfo").textContent = (selectedDataset.legacy ? tr("Legacy dataset") + " · " : "") +
    tr("Created") + " " + date(selectedDataset.created_at) + " · " + tr("Updated") + " " + date(selectedDataset.modified_at);
}
function stop() {
  scrub = null; playing = false; playbackToken++;
  pendingFrame?.abort();
  $("play").textContent = "▶";
}
function clearEpisode() {
  stop(); selectionToken++; selected = metadata = documentData = null;
  dirty = false; selectedSeparator = loopId = null; currentRevision = null; cameraViews.clear();
  $("images").replaceChildren(); $("editor").hidden = $("annotationPanel").hidden = true;
  $("empty").hidden = false;
}
async function chooseDataset(item) {
  clearEpisode(); selectedDataset = item; episodeRows = [];
  for (const filter of filters) $(filter.input).value = value(item,filter.key);
  closeFilters();
  datasetInfo(); renderDatasets(); renderEpisodes();
  $("deleteDataset").hidden = false;
  const token = selectionToken;
  const rows = await api("episodes?dataset=" + encodeURIComponent(item.dataset));
  if (token !== selectionToken) return;
  episodeRows = rows; renderEpisodes();
}
function renderEpisodes() {
  $("episodeList").replaceChildren(...episodeRows.map(item => {
    const button = node("button");
    button.append(node("strong","Episode " + String(item.index).padStart(4,"0")), node("small",date(item.saved_at)));
    button.setAttribute("aria-pressed", String(selected?.version === item.version));
    button.onclick = () => chooseEpisode(item).catch(error => message(error.message,true));
    return button;
  }));
}
function query(item = selected) { return new URLSearchParams({dataset:item.dataset,index:item.index,version:item.version}); }
function draftKey() { return "vla-annotation-" + JSON.stringify([selected.dataset, selected.index, selected.version]); }
function changed() {
  dirty = true;
  documentData.parent_revision = currentRevision;
  try { localStorage.setItem(draftKey(),JSON.stringify({document:documentData,revision:currentRevision,frame})); }
  catch { message("Browser storage is full. Save annotations before leaving.",true); }
  $("saveState").textContent = tr("Unsaved edits · kept in this browser");
}
async function chooseEpisode(item) {
  clearEpisode(); selected = item; renderEpisodes();
  const token = selectionToken;
  const [meta, annotations] = await Promise.all([api("episode?" + query(item)), api("data/annotations?" + query(item))]);
  if (token !== selectionToken) return;
  metadata = meta; documentData = annotations.document; history = annotations.history; currentRevision = documentData.revision;
  frame = 0;
  const draft = localStorage.getItem(draftKey());
  if (draft) {
    try {
      const cached = JSON.parse(draft);
      if (cached.document.source_version === item.version) {
        documentData = cached.document; currentRevision = cached.revision; dirty = true;
        frame = Math.min(meta.count - 1, Math.max(0, cached.frame || 0));
        message("Temporary edits restored.");
      }
    } catch { message("Unable to restore temporary edits.",true); }
  } else message(annotations.source_changed ? "Source recording changed. Previous annotations remain in history." : "");
  $("editor").hidden = $("annotationPanel").hidden = false; $("empty").hidden = true;
  $("episodeTitle").textContent = "Episode " + String(item.index).padStart(4,"0") + " · " + date(item.saved_at);
  $("frameInput").max = meta.count - 1;
  renderAnnotations(); await show(frame);
}
function newRecordId() {
  return Array.from(crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8,"0")).join("");
}
function parts() {
  const segments = documentData.records.filter(row => row.scope === "segment");
  return segments.length ? segments : [{id:"unlabelled", scope:"segment", start:0,
    end:metadata.count, language:"", outcome:"unknown", reviewed:false}];
}
function commitRecord(record) {
  if (record.scope === "segment" && !documentData.records.some(row => row.id === record.id))
    documentData.records.push(record);
  changed();
}
function range() {
  const part = parts().find(row => row.id === loopId);
  return part ? [part.start,part.end] : [0,metadata.count];
}
function timeAt(index) { return metadata.times?.[index] ?? index / metadata.fps; }
function updatePlayhead() {
  const position = scrub?.target ?? frame;
  const head = $("playhead");
  if (head && metadata) {
    head.style.left = (100 * position / metadata.count) + "%";
    head.setAttribute("aria-valuemin", "0");
    head.setAttribute("aria-valuemax", String(metadata.count - 1));
    head.setAttribute("aria-valuenow", String(position));
    $("playheadLabel").textContent = tr("Frame") + " " + position;
  }
  if (documentData) $("split").disabled = position !== frame || frame === 0 || parts().some(part => part.start === frame);
}
async function show(index) {
  const token = selectionToken, playToken = playbackToken;
  pendingFrame?.abort(); pendingFrame = new AbortController();
  const data = await api("episode?" + query() + "&frame=" + index, null, pendingFrame.signal);
  const decoded = await Promise.all(Object.entries(data.images).map(async ([name, encoded]) => {
    const image = new Image();
    const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], {type:"image/jpeg"}));
    try { image.src = url; await image.decode(); return [name,image]; }
    finally { URL.revokeObjectURL(url); }
  }));
  if (token !== selectionToken || playToken !== playbackToken) return;
  for (const [name,image] of decoded) {
    let view = cameraViews.get(name);
    if (!view) {
      const figure = node("figure"), canvas = node("canvas");
      figure.append(canvas,node("figcaption",name)); $("images").append(figure);
      view = {canvas,context:canvas.getContext("2d")}; cameraViews.set(name,view);
    }
    if (view.canvas.width !== image.width) view.canvas.width = image.width;
    if (view.canvas.height !== image.height) view.canvas.height = image.height;
    view.context.drawImage(image,0,0);
  }
  frame = index; $("frameInput").value = index;
  $("frameLabel").textContent = index + " / " + (metadata.count-1) + " · " + timeAt(index).toFixed(2) + " s";
  $("signals").textContent = Object.entries(data.values).map(([name,values]) => name + "   " + values.join("  ")).join("\n");
  updatePlayhead();
}
// Keep the arrow under the pointer while decoding only the latest requested preview.
function scrubTo(target) {
  if (scrub) {
    scrub.target = target;
    updatePlayhead();
    return;
  }
  stop();
  const request = {target};
  scrub = request;
  updatePlayhead();
  async function preview() {
    try {
      while (scrub === request && frame !== request.target) await show(request.target);
    } catch (error) {
      if (error.name !== "AbortError") message(error.message,true);
    } finally {
      if (scrub === request) { scrub = null; updatePlayhead(); }
    }
  }
  void preview();
}
function seek(index) {
  stop();
  const target = Math.max(0,Math.min(metadata.count-1,Number(index)));
  return show(target).catch(error => { if (error.name !== "AbortError") message(error.message,true); });
}
$("frameInput").onchange = event => seek(event.target.value);
$("back").onclick = () => seek(frame-1);
$("forward").onclick = () => seek(frame+1);
async function playFrom(index) {
  stop();
  playing = true; $("play").textContent = "⏸";
  const token = playbackToken;
  const [start,end] = range();
  let next = index >= start && index < end ? index : start;
  let origin = performance.now() - timeAt(next)*1000;
  try {
    while (playing && token === playbackToken) {
      await show(next);
      if (token !== playbackToken) return;
      if (++next >= end) {
        if (!loopId) { playing = false; $("play").textContent = "▶"; await show(0); return; }
        next = start; origin = performance.now()-timeAt(start)*1000;
      }
      await new Promise(resolve => setTimeout(resolve,Math.max(0,origin+timeAt(next)*1000-performance.now())));
      const elapsed = (performance.now()-origin)/1000;
      while (token === playbackToken && next+1 < end && timeAt(next+1) <= elapsed) next++;
    }
  } catch (error) { if (token === playbackToken && error.name !== "AbortError") { stop(); message(error.message,true); } }
}
$("play").onclick = () => {
  if (playing) stop();
  else playFrom(frame < range()[1]-1 ? frame : range()[0]);
};
function renderAnnotations() {
  $("versions").replaceChildren(new Option(tr("Latest revision"),"current"),...history.map(doc =>
    new Option(date(doc.updated_at) + " · " + doc.revision.slice(0,8),doc.revision)));
  $("saveState").textContent = dirty ? tr("Unsaved edits · kept in this browser") : documentData.updated_at ? tr("Annotations saved") + " · " + date(documentData.updated_at) : "";
  $("records").replaceChildren(...[documentData.records[0], ...parts()].map((record,index) => {
    const box = node("section",undefined,"annotation-record" + (record.id === loopId ? " selected" : ""));
    box.append(node("h3",(record.scope === "episode" ? tr("Whole episode") : tr("Subtask segment") + " " + index) + " · [" + record.start + ", " + record.end + ")"));
    const fields = node("div",undefined,"record-fields");
    const languageLabel = node("label",tr("Language description")), text = node("textarea");
    text.value = record.language; text.rows = 2; text.maxLength = 8192; languageLabel.append(text);
    text.oninput = () => { record.language = text.value; record.reviewed = false; reviewed.checked = false; commitRecord(record); };
    const resultLabel = node("label",language === "zh" ? "结果" : "Outcome"), result = node("select");
    result.append(...["unknown","success","failure"].map(value => new Option(tr(value),value))); result.value = record.outcome; resultLabel.append(result);
    const reviewedLabel = node("label",tr("Reviewed")), reviewed = node("input"); reviewed.type = "checkbox"; reviewed.checked = record.reviewed; reviewedLabel.append(reviewed);
    result.onchange = () => { record.outcome = result.value; record.reviewed = false; reviewed.checked = false; commitRecord(record); };
    reviewed.onchange = () => { record.reviewed = reviewed.checked; commitRecord(record); };
    fields.append(languageLabel,resultLabel,reviewedLabel); box.append(fields); return box;
  }));
  renderTimeline();
}
let hoverFrame = null, suppressTimelineClick = 0;
function clearSelection() {
  if (!loopId && !selectedSeparator) return;
  const wasLooping = Boolean(loopId);
  selectedSeparator = loopId = null;
  renderAnnotations();
  if (playing && wasLooping) playFrom(frame);
}
function selectPart(id) {
  if (id === loopId) return clearSelection();
  selectedSeparator = null;
  loopId = id;
  const target = range()[0];
  renderAnnotations();
  if (playing) playFrom(target);
  else seek(target);
}
function addBoundary(at) {
  stop();
  const segments = parts(), index = segments.findIndex(part => part.start < at && at < part.end);
  if (index < 0) return;
  const left = segments[index], right = {...left, id:newRecordId(), start:at, reviewed:false};
  left.end = at; left.reviewed = false; segments.splice(index + 1,0,right);
  documentData.records = [documentData.records[0], ...segments];
  loopId = null; selectedSeparator = right.id; changed(); renderAnnotations();
}
function moveBoundary(index, at) {
  const segments = parts(), left = segments[index - 1], right = segments[index];
  const boundary = Math.max(left.start + 1,Math.min(right.end - 1,Math.round(at)));
  if (boundary === left.end) return;
  left.end = right.start = boundary; left.reviewed = right.reviewed = false;
  loopId = null; selectedSeparator = right.id; changed(); renderAnnotations();
}
function frameFromPointer(event) {
  const rect = $("timeline").getBoundingClientRect();
  return Math.max(0, Math.min(metadata.count - 1, Math.round((event.clientX-rect.left)/rect.width*metadata.count)));
}
function renderTimeline() {
  const segments = parts(), length = metadata.count;
  const track = $("timelineTrack"), markers = $("timelineMarkers");
  track.replaceChildren(); markers.replaceChildren();
  const selectedPart = segments.find(part => part.id === loopId);
  $("playRange").textContent = tr(selectedPart ? "Loop segment" : "Whole episode") +
    " [" + (selectedPart?.start || 0) + ", " + (selectedPart?.end || length) + ")";
  for (const [index, part] of segments.entries()) {
    const button = node("button", String(index + 1), "interval");
    button.style.left = 100 * part.start / length + "%";
    button.style.width = 100 * (part.end-part.start) / length + "%";
    button.setAttribute("aria-pressed", String(part.id === loopId));
    button.title = "[" + part.start + ", " + part.end + ") " + part.language;
    button.onclick = () => { if (performance.now() >= suppressTimelineClick) selectPart(part.id); };
    track.append(button);
  }
  const boundaries = [0, ...segments.map(part => part.end)];
  boundaries.forEach((boundary,index) => {
    const fixed = index === 0 || index === boundaries.length - 1;
    const marker = node("button", undefined, "cut");
    marker.dataset.cut = index; marker.dataset.fixed = String(fixed);
    marker.append(node("span", String(boundary)));
    marker.style.left = 100 * boundary / length + "%";
    marker.setAttribute("aria-label", tr(fixed ? "Episode endpoint" : "Separator") + " " + boundary);
    marker.setAttribute("aria-pressed", String(!fixed && segments[index].id === selectedSeparator));
    marker.onclick = () => {
      if (performance.now() < suppressTimelineClick) return;
      loopId = null;
      selectedSeparator = fixed ? null : segments[index].id;
      stop(); renderAnnotations();
      seek(Math.min(boundary, length-1));
      markers.querySelector('[data-cut="' + index + '"]')?.focus({preventScroll:true});
    };
    marker.onkeydown = event => {
      if (fixed || !["ArrowLeft","ArrowRight"].includes(event.key)) return;
      event.preventDefault(); stop();
      moveBoundary(index,boundary+(event.key === "ArrowLeft" ? -1 : 1));
      $("timelineMarkers").querySelector('[data-cut="' + index + '"]')?.focus();
    };
    if (!fixed) marker.onpointerdown = event => {
      event.preventDefault(); stop();
      const startX = event.clientX; let moved = false;
      const move = pointer => {
        if (!moved && Math.abs(pointer.clientX-startX)<3) return;
        moved = true; moveBoundary(index,frameFromPointer(pointer));
        scrubTo(parts()[index].start);
      };
      const end = () => {
        document.removeEventListener("pointermove",move);
        document.removeEventListener("pointerup",end);
        document.removeEventListener("pointercancel",end);
        if (moved) suppressTimelineClick = performance.now() + 250;
      };
      document.addEventListener("pointermove",move);
      document.addEventListener("pointerup",end);
      document.addEventListener("pointercancel",end);
    };
    markers.append(marker);
  });
  $("merge").disabled = !segments.some((part,index) => index > 0 && part.id === selectedSeparator);
  $("split").disabled = frame === 0 || segments.some(part=>part.start === frame);
  updatePlayhead();
}
$("timelineTrack").onclick = event => {
  if (event.target === $("timelineTrack")) clearSelection();
};
$("timeline").onpointermove = event => {
  if (!metadata || event.buttons) return;
  if (event.target.closest("#timelineRuler,#playhead,.cut")) { $("addCut").hidden = true; return; }
  hoverFrame = frameFromPointer(event);
  const add = $("addCut");
  add.hidden = hoverFrame === 0 || parts().some(part => part.start === hoverFrame);
  add.style.left = 100 * hoverFrame / metadata.count + "%";
  add.querySelector("span").textContent = String(hoverFrame);
  add.title = tr("Add separator here") + " · " + hoverFrame;
};
$("timeline").onpointerleave = () => { $("addCut").hidden = true; };
$("addCut").onclick = event => {
  event.stopPropagation();
  if (hoverFrame !== null) addBoundary(hoverFrame);
  $("addCut").hidden = true;
};
$("timeline").onclick = event => {
  if (event.target === $("timeline")) clearSelection();
};
$("timelineRuler").ondblclick = event => addBoundary(frameFromPointer(event));
function startScrub(event) {
  if (!event.isPrimary || event.button !== 0) return;
  event.preventDefault();
  const control = event.currentTarget;
  control.setPointerCapture(event.pointerId);
  $("playhead").focus({preventScroll:true});
  $("addCut").hidden = true;
  const move = pointer => {
    if (pointer.pointerId === event.pointerId) scrubTo(frameFromPointer(pointer));
  };
  move(event);
  control.addEventListener("pointermove",move);
  control.addEventListener("lostpointercapture",() => control.removeEventListener("pointermove",move),{once:true});
}
$("playhead").onpointerdown = $("timelineRuler").onpointerdown = startScrub;
$("playhead").onkeydown = event => {
  if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
  event.preventDefault();
  seek(event.key === "Home" ? 0 : event.key === "End" ? metadata.count-1 : frame + (event.key === "ArrowLeft" ? -1 : 1));
};
$("split").onclick = () => addBoundary(frame);
$("merge").onclick = () => {
  const segments = parts(), index = segments.findIndex(part => part.id === selectedSeparator);
  if (index <= 0) return;
  stop(); selectedSeparator = null;
  const a = segments[index-1], b = segments[index];
  a.end = b.end; a.language = [...new Set([a.language,b.language].filter(Boolean))].join("\n");
  if (a.outcome !== b.outcome) a.outcome = "unknown";
  a.reviewed = false; segments.splice(index,1);
  documentData.records = [documentData.records[0],...segments]; changed(); renderAnnotations();
};
$("wholePlayback").onclick = clearSelection;
$("versions").onchange = async event => {
  const version = event.target.value === "current" ? history[0] : history.find(doc=>doc.revision===event.target.value);
  if (!version) return;
  if (dirty && !await confirmAction("Annotation versions","Replace current edits with this annotation version?")) return;
  stop(); selectedSeparator = loopId = null;
  documentData = structuredClone(version); changed(); renderAnnotations();
};
$("save").onclick = async () => {
  if (!documentData || saving) return;
  const token = selectionToken, key = draftKey(), source = {...selected}, records = structuredClone(documentData.records);
  saving = true; $("save").disabled = true;
  try {
    const result = await api("data/annotations",{...source,revision:currentRevision,records});
    const draft = localStorage.getItem(key);
    if (draft && JSON.stringify(JSON.parse(draft).document.records) === JSON.stringify(records)) localStorage.removeItem(key);
    if (token === selectionToken && JSON.stringify(documentData.records) === JSON.stringify(records)) {
      documentData = result.document; currentRevision = documentData.revision; history = result.history; dirty = false;
      renderAnnotations(); message("Annotations saved");
    } else if (token === selectionToken) {
      currentRevision = result.document.revision; history = result.history; changed();
    }
    await refresh();
  } catch (error) { message(error.message,true); }
  finally { saving = false; $("save").disabled = false; }
};
$("export").onclick = () => {
  const blob = new Blob([JSON.stringify({document:documentData,history},null,2)],{type:"application/json"});
  const url = URL.createObjectURL(blob), a = node("a"); a.href = url;
  a.download = selected.dataset+"-episode-"+selected.index+"-annotations.json"; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$("deleteEpisode").onclick = async () => {
  const item = {...selected};
  if (!await confirmAction("Delete episode","Move this recording to trash?")) return;
  try { await api("episode/delete",item); await chooseDataset(selectedDataset); await refresh(); }
  catch (error) { message(error.message,true); }
};
$("deleteDataset").onclick = async () => {
  const item = selectedDataset;
  if (!await confirmAction("Delete dataset","Move this dataset and all its episodes to trash?")) return;
  try {
    await api("data/delete",{dataset:item.dataset}); clearEpisode(); selectedDataset = null; episodeRows = [];
    $("datasetTitle").textContent = $("datasetInfo").textContent = ""; $("deleteDataset").hidden = true; renderEpisodes(); await refresh();
  } catch (error) { message(error.message,true); }
};
window.addEventListener("beforeunload",event => {
  if (dirty) { event.preventDefault(); event.returnValue = ""; }
});
applyTheme(); applyLanguage(); refresh().catch(error=>message(error.message,true));


document.addEventListener("settingsopening", stop);
document.addEventListener("settingschanged", () => {
  clearEpisode(); selectedDataset = null; episodeRows = [];
  $("datasetTitle").textContent = $("datasetInfo").textContent = "";
  $("deleteDataset").hidden = true;
  renderEpisodes(); refresh().catch(error => message(error.message, true));
});
async function checkService() {
  try {
    await api("health");
    $("link").textContent = tr("Service connected"); $("link").className = "chip online";
  } catch {
    $("link").textContent = tr("Connection lost"); $("link").className = "chip";
  }
}
checkService();
setInterval(() => { if (!document.hidden) checkService(); }, 5000);
