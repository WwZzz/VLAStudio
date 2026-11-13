"use strict";

(() => {
  const dataset = $("libraryDataset"), list = $("episodeList");
  const jointHistory = new Map();
  document.addEventListener("themechange", () => {
    if (selected && jointHistory.size) axes({axes:Object.fromEntries(jointHistory), now:timeAt(frame)}, $("recordedAxes"));
  });
  const episodeLabel = item => "Episode " + String(item.index).padStart(4, "0");
  const seek = $("frameSeek"), play = $("playEpisode"), remove = $("deleteEpisode");
  let rows = [], selected, meta, frame = 0, playing = false;
  let request, timer, revision = 0;
  let scopeKey, scopedDataset, choices = [];
  const suffix = $("suffix");
  const date = value => value ? new Date(value).toLocaleString(language === "zh" ? "zh-CN" : "en") : tr("Not recorded");
  function scope() {
    return {robot: $("robot").value, task: $("simulationFields").hidden ? "" : $("task").value,
      real_task: $("realTaskFields").hidden ? "" : $("real_task").value};
  }
  function chooseSuffix() {
    const item = choices.find(row => row.suffix === suffix.value);
    scopedDataset = item?.dataset;
    $("datasetDates").textContent = (item?.created_at || item?.modified_at) ?
      tr("Created") + " " + date(item.created_at) + " · " + tr("Updated") + " " + date(item.modified_at) : "";
  }
  let catalogRequest = 0, sessionKey, output;
  const retake = $("retakeEpisode");
  retake.innerHTML = transportIcon("record");
  const cameraViews = new Map();

  function buttons() {
    play.disabled = seek.disabled = !meta;
    play.innerHTML = transportIcon(playing ? "pause" : "run");
    play.title = tr(playing ? "Pause playback" : "Playback");
    play.setAttribute("aria-label", play.title);
    const session = state?.session;
    retake.hidden = !collecting;
    const unavailable = !session?.connection?.ready ? "Connect the robot to re-record" :
      !selected?.rerecordable ? "Only standalone collection episodes can be re-recorded" :
      selected.dataset !== scopedDataset ? "Connect collection to the selected dataset" :
      session.active && session.phase !== "ready" ? "Collection is not ready for a new recording" : "";
    retake.disabled = busy || !!unavailable;
    retake.title = tr(unavailable || "Re-record episode");
    retake.setAttribute("aria-label", tr("Re-record episode"));
    const locked = state?.session.active && state.session.options?.output === selected?.dataset;
    remove.disabled = !selected || locked;
    remove.title = locked ? tr("Disconnect before deleting from this dataset") : tr("Delete episode");
  }
  function pause() {
    playing = false;
    clearTimeout(timer);
    buttons();
  }
  function clearImages() {
    cameraViews.clear();
    $("recordedImages").replaceChildren();
  }
  function cancel() {
    revision++;
    request?.abort();
    pause();
  }
  function timeAt(i) { return meta.times?.[i] ?? i / meta.fps; }
  async function getFrame(params) {
    request?.abort();
    request = new AbortController();
    const query = new URLSearchParams({...selected, ...params});
    const response = await fetch("/api/episode?" + query, {signal: request.signal});
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    return data;
  }
  function display(data, index, images) {
    for (const [name, image] of images) {
      let view = cameraViews.get(name);
      if (!view) {
        const figure = el("figure", ""), canvas = el("canvas", "");
        canvas.setAttribute("role", "img");
        canvas.setAttribute("aria-label", name);
        figure.append(canvas, el("figcaption", "", name));
        view = {figure, canvas, context: canvas.getContext("2d", {alpha: false})};
        cameraViews.set(name, view);
        $("recordedImages").append(figure);
      }
      const {canvas, context} = view;
      if (canvas.width !== image.naturalWidth) canvas.width = image.naturalWidth;
      if (canvas.height !== image.naturalHeight) canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);
    }
    for (const [name, view] of cameraViews) {
      if (!Object.hasOwn(data.images, name)) {
        view.figure.remove();
        cameraViews.delete(name);
      }
    }
    const now = timeAt(index);
    const entries = Object.entries(data.values);
    const positions = entries.filter(([name]) => /(^|\/)(left_qpos|right_qpos|qpos)$/.test(name));
    if (!positions.length) positions.push(...entries.filter(([name]) => /(^|\/)joint_angles$/.test(name)));
    const traces = {};
    for (const [name, vector] of positions) {
      if (!vector.length || !vector.every(Number.isFinite)) continue;
      const history = jointHistory.get(name) || [];
      while (history.length && history[0][0] < now - 30) history.shift();
      if (!history.length || history.at(-1)[0] < now) history.push([now, vector]);
      jointHistory.set(name, history);
      traces[name] = history;
    }
    $("recordedJoints").hidden = !Object.keys(traces).length;
    if (!$("recordedJoints").hidden) axes({axes:traces, now}, $("recordedAxes"));
    $("recordedSignals").hidden = !entries.length;
    const values = entries.map(([name, vector]) => {
      const row = el("div", "signal-row");
      row.append(el("span", "", name), el("code", "", vector.join("   ")));
      return row;
    });
    $("recordedValues").replaceChildren(...values);
    frame = index;
    seek.value = index;
    $("frameNumber").textContent = (index + 1) + " / " + meta.count + " · " + timeAt(index).toFixed(2) + " s";
  }
  function failed(error, token) {
    if (error.name === "AbortError" || token !== revision) return;
    pause();
    $("playerMessage").textContent = error.message;
  }
  async function show(index, token) {
    const data = await getFrame({frame:index});
    if (token !== revision) return;
    // Keep the previous frame visible until every camera in this frame is ready.
    const urls = [];
    try {
      const images = await Promise.all(Object.entries(data.images).map(async ([name, encoded]) => {
        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        const url = URL.createObjectURL(new Blob([bytes], {type: "image/jpeg"}));
        urls.push(url);
        const image = new Image();
        image.src = url;
        await image.decode();
        return [name, image];
      }));
      if (token === revision) display(data, index, images);
    } finally {
      urls.forEach(url => URL.revokeObjectURL(url));
    }
  }
  function highlight() {
    $("currentEpisode").setAttribute("aria-pressed", String(!selected));
    for (const row of list.children)
      row.setAttribute("aria-pressed", String(row.dataset.version === selected?.version));
  }
  async function selectEpisode(item) {
    cancel();
    const token = revision;
    selected = item;
    meta = null;
    clearImages();
    $("recordedValues").replaceChildren();
    $("frameNumber").textContent = "";
    seek.value = 0;
    highlight();
    $("episodePlayer").hidden = !selected;
    document.body.classList.toggle("reviewing", !!selected);
    $("playbackTransport").hidden = remove.hidden = !selected;
    $("recordedSignals").hidden = $("recordedJoints").hidden = true;
    jointHistory.clear();
    $("recordedAxes").replaceChildren();
    heldKeys.clear();
    $("playbackTitle").textContent = selected ? selected.dataset + " · " + episodeLabel(selected) : "";
    buttons();
    $("playerMessage").textContent = selected ? tr("Loading recording") : "";
    if (!selected) return;
    try {
      const result = await getFrame({});
      if (token !== revision) return;
      meta = result;
      seek.max = meta.count - 1;
      await show(0, token);
      if (token !== revision) return;
      $("playerMessage").textContent = "";
      buttons();
    } catch (error) { failed(error, token); }
  }
  function renderList() {
    const items = rows.filter(r => r.dataset === dataset.value);
    list.replaceChildren(...items.map(item => {
      const row = el("button", "", episodeLabel(item));
      row.append(el("small", "", date(item.saved_at)));
      row.type = "button";
      row.dataset.version = item.version;
      row.setAttribute("aria-controls", "episodePlayer");
      row.onclick = () => {
        selectEpisode(item);
        $("episodePlayer").scrollIntoView({block:"nearest"});
      };
      return row;
    }));
    $("emptyEpisodes").hidden = items.length > 0;
    if (selected && !items.some(item => item.version === selected.version)) selectEpisode(null);
    highlight();
  }
  async function load() {
    const token = ++catalogRequest;
    try {
      const context = scope();
      const key = JSON.stringify(context);
      if (context.robot) {
        const items = await api("datasets", context);
        if (token !== catalogRequest) return;
        const previous = scopeKey === key ? suffix.value :
          state?.session.options?.real_task === context.real_task && state?.session.options?.robot === context.robot
          ? state.session.options.suffix || "" : "";
        scopeKey = key;
        choices = items;
        suffix.replaceChildren(...items.map(item => new Option(item.suffix || tr("Default"), item.suffix)));
        suffix.value = items.some(row => row.suffix === previous) ? previous : "";
        chooseSuffix();
      }
      const result = scopedDataset ? await api("episodes?dataset=" + encodeURIComponent(scopedDataset)) : [];
      if (token !== catalogRequest) return;
      rows = result;
      const preferred = scopedDataset;
      const names = [scopedDataset].filter(Boolean);
      dataset.replaceChildren(...names.map(n => new Option(n,n)));
      if (names.includes(preferred)) dataset.value = preferred;
      renderList();
    } catch (error) {
      if (token === catalogRequest) { sessionKey = null; notice(error.message); }
    }
  }
  play.onclick = () => {
    if (playing) { cancel(); return; }
    if (!meta) return;
    cancel();
    playing = true;
    buttons();
    const token = revision;
    let next = frame === meta.count - 1 ? 0 : frame;
    if (next === 0) jointHistory.clear();
    const start = performance.now() - timeAt(next) * 1000;
    async function tick() {
      if (revision !== token || !playing) return;
      try {
        const elapsed = (performance.now() - start) / 1000;
        while (next + 1 < meta.count && timeAt(next + 1) <= elapsed) next++;
        await show(next, token);
        if (token !== revision || !playing) return;
        if (++next >= meta.count) {
          pause();
          jointHistory.clear();
          await show(0, token);
          return;
        }
        timer = setTimeout(tick, Math.max(0, start + timeAt(next)*1000 - performance.now()));
      } catch (error) { failed(error, token); }
    }
    tick();
  };
  seek.oninput = () => {
    cancel();
    jointHistory.clear();
    const token = revision;
    show(Number(seek.value), token).catch(e => failed(e, token));
  };
  suffix.onchange = () => { chooseSuffix(); load(); };
  $("newSuffix").onclick = async () => {
    const value = window.prompt(tr("New suffix"))?.trim();
    if (!value) return;
    const context = scope();
    try {
      const item = await api("datasets/create", {...context, suffix:value});
      if (JSON.stringify(scope()) !== JSON.stringify(context)) return;
      if (![...suffix.options].some(option => option.value === value)) suffix.add(new Option(value, value));
      suffix.value = item.suffix;
      await load();
    } catch (error) { notice(tr(error.message)); }
  };
  dataset.onchange = renderList;
  $("libraryOpen").onclick = async () => {
    heldKeys.clear();
    await load();
    $("episodePanel").scrollIntoView({block:"nearest"});
    (list.querySelector('[aria-pressed="true"]') || list.firstElementChild || dataset).focus({preventScroll:true});
  };
  $("currentEpisode").onclick = event => {
    selectEpisode(null);
    if (event.detail) event.currentTarget.blur();
    refresh().catch(error => notice(error.message));
  };
  retake.onclick = async () => {
    const item = selected, session = state.session;
    if (!item || busy || retake.disabled) return;
    cancel();
    const label = item.dataset + " · " + episodeLabel(item);
    if (!await confirmAction(tr("Re-record episode"), label + " — " + tr("Re-record this episode? Saving the new recording will overwrite the original."))) return;
    busy = true;
    buttons();
    try {
      await api("episode/rerecord", {...item, connection:session.connection.id,
        id:session.active ? session.id : null, options:options()});
      selectEpisode(null);
    } catch (error) { notice(tr(error.message)); }
    finally {
      busy = false;
      await refresh();
      buttons();
    }
  };
  remove.onclick = async () => {
    if (!selected) return;
    const item = selected;
    cancel();
    if (!await confirmAction(tr("Delete episode"), item.dataset + " · Episode " + item.index + " — " + tr("Move this recording to trash?"))) return;
    try {
      await api("episode/delete", item);
      await load();
    } catch (error) { $("playerMessage").textContent = tr(error.message); }
  };
  function syncLibrary() {
    const s = state?.session;
    if (!s) return;
    const name = (s.options?.output || $("output").value).trim();
    if (name !== output) {
      output = name;
      dataset.selectedIndex = -1;
      selectEpisode(null);
    }
    const key = JSON.stringify([name, s.id, s.phase, s.episode, scope(), language]);
    if (key !== sessionKey) {
      sessionKey = key;
      load();
    }
    if (selected) buttons();
  }
  document.addEventListener("settingschanged", () => {
    selectEpisode(null);
    rows = []; dataset.replaceChildren(); sessionKey = undefined;
    load();
  });
  document.addEventListener("episode-session", syncLibrary);
  $("episodePanel").addEventListener("focusin", () => heldKeys.clear());
  syncLibrary();
  new ResizeObserver(entries => {
    document.body.style.paddingBottom = entries[0].target.offsetHeight + 16 + "px";
  }).observe($("transport"));
})();
