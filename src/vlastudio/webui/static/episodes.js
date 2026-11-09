"use strict";

(() => {
  const dialog = $("library");
  const dataset = $("libraryDataset"), episode = $("libraryEpisode");
  const seek = $("frameSeek"), play = $("playEpisode"), remove = $("deleteEpisode");
  let rows = [], selected, meta, frame = 0, playing = false;
  let request, timer, revision = 0, imageUrls = [];

  function buttons() {
    play.disabled = seek.disabled = !meta;
    play.innerHTML = transportIcon(playing ? "pause" : "run");
    play.title = tr(playing ? "Pause playback" : "Playback");
    play.setAttribute("aria-label", play.title);
    const locked = state?.session.active && state.session.options?.output === selected?.dataset;
    remove.disabled = !selected || locked;
    remove.title = locked ? tr("Disconnect before deleting from this dataset") : tr("Delete episode");
  }
  function pause() {
    playing = false;
    clearTimeout(timer);
    buttons();
  }
  function releaseImages() {
    for (const url of imageUrls) URL.revokeObjectURL(url);
    imageUrls = [];
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
  function display(data, index) {
    releaseImages();
    const views = [];
    for (const [name, encoded] of Object.entries(data.images)) {
      const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], {type:"image/jpeg"}));
      imageUrls.push(url);
      const figure = el("figure", ""), img = el("img", "");
      img.src = url;
      img.alt = name;
      figure.append(img, el("figcaption", "", name));
      views.push(figure);
    }
    $("recordedImages").replaceChildren(...views);
    const values = Object.entries(data.values).map(([name, vector]) => {
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
    if (token === revision && dialog.open) display(data, index);
  }
  async function selectEpisode() {
    cancel();
    const token = revision;
    selected = rows.find(row => row.dataset === dataset.value && String(row.index) === episode.value);
    meta = null;
    releaseImages();
    $("recordedImages").replaceChildren();
    $("recordedValues").replaceChildren();
    $("frameNumber").textContent = "";
    seek.value = 0;
    buttons();
    $("playerMessage").textContent = tr(selected ? "Loading recording" : "No recordings");
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
  function pickDataset() {
    episode.replaceChildren(...rows.filter(r => r.dataset === dataset.value).map(r =>
      new Option("Episode " + String(r.index).padStart(4,"0"), r.index)));
    return selectEpisode();
  }
  async function load(preferred) {
    cancel();
    const token = revision;
    try {
      const result = await api("episodes");
      if (token !== revision || !dialog.open) return;
      rows = result;
      const names = [...new Set(rows.map(r => r.dataset))];
      dataset.replaceChildren(...names.map(n => new Option(n,n)));
      if (names.includes(preferred)) dataset.value = preferred;
      await pickDataset();
    } catch (error) { failed(error, token); }
  }
  play.onclick = () => {
    if (playing) { cancel(); return; }
    if (!meta) return;
    cancel();
    playing = true;
    buttons();
    const token = revision;
    let next = frame === meta.count - 1 ? 0 : frame;
    const start = performance.now() - timeAt(next) * 1000;
    async function tick() {
      if (revision !== token || !playing) return;
      try {
        const elapsed = (performance.now() - start) / 1000;
        while (next + 1 < meta.count && timeAt(next + 1) <= elapsed) next++;
        await show(next, token);
        if (token !== revision || !playing) return;
        if (++next >= meta.count) { pause(); return; }
        timer = setTimeout(tick, Math.max(0, start + timeAt(next)*1000 - performance.now()));
      } catch (error) { failed(error, token); }
    }
    tick();
  };
  seek.oninput = () => {
    cancel();
    const token = revision;
    show(Number(seek.value), token).catch(e => failed(e, token));
  };
  dataset.onchange = pickDataset;
  episode.onchange = selectEpisode;
  remove.onclick = async () => {
    if (!selected) return;
    const item = selected;
    cancel();
    if (!await confirmAction(tr("Delete episode"), item.dataset + " · Episode " + item.index + " — " + tr("Move this recording to trash?"))) return;
    try {
      await api("episode/delete", item);
      await load(item.dataset);
    } catch (error) { $("playerMessage").textContent = tr(error.message); }
  };
  $("libraryOpen").onclick = () => {
    heldKeys.clear();
    dialog.showModal();
    load($("output").value);
  };
  $("libraryClose").onclick = () => dialog.close();
  dialog.addEventListener("close", () => {
    cancel();
    selected = meta = null;
    releaseImages();
  });
  new ResizeObserver(entries => {
    document.body.style.paddingBottom = entries[0].target.offsetHeight + 16 + "px";
  }).observe($("transport"));
})();
