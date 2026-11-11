"use strict";
const $ = (id) => document.getElementById(id);
const mode = location.pathname === "/infer" ? "infer" : "collect";
const collecting = mode === "collect";
try {
  $("keyboardPanel").open = localStorage.getItem("vla-keyboard-open") !== "false";
} catch {}
$("keyboardPanel").addEventListener("toggle", () => {
  try {
    localStorage.setItem("vla-keyboard-open", String($("keyboardPanel").open));
  } catch {}
});
const names = {
  record: "Start recording",
  finish: "Finish episode",
  record_pause: "Pause recording",
  record_resume: "Resume recording",
  save: "Save",
  discard: "Discard episode",
  run: "Start inference",
  pause: "Pause inference",
  reset: "Reset",
  home: "Reset",
  tele_run: "Enable teleoperation",
  tele_pause: "Pause teleoperation",
  stop: collecting ? "Disconnect teleoperation" : "Disconnect policy",
};
const phases = {
  idle: "Not connected",
  waiting_robot: "Waiting for robot",
  loading: "Connecting devices",
  ready: "Ready to record",
  recording: "Recording",
  record_paused: "Recording paused",
  save: "Awaiting save",
  infer_ready: "Policy ready",
  running: "Inference running",
  paused: "Inference paused",
  homing: "Resetting",
  busy: "Processing",
  stopping: "Disconnecting",
};
let connecting = false;
let disconnecting = false;
let robotTasks = [];
sessionStorage.removeItem("vla-key");
let state = null,
  profiles = null,
  busy = false,
  lastButtons = "";
let modelData = {},
  yaw = -0.8,
  pitch = 0.42,
  drag = null;
const feeds = new Map();
function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
}
function notice(message) {
  if (language === "en")
    message =
      Object.entries(chinese).find(([, value]) => value === message)?.[0] ||
      message;
  $("error").textContent = message;
  $("error").hidden = false;
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => ($("error").hidden = true), 6500);
}
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const v = await r.json();
  if (!r.ok) throw Error(v.error || r.status);
  return v;
}
function confirmAction(title, text) {
  return new Promise((done) => {
    $("confirmTitle").textContent = title;
    $("confirmText").textContent = text;
    const dlg = $("confirm");
    dlg.returnValue = "";
    dlg.onclose = () => done(dlg.returnValue === "yes");
    dlg.showModal();
  });
}
$("cancel").onclick = () => $("confirm").close();
$("accept").onclick = () => $("confirm").close("yes");
function fill(id, opts, empty) {
  const s = $(id);
  s.replaceChildren();
  if (empty) s.add(new Option(empty, ""));
  for (const v of opts) s.add(new Option(v, v));
}
function profilePicker(id) {
  const select = $(id), input = $(id + "Search"), list = $(id + "Options");
  let matches = [], active = -1;
  function sync() {
    input.disabled = select.disabled;
    if (input.disabled || list.hidden) close();
  }
  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    input.value = select.selectedOptions[0]?.text || "";
  }
  function highlight(index) {
    if (!matches.length) return;
    active = Math.max(0, Math.min(matches.length - 1, index));
    Array.from(list.children).forEach((row, i) =>
      row.setAttribute("aria-selected", String(i === active)));
    const row = list.children[active];
    input.setAttribute("aria-activedescendant", row.id);
    if (row.offsetTop < list.scrollTop) list.scrollTop = row.offsetTop;
    else if (row.offsetTop + row.offsetHeight > list.scrollTop + list.clientHeight)
      list.scrollTop = row.offsetTop + row.offsetHeight - list.clientHeight;
  }
  function open(prefix) {
    const query = prefix.trim().toLowerCase();
    matches = Array.from(select.options).filter(o => o.text.toLowerCase().startsWith(query));
    list.replaceChildren(...matches.map((option, i) => {
      const row = el("div", "profile-option", option.text);
      row.id = id + "-option-" + i;
      row.setAttribute("role", "option");
      row.dataset.index = i;
      return row;
    }));
    list.hidden = false;
    list.scrollTop = 0;
    input.setAttribute("aria-expanded", "true");
    active = -1;
    input.removeAttribute("aria-activedescendant");
    if (matches.length) highlight(query ? 0 : matches.indexOf(select.selectedOptions[0]));
    else list.append(el("div", "profile-empty", tr("No matching configuration")));
  }
  function choose(index) {
    if (!matches[index]) return;
    select.value = matches[index].value;
    close();
    select.dispatchEvent(new Event("change", {bubbles:true}));
    syncProfilePickers();
  }
  input.onfocus = () => { open(""); input.select(); };
  input.onclick = () => { if (list.hidden) { open(""); input.select(); } };
  input.oninput = () => open(input.value);
  input.onblur = close;
  list.onmousedown = e => e.preventDefault();
  list.onclick = e => {
    const row = e.target.closest("[role=option]");
    if (row) choose(Number(row.dataset.index));
  };
  input.onkeydown = e => {
    if (e.isComposing) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) open("");
      else highlight(active + (e.key === "ArrowDown" ? 1 : -1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (!list.hidden) choose(active);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
      input.select();
    }
  };
  return sync;
}
const profilePickers = ["robot", "teleop"].map(profilePicker);
function syncProfilePickers() { profilePickers.forEach(sync => sync()); }

function preferences() {
  try {
    return JSON.parse(localStorage.getItem("vla-" + mode)) || {};
  } catch {
    return {};
  }
}
function options() {
  return {
    mode,
    robot: $("robot").value,
    teleop: $("teleop").value,
    sync: $("sync").value,
    policy: $("policy").value,
    manager: $("manager").value,
    rate: Number($("rate").value),
    output: $("output").value,
    task: $("task").value,
    real_task: $("real_task").value,
    on_success: $("on_success").value,
  };
}
function fillRobotTasks() {
  const selected = $("real_task").value;
  $("real_task").replaceChildren(new Option(tr("No task selected"), ""));
  for (const task of robotTasks)
    $("real_task").add(new Option(task.name, task.id));
  $("real_task").value = robotTasks.some((task) => task.id === selected)
    ? selected
    : "";
}
function renderTaskPanel() {
  const benchmark = !!profiles?.robot_options?.[$("robot").value]?.tasks;
  $("simulationFields").hidden = !benchmark;
  $("realTaskFields").hidden = benchmark;
  $("taskKind").textContent = tr(benchmark ? "Benchmark task" : "Robot task");
  $("newTask").disabled = connecting || (!collecting && !!state?.session.active);
  $("realTaskLabel").textContent = tr(collecting ? "Task for next recording" : "Robot task");
  const task = robotTasks.find((task) => task.id === $("real_task").value);
  $("realTaskInstruction").textContent = task
    ? task.instruction || tr("No instruction provided.")
    : tr(
        "Select or create a robot task. Its instruction will be saved with recordings.",
      );
}
$("real_task").onchange = () => {
  renderTaskPanel();
  localStorage.setItem("vla-" + mode, JSON.stringify(options()));
};
$("newTask").onclick = () => {
  $("taskForm").reset();
  $("taskError").textContent = "";
  $("taskDialog").showModal();
};
$("cancelTask").onclick = () => $("taskDialog").close();
$("taskForm").onsubmit = async (event) => {
  event.preventDefault();
  $("saveTask").disabled = true;
  $("taskError").textContent = "";
  try {
    const task = await api("tasks", {
      name: $("taskName").value,
      instruction: $("taskInstruction").value,
    });
    robotTasks = await api("tasks");
    fillRobotTasks();
    $("real_task").value = task.id;
    renderTaskPanel();
    localStorage.setItem("vla-" + mode, JSON.stringify(options()));
    $("taskDialog").close();
  } catch (error) {
    $("taskError").textContent =
      language === "en"
        ? error.message
        : chinese[error.message] || error.message;
  } finally {
    $("saveTask").disabled = false;
  }
};
function defaults() {
  const r = $("robot").value;
  $("teleop").value = r.startsWith("rebot")
    ? "quest3_rebot_b601rs_rel_ee"
    : r.startsWith("bi_alicia")
      ? "quest3_bi_alicia_rel_ee"
      : "";
  $("sync").value = r === "rebot_b601rs_head_rel_ee" ? "rebot_b601rs_head" : "";
  $("output").value = collecting ? r : "";
  const settings = profiles.robot_options?.[r];
  $("simulationFields").hidden = !settings?.tasks;
  fill("task", settings?.tasks || []);
  if (settings?.teleop) $("teleop").value = settings.teleop;
  if (settings?.rate) $("rate").value = settings.rate;
  renderTaskPanel();
  if (!collecting && r === "dummy")
    $("policy").value = "__dummy-zero14-chunk16";
  else if (!collecting && $("policy").value.startsWith("__dummy"))
    $("policy").value = "";
}
async function init() {
  profiles = await api("catalog");
  robotTasks = await api("tasks");
  fillRobotTasks();
  fill("robot", profiles.robot.filter((name) =>
    (profiles.robot_options?.[name]?.modes || ["collect", "infer"]).includes(mode)));
  fill("teleop", profiles.teleop, tr("None \u00b7 Observations only"));
  fill("sync", profiles.recording, tr("Native synchronization"));
  fill("manager", profiles.action_manager);
  $("robot").value = collecting
    ? "rebot_b601rs_head_rel_ee"
    : "rebot_b601rs_head_qpos";
  $("manager").value = "basic";
  defaults();
  const saved = preferences();
  for (const id of [
    "robot",
    "teleop",
    "sync",
    "policy",
    "manager",
    "rate",
    "output",
    "real_task",
  ])
    if (id in saved) $(id).value = saved[id];
  const settings = profiles.robot_options?.[$("robot").value];
  $("simulationFields").hidden = !settings?.tasks;
  fill("task", settings?.tasks || []);
  if (saved.task && settings?.tasks?.includes(saved.task))
    $("task").value = saved.task;
  $("on_success").value = saved.on_success || "continue";
  renderTaskPanel();
  syncProfilePickers();
  await refresh();
}
$("robot").onchange = defaults;
$("teleop").onchange = () => {
  if ($("teleop").value !== "quest3_rebot_b601rs_rel_ee") $("sync").value = "";
};
$("robotConnection").onclick = async () => {
  const c = state?.session.connection;
  const opts = options();
  const title = c ? tr("Disconnect robot") : tr("Connect ") + opts.robot;
  const message = c ? tr("Disconnect the robot using its normal shutdown procedure?") :
    tr("This starts the selected devices. Real robots may enable or move during connection.");
  if (await confirmAction(title, message))
    await post(c ? "robot/disconnect" : "robot/connect", c ? {connection:c.id} : opts);
};
$("form").onsubmit = async (e) => {
  e.preventDefault();
  if (busy) return;
  if (state?.session.active) {
    await act("stop");
    return;
  }
  const opts = options();
  connecting = true;
  renderSession(state.session);
  try {
    await post("start", {...opts, connection: state.session.connection?.id});
    localStorage.setItem("vla-" + mode, JSON.stringify(opts));
  } finally {
    connecting = false;
    renderSession(state.session);
  }
};
$("loadAll").onclick = async () => {
  if (busy) return;
  const opts = options();
  const connection = state.session.connection;
  if (connection) {
    if (!(await confirmAction(tr("Disconnect all"), tr(
      "Disconnect all devices and discard the current unsaved episode?",
    )))) return;
    heldKeys.clear();
    disconnecting = true;
    try {
      await post("disconnect", {connection:connection.id});
    } finally {
      disconnecting = false;
      if (state) renderSession(state.session);
    }
    return;
  }
  if (!connection && !(await confirmAction(
    tr("Connect ") + opts.robot,
    tr("This starts the selected devices. Real robots may enable or move during connection."),
  ))) return;
  busy = connecting = true;
  renderSession(state.session);
  try {
    // Validate and preload the source before opening the robot hardware.
    if (!state.session.active)
      await api("start", {...opts, connection: connection?.id});
    if (!connection) await api("robot/connect", opts);
    localStorage.setItem("vla-" + mode, JSON.stringify(opts));
  } catch (e) {
    notice(e.message);
  } finally {
    busy = connecting = false;
    await refresh();
  }
};
async function post(route, data) {
  if (busy) return;
  busy = true;
  if (state) renderSession(state.session);
  try {
    await api(route, data);
    await refresh();
  } catch (e) {
    notice(e.message);
  } finally {
    busy = false;
    if (state) renderSession(state.session);
  }
}
async function act(action) {
  const sid = state.session.id;
  if (["discard", "stop"].includes(action)) {
    const msg = {
      discard: tr("Discard this episode?"),
      stop: tr(
        "Disconnect this source and keep the robot connected?",
      ),
    };
    if (!(await confirmAction(tr(names[action]), msg[action]))) return;
  }
  if (action === "home" || action === "reset") heldKeys.clear();
  const payload = {mode, id: sid, action};
  if (collecting && action === "record" && !$("realTaskFields").hidden)
    payload.real_task = $("real_task").value;
  await post("action", payload);
}
function transportIcon(key) {
  const shapes = {
    record: '<circle cx="12" cy="12" r="7" fill="currentColor"/>',
    run: '<path d="M7 4.5 20 12 7 19.5Z" fill="currentColor"/>',
    pause: '<path d="M7 5v14m10-14v14" stroke-width="4"/>',
    stop: '<rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor"/>',
    reset: '<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',
    save: '<path d="m4 12 5 5L20 6"/>',
    discard: '<path d="m6 6 12 12M18 6 6 18"/>',
  };
  return '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    shapes[({record_resume:"run",record_pause:"pause",finish:"stop",home:"reset"})[key] || key] + '</svg>';
}
$("teleSwitch").onclick = (event) => {
  if (event.detail) $("teleSwitch").blur();
  const enabled = $("teleSwitch").getAttribute("aria-checked") === "true";
  if (enabled) heldKeys.clear();
  act(enabled ? "tele_pause" : "tele_run");
};
document.addEventListener("keydown", (event) => {
  if (reviewingEpisode() || !collecting || event.key !== "Enter" || event.isComposing || event.defaultPrevented ||
      event.ctrlKey || event.altKey || event.metaKey || event.shiftKey ||
      document.querySelector("dialog[open]") ||
      event.target.closest("input,textarea,select,button,a,summary,[contenteditable=true]")) return;
  const action = $("buttons").querySelector("[data-enter]:enabled");
  if (!action) return;
  event.preventDefault();
  if (!event.repeat) action.click();
});
document.addEventListener("keydown", (event) => {
  if (reviewingEpisode() || event.code !== "KeyR" || !event.ctrlKey || !event.altKey ||
      event.repeat || event.isComposing || event.defaultPrevented ||
      event.metaKey || event.shiftKey || busy ||
      document.querySelector("dialog[open]") ||
      event.target.closest("input,textarea,select,[contenteditable=true]")) return;
  const reset = $("buttons").querySelector(".home:enabled,.reset:enabled");
  if (!reset) return;
  event.preventDefault();
  heldKeys.clear();
  reset.click();
});
function renderSession(s) {
  const other = s.active && s.mode !== mode;
  const c = s.connection;
  const selection = c || s.prepared;
  if (selection && $("robot").value !== selection.robot) {
    if (![...$("robot").options].some(o => o.value === selection.robot))
      $("robot").add(new Option(selection.robot,selection.robot));
    $("robot").value = selection.robot;
    defaults();
  }
  if (selection?.task) $("task").value = selection.task;
  if (selection?.on_success) $("on_success").value = selection.on_success;
  if (s.active && !other) {
    const settings = profiles.robot_options?.[s.robot];
    $("simulationFields").hidden = !settings?.tasks;
    const tasks = settings?.tasks || [];
    if (
      JSON.stringify([...$("task").options].map((o) => o.value)) !==
      JSON.stringify(tasks)
    )
      fill("task", tasks);
    $("sync").value = s.options.sync || "";
    for (const [key, value] of Object.entries(s.options || {}))
      if ($(key) && !(collecting && key === "real_task")) $(key).value = value;
  }
  renderTaskPanel();
  $("phase").textContent =
    (other
      ? tr(s.mode === "collect" ? "Data collection" : "Policy inference") +
        " · "
      : "") + tr(phases[s.phase]);
  for (const field of $("form").elements)
    field.disabled = s.active || connecting;
  if (collecting) $("real_task").disabled = connecting;
  for (const id of ["robot", "task", "on_success"]) $(id).disabled = !!selection || busy;
  syncProfilePickers();
  $("robotConnection").disabled = busy || (!!c && !!s.loaded?.length);
  $("robotConnection").textContent = tr(c ? c.ready ? "Disconnect robot" : "Connecting robot" : "Connect robot");
  const loading = connecting || (s.active && s.phase === "loading");
  $("launch").disabled = busy || other || (profiles.robot_options?.[$("robot").value]?.modes || ["collect","infer"]).indexOf(mode) < 0 ||
    ["recording","record_paused","save","stopping"].includes(s.phase) || connecting;
  $("launch").setAttribute(
    "aria-busy",
    String(loading || s.phase === "stopping"),
  );
  $("connectionProgress").hidden = !(loading || s.phase === "stopping");
  $("connectionLabel").textContent = tr(
    loading
      ? "Connecting devices"
      : s.phase === "stopping"
        ? "Disconnecting"
        : s.active
          ? (collecting ? "Disconnect teleoperation" : "Disconnect policy")
          : (collecting ? "Connect teleoperation" : "Connect policy"),
  );
  const allLoading = connecting || disconnecting;
  $("loadAll").disabled = busy || connecting || disconnecting || (!c &&
    !(profiles.robot_options?.[$("robot").value]?.modes || ["collect", "infer"]).includes(mode));
  $("loadAll").classList.toggle("disconnect-all", !!c);
  $("loadAll").setAttribute("aria-busy", String(allLoading));
  $("loadAllLabel").textContent = tr(disconnecting ? "Disconnecting" : connecting ? "Connecting devices" : c ? "Disconnect all" : "Connect all");
  $("loadAllProgress").hidden = !allLoading;
  const allowed = other ? [] : s.actions;
  const toggle = $("teleSwitch");
  toggle.hidden = !collecting;
  const enabled = collecting && !other && s.active && s.controls_enabled;
  toggle.setAttribute("aria-checked", String(enabled));
  toggle.disabled = busy || !allowed.includes(enabled ? "tele_pause" : "tele_run");
  const toggleLabel = tr(s.active && !other && !s.options?.teleop ? "Enable controls" : "Enable teleoperation");
  $("teleSwitchLabel").textContent = toggleLabel;
  toggle.setAttribute("aria-label", toggleLabel);
  $("episodeNumber").classList.toggle("recording-now", collecting && !other && s.phase === "recording");
  const inEpisode = s.phase === "recording" || s.phase === "record_paused";
  const current = s.active && !other && Number.isInteger(s.episode) &&
    (collecting ? inEpisode || s.phase === "save" : !!s.options?.output);
  $("currentEpisodeState").textContent = current
    ? "#" + String(s.episode).padStart(4, "0") + " · " + tr(phases[s.phase])
    : tr(s.active && !other ? phases[s.phase] : phases.idle);
  document.dispatchEvent(new Event("episode-session"));
  const recordButton = inEpisode ? "finish" : "record";
  const pauseButton = s.phase === "recording" ? "record_pause" : "record_resume";
  const enterAction = {ready: "record", recording: "finish", record_paused: "finish", save: "save"}[s.phase];
  const displayed = collecting
    ? [recordButton, pauseButton, "home"]
    : [allowed.includes("pause") ? "pause" : "run", "stop", "reset"];
  for (const key of ["save", "discard"]) if (allowed.includes(key)) displayed.push(key);
  $("episodeNumber").textContent = s.active && !other && Number.isInteger(s.episode)
    ? (s.phase === "ready" && collecting ? tr("Next episode") : "Episode") + " " + String(s.episode).padStart(4, "0")
    : "Episode —";
  const sig = JSON.stringify([s.id, allowed, displayed, enterAction, language, busy]);
  if (sig !== lastButtons) {
    lastButtons = sig;
    $("buttons").replaceChildren();
    for (const key of displayed) {
      const enter = collecting && key === enterAction;
      const label = tr(names[key]) + (enter ? " (Enter)" : "");
      const button = el("button", "transport-key " + key);
      const isReset = key === "home" || key === "reset";
      button.innerHTML = transportIcon(key);
      if (isReset) {
        button.setAttribute("aria-keyshortcuts", "Control+Alt+R");
      }
      button.title = label + (isReset ? " (Ctrl + Alt + R)" : "");
      button.setAttribute("aria-label", button.title);
      if (enter) button.setAttribute("aria-keyshortcuts", "Enter");
      button.disabled = busy || !allowed.includes(key);
      if (enter) button.dataset.enter = "true";
      button.onclick = (event) => {
        if (event.detail) button.blur();
        act(key);
      };
      $("buttons").append(button);
    }
  }
  $("status").textContent = s.robot_exit
    ? tr("Robot shutdown unconfirmed. Check device logs.")
    : other
    ? tr("A session is running on the other page.")
    : s.active && s.simulation?.paused
      ? s.robot + " · " + tr({
          limit: "Simulation reached its step limit. Reset to continue.",
          terminated: "Simulation stopped or rejected an action. Reset to continue.",
          success: "Simulation paused after task success. Reset to continue.",
        }[s.simulation.reason])
    : s.active
      ? s.robot +
        (s.tele
          ? tr(" \u00b7 Teleoperation ") +
            {
              paused: tr("paused"),
              running: tr("enabled"),
              homing: tr("resetting"),
            }[s.tele]
          : "")
      : s.exit === null
        ? tr("Waiting for devices.")
        : tr("Last exit code ") + s.exit;
  const log = s.logs.join("\n") || tr("Waiting for a session");
  if ($("log").textContent !== log) {
    $("log").textContent = log;
    if ($("follow").checked) $("log").scrollTop = $("log").scrollHeight;
  }
}
// Count distinct preview frames actually drawn, using the browser clock.
function cameraFrameRate(item, now) {
  while (item.drawTimes.length && item.drawTimes[0] <= now - 1000)
    item.drawTimes.shift();
  return now - item.createdAt >= 1000 && item.timestamped ? item.drawTimes.length : "—";
}
function reviewingEpisode() {
  return !$("episodePlayer").hidden;
}
async function updateCamera(name, item) {
  if (item.loading) return;
  item.loading = true;
  const signal = item.controller.signal;
  try {
    while (!signal.aborted && !document.hidden && !reviewingEpisode()) {
      const query = new URLSearchParams({name});
      if (item.stamp) query.set("after", item.stamp);
      const response = await fetch("/api/image?" + query, {signal, cache: "no-store"});
      if (response.status === 204) continue;
      if (!response.ok) break;
      const blob = await response.blob();
      if (signal.aborted || document.hidden || reviewingEpisode()) break;
      if (item.context) {
        const frame = await createImageBitmap(blob);
        try {
          if (signal.aborted || document.hidden || reviewingEpisode()) break;
          const canvas = item.context.canvas;
          if (canvas.width !== frame.width) canvas.width = frame.width;
          if (canvas.height !== frame.height) canvas.height = frame.height;
          item.context.drawImage(frame, 0, 0);
        } finally {
          frame.close();
        }
      } else {
        const previous = item.url;
        item.url = URL.createObjectURL(blob);
        item.img.src = item.url;
        try {
          await item.img.decode();
        } finally {
          if (previous) URL.revokeObjectURL(previous);
        }
      }
      const stamp = response.headers.get("X-Frame-Stamp");
      if (stamp && stamp !== item.stamp) {
        const now = performance.now();
        cameraFrameRate(item, now);
        item.drawTimes.push(now);
        item.timestamped = true;
      }
      item.stamp = stamp;
      // Older servers return snapshots immediately. Bound their polling rate.
      if (!item.stamp) await new Promise(resolve => setTimeout(resolve, 80));
    }
  } catch (error) {
    if (!signal.aborted) item.label.textContent = name + tr(" · Data stale");
  } finally {
    item.loading = false;
  }
}

function cameras(p) {
  for (const [name, item] of feeds) {
    if (!p.images.includes(name)) {
      item.controller.abort();
      if (item.url) URL.revokeObjectURL(item.url);
      item.node.remove();
      feeds.delete(name);
    }
  }
  const empty = $("feeds").querySelector(".empty");
  if (!p.images.length) {
    if (empty) empty.textContent = tr("Waiting for camera images");
    else $("feeds").append(el("div", "empty camera", tr("Waiting for camera images")));
    return;
  }
  empty?.remove();
  for (const name of p.images) {
    let item = feeds.get(name);
    if (!item) {
      const node = el("div", "feed"), img = el("img"), label = el("span", "", name);
      img.alt = name;
      const canvas = el("canvas");
      const context = window.createImageBitmap && canvas.getContext("2d", {alpha: false, desynchronized: true});
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", name);
      node.append(context ? canvas : img, label);
      $("feeds").append(node);
      item = {node, img, context, label, url: "", stamp: null, controller: new AbortController(),
              createdAt: performance.now(), drawTimes: [], timestamped: false};
      feeds.set(name, item);
    }
    const age = p.age[name.split("/")[0]] ?? 0;
    const fps = cameraFrameRate(item, performance.now());
    item.label.textContent = name + " · " + fps + " FPS" + (age > 2 ? tr(" · Data stale") : "");
    updateCamera(name, item);
  }
}
function chart(canvas, rows, idx) {
  const w = (canvas.width = 240),
    h = (canvas.height = 90),
    c = canvas.getContext("2d"),
    end = rows.at(-1)[0];
  let lo = Infinity,
    hi = -Infinity;
  for (const r of rows) {
    lo = Math.min(lo, r[1][idx]);
    hi = Math.max(hi, r[1][idx]);
  }
  const span = Math.max(0.02, hi - lo),
    mid = (lo + hi) / 2;
  c.strokeStyle = "#d9e1d9";
  c.beginPath();
  c.moveTo(0, h / 2);
  c.lineTo(w, h / 2);
  c.stroke();
  c.strokeStyle = "#368564";
  c.lineWidth = 2.5;
  c.beginPath();
  let last = -Infinity;
  for (const [t, q] of rows) {
    const x = w * (1 + (t - end) / 30),
      y = h / 2 - ((q[idx] - mid) / span) * (h - 10);
    if (t - last > 0.65) c.moveTo(x, y);
    else c.lineTo(x, y);
    last = t;
  }
  c.stroke();
}
function axes(p, root = $("axes")) {
  root.replaceChildren();
  for (const [key, rows] of Object.entries(p.axes)) {
    if (!rows.length) continue;
    root.append(
      el(
        "small",
        "axisname",
        key + (p.now - rows.at(-1)[0] > 2 ? tr(" \u00b7 Data stale") : ""),
      ),
    );
    const grid = el("div", "chartgrid");
    root.append(grid);
    const q = rows.at(-1)[1];
    q.forEach((v, i) => {
      const cell = el("div", "chart"),
        label = el("div");
      label.append(
        el(
          "span",
          "",
          q.length === 7 && i === 6 ? tr("Gripper") : "J" + (i + 1),
        ),
        el("span", "", v.toFixed(3)),
      );
      const canvas = el("canvas");
      cell.append(label, canvas);
      grid.append(cell);
      chart(canvas, rows, i);
    });
  }
  if (!root.children.length)
    root.append(el("div", "empty", tr("Waiting for joint feedback")));
}
function drawModel() {
  const model = modelData[$("arm").value];
  if (!model) return;
  const canvas = $("model"),
    w = (canvas.width = Math.max(320, canvas.clientWidth * 2)),
    h = (canvas.height = 440),
    c = canvas.getContext("2d");
  const project = ([x, y, z]) => [
    x * Math.cos(yaw) - y * Math.sin(yaw),
    z * Math.cos(pitch) -
      (x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(pitch),
  ];
  const lines = model.links.map((pair) => pair.map(project)),
    controlAxes = (model.control_axes || []).map((pair) => pair.map(project)),
    pts = [...lines, ...controlAxes].flat();
  if (!pts.length) return;
  const xs = pts.map((p) => p[0]),
    ys = pts.map((p) => p[1]),
    x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  const scale = Math.min(
    (w - 70) / Math.max(0.35, x1 - x0),
    (h - 55) / Math.max(0.38, y1 - y0),
  );
  const xy = (p) => [
    w / 2 + (p[0] - (x0 + x1) / 2) * scale,
    h / 2 - (p[1] - (y0 + y1) / 2) * scale,
  ];
  c.lineCap = "round";
  for (const pair of lines) {
    const [a, b] = pair.map(xy);
    c.strokeStyle = "#36504b";
    c.lineWidth = 10;
    c.beginPath();
    c.moveTo(...a);
    c.lineTo(...b);
    c.stroke();
    c.fillStyle = "#91bc62";
    c.beginPath();
    c.arc(...b, 7, 0, Math.PI * 2);
    c.fill();
  }
  controlAxes.forEach((pair, axis) => {
    const [a, b] = pair.map(xy);
    c.strokeStyle = ["#e13030", "#23a63a", "#2869eb"][axis];
    c.lineWidth = 5;
    c.beginPath();
    c.moveTo(...a);
    c.lineTo(...b);
    c.stroke();
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    c.beginPath();
    c.moveTo(...b);
    c.lineTo(
      b[0] - 13 * Math.cos(angle - 0.4),
      b[1] - 13 * Math.sin(angle - 0.4),
    );
    c.moveTo(...b);
    c.lineTo(
      b[0] - 13 * Math.cos(angle + 0.4),
      b[1] - 13 * Math.sin(angle + 0.4),
    );
    c.stroke();
    c.fillStyle = c.strokeStyle;
    c.font = "bold 18px sans-serif";
    c.fillText("+" + "XYZ"[axis], b[0] + 5, b[1] - 5);
  });
  $("modelName").textContent =
    model.name +
    (model.rest
      ? tr(" \u00b7 Model zero pose (awaiting feedback)")
      : tr(" \u00b7 Measured joint pose")) +
    tr(" \u00b7 Drag to rotate");
}
$("model").onpointerdown = (e) => {
  drag = [e.clientX, e.clientY];
  e.target.setPointerCapture(e.pointerId);
};
$("model").onpointermove = (e) => {
  if (drag) {
    yaw += (e.clientX - drag[0]) * 0.012;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
    drawModel();
  }
};
$("model").onpointerup = $("model").onpointercancel = () => (drag = null);
$("arm").onchange = drawModel;
let polling = false;
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const previousSession = state?.session.id;
    state = await api("state?mode=" + mode);
    if (
      previousSession !== state.session.id ||
      state.session.tele !== "running"
    )
      heldKeys.clear();
    $("keyboardPanel").hidden = !state.session.keyboard_arms;
    renderKeyboard();
    $("link").textContent = tr("Service connected");
    $("link").className = "chip online";
    renderSession(state.session);
    if (reviewingEpisode()) return;
    const p = state.preview;
    axes(p);
    modelData = p.models;
    const arms = Object.keys(modelData);
    if (
      Array.from($("arm").options)
        .map((x) => x.value)
        .join() !== arms.join()
    )
      fill("arm", arms);
    $("modelPanel").hidden = !arms.length;
    drawModel();
    cameras(p);
    if (p.error && !p.images.length) $("status").textContent = p.error;
  } finally {
    polling = false;
  }
}

const heldKeys = new Set();
let keyboardSending = false;
let keyboardWasEnabled = false;
const keyboardOwner = [...crypto.getRandomValues(new Uint8Array(16))]
  .map((v) => v.toString(16).padStart(2, "0"))
  .join("");
function keyboardEnabled() {
  const element = document.activeElement;
  return (
    !reviewingEpisode() &&
    state?.session.mode === mode &&
    state.session.tele === "running" &&
    state.session.keyboard_arms &&
    document.hasFocus() &&
    !document.hidden &&
    !element?.closest("input,textarea,select,#episodePanel,[contenteditable=true]") &&
    !document.querySelector("dialog[open]")
  );
}
function mappedKeys() {
  const axes = state?.session.keyboard_axes || [0, 1, 2, 3, 4, 5, 6];
  return new Set([
    ...(state?.session.keyboard_bindings || []).flatMap((arm) => arm.filter((_, axis) => axes.includes(axis))).flat(),
    ...(state?.session.keyboard_base_bindings || []).flat(),
  ]);
}
window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (
    keyboardEnabled() &&
    mappedKeys().has(key) &&
    !event.ctrlKey &&
    !event.altKey &&
    !event.metaKey
  ) {
    event.preventDefault();
    heldKeys.add(key);
  }
});
window.addEventListener("keyup", (event) =>
  heldKeys.delete(event.key.toLowerCase()),
);
window.addEventListener("blur", () => heldKeys.clear());
document.addEventListener("visibilitychange", () => heldKeys.clear());
document.addEventListener("focusin", () => {
  if (!keyboardEnabled()) heldKeys.clear();
});
setInterval(async () => {
  if (keyboardSending || !state?.session.keyboard_arms) return;
  const enabled = keyboardEnabled();
  if (!enabled && !keyboardWasEnabled) return;
  keyboardWasEnabled = enabled;
  if (!enabled) heldKeys.clear();
  keyboardSending = true;
  try {
    await api("keyboard", {
      id: state.session.id,
      owner: keyboardOwner,
      keys: [...heldKeys],
    });
  } catch {
    heldKeys.clear();
  } finally {
    keyboardSending = false;
  }
}, 50);
function renderKeyboard() {
  const bindings = state?.session.keyboard_bindings || [];
  const baseBindings = state?.session.keyboard_base_bindings || [];
  const axes = state?.session.keyboard_axes || [0, 1, 2, 3, 4, 5, 6];
  const signature = JSON.stringify([bindings, baseBindings, axes, language]);
  if ($("keyboardBindings").dataset.signature === signature) return;
  $("keyboardBindings").dataset.signature = signature;
  $("keyboardBindings").replaceChildren();
  const keyName = (key) =>
    ({
      arrowup: "↑",
      arrowdown: "↓",
      arrowleft: "←",
      arrowright: "→",
      pageup: "PageUp",
      pagedown: "PageDown",
    })[key] || key.toUpperCase();
  bindings.forEach((pairs, index) => {
    const block = el("div", "keyboard-arm");
    block.append(
      el(
        "strong",
        "",
        tr(
          bindings.length === 2
            ? index === 0
              ? "Left arm"
              : "Right arm"
            : "Arm",
        ),
      ),
    );
    const header = el("div", "key-table-header");
    header.append(el("span", ""));
    ["X", "Y", "Z"].forEach((axis, i) => {
      const cell = el("div", "key-axis-heading axis-" + "xyz"[i]);
      cell.append(el("span", "axis-name", axis));
      const signs = el("span", "key-signs");
      signs.append(el("span", "", "+"), el("span", "", "−"));
      cell.append(signs);
      header.append(cell);
    });
    block.append(header);
    ["Translation", "Rotation"].forEach((label, group) => {
      if (![0, 1, 2].some((axis) => axes.includes(group * 3 + axis))) return;
      const row = el("div", "key-table-row " + (group ? "key-rotation" : "key-translation"));
      const rowLabel = el("span", "key-group-label");
      const icon = el("span", "key-motion-icon", group ? "↻" : "↔");
      icon.setAttribute("aria-hidden", "true");
      rowLabel.append(icon, el("span", "", tr(label)));
      row.append(rowLabel);
      pairs.slice(group * 3, group * 3 + 3).forEach((pair, axis) => {
        const keys = el("span", "axis-keys axis-" + "xyz"[axis]);
        const hint = tr(group ? "Rotate about axis" : "Move along axis") + " " + "XYZ"[axis];
        keys.title = hint;
        pair.forEach((key, direction) => {
          const cap = el("kbd", direction ? "key-negative" : "key-positive", keyName(key));
          cap.setAttribute("aria-label", `${hint} ${direction ? "−" : "+"}: ${keyName(key)}`);
          keys.append(cap);
        });
        row.append(keys);
      });
      block.append(row);
    });
    const gripper = el("div", "key-gripper-row");
    gripper.append(el("span", "key-group-label", tr("Gripper")));
    pairs.slice(6).forEach((pair) => pair.forEach((key, direction) => {
      const item = el("span", "key-gripper-item");
      item.append(el("span", "", tr(direction ? "Close" : "Open")), el("kbd", "", keyName(key)));
      gripper.append(item);
    }));
    block.append(gripper);
    $("keyboardBindings").append(block);
  });
  if (baseBindings.length) {
    const block = el("div", "keyboard-arm");
    block.append(el("strong", "", tr("Mobile base (robot frame)")));
    ["Forward / backward", "Left / right", "Turn left / right"].forEach((label, index) => {
      const row = el("div", "key-gripper-row");
      row.append(el("span", "key-group-label", tr(label)));
      baseBindings[index].forEach((key) => row.append(el("kbd", "", keyName(key))));
      block.append(row);
    });
    $("keyboardBindings").append(block);
  }
}
function applyLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title =
    "VLAStudio · " + tr(collecting ? "Data collection" : "Policy inference");
  $("language").value = language;
  document
    .querySelectorAll("[data-i18n]")
    .forEach((element) => (element.textContent = tr(element.dataset.i18n)));
  $("section").textContent = collecting ? "COLLECT_DATA" : "INFER";
  $("title").textContent = tr(
    collecting ? "Teleoperation collection" : "Policy inference",
  );
  $("rateLabel").textContent = tr(
    collecting ? "Collection rate" : "Publish rate",
  );
  $("outputLabel").textContent = tr(
    collecting ? "Dataset name" : "Recording directory (optional)",
  );
  if ($("teleop").options[0]?.value === "")
    $("teleop").options[0].textContent = tr("None · Observations only");
  if ($("sync").options[0]?.value === "")
    $("sync").options[0].textContent = tr("Native synchronization");
  syncProfilePickers();
  fillRobotTasks();
  renderTaskPanel();
  lastButtons = "";
  if (state) {
    renderSession(state.session);
    axes(state.preview);
    drawModel();
    renderKeyboard();
  }
}
$("language").onchange = () => {
  heldKeys.clear();
  language = $("language").value;
  localStorage.setItem("vla-language", language);
  applyLanguage();
};
$(collecting ? "navCollect" : "navInfer").classList.add("active");
$("collectFields").hidden = !collecting;
$("inferFields").hidden = collecting;
applyLanguage();
init().catch((e) => notice(e.message));
setInterval(() => {
  if (profiles && !document.hidden)
    refresh().catch(() => {
      $("link").textContent = tr("Connection lost");
      $("link").className = "chip";
    });
}, 450);
