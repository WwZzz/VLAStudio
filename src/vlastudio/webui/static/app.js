"use strict";
const $ = (id) => document.getElementById(id);
const mode = location.pathname === "/infer" ? "infer" : "collect";
const collecting = mode === "collect";
const names = {
  record: "Start recording",
  finish: "Finish episode",
  save: "Save",
  discard: "Discard episode",
  run: "Start inference",
  pause: "Pause inference",
  reset: tr("Reset"),
  home: tr("Reset"),
  tele_run: "Enable teleoperation",
  tele_pause: "Pause teleoperation",
  stop: "Disconnect",
};
const phases = {
  idle: "Not connected",
  loading: "Loading devices",
  ready: "Ready to record",
  recording: "Recording",
  save: "Awaiting save",
  infer_ready: "Policy ready",
  running: "Inference running",
  paused: "Inference paused",
  homing: "Resetting",
  busy: "Processing",
  stopping: "Disconnecting",
};
let connecting = false;
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
    on_success: $("on_success").value,
  };
}
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
  if (!collecting && r === "dummy")
    $("policy").value = "__dummy-zero14-chunk16";
  else if (!collecting && $("policy").value.startsWith("__dummy"))
    $("policy").value = "";
}
async function init() {
  profiles = await api("catalog");
  fill("robot", profiles.robot);
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
  ])
    if (id in saved) $(id).value = saved[id];
  const settings = profiles.robot_options?.[$("robot").value];
  $("simulationFields").hidden = !settings?.tasks;
  fill("task", settings?.tasks || []);
  if (saved.task && settings?.tasks?.includes(saved.task))
    $("task").value = saved.task;
  $("on_success").value = saved.on_success || "continue";
  await refresh();
}
$("robot").onchange = defaults;
$("teleop").onchange = () => {
  if ($("teleop").value !== "quest3_rebot_b601rs_rel_ee") $("sync").value = "";
};
$("form").onsubmit = async (e) => {
  e.preventDefault();
  if (busy) return;
  if (state?.session.active) {
    await act("stop");
    return;
  }
  const opts = options();
  if (
    !(await confirmAction(
      tr("Connect ") + opts.robot,
      tr(
        "This starts the selected devices. Real robots may enable or move during connection.",
      ),
    ))
  )
    return;
  connecting = true;
  renderSession(state.session);
  try {
    await post("start", opts);
    localStorage.setItem("vla-" + mode, JSON.stringify(opts));
  } finally {
    connecting = false;
    renderSession(state.session);
  }
};
$("observe").onclick = () =>
  post("preview", {
    robot: state?.preview.profile === $("robot").value ? "" : $("robot").value,
  });
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
  const simulation = !!profiles.robot_options?.[state.session.robot]?.tasks;
  if (["reset", "home", "discard", "stop"].includes(action)) {
    const msg = {
      reset: tr(
        "Pause inference, reset the policy and robot. The current unsaved inference recording follows the native discard behavior.",
      ),
      home: simulation
        ? tr(
            "Pause teleoperation and reset the robot and simulation scene. Any current episode will finish for you to save or discard.",
          )
        : tr(
            "Pause teleoperation and reset the robot. Any current episode will finish for you to save or discard.",
          ),
      discard: tr("Discard this episode?"),
      stop: tr(
        "Disconnect the devices after completing their shutdown procedure. Save any recordings you need first.",
      ),
    };
    if (!(await confirmAction(tr(names[action]), msg[action]))) return;
  }
  await post("action", { id: sid, action });
}
function renderSession(s) {
  const other = s.active && s.mode !== mode;
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
      $(key).value = value;
  }
  $("phase").textContent =
    (other
      ? tr(s.mode === "collect" ? "Data collection" : "Policy inference") +
        " · "
      : "") + tr(phases[s.phase]);
  for (const field of $("form").elements)
    field.disabled = s.active || connecting;
  const loading = connecting || (s.active && s.phase === "loading");
  $("launch").disabled = busy || other || loading || s.phase === "stopping";
  $("launch").setAttribute(
    "aria-busy",
    String(loading || s.phase === "stopping"),
  );
  $("connectionProgress").hidden = !(loading || s.phase === "stopping");
  $("connectionLabel").textContent = tr(
    loading
      ? "Loading devices"
      : s.phase === "stopping"
        ? "Disconnecting"
        : s.active
          ? "Disconnect"
          : "Connect devices",
  );
  $("observe").disabled = s.active;
  $("observe").textContent =
    state.preview.profile === $("robot").value
      ? tr("Close read-only preview")
      : tr("Read-only preview");
  const allowed = other ? [] : s.actions.filter((action) => action !== "stop");
  const resetAction = collecting ? "home" : "reset";
  const displayed = allowed.includes(resetAction)
    ? allowed
    : [...allowed, resetAction];
  const sig = JSON.stringify([s.id, allowed, s.phase, language, busy]);
  if (sig !== lastButtons) {
    lastButtons = sig;
    $("buttons").replaceChildren();
    for (const key of displayed) {
      const b = el(
        "button",
        "",
        tr(
          key === "run" && s.phase === "paused"
            ? "Resume inference"
            : names[key],
        ),
      );
      b.onclick = () => act(key);
      b.disabled = busy || !allowed.includes(key);
      if (b.disabled)
        b.title = tr(
          s.active
            ? "Please wait for the device to become ready"
            : "Available after connecting a device that supports reset",
        );
      $("buttons").append(b);
    }
  }
  $("status").textContent = other
    ? tr("A session is running on the other page.")
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
async function cameras(p) {
  for (const [name, item] of feeds)
    if (!p.images.includes(name)) {
      URL.revokeObjectURL(item.url);
      item.node.remove();
      feeds.delete(name);
    }
  let empty = $("feeds").querySelector(".empty");
  if (!p.images.length) {
    if (!empty)
      $("feeds").append(
        el("div", "empty camera", tr("Waiting for camera images")),
      );
    else empty.textContent = tr("Waiting for camera images");
    return;
  }
  empty?.remove();
  await Promise.all(
    p.images.map(async (name) => {
      let item = feeds.get(name);
      if (!item) {
        const node = el("div", "feed"),
          img = el("img"),
          label = el("span", "", name);
        img.alt = name;
        node.append(img, label);
        $("feeds").append(node);
        item = { node, img, label, url: "" };
        feeds.set(name, item);
      }
      const age = p.age[name.split("/")[0]] ?? 0;
      item.label.textContent = name + (age > 2 ? tr(" \u00b7 Data stale") : "");
      const res = await fetch("/api/image?name=" + encodeURIComponent(name));
      if (!res.ok) return;
      const url = URL.createObjectURL(await res.blob());
      item.img.src = url;
      if (item.url) URL.revokeObjectURL(item.url);
      item.url = url;
    }),
  );
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
function axes(p) {
  const root = $("axes");
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
    pts = lines.flat();
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
    state = await api("state");
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
    await cameras(p);
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
    state?.session.mode === mode &&
    state.session.tele === "running" &&
    state.session.keyboard_arms &&
    document.hasFocus() &&
    !document.hidden &&
    !element?.closest("input,textarea,select,[contenteditable=true]") &&
    !document.querySelector("dialog[open]")
  );
}
function mappedKeys() {
  return new Set((state?.session.keyboard_bindings || []).flat(2));
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
  const signature = JSON.stringify([bindings, language]);
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
    ["Translation X / Y / Z", "Rotation X / Y / Z", "Gripper"].forEach(
      (label, group) => {
        const selected =
          group === 2 ? pairs.slice(6) : pairs.slice(group * 3, group * 3 + 3);
        block.append(
          el(
            "p",
            "",
            tr(label) +
              "  " +
              selected.map((pair) => pair.map(keyName).join(" / ")).join(" · "),
          ),
        );
      },
    );
    $("keyboardBindings").append(block);
  });
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
