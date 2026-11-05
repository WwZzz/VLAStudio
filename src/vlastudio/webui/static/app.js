"use strict";
const $ = (id) => document.getElementById(id);
const mode = location.pathname === "/infer" ? "infer" : "collect";
const collecting = mode === "collect";
const names = {
  record: "开始录制",
  finish: "结束本段",
  save: "保存",
  discard: "放弃本段",
  run: "开始推理",
  pause: "暂停推理",
  reset: "复位策略并回零",
  tele_run: "启用手柄控制",
  tele_pause: "暂停手柄控制",
  home: "机械臂回零",
  stop: "退出会话",
};
const phases = {
  idle: "未启动",
  loading: "加载中",
  ready: "等待录制",
  recording: "录制中",
  save: "等待保存",
  infer_ready: "策略就绪",
  running: "推理中",
  paused: "推理已暂停",
  homing: "正在复位",
  busy: "正在处理",
  stopping: "正在退出",
};
let secret = sessionStorage.getItem("vla-key") || "",
  state = null,
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
  $("error").textContent = message;
  $("error").hidden = false;
  clearTimeout(notice.timer);
  notice.timer = setTimeout(() => ($("error").hidden = true), 6500);
}
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: "Bearer " + secret,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  if (r.status === 401) {
    if (!$("auth").open) $("auth").showModal();
    throw Error("需要有效的访问密钥");
  }
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
  if (!collecting && r === "dummy")
    $("policy").value = "__dummy-zero14-chunk16";
  else if (!collecting && $("policy").value.startsWith("__dummy"))
    $("policy").value = "";
}
async function init() {
  profiles = await api("catalog");
  fill("robot", profiles.robot);
  fill("teleop", profiles.teleop, "无 · 仅记录观测");
  fill("sync", profiles.recording, "原生同步");
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
  await refresh();
}
$("login").onsubmit = async (e) => {
  e.preventDefault();
  secret = $("key").value.trim();
  try {
    await init();
    sessionStorage.setItem("vla-key", secret);
    $("auth").close();
  } catch (err) {
    $("loginError").textContent = err.message;
  }
};
$("robot").onchange = defaults;
$("teleop").onchange = () => {
  if ($("teleop").value !== "quest3_rebot_b601rs_rel_ee") $("sync").value = "";
};
$("form").onsubmit = async (e) => {
  e.preventDefault();
  if (busy) return;
  const opts = options();
  if (
    !(await confirmAction(
      "加载 " + opts.robot,
      "此操作启动设备；真实机械臂可能使能或执行其连接动作。确认现场就绪后继续。",
    ))
  )
    return;
  await post("start", opts);
  localStorage.setItem("vla-" + mode, JSON.stringify(opts));
};
$("observe").onclick = () =>
  post("preview", {
    robot: state?.preview.profile === $("robot").value ? "" : $("robot").value,
  });
async function post(route, data) {
  if (busy) return;
  busy = true;
  try {
    await api(route, data);
    await refresh();
  } catch (e) {
    notice(e.message);
  } finally {
    busy = false;
  }
}
async function act(action) {
  const sid = state.session.id;
  if (["reset", "home", "discard", "stop"].includes(action)) {
    const msg = {
      reset:
        "将重置策略并请求机器人回零；推理录像有未保存数据时会按原生流程丢弃。",
      home: "将请求机器人执行其回零流程。",
      discard: "这段录制将被丢弃。",
      stop: "退出运行进程。请先保存需要的数据，并等设备完成退出。",
    };
    if (!(await confirmAction(names[action], msg[action]))) return;
  }
  await post("action", { id: sid, action });
}
function renderSession(s) {
  const other = s.active && s.mode !== mode;
  if (s.active && !other)
    for (const [key, value] of Object.entries(s.options || {}))
      $(key).value = value;
  $("phase").textContent =
    (other ? (s.mode === "collect" ? "采集 · " : "推理 · ") : "") +
    phases[s.phase];
  for (const field of $("form").elements) field.disabled = s.active;
  $("observe").disabled = s.active;
  $("observe").textContent =
    state.preview.profile === $("robot").value ? "关闭只读预览" : "只读预览";
  const allowed = other ? [] : s.actions;
  const sig = JSON.stringify([s.id, allowed, s.phase]);
  if (sig !== lastButtons) {
    lastButtons = sig;
    $("buttons").replaceChildren();
    for (const key of allowed) {
      const b = el(
        "button",
        "",
        key === "run" && s.phase === "paused" ? "恢复推理" : names[key],
      );
      b.onclick = () => act(key);
      $("buttons").append(b);
    }
  }
  $("status").textContent = other
    ? "另一页面的会话正在运行。"
    : s.active
      ? s.robot +
        (s.tele
          ? " · 手柄" +
            { paused: "已暂停", running: "已启用", homing: "正在回零" }[s.tele]
          : "")
      : s.exit === null
        ? "等待加载设备。"
        : "上次退出码 " + s.exit;
  const log = s.logs.join("\n") || "等待启动会话";
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
    if (!empty) $("feeds").append(el("div", "empty camera", "等待相机画面"));
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
      item.label.textContent = name + (age > 2 ? " · 数据已停更" : "");
      const res = await fetch("/api/image?name=" + encodeURIComponent(name), {
        headers: { Authorization: "Bearer " + secret },
      });
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
        key + (p.now - rows.at(-1)[0] > 2 ? " · 数据已停更" : ""),
      ),
    );
    const grid = el("div", "chartgrid");
    root.append(grid);
    const q = rows.at(-1)[1];
    q.forEach((v, i) => {
      const cell = el("div", "chart"),
        label = el("div");
      label.append(
        el("span", "", q.length === 7 && i === 6 ? "夹爪" : "J" + (i + 1)),
        el("span", "", v.toFixed(3)),
      );
      const canvas = el("canvas");
      cell.append(label, canvas);
      grid.append(cell);
      chart(canvas, rows, i);
    });
  }
  if (!root.children.length)
    root.append(el("div", "empty", "等待读取关节反馈"));
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
    (model.rest ? " · 模型零位（等待反馈）" : " · 实测关节姿态") +
    " · 拖动旋转";
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
    state = await api("state");
    $("link").textContent = "Jetson 在线";
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
$("section").textContent = collecting ? "COLLECT_DATA" : "INFER";
$("title").textContent = collecting ? "遥操作采集" : "策略推理";
$(collecting ? "navCollect" : "navInfer").classList.add("active");
$("collectFields").hidden = !collecting;
$("inferFields").hidden = collecting;
if (!collecting) {
  $("rateLabel").textContent = "动作发布频率";
  $("outLabel").firstChild.textContent = "录像目录名（可留空）";
}
if (secret) init().catch((e) => notice(e.message));
else $("auth").showModal();
setInterval(() => {
  if (profiles && !document.hidden && !$("auth").open)
    refresh().catch(() => {
      $("link").textContent = "连接中断";
      $("link").className = "chip";
    });
}, 450);
