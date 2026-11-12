"use strict";
(() => {
  const fields = {robot:"robot", teleop:"teleop", sync:"recording", manager:"action_manager"};
  const dialog = $("configDialog"), name = $("configName"), content = $("configContent");
  const devices = new DeviceConfigEditor($("configDevices"));
  $("configAddDevice").onclick = () => devices.add();
  let item, field, original, saving = false;
  const buttons = new Map();
  function update() {
    for (const [id, button] of buttons) {
      button.disabled = !$(id).value;
      button.title = tr("View and edit selected configuration");
      button.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m4 16-1 5 5-1L20 8l-4-4L4 16Z"/><path d="m14 6 4 4"/></svg>';
      button.setAttribute("aria-label", tr("Edit configuration") + " · " +
        (id === "robot" ? tr("Robot configuration") : id === "teleop" ? tr("Teleoperation device") :
         id === "sync" ? tr("Recording synchronization") : tr("Action manager")));
    }
  }
  for (const [id, kind] of Object.entries(fields)) {
    const select = $(id);
    const parent = select.closest(".profile-picker") || select.closest("label");
    const label = parent.querySelector("label") || parent;
    const line = el("span", "config-label");
    if (label === parent && parent.tagName === "LABEL") {
      line.append(label.querySelector("span"));
      label.prepend(line);
    } else {
      // Keep the input's accessible label intact.
      const title = label.cloneNode(true);
      label.replaceWith(line);
      line.append(title);
    }
    const button = el("button", "config-edit", tr("Edit"));
    button.type = "button";
    button.onclick = () => open(id, kind);
    line.append(button); buttons.set(id, button);
    select.addEventListener("change", update);
  }
  function error(message) {
    $("configError").textContent = tr(message);
    $("configError").hidden = !message;
  }
  async function open(id, kind) {
    field = id; original = $(id).value;
    if (!original) return;
    heldKeys.clear();
    try {
      const data = await api("config?" + new URLSearchParams({kind, name:original}));
      item = data;
      name.value = item.name; content.value = item.content;
      const structured = kind !== "recording";
      $("configDevices").hidden = $("configAddDevice").hidden = !structured;
      $("configYaml").hidden = structured;
      content.required = !structured;
      if (structured) devices.load(item.document, item.device_types);
      $("configLocation").textContent = "configs/" + kind + "/" + item.name + ".yaml";
      $("configUsage").textContent = tr(item.in_use ?
        "This configuration is connected. Save a copy now, or disconnect before overwriting. Changes apply on the next connection." :
        "Edit the parameters below. Changes apply on the next connection.");
      error(""); saving = false; controls(); dialog.showModal();
    } catch (e) { notice(e.message); }
  }
  function controls() {
    const target = name.value.trim();
    $("configSave").disabled = saving || !target || (item?.in_use && target === item.name);
    $("configCopy").disabled = saving;
    $("configClose").disabled = saving;
    name.disabled = content.disabled = saving;
    $("configAddDevice").disabled = saving;
    $("configDevices").inert = saving;
    $("configSave").textContent = tr(saving ? "Saving" : "Save configuration");
  }
  name.oninput = controls;
  async function save() {
    if (saving || !item) return;
    saving = true; controls(); error("");
    try {
      const payload = {kind:item.kind, name:item.name, version:item.version, target:name.value.trim(),
        content:item.kind === "recording" ? content.value : devices.value()};
      const saved = await api("config", payload);
      const previous = options();
      profiles = await api("catalog");
      fill("robot", profiles.robot.filter(n => (profiles.robot_options?.[n]?.modes || ["collect","infer"]).includes(mode)));
      fill("teleop", profiles.teleop, tr("None · Observations only"));
      fill("sync", profiles.recording, tr("Native synchronization"));
      fill("manager", profiles.action_manager);
      for (const id of Object.keys(fields)) $(id).value = previous[id] || "";
      if (!$(field).disabled && previous[field] === original) {
        $(field).value = saved.name;
        if (saved.name !== original) $(field).dispatchEvent(new Event("change", {bubbles:true}));
      }
      syncProfilePickers(); update();
      localStorage.setItem("vla-" + mode, JSON.stringify(options()));
      dialog.close();
    } catch (e) { error(e.message); }
    finally { saving = false; controls(); }
  }
  $("configForm").onsubmit = event => {event.preventDefault(); save();};
  $("configCopy").onclick = () => {
    if (name.value.trim() === item.name) {
      name.value = item.name.slice(0, 72) + "_custom";
      controls(); name.focus(); name.select();
      return;
    }
    save();
  };
  $("configClose").onclick = () => dialog.close();
  dialog.addEventListener("cancel", event => {if (saving) event.preventDefault();});
  document.addEventListener("episode-session", update);
  $("language").addEventListener("change", update);
  update();
})();
