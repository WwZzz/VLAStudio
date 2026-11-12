"use strict";
(() => {
  const fields = {robot:"robot", teleop:"teleop", sync:"recording", manager:"action_manager"};
  const dialog = $("configDialog"), name = $("configName"), content = $("configContent");
  const devices = new DeviceConfigEditor($("configDevices"));
  $("configAddDevice").onclick = () => {
    const card = devices.addCopy({type:"", name:"device", args:{}});
    card.querySelector('input[aria-label="type"]').focus();
  };
  let library = [];
  function renderLibrary() {
    $("deviceLibrarySearch").placeholder = tr("Search devices");
    $("deviceLibrarySearch").setAttribute("aria-label", tr("Search devices"));
    const query = $("deviceLibrarySearch").value.trim().toLowerCase();
    const tree = $("deviceLibraryTree"); tree.replaceChildren();
    const filtered = library.filter(row =>
      [row.category,row.type,row.name,...row.sources.map(s=>(s.root || "")+"/"+s.config)].join(" ").toLowerCase().includes(query));
    $("deviceLibraryStatus").textContent = filtered.length ? "" : tr("No matching devices");
    for (const category of ["Bench","Robots","Cameras","Teleoperation","Action managers","Other devices"]) {
      const rows = filtered.filter(row=>row.category===category);
      if (!rows.length) continue;
      const group = el("details","device-library-group"), title = el("summary","",tr(category) + " · " + rows.length);
      group.open = !!query; group.append(title);
      for (const type of [...new Set(rows.map(row=>row.type))]) {
        const branch = el("details","device-library-type");
        const typeTitle = el("summary","",type.split(".").at(-1));typeTitle.title=type;branch.open=!!query;branch.append(typeTitle);
        for (const row of rows.filter(r=>r.type===type)) {
          const entry = el("div","device-library-entry"), info=el("div","");
          const preview=el("details",""), previewTitle=el("summary","",String(row.name));
          preview.append(previewTitle,el("pre","",JSON.stringify(row.device,null,2)));
          info.append(preview,el("small","",row.sources.map(s=>(s.root ? s.root+"/" : "")+s.kind+"/"+s.config).join(", ")));
          const add=el("button","",tr("Add"));add.type="button";
          add.onclick=()=>{
            const copy=JSON.parse(JSON.stringify(row.device));
            const card=devices.addCopy(copy);
            $("deviceLibrary").hidden=true;
            card.scrollIntoView({block:"nearest"});
          };
          entry.append(info,add);branch.append(entry);
        }
        group.append(branch);
      }
      tree.append(group);
    }
  }
  $("configImportDevice").onclick=async()=>{
    $("deviceLibrary").hidden=false;
    $("deviceLibraryStatus").textContent=tr("Loading devices");
    $("deviceLibraryTree").replaceChildren();
    try {library=await api("device-library");renderLibrary();$("deviceLibrarySearch").focus();}
    catch(e) {$("deviceLibraryStatus").textContent=e.message;}
  };
  $("deviceLibrarySearch").oninput=renderLibrary;
  $("deviceLibraryClose").onclick=()=>{$("deviceLibrary").hidden=true;};
  let item, field, original, saving = false, loading = false, revision = 0;
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
    const token = ++revision;
    item = null; loading = true; saving = false;
    name.value = original; content.value = "";
    $("configLocation").textContent = "configs/" + kind + "/" + original + ".yaml";
    $("configUsage").textContent = tr("Loading configuration");
    $("configDevices").hidden = $("configDeviceActions").hidden = $("configYaml").hidden = true;
    $("deviceLibrary").hidden = true;
    $("configLoading").hidden = false;
    error(""); controls(); dialog.showModal();
    try {
      const data = await api("config?" + new URLSearchParams({kind, name:original}));
      if (revision !== token || !dialog.open) return;
      item = data;
      $("configLocation").textContent = item.path;
      name.value = item.name; content.value = item.content;
      const structured = kind !== "recording";
      $("configDevices").hidden = $("configDeviceActions").hidden = !structured;
      $("deviceLibrarySearch").value = "";
      $("configYaml").hidden = structured;
      content.required = !structured;
      if (structured) {
        const rows = Array.isArray(item.document) ? item.document : [item.document];
        devices.load(item.document, [...new Set(rows.map(row=>row.type))]);
        // Type suggestions are optional and load independently of the selected profile.
        api("device-types").then(types=>{
          if (revision === token && dialog.open) devices.updateTypes(types);
        }).catch(()=>{});
      }
      $("configUsage").textContent = tr(item.in_use ?
        "This configuration is connected. Save a copy now, or disconnect before overwriting. Changes apply on the next connection." :
        "Edit the parameters below. Changes apply on the next connection.");
    } catch (e) {
      if (revision === token && dialog.open) error(e.message);
    } finally {
      if (revision === token) {
        loading = false;
        $("configLoading").hidden = true;
        controls();
      }
    }
  }
  function controls() {
    const target = name.value.trim();
    $("configSave").disabled = loading || !item || saving || !target || (item?.in_use && target === item.name);
    $("configCopy").disabled = loading || !item || saving;
    $("configClose").disabled = saving;
    name.disabled = content.disabled = loading || saving;
    $("configAddDevice").disabled = $("configImportDevice").disabled = saving;
    $("deviceLibrary").inert = saving;
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
  dialog.addEventListener("close", () => {revision++; loading = false;});
  dialog.addEventListener("cancel", event => {if (saving) event.preventDefault();});
  document.addEventListener("episode-session", update);
  $("language").addEventListener("change", update);
  update();
})();
