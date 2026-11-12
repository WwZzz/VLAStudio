"use strict";
(() => {
  const dialog=$("settingsDialog"), token=$("settingsToken");
  let settings, pending=false, revision=0;
  const roots=$("settingsRootList");
  function rootLabels() {
    const rows=[...roots.children];
    rows.forEach((row,index)=>{
      row.querySelector("input").ariaLabel=tr("Configuration directory")+" "+(index+1);
      for (const button of row.querySelectorAll("button")) {
        const action=button.dataset.action;
        button.title=button.ariaLabel=tr(action);
        button.disabled=(action==="Move up" && index===0) || (action==="Move down" && index===rows.length-1);
      }
    });
  }
  function addRoot(value) {
    const row=document.createElement("div"), input=document.createElement("input");
    input.value=value;input.spellcheck=false;input.autocomplete="off";
    row.append(input);
    for (const [action,symbol] of [["Move up","↑"],["Move down","↓"],["Remove directory","×"]]) {
      const button=document.createElement("button");
      button.type="button";button.dataset.action=action;button.textContent=symbol;
      button.onclick=()=>{
        if (action==="Remove directory") {
          const next=row.nextElementSibling || row.previousElementSibling;
          row.remove();
          (next?.querySelector("input") || $("settingsAddRoot")).focus();
        } else {
          if (action==="Move up") roots.insertBefore(row,row.previousElementSibling);
          else roots.insertBefore(row.nextElementSibling,row);
          input.focus();
        }
        rootLabels();
      };
      row.append(button);
    }
    roots.append(row);
    return input;
  }
  $("settingsAddRoot").onclick=()=>{const input=addRoot("");rootLabels();input.focus();};
  function controls() {
    $("settingsSave").disabled=pending || !settings;
    for (const id of ["settingsData","settingsConfigs","settingsUsername","settingsToken","settingsClearToken"])
      $(id).disabled=pending;
  }
  function failure(message) {
    $("settingsError").textContent=tr(message);
    $("settingsError").hidden=!message;
  }
  function labels() {
    rootLabels();
    $("settingsOpen").title=$("settingsOpen").ariaLabel=tr("Settings");
    if (settings) $("settingsTokenState").textContent=tr(settings.token_set ?
      "Token configured. Leave blank to keep it." : "Enter a token to configure Hugging Face access.");
  }
  $("settingsOpen").onclick=async()=>{
    heldKeys.clear();
    const request=++revision;
    roots.replaceChildren();
    settings=null;pending=true;token.value="";failure("");controls();
    $("settingsTokenState").textContent=tr("Loading settings");
    dialog.showModal();
    try {
      const result=await api("settings");
      if (revision!==request || !dialog.open) return;
      settings=result;
      $("settingsData").value=result.data_dir;
      for (const path of result.config_roots) addRoot(path);
      $("settingsUsername").value=result.hf_username;
      $("settingsClearToken").checked=false;
      labels();
    } catch(e) {failure(e.message);}
    finally {if (revision===request) {pending=false;controls();}}
  };
  $("settingsClose").onclick=()=>dialog.close();
  dialog.addEventListener("close",()=>{revision++;token.value="";});
  $("settingsForm").onsubmit=async event=>{
    event.preventDefault();
    if (pending || !settings) return;
    pending=true;controls();failure("");
    try {
      const payload={version:settings.version,data_dir:$("settingsData").value,
        config_roots:[...roots.querySelectorAll("input")].map(input=>input.value.trim()).filter(Boolean),hf_username:$("settingsUsername").value,
        clear_token:$("settingsClearToken").checked};
      if (token.value) payload.hf_token=token.value.trim();
      settings=await api("settings",payload);
      token.value="";
      const previous=options();
      profiles=await api("catalog");
      fill("robot",profiles.robot.filter(n=>(profiles.robot_options?.[n]?.modes||["collect","infer"]).includes(mode)));
      fill("teleop",profiles.teleop,tr("None · Observations only"));
      fill("sync",profiles.recording,tr("Native synchronization"));
      fill("manager",profiles.action_manager);
      for (const id of ["robot","teleop","sync","manager"]) {
        const value=previous[id] || "";
        if ([...$(id).options].some(option=>option.value===value)) $(id).value=value;
      }
      syncProfilePickers();renderTaskPanel();
      localStorage.setItem("vla-"+mode,JSON.stringify(options()));
      document.dispatchEvent(new Event("settingschanged"));
      dialog.close();await refresh();
    } catch(e) {failure(e.message);}
    finally {pending=false;controls();}
  };
  dialog.addEventListener("cancel",event=>{if(pending)event.preventDefault();});
  $("language").addEventListener("change",labels);
  labels();
})();
