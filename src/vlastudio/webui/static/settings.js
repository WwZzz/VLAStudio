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
  const picker=$("folderDialog");
  let folder=null, browseRevision=0, browseTarget="settingsData", creating=false;
  const createForm=$("folderCreateForm"), folderName=$("folderName");
  function folderControls() {
    $("folderNew").disabled=$("folderChoose").disabled=creating || !folder;
    $("folderUp").disabled=creating || !folder?.parent;
    $("folderHome").disabled=$("folderCancel").disabled=creating;
    for (const button of $("folderList").querySelectorAll("button")) button.disabled=creating;
    for (const control of createForm.elements) control.disabled=creating;
  }
  async function browse(path, initial=false) {
    const request=++browseRevision;
    folder=null;createForm.hidden=true;folderName.value="";folderControls();
    $("folderPath").textContent=path;
    $("folderList").replaceChildren();
    $("folderChoose").disabled=$("folderUp").disabled=true;
    $("folderStatus").textContent=tr("Loading directories…");
    try {
      const result=await api("settings/directories",{path,initial});
      if (!picker.open || request!==browseRevision) return;
      folder=result;
      $("folderPath").textContent=folder.path;
      $("folderUp").disabled=!folder.parent;
      $("folderChoose").disabled=false;
      $("folderStatus").textContent=folder.folders.length ? "" : tr("No subdirectories");
      const list=document.createDocumentFragment();
      for (const child of folder.folders) {
        const button=document.createElement("button");
        button.type="button";
        button.textContent=child.name;
        button.onclick=()=>browse(child.path);
        list.append(button);
      }
      $("folderList").append(list);
      folderControls();
      $("folderList").scrollTop=0;
      ($("folderList").firstElementChild || $("folderChoose")).focus({preventScroll:true});
    } catch (error) {
      if (picker.open && request===browseRevision) $("folderStatus").textContent=tr(error.message);
    }
  }
  $("folderNew").onclick=()=>{
    if (!folder || creating) return;
    createForm.hidden=false;folderName.value="";folderName.focus();
  };
  $("folderCreateCancel").onclick=()=>{createForm.hidden=true;$("folderNew").focus();};
  createForm.onsubmit=async event=>{
    event.preventDefault();
    if (!folder || creating || !folderName.value.trim()) return;
    const request=++browseRevision;
    creating=true;folderControls();$("folderStatus").textContent=tr("Creating folder…");
    try {
      const result=await api("settings/directories/create",{path:folder.path,name:folderName.value.trim()});
      if (!picker.open || request!==browseRevision) return;
      creating=false;
      await browse(result.path);
    } catch(error) {
      if (picker.open && request===browseRevision) $("folderStatus").textContent=tr(error.message);
    } finally {creating=false;folderControls();}
  };
  picker.addEventListener("cancel",event=>{if (creating) event.preventDefault();});
  $("settingsBrowse").onclick=()=>{
    browseTarget="settingsData";
    picker.showModal();
    browse($("settingsData").value.trim() || settings.data_dir,true);
  };
  $("settingsCheckpointBrowse").onclick=()=>{
    browseTarget="settingsCheckpoint";
    picker.showModal();
    browse($(browseTarget).value.trim() || settings.checkpoint_dir,true);
  };
  $("folderUp").onclick=()=>{if (folder?.parent) browse(folder.parent);};
  $("folderHome").onclick=()=>browse("");
  $("folderCancel").onclick=()=>picker.close();
  $("folderChoose").onclick=()=>{
    if (!folder) return;
    $(browseTarget).value=folder.path;
    picker.close();
  };
  picker.addEventListener("close",()=>{browseRevision++;folder=null;createForm.hidden=true;folderName.value="";folderControls();});
  function controls() {
    $("settingsBrowse").disabled=$("settingsCheckpointBrowse").disabled=pending || !settings;
    $("settingsSave").disabled=pending || !settings;
    for (const id of ["settingsData","settingsCheckpoint","settingsConfigs","settingsUsername","settingsToken","settingsClearToken"])
      $(id).disabled=pending;
  }
  function failure(message) {
    $("settingsError").textContent=tr(message);
    $("settingsError").hidden=!message;
  }
  function labels() {
    $("folderUp").title=$("folderUp").ariaLabel=tr("Go to parent directory");
    rootLabels();
    $("settingsOpen").title=$("settingsOpen").ariaLabel=tr("Settings");
    if (settings) $("settingsTokenState").textContent=tr(settings.token_set ?
      "Token configured. Leave blank to keep it." : "Enter a token to configure Hugging Face access.");
  }
  $("settingsOpen").onclick=async()=>{
    document.dispatchEvent(new Event("settingsopening"));
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
      $("settingsCheckpoint").value=result.checkpoint_dir;
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
      const payload={version:settings.version,data_dir:$("settingsData").value,checkpoint_dir:$("settingsCheckpoint").value,
        config_roots:[...roots.querySelectorAll("input")].map(input=>input.value.trim()).filter(Boolean),hf_username:$("settingsUsername").value,
        clear_token:$("settingsClearToken").checked};
      if (token.value) payload.hf_token=token.value.trim();
      settings=await api("settings",payload);
      token.value="";
      document.dispatchEvent(new Event("settingschanged"));
      dialog.close();
    } catch(e) {failure(e.message);}
    finally {pending=false;controls();}
  };
  dialog.addEventListener("cancel",event=>{if(pending)event.preventDefault();});
  $("language").addEventListener("change",labels);
  labels();
})();
