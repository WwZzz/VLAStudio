"use strict";
(() => {
  Object.assign(chinese, {
    "Saved changes apply to new training requests. Submitted runs keep their configuration snapshots.":"保存的修改用于后续训练请求，已提交任务保留原配置快照。",
    "Configuration must be a nonempty mapping":"配置必须是非空的参数映射",
    "Policy configuration needs a type":"策略配置需要填写 type",
    "Policy args must be a mapping":"策略 args 必须是参数映射",
    "Dataset configuration needs a nonempty datasets or vqa list":"数据集配置需要非空的 datasets 或 vqa 列表",
  });
  const dialog=$("configDialog"), name=$("configName"), content=$("configContent");
  const fields={policy:"Policy",task:"Dataset configuration",training:"Training configuration"};
  let item=null, field=null, revision=0, loading=false, saving=false;
  const pencil='<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m4 16-1 5 5-1L20 8l-4-4L4 16Z"/><path d="m14 6 4 4"/></svg>';
  function labels(){
    for(const [kind,title] of Object.entries(fields)){
      const button=$("edit-"+kind);button.disabled=!$(kind).value;
      button.title=tr("View and edit selected configuration");
      button.ariaLabel=tr("Edit configuration")+" · "+tr(title);button.innerHTML=pencil;
    }
    $("configLoading").ariaLabel=tr("Loading configuration");
  }
  function failure(message){$("configError").textContent=tr(message);$("configError").hidden=!message;}
  function controls(){
    name.disabled=content.disabled=loading||saving;
    $("configSave").disabled=loading||saving||!item||!name.value.trim();
    $("configCopy").disabled=loading||saving||!item;
    $("configClose").disabled=saving;
    $("configSave").textContent=tr(saving?"Saving":"Save configuration");
    $("configLoading").hidden=!loading;
  }
  async function open(kind){
    const selected=$(kind).value;if(!selected)return;
    const token=++revision;field=kind;item=null;loading=true;name.value=selected;content.value="";failure("");
    $("configLocation").textContent="configs/"+kind+"/"+selected+".yaml";
    controls();dialog.showModal();
    try{
      const data=await api("config?"+new URLSearchParams({kind,name:selected}));
      if(token!==revision||!dialog.open)return;
      item=data;name.value=data.name;content.value=data.content;$("configLocation").textContent=data.path;
    }catch(e){if(token===revision&&dialog.open)failure(e.message);}
    finally{if(token===revision){loading=false;controls();if(item)content.focus();}}
  }
  async function save(){
    if(saving||loading||!item)return;
    saving=true;controls();failure("");
    try{
      const saved=await api("config",{kind:item.kind,name:item.name,version:item.version,target:name.value.trim(),content:content.value});
      item=saved;name.value=saved.name;
      await catalog();$(field).value=saved.name;$(field).dispatchEvent(new Event("change",{bubbles:true}));
      dialog.close();
    }catch(e){failure(e.message);}
    finally{saving=false;controls();}
  }
  for(const kind of Object.keys(fields)){
    $("edit-"+kind).onclick=()=>open(kind);$(kind).addEventListener("change",labels);
  }
  $("configForm").onsubmit=event=>{event.preventDefault();save();};
  $("configCopy").onclick=()=>{
    if(name.value.trim()===item.name){const parts=item.name.split("/");parts.push(parts.pop().slice(0,100)+"_custom");name.value=parts.join("/");controls();name.focus();name.select();}
    else $("configForm").requestSubmit();
  };
  name.oninput=controls;
  $("configClose").onclick=()=>dialog.close();
  dialog.addEventListener("close",()=>{revision++;loading=false;});
  dialog.addEventListener("cancel",event=>{if(saving)event.preventDefault();});
  content.addEventListener("keydown",event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"){event.preventDefault();$("configForm").requestSubmit();}});
  document.addEventListener("trainingcatalogchanged",labels);
  $("language").addEventListener("change",()=>{labels();controls();});
  applyLanguage();labels();controls();
})();
