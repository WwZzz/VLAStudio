"use strict";
(() => {
  Object.assign(chinese,{'Collapse folders':'折叠文件夹','Expand folders':'展开文件夹','files':'个文件','No saved files yet.':'暂无已保存文件。'});
  const box=document.getElementById('artifacts'),states=new Map();
  let signature='',currentId;
  function folderState(id){if(!states.has(id))states.set(id,new Set(['model']));return states.get(id);}
  function render(id,files){
    document.getElementById('artifactCount').textContent=files.length+' '+tr('files');
    const next=JSON.stringify([id,files,document.documentElement.lang]);if(signature===next)return;
    const scroll=id===currentId?box.scrollTop:0;signature=next;currentId=id;box.replaceChildren();
    if(!files.length){box.textContent=tr('No saved files yet.');return;}
    const tree=new Map(),expanded=folderState(id);
    for(const file of files){const parts=file.split('/');let node=tree;parts.forEach((part,index)=>{if(index===parts.length-1)node.set(part,{file});else{if(!node.has(part))node.set(part,{children:new Map()});node=node.get(part).children;}});}
    function append(parent,entries,prefix=''){
      for(const [name,node]of [...entries].sort((a,b)=>Number(!!b[1].children)-Number(!!a[1].children)||a[0].localeCompare(b[0],undefined,{numeric:true}))){
        const path=prefix?prefix+'/'+name:name;
        if(node.children){
          const details=document.createElement('details'),summary=document.createElement('summary'),label=document.createElement('span');
          details.className='artifact-folder';details.dataset.path=path;details.open=expanded.has(path);label.textContent=name;summary.append(label);details.append(summary);
          details.addEventListener('toggle',()=>{if(details.isConnected){if(details.open)expanded.add(path);else expanded.delete(path);}});
          const children=document.createElement('div');children.className='artifact-children';append(children,node.children,path);details.append(children);parent.append(details);
        }else{
          const link=document.createElement('a');link.className='artifact-file';link.textContent=name;link.title=node.file;link.href='/api/training/artifact?'+new URLSearchParams({id,file:node.file});link.download='';parent.append(link);
        }
      }
    }
    append(box,tree);box.scrollTop=scroll;
  }
  function toggle(open){if(!currentId)return;const state=folderState(currentId);for(const node of box.querySelectorAll('details')){node.open=open;if(open)state.add(node.dataset.path);else state.delete(node.dataset.path);}}
  document.getElementById('collapseArtifacts').onclick=()=>toggle(false);
  document.getElementById('expandArtifacts').onclick=()=>toggle(true);
  window.TrainingArtifacts={render};
})();
