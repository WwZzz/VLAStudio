"use strict";
window.EnvironmentSelection=(()=>{
 const path=value=>(value||'').replace(/\\/g,'/').replace(/\/+$/,'');
 function select(rows,context,runs=[]){
  const selected=new Map(),missing=[];
  function add(roles,key){if(!key)return;const matches=rows.filter(row=>roles.some(role=>(row.bindings[role]||[]).includes(key)));if(matches.length)matches.forEach(row=>selected.set(row.id,row));else missing.push(key);}
  if(context.mode==='train')add(['training'],context.policy);
  if(['infer','collect'].includes(context.mode))add(['simulation','robot'],context.robot);
  if(context.mode==='collect')add(['teleop'],context.teleop);
  if(context.mode==='infer'&&context.policy){
   let key=context.policy;
   const direct=rows.some(row=>(row.bindings.inference||[]).includes(key));
   if(!direct){
    if(/^(https?|wss?|tcp|grpc):\/\//.test(key)||/^[\w.-]+:\d+$/.test(key))key='vlastudio.policy.remote';
    else{const model=path(key),run=runs.filter(row=>row.output_dir&&(model===path(row.output_dir)||model.startsWith(path(row.output_dir)+'/'))).sort((a,b)=>b.output_dir.length-a.output_dir.length)[0];if(run)key=run.policy;}
   }
   add(['inference'],key);
  }
  return {rows:[...selected.values()],missing};
 }
 return {select};
})();
// Settings UI
(() => {
 const panel=$('settingsInterpretersPanel');if(!panel)return;
 Object.assign(chinese, {'Install specified components':'安装指定组件','Choose components independently of the current configuration.':'这里可以独立于当前配置，选择其他组件进行检查或安装。','Check selected components':'检查勾选组件','Install selected components':'安装勾选组件','Clear selection':'清空选择','Select components from the catalog.':'请从组件列表中勾选。','Selected components':'已勾选组件','Select component':'勾选组件','Current configuration environments':'当前配置的环境','Check selected configuration':'检查当前配置','Check and install selected configuration':'检查并安装当前配置','Current configuration':'当前配置','Choose a configuration on the training, inference or collection page first.':'请先在训练、推理或采集页面选择配置。','Cannot identify one of the selected configurations. Select a known configuration before installing.':'无法识别当前选择的某项配置，请先选择已知配置再安装。','Component environments':'组件环境','Check and install all':'检查并安装全部','Check all':'检查全部','Install / repair':'安装 / 修复','Check':'检查','Stop installation':'停止安装','All components':'全部组件','Policies':'策略','Simulations':'仿真','Robots':'机器人','Teleoperators':'遥操作','Search components':'搜索组件','Open installation log':'查看安装日志','Checking imports before installation. Valid environments are reused. New environments use uv under $VLASTUDIO/envs/.':'安装前先检查导入；可用环境直接复用。新环境统一由 uv 安装在 $VLASTUDIO/envs/ 下。','Installed environments are used automatically unless you select another interpreter above.':'安装成功的环境会自动使用；上方手动指定的解释器优先。','queued':'排队中','checking':'检查中','installing':'安装中','ready':'已安装并验证','reused':'可用，已跳过安装','missing':'缺少依赖','unavailable':'暂不可安装','needs_license':'需要同意许可证','cancelled':'已停止','interrupted':'已中断','failed':'失败','registered':'此前已验证','unchecked':'尚未检查','Environment operation submitted':'环境任务已提交','Environment queue completed':'环境队列已完成','I have reviewed and accept':'我已阅读并同意','Some environments need license acceptance before installation. Other environments will continue.':'部分环境需先同意许可证，其他环境会继续处理。','No matching components':'没有匹配的组件','Loading components…':'正在读取组件…'});
 const section=document.createElement('section');section.dataset.environmentInstaller='1';section.id='environmentInstaller';
 section.innerHTML='<h4 data-i18n="Current configuration environments"></h4><small id="environmentScope" role="status"></small><p data-i18n="Checking imports before installation. Valid environments are reused. New environments use uv under $VLASTUDIO/envs/."></p><small data-i18n="Installed environments are used automatically unless you select another interpreter above."></small><div class="environment-toolbar"><button type="button" id="environmentsInstall" class="primary" data-i18n="Check and install selected configuration"></button><button type="button" id="environmentsCheck" data-i18n="Check selected configuration"></button><button type="button" id="environmentsCancel" data-i18n="Stop installation" hidden></button></div><small id="environmentsStatus" role="status"></small><div id="environmentComponents"></div><details id="environmentBrowse"><summary data-i18n="Install specified components"></summary><p data-i18n="Choose components independently of the current configuration."></p><div class="environment-toolbar"><select id="environmentKind" aria-label="Component type"></select><input id="environmentSearch" type="search" autocomplete="off" placeholder="Search components" aria-label="Search components"></div><small id="environmentPickedSummary" class="environment-select-summary" role="status"></small><div class="environment-toolbar"><button type="button" id="environmentPickedCheck" data-i18n="Check selected components"></button><button type="button" id="environmentPickedInstall" data-i18n="Install selected components"></button><button type="button" id="environmentPickedClear" data-i18n="Clear selection"></button></div><div id="environmentAvailableComponents" class="environment-catalog"></div></details><details id="environmentLog" hidden><summary data-i18n="Open installation log"></summary><pre id="environmentLogText"></pre></details>';
 const bindings=$('interpreterBindings');bindings.after(section);
 let data=null,pollTimer=null,loading=false,submitting=false,logSelection=null,runs=[],scopeTimer=null,lastScope='';
 const accepted=new Set(),picked=new Set();
 const kind=$('environmentKind');for(const [value,label]of [['','All components'],['policy','Policies'],['simulation','Simulations'],['robot','Robots'],['teleop','Teleoperators']])kind.add(new Option(tr(label),value));
 const active=()=>data?.job&&['queued','running'].includes(data.job.status);
 const rows=()=>data?.components||[];
 function context(){return {mode:location.pathname.slice(1)||'collect',policy:$('policy')?.value||'',robot:$('robot')?.value||'',teleop:$('teleop')?.value||''};}
 function scope(){return window.EnvironmentSelection.select(rows(),context(),runs);}
 function filtered(){return scope().rows;}
 function controls(){const busy=submitting||active();$('environmentsInstall').disabled=$('environmentsCheck').disabled=busy||!scope().rows.length||!!scope().missing.length;$('environmentsCancel').hidden=!active();$('environmentsCancel').disabled=submitting;$('environmentPickedCheck').disabled=$('environmentPickedInstall').disabled=busy||!picked.size;$('environmentPickedClear').disabled=!picked.size;}
 async function start(ids,action,source='current'){const allowed=new Set((source==='specified'?rows():scope().rows).map(row=>row.id));if(submitting||active()||!ids.length||ids.some(id=>!allowed.has(id)))return;submitting=true;controls();try{await api('settings/environments/start',{components:ids,action,accepted_licenses:[...accepted]});$('environmentsStatus').textContent=tr('Environment operation submitted');await refresh();}catch(e){$('environmentsStatus').textContent=e.message;}finally{submitting=false;render();}}
 async function showLog(id){if(!data?.job)return;logSelection=id;$('environmentLog').hidden=false;$('environmentLog').open=true;try{const result=await api('settings/environments/log?'+new URLSearchParams({id:data.job.id,component:id}));if(logSelection===id)$('environmentLogText').textContent=result.log||'—';}catch(e){$('environmentLogText').textContent=e.message;}}
 function renderRows(box,visible,results,source){
  box.replaceChildren();if(!visible.length){box.textContent=tr('No matching components');return;}
  for(const component of visible){
   const result=results.get(component.id)||component.saved_result,card=document.createElement('article');card.className='environment-component';card.dataset.component=component.id;
   const title=document.createElement('div');title.className='environment-component-title';const name=document.createElement('strong'),kind=document.createElement('small');name.textContent=component.label;kind.textContent=component.kind;title.append(name,kind);
   const status=document.createElement('span');status.className='environment-result';status.textContent=tr(result?.status||(component.issue?'unavailable':'unchecked'));status.dataset.state=result?.status||'unchecked';
   const actions=document.createElement('div');actions.className='environment-row-actions';for(const [text,action]of [['Check','check'],['Install / repair','install']]){const button=document.createElement('button');button.type='button';button.textContent=tr(text);button.disabled=submitting||!!active()||!!component.issue;button.onclick=()=>start([component.id],action,source);actions.append(button);}
   const header=document.createElement('div');header.className='environment-component-heading';if(source==='specified'){const check=document.createElement('input');check.type='checkbox';check.className='environment-pick';check.setAttribute('aria-label',tr('Select component')+' · '+component.label);check.checked=picked.has(component.id);check.disabled=submitting||!!active()||!!component.issue;check.onchange=()=>{if(check.checked)picked.add(component.id);else picked.delete(component.id);selectionLabels();controls();};header.append(check);}header.append(title,status,actions);card.append(header);
   const note=document.createElement('small');note.className='environment-component-note';note.textContent=result?.message||component.issue||'$VLASTUDIO/envs/'+component.environment;note.title=result?.python||'';card.append(note);
   if(result&&['installing','failed','ready','needs_license','unavailable'].includes(result.status)){const button=document.createElement('button');button.type='button';button.className='environment-log-button';button.textContent=tr('Open installation log');button.onclick=()=>showLog(component.id);card.append(button);}
   for(const [id,url]of Object.entries(component.licenses)){const label=document.createElement('label');label.className='environment-license';const check=document.createElement('input');check.type='checkbox';check.checked=accepted.has(id);check.onchange=()=>{if(check.checked)accepted.add(id);else accepted.delete(id);};const text=document.createElement('span');text.textContent=tr('I have reviewed and accept')+' ';const link=document.createElement('a');link.textContent=id;link.href=url;link.target='_blank';link.rel='noopener noreferrer';label.append(check,text,link);card.append(label);}
   box.append(card);
  }
 }
 function selectionLabels(){const chosen=rows().filter(row=>picked.has(row.id));$('environmentPickedSummary').textContent=chosen.length?tr('Selected components')+' ('+chosen.length+'): '+chosen.map(row=>row.label).join(' · '):tr('Select components from the catalog.');}
 function render(){
  if(!data)return;controls();const current=scope();$('environmentScope').textContent=current.missing.length?tr('Cannot identify one of the selected configurations. Select a known configuration before installing.'):(current.rows.length?tr('Current configuration')+': '+current.rows.map(row=>row.label).join(' · '):tr('Choose a configuration on the training, inference or collection page first.'));
  const results=new Map((data.job?.rows||[]).map(row=>[row.component.id,row]));renderRows($('environmentComponents'),current.rows,results,'current');
  const query=$('environmentSearch').value.toLowerCase();const available=rows().filter(row=>(!kind.value||row.kind===kind.value)&&(!query||[row.label,row.kind,...Object.values(row.bindings).flat()].join(' ').toLowerCase().includes(query)));renderRows($('environmentAvailableComponents'),available,results,'specified');selectionLabels();
  if(data.job){const completed=data.job.rows.filter(row=>!['queued','checking','installing'].includes(row.status)).length;$('environmentsStatus').textContent=completed+'/'+data.job.rows.length+' · '+tr(active()?data.job.status:data.job.status==='completed'?'Environment queue completed':data.job.status);}
 }
 async function refresh(){if(loading)return;loading=true;try{data=await api('settings/environments');if(context().mode==='infer')runs=await api('training/runs');window.InterpreterSettings.setManaged(data.interpreters);render();if(logSelection&&active())await showLog(logSelection);}catch(e){$('environmentsStatus').textContent=e.message;}finally{loading=false;clearTimeout(pollTimer);if($('settingsDialog').open&&active())pollTimer=setTimeout(refresh,2000);}}
 $('environmentsInstall').onclick=()=>start(filtered().map(row=>row.id),'install');$('environmentsCheck').onclick=()=>start(filtered().map(row=>row.id),'check');
 $('environmentPickedCheck').onclick=()=>start([...picked],'check','specified');$('environmentPickedInstall').onclick=()=>start([...picked],'install','specified');$('environmentPickedClear').onclick=()=>{picked.clear();render();};kind.onchange=render;$('environmentSearch').oninput=render;
 $('environmentsCancel').onclick=async()=>{if(!data?.job)return;submitting=true;controls();try{await api('settings/environments/cancel',{id:data.job.id});await refresh();}catch(e){$('environmentsStatus').textContent=e.message;}finally{submitting=false;controls();}};
 $('settingsDialog').addEventListener('close',()=>{clearTimeout(pollTimer);clearInterval(scopeTimer);scopeTimer=null;logSelection=null;});
 section.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=tr(el.dataset.i18n));
 controls();
 window.EnvironmentInstaller={refresh(){if(!scopeTimer)scopeTimer=setInterval(()=>{const signature=JSON.stringify(context());if(signature!==lastScope){lastScope=signature;if(data)render();if(context().mode==='infer')refresh();}},500);return refresh();}};
})();
