"use strict";
const $=id=>document.getElementById(id);
Object.assign(chinese, {'Resume':'继续训练','Resume requested':'已提交恢复训练请求','Resume from checkpoint at step':'将从检查点继续，步数','No checkpoint yet. Resume retries the original configuration from the beginning.':'尚无检查点。继续训练会使用原配置从头重试。','Fix the reported error before resuming.':'请先解决报错原因，再继续训练。','Another training run is active.':'当前已有其他训练任务运行。'});
Object.assign(chinese, {'Copy model path':'复制模型路径','Copied successfully':'复制成功','Copy failed. Select the path and copy it manually.':'复制失败，请选中路径手动复制。'});
Object.assign(chinese, {"Last checked":"最近检查", "Last output":"最近输出", "No new output":"暂无新输出", "Waiting for log output":"等待日志输出", "Open full log":"打开完整日志", "No log output yet.":"尚无日志输出。", "Select a training run to view its logs and metrics.":"选择训练任务以查看日志和指标。", "Dataset download failed. Check the endpoint and retry.":"数据集下载失败，请检查下载地址后重试。", "Download final model":"下载最终模型", "New training run":"新建训练任务","Run name":"任务名称","Policy":"策略","Dataset configuration":"数据集配置","Dataset configurations define data paths, cameras and action format.":"数据集配置包含数据路径、相机及动作格式。","Training configuration":"训练配置","Training parameters":"训练参数","Leave blank to use the selected configuration.":"留空则沿用所选配置。","Runs execute on this WebUI machine using its Python environment.":"训练在运行 WebUI 的机器上，使用该机器当前的 Python 环境执行。","Start training":"开始训练","Training runs":"训练任务","Stop training":"停止训练","Request configuration":"请求配置","Training curves":"训练曲线","Metrics appear after the first logging step.":"首次记录训练指标后显示曲线。","Training log":"训练日志","Download log":"下载日志","Model weights and artifacts":"模型权重与产物","Saved files appear here as training progresses.":"训练生成的文件会显示在这里。","Refresh":"刷新","No training runs":"暂无训练任务","starting":"启动中","running":"训练中","stopping":"正在停止","completed":"已完成","failed":"失败","cancelled":"已停止","interrupted":"已中断","Max steps":"训练步数","Batch size":"批大小","Learning rate":"学习率","Gradient accumulation":"梯度累积步数","Logging steps":"日志间隔步数","Save steps":"保存间隔步数","Data loader workers":"数据加载进程数","Seed":"随机种子","Training request submitted":"训练请求已提交"});
async function api(path,body){const response=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error||response.status);return data;}
const parameters=[['max_steps','Max steps',1,100000000,1],['per_device_train_batch_size','Batch size',1,65536,1],['learning_rate','Learning rate',0.000000001,1,'any'],['gradient_accumulation_steps','Gradient accumulation',1,65536,1],['logging_steps','Logging steps',1,1000000,1],['save_steps','Save steps',1,100000000,1],['dataloader_num_workers','Data loader workers',0,256,1],['seed','Seed',0,2147483647,1]];
for(const [key,label,min,max,step] of parameters){const row=document.createElement('label'),span=document.createElement('span'),input=document.createElement('input');span.dataset.i18n=label;span.textContent=label;input.id=key;input.type='number';input.min=min;input.max=max;input.step=step;row.append(span,input);$('parameters').append(row);}
let selected=localStorage.getItem('vla-training-run'),rows=[],current=null,polling=false,submitting=false;
let copiedRun=null,copyTimer;
$('copyModelPath').onclick=async()=>{
 const run=current?.run,path=run?.output_dir;if(!path)return;
 try{
  if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(path);
  else{
   const input=document.createElement('textarea');input.value=path;input.setAttribute('readonly','');input.style.position='fixed';input.style.opacity='0';document.body.append(input);
   try{input.select();if(!document.execCommand('copy'))throw Error('Copy failed');}finally{input.remove();$('copyModelPath').focus();}
  }
  if(current?.run.id!==run.id)return;
  copiedRun=run.id;$('copyModelStatus').textContent=tr('Copied successfully');
  clearTimeout(copyTimer);copyTimer=setTimeout(()=>{copiedRun=null;$('copyModelStatus').textContent='';},3000);
 }catch(e){if(current?.run.id===run.id)$('copyModelStatus').textContent=tr('Copy failed. Select the path and copy it manually.');}
};
const active=row=>['starting','running','stopping'].includes(row.status);
function message(text,error=false){$('message').textContent=tr(text);$('message').classList.toggle('error',error);}
function renderRuns(){
 window.TrainingCharts.setRuns(rows,selected,id=>{selected=id;localStorage.setItem('vla-training-run',id);current=null;renderRuns();loadDetail().catch(e=>message(e.message,true));});
 $('startTraining').disabled=submitting||rows.some(active);
}
function url(file){return '/api/training/artifact?'+new URLSearchParams({id:selected,file});}
function renderDetail(){
 if(!current)return;
 const {run,metrics,log,files}=current;
 $('detailEmpty').hidden=true;$('runDetail').hidden=false;$('runTitle').textContent=run.name;
 const latest=metrics.at(-1),step=latest?.step||0,max=run.max_steps||run.overrides.max_steps;
 $('runState').textContent=tr(run.status)+(step?' · step '+step:'')+(run.exit_code!==undefined?' · exit '+run.exit_code:'')+(run.error?' · '+run.error:'');
 const failed=run.status==='failed';
 const lastLine=(log||'').trim().split('\n').filter(Boolean).at(-1)||run.error||'';
 const networkFailure=/huggingface|Hub/.test(log||'')&&/timed out|LocalEntryNotFoundError|ConnectTimeout/.test(log||'');
 $('runError').hidden=!failed||!lastLine;
 $('runError').textContent=failed?(networkFailure?tr('Dataset download failed. Check the endpoint and retry.'):lastLine.slice(-600)):'';
 $('openLog').href='/api/training/log?'+new URLSearchParams({id:run.id});
 $('runPath').textContent=run.output_dir;
 $('trainingInterpreter').textContent='Python: '+(run.python||'—');
 $('copyModelPath').disabled=!run.output_dir;
 if(copiedRun!==run.id)$('copyModelStatus').textContent='';
 $('progress').hidden=!active(run);if(max>0)$('progress').value=Math.min(100,step/max*100);else $('progress').removeAttribute('value');
 $('stopTraining').disabled=!active(run)||run.status==='stopping';
 const resumable=current.resume?.available??['failed','cancelled','interrupted'].includes(run.status);
 $('resumeTraining').hidden=!resumable;$('stopTraining').hidden=resumable;
 $('resumeTraining').disabled=submitting||rows.some(active);
 $('resumeHint').hidden=!resumable;
 $('resumeHint').textContent=resumable?(current.resume?.checkpoint?tr('Resume from checkpoint at step')+' '+current.resume.step:tr('No checkpoint yet. Resume retries the original configuration from the beginning.'))+' '+(rows.some(active)?tr('Another training run is active.'):run.status==='failed'?tr('Fix the reported error before resuming.'):''):'';
 $('runConfig').textContent=JSON.stringify({policy:run.policy,dataset:run.task,training:run.training,overrides:run.overrides,data_directory:run.data_dir,python:run.python},null,2);
 window.TrainingCharts.setCurrent(current);
 const clock=value=>value?new Date(value).toLocaleTimeString(language==='zh'?'zh-CN':'en',{hour12:false}):'—';
 const silence=current.log_updated_at&&current.checked_at?Math.max(0,Math.floor((Date.parse(current.checked_at)-Date.parse(current.log_updated_at))/1000)):0;
 $('logStatus').textContent=tr('Last checked')+' '+clock(current.checked_at)+' · '+tr('Last output')+' '+clock(current.log_updated_at)+' · '+(current.log_bytes??0)+' B'+(active(run)&&silence>=10?' · '+tr('No new output')+' '+silence+' s':'');
 window.TrainingLog.update({id:run.id,text:log||tr('No log output yet.'),bytes:current.log_bytes});
 $('downloadModel').hidden=run.status!=='completed'||!files.some(file=>file.startsWith('model/'));$('downloadModel').href='/api/training/model?'+new URLSearchParams({id:selected});
 $('downloadLog').href=url('training.log');window.TrainingArtifacts.render(run.id,files);
}
async function loadDetail(){const id=selected;if(!id||!window.VLAView.active())return;const data=await api('training/run?'+new URLSearchParams({id}));if(id!==selected||!window.VLAView.active())return;current=data;renderDetail();await window.TrainingCharts.refresh();}
async function refresh(){if(polling||!window.VLAView.active())return;polling=true;try{const data=await api('training/runs');if(!window.VLAView.active())return;rows=data;if(!rows.some(row=>row.id===selected))selected=rows[0]?.id;if(selected)localStorage.setItem('vla-training-run',selected);renderRuns();await loadDetail();if(!window.VLAView.active())return;$('link').textContent=tr('Service connected');}catch(e){if(window.VLAView.active()){$('link').textContent=tr('Connection lost');message(e.message,true);}}finally{polling=false;}}
async function catalog(){const data=await api('training/catalog');for(const key of ['policy','task','training']){const select=$(key),old=select.value;select.replaceChildren();for(const name of data[key]){const o=document.createElement('option');o.value=o.textContent=name;select.append(o);}if(data[key].includes(old))select.value=old;else if(data[key].includes(key==='policy'?'act':key==='training'?'default':'sim_transfer_cube_scripted'))select.value=key==='policy'?'act':key==='training'?'default':'sim_transfer_cube_scripted';}document.dispatchEvent(new Event('trainingcatalogchanged'));}
$('trainForm').onsubmit=async event=>{event.preventDefault();if(submitting)return;submitting=true;renderRuns();message('');try{const overrides={};for(const [key]of parameters){if($(key).value!=='')overrides[key]=Number($(key).value);}const run=await api('training/start',{name:$('runName').value,policy:$('policy').value,task:$('task').value,training:$('training').value,overrides});selected=run.id;window.TrainingCharts.include(run.id);message('Training request submitted');await refresh();}catch(e){message(e.message,true);}finally{submitting=false;renderRuns();}};
$('stopTraining').onclick=async()=>{const id=selected;$('stopTraining').disabled=true;try{await api('training/stop',{id});await refresh();}catch(e){message(e.message,true);}};
$('resumeTraining').onclick=async()=>{if(submitting)return;const id=selected,policy=current?.run.policy;submitting=true;renderRuns();renderDetail();message('');try{const prefs=await api('settings'),bindings=prefs.interpreter_bindings||{},interpreter=(prefs.interpreter_overrides||[]).find(row=>row.role==='training'&&row.config===policy)?.interpreter||bindings.training||bindings.default;await api('training/resume',{id,...(interpreter?{interpreter}:{})});message('Resume requested');await refresh();}catch(e){message(e.message,true);}finally{submitting=false;renderRuns();renderDetail();}};
$('refresh').onclick=()=>refresh();
function applyLanguage(){document.documentElement.lang=language==='zh'?'zh-CN':'en';$('language').value=language;document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=tr(el.dataset.i18n));renderRuns();renderDetail();theme();window.TrainingLog.labels();}
function theme(){const dark=document.documentElement.dataset.theme==='dark',b=$('themeToggle');b.textContent=dark?'☀':'☾';b.title=b.ariaLabel=tr(dark?'Light mode':'Dark mode');b.setAttribute('aria-pressed',String(dark));}
window.VLATheme.apply();
$('themeToggle').onclick=()=>{const value=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=value;localStorage.setItem('vla-theme',value);theme();};
$('language').onchange=()=>{language=$('language').value;localStorage.setItem('vla-language',language);applyLanguage();};
function refreshSettings(){catalog().catch(e=>message(e.message,true));}
document.addEventListener('settingschanged',refreshSettings);
document.addEventListener('vla-settingsrefresh',refreshSettings);
document.addEventListener('vla-viewvisibility',()=>{if(window.VLAView.active())refresh();});
document.addEventListener('vla-preferenceschange',()=>{
 language=localStorage.getItem('vla-language')==='zh'?'zh':'en';
 window.VLATheme.apply();applyLanguage();
});
applyLanguage();catalog().then(refresh).catch(e=>message(e.message,true));setInterval(refresh,2500);
