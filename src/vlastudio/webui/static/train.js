"use strict";
const $=id=>document.getElementById(id);
Object.assign(chinese, {"Last checked":"最近检查", "Last output":"最近输出", "No new output":"暂无新输出", "Waiting for log output":"等待日志输出", "Open full log":"打开完整日志", "No log output yet.":"尚无日志输出。", "Select a training run to view its logs and metrics.":"选择训练任务以查看日志和指标。", "Dataset download failed. Check the endpoint and retry.":"数据集下载失败，请检查下载地址后重试。", "Download final model":"下载最终模型", "New training run":"新建训练任务","Run name":"任务名称","Policy":"策略","Dataset configuration":"数据集配置","Dataset configurations define data paths, cameras and action format.":"数据集配置包含数据路径、相机及动作格式。","Training configuration":"训练配置","Training parameters":"训练参数","Leave blank to use the selected configuration.":"留空则沿用所选配置。","Runs execute on this WebUI machine using its Python environment.":"训练在运行 WebUI 的机器上，使用该机器当前的 Python 环境执行。","Start training":"开始训练","Training runs":"训练任务","Stop training":"停止训练","Request configuration":"请求配置","Training curves":"训练曲线","Metrics appear after the first logging step.":"首次记录训练指标后显示曲线。","Training log":"训练日志","Download log":"下载日志","Model weights and artifacts":"模型权重与产物","Saved files appear here as training progresses.":"训练生成的文件会显示在这里。","Refresh":"刷新","No training runs":"暂无训练任务","starting":"启动中","running":"训练中","stopping":"正在停止","completed":"已完成","failed":"失败","cancelled":"已停止","interrupted":"已中断","Max steps":"训练步数","Batch size":"批大小","Learning rate":"学习率","Gradient accumulation":"梯度累积步数","Logging steps":"日志间隔步数","Save steps":"保存间隔步数","Data loader workers":"数据加载进程数","Seed":"随机种子","Training request submitted":"训练请求已提交"});
async function api(path,body){const response=await fetch('/api/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error||response.status);return data;}
const parameters=[['max_steps','Max steps',1,100000000,1],['per_device_train_batch_size','Batch size',1,65536,1],['learning_rate','Learning rate',0.000000001,1,'any'],['gradient_accumulation_steps','Gradient accumulation',1,65536,1],['logging_steps','Logging steps',1,1000000,1],['save_steps','Save steps',1,100000000,1],['dataloader_num_workers','Data loader workers',0,256,1],['seed','Seed',0,2147483647,1]];
for(const [key,label,min,max,step] of parameters){const row=document.createElement('label'),span=document.createElement('span'),input=document.createElement('input');span.dataset.i18n=label;span.textContent=label;input.id=key;input.type='number';input.min=min;input.max=max;input.step=step;row.append(span,input);$('parameters').append(row);}
let selected=localStorage.getItem('vla-training-run'),rows=[],current=null,polling=false,submitting=false;
const active=row=>['starting','running','stopping'].includes(row.status);
function message(text,error=false){$('message').textContent=tr(text);$('message').classList.toggle('error',error);}
function renderRuns(){
 $('runs').replaceChildren();
 if(!rows.length)$('runs').textContent=tr('No training runs');
 for(const row of rows){const button=document.createElement('button');button.type='button';button.className='run-row';button.setAttribute('aria-pressed',String(row.id===selected));const name=document.createElement('strong'),state=document.createElement('small');name.textContent=row.name;state.textContent=tr(row.status)+' · '+new Date(row.created_at).toLocaleString(language==='zh'?'zh-CN':'en');button.append(name,state);button.onclick=()=>{selected=row.id;localStorage.setItem('vla-training-run',selected);current=null;renderRuns();loadDetail().catch(e=>message(e.message,true));};$('runs').append(button);}
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
 $('progress').hidden=!active(run);if(max>0)$('progress').value=Math.min(100,step/max*100);else $('progress').removeAttribute('value');
 $('stopTraining').disabled=!active(run)||run.status==='stopping';
 $('runConfig').textContent=JSON.stringify({policy:run.policy,dataset:run.task,training:run.training,overrides:run.overrides,data_directory:run.data_dir,python:run.python},null,2);
 const previous=$('metric').value,keys=[...new Set(metrics.flatMap(row=>Object.keys(row)))].filter(key=>!['step','time','epoch'].includes(key));
 $('metric').replaceChildren();for(const key of keys){const o=document.createElement('option');o.value=o.textContent=key;$('metric').append(o);}if(keys.includes(previous))$('metric').value=previous;else if(keys.includes('loss'))$('metric').value='loss';
 chart();
 const clock=value=>value?new Date(value).toLocaleTimeString(language==='zh'?'zh-CN':'en',{hour12:false}):'—';
 const silence=current.log_updated_at&&current.checked_at?Math.max(0,Math.floor((Date.parse(current.checked_at)-Date.parse(current.log_updated_at))/1000)):0;
 $('logStatus').textContent=tr('Last checked')+' '+clock(current.checked_at)+' · '+tr('Last output')+' '+clock(current.log_updated_at)+' · '+(current.log_bytes??0)+' B'+(active(run)&&silence>=10?' · '+tr('No new output')+' '+silence+' s':'');
 window.TrainingLog.update({id:run.id,text:log||tr('No log output yet.'),bytes:current.log_bytes});
 $('downloadModel').hidden=run.status!=='completed'||!files.some(file=>file.startsWith('model/'));$('downloadModel').href='/api/training/model?'+new URLSearchParams({id:selected});
 $('downloadLog').href=url('training.log');$('artifacts').replaceChildren();
 for(const file of files){const a=document.createElement('a');a.href=url(file);a.textContent=file;a.download='';$('artifacts').append(a);}
}
function svg(tag,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,value);if(text!==undefined)el.textContent=text;return el;}
function chart(){
 const key=$('metric').value,points=(current?.metrics||[]).filter(row=>Number.isFinite(row[key])&&Number.isFinite(row.step)),canvas=$('chart');canvas.replaceChildren();$('noMetrics').hidden=points.length>0;canvas.toggleAttribute('hidden',!points.length);$('metricValue').textContent='';if(!points.length)return;
 const xs=points.map(p=>p.step),ys=points.map(p=>p[key]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const x=v=>70+(v-xmin)/(xmax-xmin||1)*660,y=v=>215-(v-ymin)/(ymax-ymin||1)*185;
 canvas.append(svg('path',{d:'M70 20V215H735',fill:'none',stroke:'currentColor',opacity:'.4'}));
 for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4;canvas.append(svg('text',{x:62,y:220-i*185/4,'text-anchor':'end',fill:'currentColor','font-size':12},v.toPrecision(4)));}
 canvas.append(svg('text',{x:70,y:245,fill:'currentColor','font-size':12},xmin),svg('text',{x:730,y:245,'text-anchor':'end',fill:'currentColor','font-size':12},xmax+' step'));
 canvas.append(svg('polyline',{points:points.map(p=>x(p.step)+','+y(p[key])).join(' '),fill:'none',stroke:'#2389de','stroke-width':2}));
 for(const p of points){const dot=svg('circle',{cx:x(p.step),cy:y(p[key]),r:3,fill:'#2389de'});dot.append(svg('title',{},'step '+p.step+' · '+key+': '+p[key]));canvas.append(dot);}
 const last=points.at(-1);$('metricValue').textContent=key+': '+last[key]+' · step '+last.step;
}
async function loadDetail(){const id=selected;if(!id)return;const data=await api('training/run?'+new URLSearchParams({id}));if(id!==selected)return;current=data;renderDetail();}
async function refresh(){if(polling)return;polling=true;try{rows=await api('training/runs');if(!rows.some(row=>row.id===selected))selected=rows[0]?.id;if(selected)localStorage.setItem('vla-training-run',selected);renderRuns();await loadDetail();$('link').textContent=tr('Service connected');}catch(e){$('link').textContent=tr('Connection lost');message(e.message,true);}finally{polling=false;}}
async function catalog(){const data=await api('training/catalog');for(const key of ['policy','task','training']){const select=$(key),old=select.value;select.replaceChildren();for(const name of data[key]){const o=document.createElement('option');o.value=o.textContent=name;select.append(o);}if(data[key].includes(old))select.value=old;else if(data[key].includes(key==='policy'?'act':key==='training'?'default':'sim_transfer_cube_scripted'))select.value=key==='policy'?'act':key==='training'?'default':'sim_transfer_cube_scripted';}document.dispatchEvent(new Event('trainingcatalogchanged'));}
$('trainForm').onsubmit=async event=>{event.preventDefault();if(submitting)return;submitting=true;renderRuns();message('');try{const overrides={};for(const [key]of parameters){if($(key).value!=='')overrides[key]=Number($(key).value);}const run=await api('training/start',{name:$('runName').value,policy:$('policy').value,task:$('task').value,training:$('training').value,overrides});selected=run.id;message('Training request submitted');await refresh();}catch(e){message(e.message,true);}finally{submitting=false;renderRuns();}};
$('stopTraining').onclick=async()=>{const id=selected;$('stopTraining').disabled=true;try{await api('training/stop',{id});await refresh();}catch(e){message(e.message,true);}};
$('metric').onchange=chart;$('refresh').onclick=()=>refresh();
function applyLanguage(){document.documentElement.lang=language==='zh'?'zh-CN':'en';$('language').value=language;document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=tr(el.dataset.i18n));renderRuns();renderDetail();theme();window.TrainingLog.labels();}
function theme(){const dark=document.documentElement.dataset.theme==='dark',b=$('themeToggle');b.textContent=dark?'☀':'☾';b.title=b.ariaLabel=tr(dark?'Light mode':'Dark mode');b.setAttribute('aria-pressed',String(dark));}
window.VLATheme.apply();
$('themeToggle').onclick=()=>{const value=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=value;localStorage.setItem('vla-theme',value);theme();};
$('language').onchange=()=>{language=$('language').value;localStorage.setItem('vla-language',language);applyLanguage();};
document.addEventListener('settingschanged',()=>catalog().catch(e=>message(e.message,true)));
applyLanguage();catalog().then(refresh).catch(e=>message(e.message,true));setInterval(refresh,2500);
