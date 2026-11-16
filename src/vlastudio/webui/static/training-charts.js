"use strict";
(() => {
  Object.assign(chinese,{'Visible metrics':'显示指标','Smoothing':'平滑','Check runs to compare. Click a name for details.':'勾选任务对比曲线，点击名称查看详情。','Select all':'全选','Clear selection':'清空选择','Compare run':'对比任务','Compared runs':'对比任务','runs selected':'个任务已勾选','Check at least one run to display curves.':'请先勾选要展示曲线的任务。','Select metrics to display.':'请勾选需要展示的指标。','Metrics appear after the first logging step.':'首次记录训练指标后显示曲线。','No data for this metric.':'此指标暂无数据。','No metrics yet':'暂无指标','Hide metric':'隐藏指标','Expand chart':'展开图表','Restore chart':'还原图表','Loading curves…':'正在加载曲线…','Curve unavailable':'曲线读取失败','Step':'步数'});
  const $=id=>document.getElementById(id),palette=['#48a8ec','#f4ac62','#a58cf0','#54c5a0','#ea829e','#d6c65e','#5fceda','#cb9d78','#a0c96c','#da8de4','#f28a70','#7d9cdd'];
  Object.assign(chinese,{'Raw':'原始值','Smoothed':'平滑值'});
  function stored(key){try{const data=JSON.parse(localStorage.getItem(key));return Array.isArray(data)?new Set(data.filter(x=>typeof x==='string')):null;}catch{return null;}}
  let compared=stored('vla-compared-runs'),visible=stored('vla-visible-metrics'),runs=[],focus='',selectDetail,sidebarSignature='',choiceSignature='',chartSignature='',loading=0;
  const cache=new Map(),pending=new Map(),errors=new Map(),colors=new Map(),expanded=new Set();
  let smoothing=Number(localStorage.getItem('vla-chart-smoothing')||0);if(!Number.isFinite(smoothing)||smoothing<0||smoothing>.95)smoothing=0;
  $('chartSmoothing').value=smoothing;$('smoothingValue').value=smoothing.toFixed(2);
  const saveCompared=()=>localStorage.setItem('vla-compared-runs',JSON.stringify([...compared]));
  const saveVisible=()=>localStorage.setItem('vla-visible-metrics',JSON.stringify([...visible]));
  function color(id){if(!colors.has(id))colors.set(id,palette[colors.size%palette.length]);return colors.get(id);}
  function shortDate(row){return new Date(row.created_at).toLocaleString(document.documentElement.lang==='zh'?'zh-CN':'en',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});}
  const runLabel=row=>row.name+' · '+shortDate(row);
  function selectedRuns(){return runs.filter(row=>compared?.has(row.id));}
  function chooseRun(id,checked){if(checked)compared.add(id);else compared.delete(id);saveCompared();renderSidebar();render();refresh();}
  function renderSidebar(){
    const signature=JSON.stringify([runs.map(r=>[r.id,r.name,r.status,r.created_at]),focus,[...(compared||[])],document.documentElement.lang]);
    if(sidebarSignature===signature)return;sidebarSignature=signature;$('runs').replaceChildren();
    if(!runs.length){$('runs').textContent=tr('No training runs');return;}
    for(const row of runs){
      const entry=document.createElement('div');entry.className='run-row';entry.classList.toggle('focused',row.id===focus);entry.dataset.run=row.id;
      const check=document.createElement('input');check.type='checkbox';check.checked=compared.has(row.id);check.setAttribute('aria-label',tr('Compare run')+' '+runLabel(row));check.style.accentColor=color(row.id);check.onchange=()=>chooseRun(row.id,check.checked);
      const swatch=document.createElement('span');swatch.className='run-color';swatch.style.backgroundColor=color(row.id);
      const button=document.createElement('button');button.type='button';button.className='run-detail-button';button.setAttribute('aria-pressed',String(row.id===focus));button.title=runLabel(row)+' · '+tr(row.status);
      const name=document.createElement('strong'),meta=document.createElement('small');name.textContent=row.name;meta.textContent=tr(row.status)+' · '+shortDate(row);button.append(name,meta);button.onclick=()=>selectDetail(row.id);
      entry.append(check,swatch,button);$('runs').append(entry);
    }
  }
  function setRuns(rows,selected,onSelect){
    runs=rows;focus=selected;selectDetail=onSelect;
    if(compared===null&&rows.length)compared=new Set(selected?[selected]:[]);
    const known=new Set(rows.map(r=>r.id));if(rows.length&&compared)for(const id of compared)if(!known.has(id))compared.delete(id);
    for(const row of rows)color(row.id);
    renderSidebar();render();
  }
  function include(id){if(compared===null)compared=new Set();compared.add(id);saveCompared();}
  function setCurrent(detail){cache.set(detail.run.id,{run:detail.run,metrics:detail.metrics});errors.delete(detail.run.id);render();}
  async function fetchRun(row){
    if(pending.has(row.id))return pending.get(row.id);
    const previous=cache.get(row.id);if(previous&&previous.run.status===row.status&&!['starting','running','stopping'].includes(row.status))return;
    const request=(async()=>{try{cache.set(row.id,await api('training/metrics?'+new URLSearchParams({id:row.id})));errors.delete(row.id);}catch(e){errors.set(row.id,e.message);}finally{pending.delete(row.id);}})();pending.set(row.id,request);return request;
  }
  async function refresh(){loading++;status();try{await Promise.all(selectedRuns().map(fetchRun));}finally{loading--;render();}}
  function status(){const failures=selectedRuns().filter(r=>errors.has(r.id));$('comparisonStatus').textContent=loading?tr('Loading curves…'):failures.length?tr('Curve unavailable')+': '+failures.map(r=>r.name).join(', '):selectedRuns().length+' '+tr('runs selected');$('comparisonStatus').title=failures.map(r=>r.name+': '+errors.get(r.id)).join('\n');}
  function metricKeys(){const priority=key=>{const n=['loss','eval_loss','grad_norm','learning_rate'].indexOf(key);return n<0?100:n;};return [...new Set(selectedRuns().flatMap(row=>(cache.get(row.id)?.metrics||[]).flatMap(p=>Object.keys(p).filter(k=>!['step','time','epoch'].includes(k)&&typeof p[k]==='number'&&Number.isFinite(p[k])))))].sort((a,b)=>priority(a)-priority(b)||a.localeCompare(b));}
  function chooseMetric(key,checked){if(checked)visible.add(key);else visible.delete(key);saveVisible();render();}
  function renderChoices(keys){
    const signature=JSON.stringify([keys,[...visible],document.documentElement.lang]);if(signature===choiceSignature)return;choiceSignature=signature;
    const box=$('metricChoices');box.replaceChildren();
    if(!keys.length)box.textContent=tr('No metrics yet');
    for(const key of keys){const label=document.createElement('label'),check=document.createElement('input'),text=document.createElement('span');check.type='checkbox';check.checked=visible.has(key);check.onchange=()=>chooseMetric(key,check.checked);text.textContent=key;label.append(check,text);box.append(label);}
  }
  function render(){
    status();const keys=metricKeys();if(visible===null){visible=new Set(keys.filter(k=>['loss','grad_norm','learning_rate','eval_loss'].includes(k)));if(!visible.size&&keys.length)visible=new Set(keys.slice(0,3));if(!keys.length)visible=null;}
    const selectedMetrics=keys.filter(k=>visible?.has(k));
    if(visible!==null)renderChoices(keys);else{$('metricChoices').textContent=tr('No metrics yet');choiceSignature='';}
    const chosen=selectedRuns();const legend=$('chartLegend');legend.replaceChildren();legend.setAttribute('aria-label',tr('Compared runs'));
    for(const row of chosen){const label=document.createElement('span'),dot=document.createElement('i');dot.style.backgroundColor=color(row.id);label.append(dot,document.createTextNode(runLabel(row)));if(!(cache.get(row.id)?.metrics.length))label.className='no-run-metrics';legend.append(label);}
    $('noMetrics').hidden=selectedMetrics.length>0;
    $('noMetrics').textContent=!chosen.length?tr('Check at least one run to display curves.'):!keys.length?tr('Metrics appear after the first logging step.'):tr('Select metrics to display.');
    const signature=JSON.stringify([selectedMetrics,chosen.map(r=>[r.id,r.name,r.created_at,cache.get(r.id)?.metrics]),smoothing,[...expanded],document.documentElement.lang]);
    if(chartSignature===signature)return;chartSignature=signature;renderCards(selectedMetrics,chosen);
  }
  const svg=(tag,attrs,text)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;return el;};
  const number=v=>Math.abs(v)>=10000||Math.abs(v)>0&&Math.abs(v)<.001?v.toExponential(1):Number(v.toPrecision(4)).toLocaleString('en',{maximumFractionDigits:5});
  function points(row,key){return (cache.get(row.id)?.metrics||[]).filter(p=>Number.isFinite(p.step)&&Number.isFinite(p[key])).sort((a,b)=>a.step-b.step);}
  function smoothed(values){let last;return values.map(p=>{last=last===undefined?p.value:smoothing*last+(1-smoothing)*p.value;return {...p,value:last};});}
  function nearest(values,step){let low=0,high=values.length;while(low<high){const mid=(low+high)>>1;if(values[mid].step<step)low=mid+1;else high=mid;}const a=values[Math.max(0,low-1)],b=values[Math.min(values.length-1,low)];return Math.abs(a.step-step)<=Math.abs(b.step-step)?a:b;}
  const plots=[];
  function renderCards(keys,chosen){
    plots.length=0;const grid=$('chartsGrid');grid.replaceChildren();
    for(const key of keys){
      const card=document.createElement('section');card.className='metric-card';card.dataset.metric=key;card.classList.toggle('chart-expanded',expanded.has(key));
      const heading=document.createElement('div');heading.className='metric-heading';const title=document.createElement('h4');title.textContent=key;heading.append(title);
      const expand=document.createElement('button');expand.type='button';expand.textContent=expanded.has(key)?'↙':'↗';expand.ariaLabel=tr(expanded.has(key)?'Restore chart':'Expand chart')+' '+key;expand.title=expand.ariaLabel;expand.onclick=()=>{if(expanded.has(key))expanded.delete(key);else expanded.add(key);render();};
      const hide=document.createElement('button');hide.type='button';hide.textContent='×';hide.ariaLabel=hide.title=tr('Hide metric')+' '+key;hide.onclick=()=>{chooseMetric(key,false);$('metricPicker').querySelector('summary').focus();};heading.append(expand,hide);card.append(heading);
      const plot=document.createElement('div');plot.className='metric-plot';const canvas=svg('svg',{role:'img',tabindex:'0','aria-label':key+' · '+tr('Compared runs')});const tooltip=document.createElement('div');tooltip.className='chart-tooltip';tooltip.hidden=true;plot.append(canvas,tooltip);card.append(plot);
      const values=document.createElement('div');values.className='metric-values';
      const series=chosen.map(row=>({row,points:points(row,key)})).filter(s=>s.points.length);
      for(const s of series){const line=document.createElement('div'),dot=document.createElement('i'),name=document.createElement('span'),value=document.createElement('strong');dot.style.backgroundColor=color(s.row.id);name.textContent=runLabel(s.row);value.textContent=number(s.points.at(-1)[key]);line.title=name.textContent;line.append(dot,name,value);values.append(line);}card.append(values);grid.append(card);
      if(!series.length){plot.textContent=tr('No data for this metric.');continue;}
      const draw=()=>drawPlot(canvas,tooltip,series,key);plots.push(draw);draw();
    }
  }
  function drawPlot(canvas,tooltip,series,key){
    const width=Math.max(240,Math.round(canvas.getBoundingClientRect().width)),height=220,left=58,right=width-16,top=14,bottom=181;
    canvas.setAttribute('viewBox',`0 0 ${width} ${height}`);canvas.replaceChildren();
    const curves=series.map(s=>({...s,values:smoothed(s.points.map(p=>({step:p.step,value:p[key]})))}));
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    for(const s of curves)for(const p of s.values){xmin=Math.min(xmin,p.step);xmax=Math.max(xmax,p.step);ymin=Math.min(ymin,p.value);ymax=Math.max(ymax,p.value);}
    const pad=(ymax-ymin)*.06||Math.abs(ymax)*.05||1;ymin-=pad;ymax+=pad;
    const x=v=>left+(v-xmin)/(xmax-xmin||1)*(right-left),y=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
    for(let i=0;i<=4;i++){const val=ymin+(ymax-ymin)*i/4,yy=y(val);canvas.append(svg('line',{x1:left,y1:yy,x2:right,y2:yy,class:'chart-grid-line'}),svg('text',{x:left-7,y:yy+4,'text-anchor':'end',class:'chart-tick'},number(val)));}
    for(let i=0;i<=3;i++){const val=xmin+(xmax-xmin)*i/3;canvas.append(svg('text',{x:x(val),y:bottom+19,'text-anchor':i===0?'start':i===3?'end':'middle',class:'chart-tick'},Math.round(val).toLocaleString('en')));}
    canvas.append(svg('text',{x:right,y:height-5,'text-anchor':'end',class:'chart-tick'},tr('Step')));
    for(const s of curves){const stride=Math.max(1,Math.ceil(s.values.length/1200)),sample=s.values.filter((_,i)=>i%stride===0||i===s.values.length-1);canvas.append(svg('polyline',{points:sample.map(p=>x(p.step)+','+y(p.value)).join(' '),fill:'none',stroke:color(s.row.id),'stroke-width':'1.8','stroke-linejoin':'round',class:'metric-series','data-run':s.row.id}));if(s.values.length===1){const p=s.values[0];canvas.append(svg('circle',{cx:x(p.step),cy:y(p.value),r:3,fill:color(s.row.id)}));}}
    const cross=svg('line',{x1:left,y1:top,x2:left,y2:bottom,class:'chart-crosshair',visibility:'hidden'});canvas.append(cross);
    function inspect(step){
      cross.setAttribute('x1',x(step));cross.setAttribute('x2',x(step));cross.setAttribute('visibility','visible');tooltip.replaceChildren();const header=document.createElement('strong');header.textContent=tr('Step')+' '+Math.round(step);tooltip.append(header);
      for(const s of curves){const point=nearest(s.points,step),filtered=nearest(s.values,step),row=document.createElement('div'),dot=document.createElement('i'),text=document.createElement('span');dot.style.backgroundColor=color(s.row.id);text.textContent=runLabel(s.row)+': '+(smoothing?tr('Smoothed')+' '+number(filtered.value)+' / '+tr('Raw')+' ':'')+number(point[key])+' · '+point.step;row.append(dot,text);tooltip.append(row);}tooltip.hidden=false;
    }
    canvas.onpointermove=e=>{const box=canvas.getBoundingClientRect(),px=Math.min(right,Math.max(left,(e.clientX-box.left)*width/box.width));inspect(xmin+(px-left)/(right-left)*(xmax-xmin));};
    canvas.onpointerleave=()=>{tooltip.hidden=true;cross.setAttribute('visibility','hidden');};
    let cursor=0;canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Escape'].includes(e.key))return;e.preventDefault();if(e.key==='Escape'){canvas.onpointerleave();return;}cursor=Math.max(0,Math.min(curves[0].points.length-1,cursor+(e.key==='ArrowRight'?1:-1)));inspect(curves[0].points[cursor].step);};
    canvas.onblur=canvas.onpointerleave;
  }
  let resizeFrame;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>plots.forEach(draw=>draw()));}).observe($('chartsGrid'));
  $('chartSmoothing').oninput=()=>{smoothing=Number($('chartSmoothing').value);$('smoothingValue').value=smoothing.toFixed(2);localStorage.setItem('vla-chart-smoothing',smoothing);render();};
  $('compareAll').onclick=()=>{compared=new Set(runs.map(r=>r.id));saveCompared();renderSidebar();render();refresh();};
  $('compareNone').onclick=()=>{compared=new Set();saveCompared();renderSidebar();render();};
  window.TrainingCharts={setRuns,setCurrent,refresh,include};
})();
