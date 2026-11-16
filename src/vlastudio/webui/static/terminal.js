"use strict";
(() => {
  const button=document.getElementById('terminalOpen');
  if(!button)return;
  if(window.parent!==window&&window.parent.VLAWorkspace){
    Object.assign(chinese,{'Terminal':'终端','Copy model path':'复制模型路径','Copied successfully':'复制成功','Copy failed. Select the path and copy it manually.':'复制失败，请选中路径手动复制。'});
    const parent=window.parent;
    function update(){button.setAttribute('aria-expanded',String(parent.VLATerminal?.visible||false));button.title=button.ariaLabel=tr('Terminal');}
    button.onclick=()=>parent.VLATerminal?.toggle();
    parent.addEventListener('vla-terminal-visibility',update);parent.addEventListener('vla-terminal-ready',update);
    window.addEventListener('pagehide',()=>{parent.removeEventListener('vla-terminal-visibility',update);parent.removeEventListener('vla-terminal-ready',update);});
    document.addEventListener('click',e=>{
      const link=e.target.closest('a[href]');
      if(!link||e.defaultPrevented||e.button!==0||e.ctrlKey||e.metaKey||e.altKey||e.shiftKey||(link.target&&link.target!=='_self')||link.hasAttribute('download'))return;
      const url=new URL(link.href,location.href);
      if(url.origin!==location.origin||!['/collect','/infer','/data','/train'].includes(url.pathname))return;
      e.preventDefault();parent.VLAWorkspace.navigate(url.href);
    });
    update();return;
  }
  Object.assign(chinese,{'Terminal':'终端','Close terminal':'关闭终端','New terminal':'新建终端','Hide terminal panel':'收起终端面板','Resize terminal panel':'调整终端面板高度','No terminals open':'暂无终端，点击 + 新建终端','Up to 4 terminals':'最多同时打开 4 个终端','Connecting terminal…':'正在连接终端…','Terminal connected':'终端已连接','Terminal session ended':'终端会话已结束','Reopen terminal':'重新打开终端','Copy model path':'复制模型路径','Copied successfully':'复制成功','Copy failed. Select the path and copy it manually.':'复制失败，请选中路径手动复制。'});
  chinese['Terminal session has ended; reopen the terminal']='终端会话已失效，请重新打开终端';
  const panel=document.createElement('section');panel.id='terminalPanel';panel.hidden=true;
  panel.setAttribute('aria-labelledby','terminalTitle');
  panel.innerHTML='<div id="terminalResize" role="separator" tabindex="0" aria-orientation="horizontal" aria-controls="terminalPanel"></div><div class="terminal-heading"><h2 id="terminalTitle"></h2><div id="terminalTabs" role="tablist"></div><div class="terminal-controls"><button id="terminalNew" type="button">+</button><button id="terminalReconnect" type="button" hidden></button><button id="terminalClose" type="button">⌄</button></div></div><div id="terminalScreen"><p id="terminalEmpty"></p></div><div class="terminal-footer"><p id="terminalCwd"></p><p id="terminalStatus" role="status" aria-live="polite"></p></div>';
  document.body.append(panel);
  const screen=document.getElementById('terminalScreen'),status=document.getElementById('terminalStatus'),cwd=document.getElementById('terminalCwd'),tabs=document.getElementById('terminalTabs'),empty=document.getElementById('terminalEmpty'),reconnect=document.getElementById('terminalReconnect'),add=document.getElementById('terminalNew'),hide=document.getElementById('terminalClose'),handle=document.getElementById('terminalResize');
  button.setAttribute('aria-controls',panel.id);button.setAttribute('aria-expanded','false');
  const workspaceKey='vla-terminal-workspace-v1';
  const sessions=[];let active=null,serial=0,assets,resizeTimer,panelHeight=300,restoring=true,leaving=false,skipSave=false;
  let saved=null;
  try{const value=JSON.parse(sessionStorage.getItem(workspaceKey));if(value?.version===1&&Array.isArray(value.sessions))saved=value;}catch{}
  function persist(force=false){
    if(restoring||leaving&&!force)return;
    try{sessionStorage.setItem(workspaceKey,JSON.stringify({version:1,visible:!panel.hidden,active,serial,height:panelHeight,sessions:sessions.filter(s=>s.id&&!s.closed).map(s=>({id:s.id,number:s.number,cwd:s.cwd,cols:s.terminal?.cols||80,rows:s.terminal?.rows||24}))}));}catch{}
  }
  try{panelHeight=Number(localStorage.getItem('vla-terminal-height'))||300;}catch{}
  const selected=()=>sessions.find(s=>s.key===active);
  const title=s=>tr('Terminal')+' '+s.number;
  function iconLabel(element,text){element.title=tr(text);element.setAttribute('aria-label',tr(text));}
  function labels(){
    iconLabel(button,'Terminal');iconLabel(add,sessions.length>=4?'Up to 4 terminals':'New terminal');iconLabel(hide,'Hide terminal panel');iconLabel(handle,'Resize terminal panel');
    document.getElementById('terminalTitle').textContent=tr('Terminal');tabs.setAttribute('aria-label',tr('Terminal'));empty.textContent=tr('No terminals open');reconnect.textContent=tr('Reopen terminal');
    for(const s of sessions){s.tab.textContent=title(s);iconLabel(s.close,'Close terminal');s.close.setAttribute('aria-label',tr('Close terminal')+' '+s.number);}
    renderStatus();
  }
  function renderStatus(){const s=selected();status.textContent=s?tr(s.status):'';status.classList.toggle('error',Boolean(s?.error));cwd.textContent=s?.cwd||'';cwd.title=cwd.textContent;reconnect.hidden=!s||!s.error&&!s.ended;reconnect.disabled=Boolean(s?.opening);add.disabled=sessions.length>=4;iconLabel(add,sessions.length>=4?'Up to 4 terminals':'New terminal');}
  function show(s,text,error=false){if(s.closed)return;s.status=text;s.error=error;s.tab.classList.toggle('terminal-ended',Boolean(s.ended));if(s.key===active)renderStatus();}
  new MutationObserver(labels).observe(document.documentElement,{attributes:true,attributeFilter:['lang']});labels();
  async function request(action,payload={}){const response=await fetch('/api/terminal/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error||response.status);return data;}
  function script(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>{s.remove();reject(Error('Could not load terminal assets'));};document.head.append(s);});}
  async function load(){if(!assets)assets=(async()=>{await script('/vendor/xterm.js');await script('/vendor/addon-fit.js');})().catch(e=>{assets=null;throw e;});return assets;}
  function resize(){
    const s=selected();if(panel.hidden||!s?.terminal)return;
    s.fit.fit();
    s.dimensions={cols:s.terminal.cols,rows:s.terminal.rows};syncSize(s).catch(e=>show(s,e.message,true));
  }
  function syncSize(s){
    if(s.resizing)return s.resizing;
    if(!s.id||s.ended||s.closed||!s.dimensions)return Promise.resolve();
    const id=s.id;
    // Keep rapid drag updates in order and send the latest size before input.
    s.resizing=(async()=>{while(!s.closed&&id===s.id){const dimensions=s.dimensions,key=dimensions.cols+':'+dimensions.rows;if(key===s.size)break;await request('resize',{id,...dimensions});s.size=key;}})().finally(()=>{s.resizing=null;});
    return s.resizing;
  }
  function maximumHeight(){return Math.max(140,Math.min(innerHeight*.72,innerHeight-(document.querySelector('body > header')?.getBoundingClientRect().height||66)-90));}
  function setHeight(value,persist=false){
    panelHeight=Math.round(Math.max(Math.min(180,maximumHeight()),Math.min(value,maximumHeight())));
    panel.style.height=panelHeight+'px';document.body.style.setProperty('--terminal-panel-height',panel.hidden?'0px':panelHeight+'px');
    handle.setAttribute('aria-valuemin',Math.min(180,Math.round(maximumHeight())));handle.setAttribute('aria-valuemax',Math.round(maximumHeight()));handle.setAttribute('aria-valuenow',panelHeight);
    if(persist)try{localStorage.setItem('vla-terminal-height',String(panelHeight));}catch{}
    resize();
  }
  function select(s,focus=true){
    active=s.key;
    for(const item of sessions){const current=item===s;item.host.hidden=!current;item.tab.setAttribute('aria-selected',String(current));item.tab.tabIndex=current?0:-1;}
    renderStatus();resize();persist();if(focus&&!panel.hidden)s.terminal?.focus();
  }
  function setVisible(visible,createIfEmpty=true){
    panel.hidden=!visible;document.body.classList.toggle('terminal-visible',visible);button.setAttribute('aria-expanded',String(visible));
    setHeight(panelHeight);
    if(visible){if(!sessions.length&&createIfEmpty)create();else{resize();selected()?.terminal?.focus();}}else button.focus();
    persist();
    window.dispatchEvent(new CustomEvent('vla-terminal-visibility',{detail:{visible}}));
  }
  function flushInput(s){
    if(s.sending)return s.inputPromise;
    if(!s.id||s.closed||s.ended)return Promise.resolve();const id=s.id;s.sending=true;
    s.inputPromise=(async()=>{
      try{await syncSize(s);while(s.pending.length&&!s.closed&&id===s.id){await request('write',{id,data:s.pending.splice(0,1000).join('')});}}
      catch(e){show(s,e.message,true);}
      finally{s.sending=false;if(s.pending.length&&!s.closed&&!s.error&&!leaving)s.inputTimer=setTimeout(()=>flushInput(s),20);}
    })();return s.inputPromise;
  }
  async function poll(s,run,id){
    try{
      while(!s.closed&&run===s.generation&&id===s.id){
        const data=await request('read',{id,cursor:s.cursor});if(s.closed||run!==s.generation)return;
        if(data.truncated)s.terminal.write('\r\n[Earlier output truncated]\r\n');s.cursor=data.cursor;
        if(data.data)s.terminal.write(Uint8Array.from(atob(data.data),c=>c.charCodeAt(0)));
        if(data.exited){s.ended=true;show(s,'Terminal session ended');return;}
        await new Promise(resolve=>setTimeout(resolve,panel.hidden||active!==s.key?750:data.data?30:180));
      }
    }catch(e){if(!s.closed&&run===s.generation)show(s,e.message,true);}
  }
  async function connect(s,existing=null){
    const run=++s.generation;s.opening=true;s.ended=false;show(s,'Connecting terminal…');renderStatus();
    try{
      await load();if(s.closed||run!==s.generation)return;
      if(!s.terminal){
        s.terminal=new Terminal({cols:existing?.cols||80,rows:existing?.rows||24,cursorBlink:true,convertEol:false,fontFamily:'Consolas, monospace',fontSize:13,scrollback:5000,theme:{background:'#111b18',foreground:'#d8e6df',cursor:'#a1c8b2'},allowProposedApi:false});
        s.fit=new FitAddon.FitAddon();s.terminal.loadAddon(s.fit);s.terminal.open(s.host);
        s.terminal.onData(data=>{if(!s.id||s.ended||s.closed)return;s.pending.push(...Array.from(data));clearTimeout(s.inputTimer);s.inputTimer=setTimeout(()=>flushInput(s),20);});
      }
      if(!panel.hidden&&active===s.key)s.fit.fit();
      const old=s.id;
      if(!existing){s.id=null;persist();if(old)await request('close',{id:old});}
      if(s.closed||run!==s.generation)return;
      let session;
      if(existing){await request('read',{id:existing.id,cursor:0});session=existing;}
      else session=await request('open',{cols:s.terminal.cols,rows:s.terminal.rows});
      if(s.closed||run!==s.generation){if(!existing)request('close',{id:session.id}).catch(()=>{});return;}
      s.id=session.id;s.cursor=0;s.cwd=session.cwd;s.pending=[];s.size='';
      persist();
      show(s,'Terminal connected');if(active===s.key){resize();if(!panel.hidden)s.terminal.focus();}poll(s,run,s.id);
    }catch(e){if(!s.closed&&run===s.generation)show(s,e.message,true);}
    finally{if(!s.closed&&run===s.generation){s.opening=false;renderStatus();}}
  }
  function create(existing=null){
    if(sessions.length>=4)return;
    const number=existing?existing.number:serial+1;serial=Math.max(serial,number);
    const s={key:'terminal-'+number,number,id:existing?.id||null,cursor:0,generation:0,pending:[],sending:false,closed:false,ended:false,opening:false,cwd:existing?.cwd||'',status:'',error:false};
    s.host=document.createElement('div');s.host.className='terminal-session';s.host.id=s.key;s.host.setAttribute('role','tabpanel');s.host.setAttribute('aria-labelledby',s.key+'-tab');
    const row=document.createElement('div');row.className='terminal-tab';s.row=row;
    s.tab=document.createElement('button');s.tab.type='button';s.tab.id=s.key+'-tab';s.tab.setAttribute('role','tab');s.tab.setAttribute('aria-controls',s.host.id);s.tab.onclick=()=>select(s);
    s.close=document.createElement('button');s.close.type='button';s.close.className='terminal-tab-close';s.close.textContent='×';s.close.onclick=()=>close(s);
    s.tab.onkeydown=e=>{const index=sessions.indexOf(s),next=e.key==='ArrowRight'?(index+1)%sessions.length:e.key==='ArrowLeft'?(index+sessions.length-1)%sessions.length:e.key==='Home'?0:e.key==='End'?sessions.length-1:null;if(next!==null){e.preventDefault();select(sessions[next],false);sessions[next].tab.focus();}};
    row.append(s.tab,s.close);tabs.append(row);screen.append(s.host);sessions.push(s);empty.hidden=true;labels();select(s);s.connection=connect(s,existing);
  }
  function close(s){
    if(s.closed)return;s.closed=true;++s.generation;clearTimeout(s.inputTimer);s.pending=[];
    const index=sessions.indexOf(s);sessions.splice(index,1);s.row.remove();s.host.remove();s.terminal?.dispose();
    const id=s.id;s.id=null;if(id)request('close',{id}).catch(()=>{});
    if(active===s.key){active=null;if(sessions.length)select(sessions[Math.min(index,sessions.length-1)]);}
    empty.hidden=sessions.length>0;labels();persist();if(!sessions.length)add.focus();
  }
  new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(resize,80);}).observe(screen);
  const transport=document.getElementById('transport');
  if(transport)new ResizeObserver(()=>{document.body.style.setProperty('--terminal-transport-space',(transport.offsetHeight+16)+'px');}).observe(transport);
  window.addEventListener('resize',()=>{if(!panel.hidden)setHeight(panelHeight);});
  let drag=null;
  handle.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();drag={y:e.clientY,height:panelHeight};handle.setPointerCapture(e.pointerId);document.body.classList.add('terminal-resizing');});
  handle.addEventListener('pointermove',e=>{if(drag)setHeight(drag.height+drag.y-e.clientY);});
  function endDrag(){if(!drag)return;drag=null;document.body.classList.remove('terminal-resizing');setHeight(panelHeight,true);persist();}
  handle.addEventListener('pointerup',endDrag);handle.addEventListener('pointercancel',endDrag);handle.addEventListener('lostpointercapture',endDrag);
  handle.addEventListener('keydown',e=>{const step=e.shiftKey?50:20,value=e.key==='ArrowUp'?panelHeight+step:e.key==='ArrowDown'?panelHeight-step:e.key==='Home'?180:e.key==='End'?maximumHeight():null;if(value!==null){e.preventDefault();setHeight(value,true);}});
  button.onclick=()=>setVisible(panel.hidden);hide.onclick=()=>setVisible(false);add.onclick=()=>create();reconnect.onclick=()=>{const s=selected();if(s&&!s.opening)s.connection=connect(s);};
  // A route change detaches this view; only the explicit × closes its shell.
  window.addEventListener('pagehide',()=>{
    if(skipSave)return;persist(true);leaving=true;
    for(const s of sessions){++s.generation;clearTimeout(s.inputTimer);}
  });
  window.addEventListener('pageshow',e=>{if(e.persisted){skipSave=true;location.reload();}});
  let navigating=false;
  document.addEventListener('click',e=>{
    const link=e.target.closest('a[href]');
    if(!link||e.defaultPrevented||e.button!==0||e.ctrlKey||e.metaKey||e.altKey||e.shiftKey||link.target&&link.target!=='_self'||link.hasAttribute('download'))return;
    const url=new URL(link.href,location.href);
    if(url.origin!==location.origin||!['/collect','/infer','/data','/train'].includes(url.pathname))return;
    if(!sessions.some(s=>s.opening||s.sending||s.pending.length))return;
    e.preventDefault();if(navigating)return;navigating=true;
    Promise.all(sessions.map(async s=>{await s.connection;clearTimeout(s.inputTimer);await flushInput(s);})).then(()=>{persist();location.assign(url.href);}).catch(()=>{navigating=false;});
  });
  if(saved){
    const numbers=new Set(),ids=new Set();
    for(const item of saved.sessions.slice(0,4)){
      if(!item||typeof item.id!=='string'||item.id.length>128||!Number.isSafeInteger(item.number)||item.number<1||numbers.has(item.number)||ids.has(item.id))continue;
      numbers.add(item.number);ids.add(item.id);create({id:item.id,number:item.number,cwd:typeof item.cwd==='string'?item.cwd:'',cols:Number.isInteger(item.cols)&&item.cols>=10&&item.cols<=500?item.cols:80,rows:Number.isInteger(item.rows)&&item.rows>=2&&item.rows<=200?item.rows:24});
    }
    serial=Math.max(serial,Number.isSafeInteger(saved.serial)?saved.serial:0);
    if(Number.isFinite(saved.height))panelHeight=saved.height;
    const current=sessions.find(s=>s.key===saved.active)||sessions[0];if(current)select(current,false);
    if(saved.visible===true)setVisible(true,false);
  }
  restoring=false;persist();
  window.VLATerminal={toggle:()=>setVisible(panel.hidden),get visible(){return !panel.hidden;}};
  window.dispatchEvent(new Event('vla-terminal-ready'));
})();
