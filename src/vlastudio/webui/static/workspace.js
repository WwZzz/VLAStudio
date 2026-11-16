"use strict";
(() => {
  const routes=new Set(['/collect','/infer','/data','/train']);
  const pages=new Map(), header=document.querySelector('body > header');
  let current='',active;
  function publish(record) {
    if(!record || record!==active || !record.ready)return;
    const doc=record.frame.contentDocument;
    document.title=doc.title;
    const source=doc.getElementById('link'), target=document.getElementById('link');
    if(source){target.textContent=source.textContent;target.className=source.className;}
    const theme=doc.getElementById('themeToggle'), button=document.getElementById('themeToggle');
    if(theme){button.innerHTML=theme.innerHTML;button.title=button.ariaLabel=theme.title;button.setAttribute('aria-pressed',theme.getAttribute('aria-pressed'));}
  }
  function labels(){
    language=localStorage.getItem('vla-language')==='zh'?'zh':'en';
    document.documentElement.lang=language==='zh'?'zh-CN':'en';
    document.getElementById('language').value=language;
    header.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=tr(el.dataset.i18n));
    document.getElementById('settingsOpen').title=document.getElementById('settingsOpen').ariaLabel=tr('Settings');
    window.VLATheme.apply();
  }
  function modalOpen(){
    const modal=active?.frame.contentDocument?.querySelector('dialog[open]');
    if(modal){active.frame.contentWindow.focus();modal.focus();return true;}
    return false;
  }
  function navigate(value,push=true){
    const url=new URL(value,location.href);
    if(url.origin!==location.origin||!routes.has(url.pathname))return;
    url.searchParams.delete('workspace_frame');
    const path=url.pathname+url.search+url.hash;
    if(path===current)return;
    if(push&&modalOpen())return;
    const key=url.pathname+url.search;
    let record=pages.get(key);
    if(active && active!==record){
      const previous=active;active=null;
      previous.frame.contentWindow.document.dispatchEvent(new Event('vla-viewchange'));
      previous.frame.hidden=true;previous.frame.removeAttribute('id');
    }
    current=path;
    if(push)history.pushState(null,'',path);
    header.querySelectorAll('nav a').forEach(link=>{
      const selected=new URL(link.href).pathname===url.pathname;
      link.classList.toggle('active',selected);
      if(selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
    });
    if(!record){
      const frame=pages.size ? document.createElement('iframe') : document.getElementById('workspaceFrame');
      frame.className='workspace-frame';frame.title=tr(header.querySelector('nav a.active').dataset.i18n);
      frame.allow='clipboard-read; clipboard-write; fullscreen; xr-spatial-tracking';
      record={frame,ready:false,invalidated:false};pages.set(key,record);active=record;
      frame.addEventListener('load',()=>{
        const doc=frame.contentDocument;if(!doc?.querySelector('main'))return;
        record.ready=true;
        record.observer?.disconnect();
        record.observer=new MutationObserver(()=>publish(record));
        for(const node of [doc.querySelector('title'),doc.querySelector('body > header')])
          if(node)record.observer.observe(node,{subtree:true,childList:true,characterData:true,attributes:true});
        doc.addEventListener('settingschanged',()=>{
          for(const other of pages.values())if(other!==record){
            if(other.ready)other.frame.contentDocument.dispatchEvent(new Event('vla-settingsinvalidated'));
            else other.invalidated=true;
          }
        });
        if(record.invalidated){
          record.invalidated=false;
          doc.dispatchEvent(new Event('vla-settingsinvalidated'));
        }
        publish(record);
        doc.dispatchEvent(new Event('vla-viewchange'));
      });
      if(!frame.isConnected)document.getElementById('workspaceViews').append(frame);
      const embedded=new URL(path,location.origin);embedded.searchParams.set('workspace_frame','1');
      frame.contentWindow.location.replace(embedded.href);
    }else{
      active=record;record.frame.hidden=false;
      if(record.ready)record.frame.contentDocument.dispatchEvent(new Event('vla-viewchange'));
      publish(record);
    }
    record.frame.id='workspaceFrame';record.frame.hidden=false;
    if(record.ready){
      const embedded=new URL(path,location.origin);embedded.searchParams.set('workspace_frame','1');
      record.frame.contentWindow.history.replaceState(null,'',embedded.href);
      if(url.hash)record.frame.contentDocument.getElementById(decodeURIComponent(url.hash.slice(1)))?.scrollIntoView();
    }
  }
  window.VLAWorkspace={navigate,isActive:win=>active?.frame.contentWindow===win};
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href]');
    if(!link||event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||(link.target&&link.target!=='_self')||link.hasAttribute('download'))return;
    const url=new URL(link.href,location.href);
    if(url.origin!==location.origin||!routes.has(url.pathname))return;
    event.preventDefault();navigate(url.href);
  });
  for(const id of ['settingsOpen','themeToggle'])document.getElementById(id).onclick=()=>{
    if(active?.ready&&!modalOpen())active.frame.contentDocument.getElementById(id).click();
  };
  document.getElementById('language').onchange=event=>{
    if(!active?.ready||modalOpen()){labels();return;}
    const select=active.frame.contentDocument.getElementById('language');
    select.value=event.target.value;select.dispatchEvent(new Event('change'));
    labels();
  };
  window.addEventListener('popstate',()=>navigate(location.href,false));
  window.addEventListener('storage',event=>{if(['vla-language','vla-theme'].includes(event.key)){labels();publish(active);}});
  window.addEventListener('vla-terminal-visibility',event=>{if(!event.detail.visible)document.getElementById('terminalOpen').focus();});
  labels();navigate(routes.has(location.pathname)?location.href:'/collect',false);
})();
