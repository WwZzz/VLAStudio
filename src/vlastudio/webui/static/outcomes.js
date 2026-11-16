"use strict";
Object.assign(chinese,{'Outcome':'结果','Unlabeled':'未标注','Success':'成功','Failure':'失败','All':'全部','Result updated':'结果已更新','Success and save (1)':'成功并保存（1）','Failure and save (2)':'失败并保存（2）','Save or cancel your annotation edits first.':'请先保存详细标注中的修改，再快捷修改结果。','Recording saved; label was not saved. Update its result in Data management.':'轨迹已保存，但结果标注失败，请在数据管理中补标。','Save confirmation timed out. Check this episode in Data management before retrying.':'保存确认超时，请先在数据管理检查该轨迹。'});
window.VLAOutcomes={
  async list(dataset){
    const query='?dataset='+encodeURIComponent(dataset);
    const [response,labels]=await Promise.all([fetch('/api/episodes'+query),this.request('outcomes'+query)]);
    const rows=await response.json();if(!response.ok)throw Error(rows.error||response.status);
    return rows.map(row=>({...row,...labels.find(r=>r.index===row.index&&r.version===row.version)}));
  },
  label:value=>tr(({unknown:'Unlabeled',success:'Success',failure:'Failure'})[value]||'Unlabeled'),
  async request(path,body){
    const response=await fetch('/api/episode/'+path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
    const result=await response.json();if(!response.ok)throw Error(result.error||response.status);return result;
  },
  filter(list,render){
    const group=document.createElement('div');group.className='outcome-filter';group.setAttribute('role','group');group.setAttribute('aria-label',tr('Outcome'));group.value='';
    for(const value of ['','success','failure','unknown']){
      const button=document.createElement('button');button.type='button';button.textContent=value?this.label(value):tr('All');button.dataset.value=value;
      button.setAttribute('aria-pressed',String(!value));
      button.onclick=()=>{group.value=value;for(const b of group.children)b.setAttribute('aria-pressed',String(b===button));render();};group.append(button);
    }
    list.before(group);return group;
  },
  episodeButton(button,item){
    const title=document.createElement('span');title.className='episode-row-title';title.textContent='Episode '+String(item.index).padStart(4,'0');
    const stamp=document.createElement('small');const date=new Date(item.saved_at);
    stamp.textContent=Number.isNaN(date.getTime())?'—':new Intl.DateTimeFormat(language==='zh'?'zh-CN':'en-GB',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).format(date);
    stamp.title=Number.isNaN(date.getTime())?'':date.toLocaleString();
    button.replaceChildren(title);
    if(item.recording_source==='inference'){const badge=document.createElement('span');badge.className='recording-source inference';badge.textContent='infer';badge.title=tr('Inference recording');button.append(badge);}
    button.append(stamp);
  },
  picker(anchor,choose){
    if(!this.popup){
      const popup=document.createElement('div');popup.id='outcomePicker';popup.setAttribute('popover','auto');popup.setAttribute('role','group');popup.setAttribute('aria-label',tr('Outcome'));document.body.append(popup);this.popup=popup;
      popup.addEventListener('toggle',e=>{if(e.newState==='closed')this.pickerAnchor?.setAttribute('aria-expanded','false');});
      window.addEventListener('resize',()=>popup.hidePopover());
      document.addEventListener('scroll',()=>popup.hidePopover(),true);
    }
    const popup=this.popup;popup.hidePopover();this.pickerAnchor=anchor;popup.replaceChildren();
    for(const value of ['success','failure','unknown']){
      const option=document.createElement('button');option.type='button';option.className='outcome-swatch '+value;option.dataset.value=value;option.textContent=value==='success'?'S':value==='failure'?'F':'';
      option.title=option.ariaLabel=this.label(value);option.onclick=()=>{popup.hidePopover();anchor.focus();choose(value);};popup.append(option);
    }
    const rect=anchor.getBoundingClientRect();popup.style.left=Math.max(8,Math.min(innerWidth-84,rect.right-76))+'px';popup.style.top=(rect.bottom+34<innerHeight?rect.bottom+5:rect.top-33)+'px';popup.showPopover();anchor.setAttribute('aria-expanded','true');popup.firstElementChild.focus({preventScroll:true});
  },
  control(item,updated,failed,canEdit=()=>true){
    const button=document.createElement('button');button.type='button';button.setAttribute('aria-haspopup','true');button.setAttribute('aria-expanded','false');
    let saving=false;
    const sync=()=>{const value=item.outcome||'unknown';button.className='outcome-badge outcome-swatch '+value;button.dataset.outcome=value;button.textContent=value==='success'?'S':value==='failure'?'F':'';button.title=button.ariaLabel='Episode '+item.index+' · '+this.label(value);button.disabled=saving;};
    const change=async outcome=>{
      if(saving||!canEdit()||outcome===(item.outcome||'unknown'))return;
      button.closest('.episode-outcome-row')?.querySelector('.outcome-error')?.remove();saving=true;sync();
      try{
        const result=await this.request('outcome',{dataset:item.dataset,index:item.index,version:item.version,revision:item.annotation_revision??null,outcome});
        item.outcome=outcome;item.annotation_revision=result.document.revision;sync();await updated(item,result);
      }catch(e){const row=button.closest('.episode-outcome-row');if(row){const error=document.createElement('span');error.className='outcome-error';error.setAttribute('role','status');error.textContent=language==='zh'?'未保存：'+tr(e.message):'Not saved: '+tr(e.message);row.append(error);}failed(e.message);}finally{saving=false;sync();}
    };
    button.onclick=e=>{e.stopPropagation();if(!saving)this.picker(button,change);};sync();return button;
  }
};
