"use strict";
(() => {
  Object.assign(chinese, {
    "Find in log":"搜索日志", "Search log":"搜索日志", "Previous match":"上一个匹配", "Next match":"下一个匹配", "Close search":"关闭搜索",
    "Search full log…":"搜索完整日志…", "Loading full log…":"正在读取完整日志…", "No matches":"没有匹配", "Enter a keyword to search the full log.":"输入关键词以搜索完整日志。",
    "Showing context around the selected match.":"正在显示所选匹配位置的上下文。", "Too many matches; refine your keyword.":"匹配过多，请缩小关键词范围。"
  });
  const byId=id=>document.getElementById(id), panel=byId("logPanel"), output=byId("log"), bar=byId("logSearchBar"), query=byId("logQuery");
  let runId=null, preview="", expectedBytes=0, raw="", text="", loaded=0, decoder=new TextDecoder(), matches=[], index=-1;
  let loading=false, request=null, revision=0, error="", truncated=false, savedScroll=0;
  const clean=value=>value.replace(/\x1b\[[0-9;]*[A-Za-z]/g,"").replace(/\r\n?/g,"\n");
  function labels() {
    byId("findLog").title=tr("Find in log")+" (Ctrl+F)";
    query.placeholder=tr("Search full log…");query.ariaLabel=tr("Search log");
    for(const [id,key,shortcut] of [["logPrevious","Previous match","Shift+Enter"],["logNext","Next match","Enter"],["logSearchClose","Close search","Esc"]]) {
      byId(id).ariaLabel=tr(key);byId(id).title=tr(key)+" ("+shortcut+")";
    }
    status();
  }
  function status() {
    byId("logMatchCount").textContent=query.value?(matches.length?(index+1)+" / "+matches.length+(truncated?"+":""):tr("No matches")):"";
    byId("logPrevious").disabled=byId("logNext").disabled=!matches.length;
    byId("logSearchState").textContent=error||(loading?tr("Loading full log…"):truncated?tr("Too many matches; refine your keyword."):matches.length?tr("Showing context around the selected match."):query.value?"":tr("Enter a keyword to search the full log."));
  }
  function displayPreview(follow=true) {
    const pinned=output.scrollHeight-output.scrollTop-output.clientHeight<40;
    if(output.textContent!==preview){output.textContent=preview;if(follow&&pinned)output.scrollTop=output.scrollHeight;}
  }
  function drawMatch(scroll=true) {
    status();
    if(index<0||!matches.length){displayPreview(false);return;}
    const selected=matches[index];
    let start=Math.max(0,selected.start-1600),end=Math.min(text.length,selected.end+4000);
    const lineStart=text.lastIndexOf("\n",start);if(lineStart>=0&&start-lineStart<200)start=lineStart+1;
    const fragment=document.createDocumentFragment();if(start)fragment.append(document.createTextNode("…\n"));
    let cursor=start;
    for(let n=0;n<matches.length;n++){
      const match=matches[n];if(match.end<=start)continue;if(match.start>=end)break;
      fragment.append(document.createTextNode(text.slice(cursor,Math.max(cursor,match.start))));
      const mark=document.createElement("mark");mark.textContent=text.slice(Math.max(start,match.start),Math.min(end,match.end));
      if(n===index){mark.className="current-match";mark.setAttribute("aria-current","true");}
      fragment.append(mark);cursor=Math.min(end,match.end);
    }
    fragment.append(document.createTextNode(text.slice(cursor,end)+(end<text.length?"\n…":"")));output.replaceChildren(fragment);
    if(scroll){const mark=output.querySelector(".current-match");if(mark)output.scrollTop+=mark.getBoundingClientRect().top-output.getBoundingClientRect().top-output.clientHeight/2;}
  }
  function find(preserve=false) {
    const previous=preserve&&index>=0?matches[index]?.start:null;
    matches=[];truncated=false;
    if(query.value){
      const literal=query.value.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),pattern=new RegExp(literal,"giu");let match;
      while((match=pattern.exec(text))!==null){
        if(matches.length>=20000){truncated=true;break;}
        matches.push({start:match.index,end:match.index+match[0].length});
      }
    }
    index=matches.length?Math.max(0,previous===null?0:matches.findIndex(m=>m.start===previous)):-1;
    drawMatch(!preserve);
  }
  async function load() {
    if(loading||bar.hidden||!runId)return;
    loading=true;error="";status();const token=revision,id=runId;request=new AbortController();
    let goal=expectedBytes||Infinity;
    try{
      do{
        const response=await fetch('/api/training/log?'+new URLSearchParams({id}),{headers:{Range:'bytes='+loaded+'-'+(loaded+1048575)},cache:'no-store',signal:request.signal});
        if(response.status===416&&response.headers.get('Content-Range')==='bytes */0'){goal=0;break;}
        if(!response.ok){let message=response.statusText;try{message=(await response.json()).error||message;}catch{}throw Error(message);}
        const range=/bytes (\d+)-(\d+)\/(\d+)/.exec(response.headers.get('Content-Range')||'');
        const bytes=new Uint8Array(await response.arrayBuffer());
        if(token!==revision||bar.hidden||id!==runId)return;
        if(response.status===200){raw="";loaded=0;decoder=new TextDecoder();}
        raw+=decoder.decode(bytes,{stream:true});loaded+=bytes.byteLength;text=clean(raw);
        const total=range?Number(range[3]):loaded;goal=Math.min(goal,total);find(true);
        if(!bytes.byteLength)break;
      }while(loaded<goal);
    }catch(e){if(e.name!=='AbortError'&&token===revision){error=e.message;status();}}
    finally{if(token===revision){loading=false;request=null;status();}}
  }
  function clearCache(){revision++;request?.abort();request=null;loading=false;raw=text="";loaded=0;decoder=new TextDecoder();matches=[];index=-1;error="";}
  function open() {
    if(!runId)return;
    if(bar.hidden){savedScroll=output.scrollTop;bar.hidden=false;}
    const selection=window.getSelection();if(selection&&!selection.isCollapsed&&panel.contains(selection.anchorNode)){const value=selection.toString();if(value.length<=200&&!value.includes("\n"))query.value=value;}
    query.focus();query.select();find(true);load();
  }
  function close(){bar.hidden=true;revision++;request?.abort();request=null;loading=false;output.textContent=preview;output.scrollTop=savedScroll;output.focus();}
  function move(direction){if(matches.length){index=(index+direction+matches.length)%matches.length;drawMatch();}}
  byId("findLog").onclick=open;byId("logSearchClose").onclick=close;byId("logPrevious").onclick=()=>move(-1);byId("logNext").onclick=()=>move(1);
  query.oninput=()=>find();
  query.onkeydown=event=>{if(event.isComposing)return;if(event.key==='Enter'){event.preventDefault();move(event.shiftKey?-1:1);}else if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close();}};
  panel.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='f'){event.preventDefault();open();}else if(event.key==='Escape'&&!bar.hidden){event.preventDefault();close();}});
  window.TrainingLog={labels,update(info){
    const changed=runId!==info.id;
    if(changed){clearCache();runId=info.id;}
    if(!changed&&info.bytes!==undefined&&info.bytes<expectedBytes)clearCache();
    preview=info.text;expectedBytes=info.bytes??0;
    if(bar.hidden)displayPreview();else if(changed||loaded<expectedBytes||!loaded)load();
  }};
  labels();
})();
